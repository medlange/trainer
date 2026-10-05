# SPDX-License-Identifier: Apache-2.0
"""`/usr/local/bin/medos-trainer` -- the process `spec.argv` points at.

FIVE SUBCOMMANDS, AND WHY EACH ONE IS A SEPARATE PROCESS

    plan                 phase 1. Derives the fingerprint from the FIT partition and
                         writes it. Exits before the platform moves the row to `RUNNING`,
                         which is what makes `MOS-TRAIN-135`'s freeze a fact and not a
                         race.
    fit                  phase 2. Trains against the frozen plan and writes the bundle.
    declare-environment  the nine keys of `MEDOS_TRAINING_ENVIRONMENT`, observed. Run as
                         a one-shot before the API starts; see docker-compose.yml.
    execute              the supervisor: pick up runs a PERSON submitted and drive them.
    doctor               what this image can and cannot do right now, as a document. It
                         is the answer to "is the GPU attached" that does not require
                         starting a training run to find out.

EVERY EXIT WRITES A REASON. `medos.training.runs.fail` refuses an empty one
(`MOS-REL-051`: "a FAILED training run MUST name a reason"), and the reason has to
survive the process, so the two phases write `result.json` BEFORE they exit -- including
when they are failing, including when they are failing on a contract violation. The exit
code is a summary of what is in that file.

NOTHING HERE TAKES A COHORT ARGUMENT. There is no `--split`, no `--partition`, no
`--dataset`, no `--bucket` and no `--glob` on any subcommand: chapter 17 acceptance check
15 requires the training container's resolver to reject a filesystem path, a bucket prefix
and a glob as a cohort argument, and the strongest form of that is having nowhere to put
one. `--run-dir` names a directory the PLATFORM created and whose contents the platform
wrote.

Spec: MOS-TRAIN-121, MOS-TRAIN-122, MOS-TRAIN-135, MOS-TRAIN-141, MOS-REL-051,
MOS-SEC-033, chapter 17 acceptance check 15.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from medos.sdk.contract import (
    RUN_DIRECTORY,
    ContractViolation,
    RunDirectory,
    failure_document,
    success_document,
)

# SAFE AT MODULE SCOPE, and that is a property of `port.py` rather than a convenience: it
# imports no torch, no nnU-Net and no backend, so importing it cannot pull `nnunetv2` and
# bind the three roots `nnunetv2/paths.py` reads at import time. Every other trainer import
# in this file is deliberately inside a function for exactly that reason.
from medos_trainer import port

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _logging() -> None:
    logging.basicConfig(
        level=os.environ.get("MEDOS_TRAINER_LOG_LEVEL", "INFO").upper(),
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )


def _refusals_of(exc: BaseException) -> list[dict[str, Any]]:
    """`medos.sdk.errors.Refusal`s, as documents, when the failure was one.

    The console renders the engine's refusal vocabulary already (`MOS-API-112` pins the
    wire form). A refusal raised inside the trainer is the same kind of fact as one raised
    at submit, and rendering it as a second kind of error would make the operator learn
    two vocabularies for one thing.
    """
    refusals = getattr(exc, "refusals", None)
    if not refusals:
        return []
    out: list[dict[str, Any]] = []
    for refusal in refusals:
        as_dict = getattr(refusal, "as_dict", None)
        out.append(dict(as_dict()) if callable(as_dict) else {"message": str(refusal)})
    return out


def _produced_by() -> dict[str, Any]:
    """The build stamp, for the result document. WHICH SOFTWARE WROTE THIS.

    THE DEFECT THIS CLOSES COST AN AFTERNOON. A source tree was fixed and the OLD image
    was run -- twice -- and both times it was noticed only by a side effect: the results
    folder was named after the trainer class that should no longer have been constructed.
    The image knows its own commit, its own dirty flag and its own inventory digest and
    has since `stamp.py` was written; nothing wrote them into the artifact, so "what
    trained this" had to be inferred from a directory name instead of read.

    RECORDED, NOT ASSERTED -- the same reading `MOS-TRAIN-126` applies to the binding. A
    stamp that cannot be read is reported as an error in place of the values, exactly as
    `doctor` reports it, and never silently omitted: a missing key and a key that says
    "this ran outside an image" are different facts and only one of them is alarming.
    """
    from medos_trainer.stamp import read_stamp

    try:
        stamp = read_stamp()
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    # `image_inventory` is the per-file listing the digest is taken over -- thousands of
    # entries. The digest is the identifier; the listing is how it was computed, and it
    # stays in the image where `doctor` can print it.
    return {k: v for k, v in stamp.items() if k != "image_inventory"}


def _write_result(run: RunDirectory, document: dict[str, Any]) -> None:
    """The phase's record: to `result.json` AND to a file only this phase writes.

    ONE WRITER, so the stamp cannot be forgotten. Three call sites write a result -- two
    successes and the failure path -- and a member added to two of three is a provenance
    record that is absent exactly when a run went wrong, which is when it is read.

    AND TWO DESTINATIONS, BECAUSE ONE WAS LOSING HALF THE RECORD. Both phases wrote
    `result.json`; the fit overwrote the plan's. So after a completed run the question
    "which image derived this plan" had no answer in the run directory -- the same
    question `produced_by` was added to stop inferring from a directory name. It came up
    for real when the plans had to be re-derived after an nnU-Net bump.

    `result.json` keeps its meaning: the latest phase, which is what the executor polls
    and what `medos.training.runs` reads. `result-<phase>.json` is the history. A phase
    that is neither `plan` nor `fit` writes only `result.json` rather than inventing a
    member name -- the run directory is a closed set of paths, and a file nobody declared
    is a file nobody reads.
    """
    stamped = {**document, "produced_by": _produced_by()}
    run.write("result", stamped)
    role = f"result_{document.get('phase')}"
    if role in RUN_DIRECTORY:
        run.write(role, stamped)


def _parse_bound_spec(run: RunDirectory) -> None:
    """`preprocessing.json` through the platform's own parser, before a GPU is taken.

    Raises whatever `parse_spec` raises, unwrapped. The parser's refusal already names
    the field and the requirement behind it, and `_refusals_of` renders it into
    `result.json` in the vocabulary `MOS-API-112` pins; a second sentence from here would
    be a second vocabulary for one fact.

    The import is local because `medos.sdk.spec` pulls the platform package, and
    `contract.py` -- which this module leans on for everything else -- deliberately does
    not. Keeping the dependency inside the one function that needs it is what lets the
    run-directory contract stay pure Python over one directory.
    """
    from medos.sdk.spec import parse_spec

    parse_spec(run.spec_document())


def _phase(phase: str, run_dir: str) -> int:
    """Run one phase, and write `result.json` whatever happens."""
    run = RunDirectory(run_dir)
    try:
        request = run.request()
        # THE PORT DECIDES WHICH DRIVER RUNS, and it is the only thing that does. This was
        # a bare `!= "nnunet"` against a string literal, with no test tying it to
        # `medos_trainer.BACKEND_KIND` -- so the declaration the image makes about itself
        # and the refusal it enforces could disagree, and the refusal could not name what
        # the image does implement.
        try:
            backend = port.resolve(request.backend_kind)
        except LookupError as exc:
            raise ContractViolation(str(exc)) from exc

        # THE SPEC IS PARSED HERE, WHERE REFUSING IT IS FREE.
        #
        # `preprocessing.json` is the deployment's registered `PreprocessingSpec` -- the
        # template `packaging.derive_spec_document` merges the derived fields into. Until
        # this call existed the FIRST thing to parse it was that merge, which runs after
        # the fit: a template missing `backend`, `inverse` or `golden_fixture` was
        # refused at the END of a run, having already spent the preprocessing and every
        # epoch. That is not hypothetical -- a run reached "Training done." and died on
        # `KeyError: 'backend'`, and the answer had been sitting in the run directory
        # since before the first volume was read.
        #
        # WHAT IT DOES NOT PROVE, SAID PLAINLY. `plan["spec_fields"]` overwrites the
        # spacing, the patch and the normalisation, and `derive_spec_document` rewrites
        # `backend` and `golden_fixture` wholesale. So this parse is WEAKER than the one
        # at the end and does not replace it. It proves the part that is the
        # deployment's own -- `io`, `inverse`, the geometry, the label set -- which no
        # derivation supplies and which therefore cannot become valid later.
        _parse_bound_spec(run)


        # BEFORE anything imports `nnunetv2`. `nnunetv2/paths.py` reads its three roots
        # from the environment AT MODULE IMPORT and binds them to constants, so setting
        # them later has no effect and the run writes its dataset into the image's
        # default root instead of into this run's directory. `backend.py` imports
        # nnunetv2 only inside `derive_plan` and `fit`, which is what makes this call
        # site early enough -- and why it is a separate function with a docstring rather
        # than three `os.environ` lines somewhere in the middle of the backend.
        work = run.root / "work"
        backend.prepare_workspace(work)
        determinism = backend.apply_determinism(request)

        if phase == "plan":
            plan = backend.derive_plan(run, request, work=work)
            run.write("plan", plan)
            _write_result(run, success_document(
                "plan",
                fingerprint_digest=plan["fingerprint_digest"],
                plan_digest=plan["fingerprint_digest"],
                hyperparameters=plan["hyperparameters"],
                fit_cases=plan["fit_cases"],
                select_cases=plan["select_cases"],
                determinism=determinism,
            ))
            return 0

        plan_path = run.path("plan")
        if not plan_path.is_file():
            raise ContractViolation(
                f"{plan_path.name} is absent, so no plan was frozen. MOS-TRAIN-135 "
                "requires the derived plan to be frozen at run START; this phase does "
                "not derive one, because MOS-TRAIN-225's argument is that a "
                "re-derivation is invisible and the only defence that holds is having no "
                "code path that can produce one"
            )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        # WHICH BACKEND FROZE THIS PLAN. `plan.json` has always recorded it and nothing has
        # ever read it back. In a one-backend image that is merely unused; with a second
        # one it is the difference between fitting against your own frozen plan and
        # fitting against somebody else's, with every structural check passing --
        # MOS-TRAIN-225's invisible re-derivation arriving through a different door.
        froze = str(dict(plan.get("backend") or {}).get("kind", ""))
        if froze != request.backend_kind:
            raise ContractViolation(
                f"{plan_path.name} was frozen by backend {froze!r} and this run binds "
                f"{request.backend_kind!r}. MOS-TRAIN-135 freezes the plan at run start so "
                "that the fit cannot re-derive one; a plan another backend derived is a "
                "re-derivation that already happened, elsewhere"
            )
        return _fit(run, request, plan, determinism, backend)
    except BaseException as exc:  # noqa: BLE001 - every exit records a reason
        _write_result(run, failure_document(
            phase,
            reason=f"{type(exc).__name__}: {exc}",
            refusals=_refusals_of(exc),
        ))
        traceback.print_exc()
        return 1


def _fit(
    run: RunDirectory, request: Any, plan: dict[str, Any], determinism: dict[str, Any],
    backend: Any,
) -> int:
    from medos_trainer import packaging

    # TYPED AT THE BOUNDARY, not indexed seven times below. An absent member used to raise
    # `KeyError` here -- after the whole fit had been paid for -- one member at a time.
    fitted = port.FitResult.from_mapping(backend.fit(run, request, plan))

    spec_document = packaging.derive_spec_document(
        run.spec_document(), plan, code_commit=_code_commit()
    )
    weights = packaging.torchscript_bytes(
        fitted.network, patch=fitted.patch_size, channels=1
    )
    # THE SERVED ARTIFACT, beside the card: ONNX for the normative Triton path
    # (MOS-OPS-071 forbids server-side conversion; ConversionRun owns the equivalence
    # evidence). Export needs the `onnx` tooling package, like `medos/tools` — a training
    # environment without it still gets the bundle and the TorchScript card.
    onnx_blob: bytes | None = None
    try:
        onnx_blob = packaging.onnx_bytes(fitted.network, patch=fitted.patch_size, channels=1)
    except ImportError as exc:  # pragma: no cover - depends on the ambient env
        print(f"onnx export skipped ({exc}); the card will name the bundle only")
    checkpoint = fitted.checkpoint.read_bytes()

    import numpy as np
    import torch

    report = packaging.write_candidate_bundle(
        run.path("bundle"),
        spec_document=spec_document,
        weights=weights,
        checkpoint=checkpoint,
        patch=fitted.patch_size,
        channels=1,
        classes=fitted.classes,
        versions={"torch": str(torch.__version__), "numpy": str(np.__version__)},
    )
    import hashlib

    modelcard_path = packaging.write_modelcard(
        run.root,
        spec_document=spec_document,
        bundle_dir=run.path("bundle").name,
        weights_file=report.weights_path,
        weights_digest=report.weights_digest,
        versions={"torch": str(torch.__version__), "numpy": str(np.__version__)},
        stamp=_produced_by(),
        onnx_file="model.onnx" if onnx_blob is not None else None,
        onnx_digest=(
            "sha256:" + hashlib.sha256(onnx_blob).hexdigest() if onnx_blob is not None else None
        ),
    )
    if onnx_blob is not None:
        (Path(run.root) / "model.onnx").write_bytes(onnx_blob)

    _write_result(run, success_document(
        "fit",
        bundle={
            "path": run.path("bundle").name,
            "layout": report.layout_convention,
            "bundle_digest": report.bundle_digest,
            "weights_path": report.weights_path,
            "weights_digest": report.weights_digest,
            "file_count": len(report.files),
        },
        modelcard={"path": modelcard_path.name, "format": "medlange.modelcard/1"},
        preprocessing_spec={
            "id": spec_document["id"],
            "version": spec_document["version"],
            "derived_digest": _canonical_digest(spec_document),
            "golden_fixture_tensor_sha256":
                spec_document["golden_fixture"]["output_tensor_sha256"],
        },
        # MOS-TRAIN-224: the derived TRAINING batch size. It belongs in
        # `TrainingRun.hyperparameters` and that column is sealed at submit on this
        # deployment; see trainer/README.md. Recorded here so it is recorded
        # somewhere a reviewer reads rather than nowhere.
        hyperparameters={**dict(plan["hyperparameters"]), **dict(fitted.budget)},
        training={
            "seconds": fitted.seconds,
            "device": fitted.device,
            "epochs": fitted.budget["max_epochs"],
            "iterations_per_epoch": fitted.budget["iterations_per_epoch"],
        },
        determinism=determinism,
    ))
    return 0


def _canonical_digest(document: dict[str, Any]) -> str:
    from medos.sdk.canonical import canonical_bytes, sha256_hex

    return "sha256:" + sha256_hex(canonical_bytes(document))


def _predict(args: argparse.Namespace) -> int:
    """The vanilla stack's standalone inference: bundle in, prediction out.

    EXISTS BECAUSE THE AUDIT NAMED ITS ABSENCE: a trainer a stranger cannot
    run a trained model with is a component, not a framework. No platform,
    no run directory, no database -- a checkpoint directory from `fit`, one
    case `.npz`, one output `.npz`.
    """
    import numpy as np

    from medos_trainer.vanilla.data import load_case_npz
    from medos_trainer.vanilla.infer import load_predictor

    case = load_case_npz(args.input)
    predictor = load_predictor(
        args.checkpoint_dir, overlap=args.overlap,
        batch_size=args.batch_size, device=args.device,
    )
    label, probabilities = predictor.predict(case.image)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, label=label, probabilities=probabilities)
    print(f"wrote {out} label={label.shape} probabilities={probabilities.shape}")
    return 0


def _code_commit() -> str:
    from medos_trainer.stamp import read_stamp

    return str(read_stamp()["code_commit"])


def _declare_environment(out: str | None, bindings: str | None, allow_cpu: bool) -> int:
    from medos_trainer.environment import declare

    document = declare(bindings_path=bindings, allow_cpu=allow_cpu or None)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and then moved, so a reader that opens the file while this is
        # running never sees half a declaration and reports eight of nine keys missing.
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        # 0644: the API container reads it as a different user through a shared volume.
        target.chmod(0o644)
        print(f"wrote {target}")
    print(text)
    return 0


def _doctor() -> int:
    """What this image can do, as a document. No training run required to find out."""
    from medos_trainer import BACKEND_KIND
    from medos_trainer.environment import HardwareUnavailable, observe_hardware
    from medos_trainer.stamp import read_stamp

    report: dict[str, Any] = {"backend": BACKEND_KIND}
    try:
        report["stamp"] = {
            k: v for k, v in read_stamp().items() if k != "image_inventory"
        }
    except (FileNotFoundError, ValueError) as exc:
        report["stamp"] = {"error": str(exc)}
    try:
        report["hardware"] = observe_hardware()
        report["can_fit"] = True
    except HardwareUnavailable as exc:
        report["hardware"] = {"error": str(exc)}
        report["can_fit"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("can_fit") else 2


def _execute(args: argparse.Namespace) -> int:
    """The supervisor. Reads the deployment's own configuration and drives one run."""
    import psycopg
    from medos.training.orchestrator import LocalProcessOrchestrator
    from medos.training.supervisor import execute_pending, execute_run
    from psycopg.rows import dict_row

    from medos_trainer.environment import declare, preprocessing_bindings

    dsn = os.environ.get("MEDOS_DATABASE_URL", "")
    if not dsn:
        print("MEDOS_DATABASE_URL is unset; the supervisor reads the run rows", file=sys.stderr)
        return 2
    tenant_id = os.environ.get("MEDOS_TENANT_ID") or None

    environment = declare(bindings_path=args.bindings, allow_cpu=None)
    bindings = preprocessing_bindings(args.bindings)
    spec_documents = {
        capability: _spec_document_for(binding["spec_document"])
        for capability, binding in bindings.items()
    }

    orchestrator = LocalProcessOrchestrator(root=args.root)
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=False) as conn:
        if args.training_run_id:
            outcomes = (
                execute_run(
                    conn,
                    public_id=args.training_run_id,
                    root=args.root,
                    environment=environment,
                    spec_document=spec_documents[_capability_of(conn, args, tenant_id)],
                    orchestrator=orchestrator,
                    tenant_id=tenant_id,
                ),
            )
        elif args.watch:
            return _watch(
                conn, args, environment=environment, spec_documents=spec_documents,
                orchestrator=orchestrator, tenant_id=tenant_id,
            )
        else:
            outcomes = execute_pending(
                conn,
                root=args.root,
                environment=environment,
                spec_documents=spec_documents,
                orchestrator=orchestrator,
                limit=args.limit,
                tenant_id=tenant_id,
            )
    print(json.dumps([o.__dict__ for o in outcomes], indent=2, sort_keys=True))
    return 0 if all(o.state == "SUCCEEDED" for o in outcomes) else 1


def _watch(
    conn: Any,
    args: argparse.Namespace,
    *,
    environment: dict[str, Any],
    spec_documents: dict[str, Any],
    orchestrator: Any,
    tenant_id: str | None,
) -> int:
    """Keep picking up runs a PERSON submitted, `--watch` seconds apart. Never exits 0.

    THIS IS NOT `MOS-TRAIN-194`'S FORBIDDEN CONSTRUCTION AND THE DIFFERENCE IS THE WHOLE
    POINT. What that requirement forbids is "a drift alert that STARTS a training run,
    which produces a candidate, which passes a gate, which promotes itself". Nothing here
    starts one: this loop reads rows that are already `PENDING`, and a row is `PENDING`
    because a named person pressed one control on the console (`MOS-UI-147`). It is the
    same relationship `medos-worker`'s claim loop has to a `Job` a person created.

    There is no drift input, no monitoring callback, no retry-on-failure and no promotion
    verb reachable from here -- `MOS-TRAIN-189`'s call-graph property is asserted over
    this package's import closure by `tests/gate/test_no_auto_promote.py`, and a FAILED
    run stays failed until a person submits a different one.
    """
    import time

    from medos.training.supervisor import execute_pending

    log = logging.getLogger("medos_trainer")
    log.info("supervisor watching for PENDING runs every %ss", args.watch)
    while True:
        try:
            for outcome in execute_pending(
                conn,
                root=args.root,
                environment=environment,
                spec_documents=spec_documents,
                orchestrator=orchestrator,
                limit=args.limit,
                tenant_id=tenant_id,
            ):
                log.info("run=%s state=%s %s", outcome.public_id, outcome.state, outcome.detail)
        except Exception:  # noqa: BLE001 - a supervisor that dies stops every later run
            # MOS-REL-051: never silent. The traceback goes to the run log the operator
            # reads, and the loop continues, because the alternative is a container that
            # exits on one bad row and leaves every subsequent submission in PENDING with
            # no explanation on the console.
            conn.rollback()
            log.exception("the supervisor could not complete a pass")
        time.sleep(float(args.watch))


def _capability_of(conn: Any, args: argparse.Namespace, tenant_id: str | None) -> str:
    from medos.training import runs as tr

    return str(tr.get(conn, args.training_run_id, tenant_id=tenant_id)["capability_id"])


def _spec_document_for(reference: str) -> dict[str, Any]:
    from medos_trainer.environment import _load_spec_document

    return _load_spec_document(reference)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medos-trainer",
        description="Fit a MedicalOS training run. nnU-Net v2; MOS-TRAIN-135's freeze.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for phase in ("plan", "fit"):
        one = sub.add_parser(phase, help=f"phase {phase} of one training run")
        one.add_argument(
            "--run-dir", required=True,
            help="the run directory the platform created. The ONLY input argument: "
                 "chapter 17 acceptance check 15 forbids a path, a prefix or a glob as a "
                 "cohort argument, and there is nowhere here to put one",
        )

    declare_parser = sub.add_parser(
        "declare-environment", help="emit MEDOS_TRAINING_ENVIRONMENT's nine keys"
    )
    declare_parser.add_argument("--out", default=None)
    declare_parser.add_argument("--bindings", default=None)
    declare_parser.add_argument(
        "--allow-cpu", action="store_true",
        help="record a deployment with no GPU AS one, instead of refusing",
    )

    execute_parser = sub.add_parser("execute", help="drive submitted runs to completion")
    execute_parser.add_argument("--root", default="/var/lib/medos-trainer/runs")
    execute_parser.add_argument("--bindings", default=None)
    execute_parser.add_argument("--limit", type=int, default=1)
    execute_parser.add_argument("--training-run-id", default=None)
    execute_parser.add_argument(
        "--watch", type=float, default=None,
        help="keep picking up PENDING runs this many seconds apart. It STARTS no run; "
             "see _watch's docstring for why that is MOS-TRAIN-194's actual line",
    )

    sub.add_parser("doctor", help="what this image can and cannot do right now")

    predict_parser = sub.add_parser(
        "predict",
        help="vanilla stack: run a saved bundle over one case (sliding window)",
    )
    predict_parser.add_argument(
        "--checkpoint-dir", required=True,
        help="directory save_inference_bundle wrote (model.pt, net_config.json, "
             "fit_plan.json)",
    )
    predict_parser.add_argument("--input", required=True, help="case .npz (image)")
    predict_parser.add_argument("--output", required=True, help="output .npz")
    predict_parser.add_argument("--overlap", type=float, default=0.5)
    predict_parser.add_argument("--batch-size", type=int, default=2)
    predict_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    _logging()
    args = build_parser().parse_args(argv)
    if args.command in ("plan", "fit"):
        return _phase(args.command, args.run_dir)
    if args.command == "declare-environment":
        return _declare_environment(args.out, args.bindings, args.allow_cpu)
    if args.command == "execute":
        return _execute(args)
    if args.command == "predict":
        return _predict(args)
    return _doctor()


if __name__ == "__main__":  # pragma: no cover - a process entry point
    raise SystemExit(main())
