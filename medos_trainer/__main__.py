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

from medos_trainer.contract import (
    ContractViolation,
    RunDirectory,
    failure_document,
    success_document,
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _logging() -> None:
    logging.basicConfig(
        level=os.environ.get("MEDOS_TRAINER_LOG_LEVEL", "INFO").upper(),
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )


def _refusals_of(exc: BaseException) -> list[dict[str, Any]]:
    """`medicalos_preprocessing.errors.Refusal`s, as documents, when the failure was one.

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


def _phase(phase: str, run_dir: str) -> int:
    """Run one phase, and write `result.json` whatever happens."""
    run = RunDirectory(run_dir)
    try:
        request = run.request()
        if request.backend_kind != "nnunet":
            raise ContractViolation(
                f"this image trains {'nnunet'!r} and the run binds "
                f"{request.backend_kind!r}. MOS-REL-037 forbids a compatibility range on "
                "a recorded version and the same argument applies to a backend: an image "
                "that pretended to be a second backend would record a version for a "
                "planner that did not run"
            )
        from medos_trainer import backend as nnunet

        # BEFORE anything imports `nnunetv2`. `nnunetv2/paths.py` reads its three roots
        # from the environment AT MODULE IMPORT and binds them to constants, so setting
        # them later has no effect and the run writes its dataset into the image's
        # default root instead of into this run's directory. `backend.py` imports
        # nnunetv2 only inside `derive_plan` and `fit`, which is what makes this call
        # site early enough -- and why it is a separate function with a docstring rather
        # than three `os.environ` lines somewhere in the middle of the backend.
        work = run.root / "work"
        nnunet.prepare_workspace(work)
        determinism = nnunet.apply_determinism(request)

        if phase == "plan":
            plan = nnunet.derive_plan(run, request, work=work)
            run.write("plan", plan)
            run.write("result", success_document(
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
        return _fit(run, request, plan, determinism, nnunet)
    except BaseException as exc:  # noqa: BLE001 - every exit records a reason
        run.write("result", failure_document(
            phase,
            reason=f"{type(exc).__name__}: {exc}",
            refusals=_refusals_of(exc),
        ))
        traceback.print_exc()
        return 1


def _fit(
    run: RunDirectory, request: Any, plan: dict[str, Any], determinism: dict[str, Any],
    nnunet: Any,
) -> int:
    from medos_trainer import packaging

    fitted = nnunet.fit(run, request, plan)

    spec_document = packaging.derive_spec_document(
        run.spec_document(), plan, code_commit=_code_commit()
    )
    weights = packaging.torchscript_bytes(
        fitted["network"], patch=fitted["patch_size"], channels=1
    )
    checkpoint = Path(fitted["checkpoint"]).read_bytes()

    import numpy as np
    import torch

    report = packaging.write_candidate_bundle(
        run.path("bundle"),
        spec_document=spec_document,
        weights=weights,
        checkpoint=checkpoint,
        patch=fitted["patch_size"],
        channels=1,
        classes=int(fitted["label_manager_classes"]),
        versions={"torch": str(torch.__version__), "numpy": str(np.__version__)},
    )

    run.write("result", success_document(
        "fit",
        bundle={
            "path": run.path("bundle").name,
            "layout": report.layout_convention,
            "bundle_digest": report.bundle_digest,
            "weights_path": report.weights_path,
            "weights_digest": report.weights_digest,
            "file_count": len(report.files),
        },
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
        hyperparameters={**dict(plan["hyperparameters"]), **dict(fitted["budget"])},
        training={
            "seconds": fitted["seconds"],
            "device": fitted["device"],
            "epochs": fitted["budget"]["max_epochs"],
            "iterations_per_epoch": fitted["budget"]["iterations_per_epoch"],
        },
        determinism=determinism,
    ))
    return 0


def _canonical_digest(document: dict[str, Any]) -> str:
    from medicalos_preprocessing.canonical import canonical_bytes, sha256_hex

    return "sha256:" + sha256_hex(canonical_bytes(document))


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
    from psycopg.rows import dict_row

    from medos.training.orchestrator import LocalProcessOrchestrator
    from medos_trainer.environment import declare, preprocessing_bindings
    from medos_trainer.executor import execute_pending, execute_run

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

    from medos_trainer.executor import execute_pending

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
    return _doctor()


if __name__ == "__main__":  # pragma: no cover - a process entry point
    raise SystemExit(main())
