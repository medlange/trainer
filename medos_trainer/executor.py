# SPDX-License-Identifier: Apache-2.0
"""The trainer deployable's supervisor: it EXECUTES a run a person submitted.

WHAT IT IS NOT, STATED FIRST BECAUSE THE DISTINCTION IS A REQUIREMENT
-----------------------------------------------------------------------
`MOS-TRAIN-194` forbids "Automated retraining triggered by a monitoring signal ... A
drift alert that starts a training run, which produces a candidate, which passes a gate,
which promotes itself" and calls it "four individually reasonable components" assembled
into the forbidden path. `tests/gate/test_no_auto_promote.py` check 4 greps `medos/` and
`deploy/` for a scheduler in the same module as a training or promotion entrypoint.

This module SUBMITS NOTHING. It has no call to the submit path, no drift callback, no
alert hook and no promotion verb; it reads rows that are already `PENDING` because a
named person pressed one control on the console, and it drives them to a terminal state.
The distinction is exactly `MOS-TRAIN-194`'s: what is forbidden is a machine deciding
that a run should happen. Executing a run a person decided on is the orchestrator's job
and `MOS-TRAIN-122` puts start and completion in its hands.

WHY IT LIVES IN `deploy/` AND NOT IN `medos/`
-----------------------------------------------
It runs in the TRAINER image, beside torch, and `medos/` must stay importable without
torch (`MOS-TRAIN-225`, and `medos/training/chain.py`'s whole design). A supervisor in
`medos/worker/` would be a supervisor the API image imports. It is deployment code: it
knows this deployment's database URL, this deployment's image staging, and which
orchestrator driver this deployment runs -- three things `MOS-REL-046` says are injected
and never ambient in the library.

WHERE THE PIXELS COME FROM, AND WHERE THEY DO NOT
---------------------------------------------------
`MOS-TRAIN-068` and `MOS-TRAIN-199` permit the pipeline to acquire imaging ONLY from the
de-identified side of the Gateway, as the `dataset_export` consumer class.
`medos/training/retrieval.py` already implements that boundary and already reports what
it finds: on the shipped deployment `MEDOS_DATASET_EXPORT_TOKEN` is unset and
`medos/gateway/app.py` answers `503 DEID_NOT_IMPLEMENTED` for that class anyway, because
the de-identification stage does not exist and `MOS-DATA-037` requires the egress to fail
closed rather than emit identified data.

So on this deployment the `gateway` stager CANNOT produce a volume, and the run FAILS
with that reason written into `failure_reason` where an operator reads it. That is the
honest outcome and it is not worked around: the alternative -- borrowing the worker's
credential, which the Gateway would resolve as `platform_writer` -- would put IDENTIFIED
pixels into a training corpus while every response looked correct, and
`medos/training/retrieval.py` names that as "the worst combination available".

The `directory` stager is the other route and it is deliberately dumb: a site that
exported its corpus out of band, under its own de-identification, points
`MEDOS_TRAINER_IMAGE_ROOT` at it. What that buys is that the fitting half of this system
is measurable on a deployment whose Gateway cannot yet de-identify; what it does NOT do is
assert anything about the provenance of what it finds there. `MOS-EVID-021`'s
de-identification status is recorded at SEAL time, on the `DatasetVersion`, by
`medos/training/seal.py`, and nothing here can add to it.

Spec: MOS-TRAIN-068, MOS-TRAIN-121, MOS-TRAIN-122, MOS-TRAIN-124, MOS-TRAIN-135,
MOS-TRAIN-141, MOS-TRAIN-194, MOS-TRAIN-199, MOS-REL-046, MOS-REL-051, MOS-SEC-033.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from medicalos_preprocessing.contract import CONTRACT_VERSION, RUN_DIRECTORY, RunDirectory

__all__ = [
    "STAGERS",
    "ExecutionError",
    "RunOutcome",
    "execute_pending",
    "execute_run",
    "stage_from_directory",
]

log = logging.getLogger("medos_trainer.executor")

#: How long one phase may run before the orchestrator kills it. A training run is hours,
#: so the default is generous and is configuration rather than a constant
#: (`MOS-REL-046`). It exists at all because `LocalProcessOrchestrator.poll` turns an
#: overrun into `FAILED` with the bound named, and a phase that hangs forever is a run
#: nobody can cancel from the console (`MOS-UI-156`).
_DEFAULT_TIMEOUT: Final[int] = 60 * 60 * 24


class ExecutionError(RuntimeError):
    """The run could not be driven. Always carries the reason `runs.fail` will record."""


@dataclass(frozen=True)
class RunOutcome:
    """What one run did. Returned rather than logged, so a caller decides what is fatal."""

    public_id: str
    state: str
    detail: str
    bundle_digest: str | None = None
    fingerprint_digest: str | None = None
    run_directory: str | None = None


# =====================================================================================
# Staging
# =====================================================================================
def stage_from_directory(
    case_key: str, target: Path, *, root: str | os.PathLike[str]
) -> tuple[str, str | None]:
    """`<root>/<case_key>/{image,label}.nii.gz`, copied into the run directory.

    The layout is fixed and is NOT configurable by the run: `MOS-TRAIN-141`'s argument is
    that "a container that can name its own data can name the test partition", and a
    pattern read from a request would be exactly that. The root is deployment
    configuration; the case key comes from the frozen split manifest.
    """
    source = Path(root) / case_key
    image = source / "image.nii.gz"
    if not image.is_file():
        raise ExecutionError(
            f"case {case_key} has no image at {image}. MEDOS_TRAINER_IMAGE_ROOT names "
            "an export this deployment placed; a case in the sealed cohort that is not "
            "in it would be silently dropped from the fit, and the run's "
            "dataset_version_digest would describe a cohort this fit did not see"
        )
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, target / "image.nii.gz")
    label = source / "label.nii.gz"
    if label.is_file():
        shutil.copyfile(label, target / "label.nii.gz")
        return "image.nii.gz", "label.nii.gz"
    return "image.nii.gz", None


def _stage_from_gateway(case_key: str, target: Path, **_: Any) -> tuple[str, str | None]:
    """The `dataset_export` route. Reports why it cannot, rather than falling back.

    See the module docstring. This function deliberately contains no retrieval code yet
    and no credential fallback: `medos/training/retrieval.py` owns the boundary, reports
    the reason, and the reason on the shipped deployment is that the Gateway refuses the
    only consumer class the pipeline may use.
    """
    from medos.training.retrieval import gateway_from_env

    gateway, reason = gateway_from_env()
    if gateway is None:
        raise ExecutionError(
            f"case {case_key} cannot be retrieved: {reason}. MOS-TRAIN-068 and "
            "MOS-TRAIN-199 permit the pipeline to acquire imaging under the "
            "dataset_export consumer class and no other, and this deployment holds no "
            "such credential. Set MEDOS_TRAINER_STAGER=directory with an out-of-band "
            "export, or give the Gateway a de-identification stage"
        )
    raise ExecutionError(
        f"case {case_key}: a dataset_export credential is present but this image ships "
        "no DICOM-to-NIfTI staging for it. medos/training/retrieval.py retrieves and "
        "digests series; converting a retrieved series into the volume a trainer reads "
        "is the piece that is not built, and it is named here rather than approximated"
    )


#: The two routes, as data, so the deployment's choice is one name and not a code path.
STAGERS: Final[dict[str, Any]] = {
    "gateway": _stage_from_gateway,
    "directory": stage_from_directory,
}


# =====================================================================================
# Driving one run
# =====================================================================================
def _cohort_lines(
    run_directory: RunDirectory,
    cases: Sequence[Any],
    *,
    role: str,
    stager: str,
    image_root: str | None,
) -> int:
    """Stage every case of one partition and write its `cohort/*.jsonl`. Never partial.

    A case that cannot be staged aborts the whole run. `MOS-EVID-028` makes the split "a
    frozen manifest of rows"; training on the subset that happened to be retrievable is
    a different cohort with the same `dataset_version_digest`, which is the one thing the
    digest exists to prevent.
    """
    path = run_directory.path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = run_directory.path("images")
    stage = STAGERS[stager]

    lines: list[str] = []
    for case in cases:
        target = images / case.case_key
        relative = f"{RUN_DIRECTORY['images']}/{case.case_key}"
        if stager == "directory":
            image, label = stage(case.case_key, target, root=image_root)
        else:
            image, label = stage(case.case_key, target)
        lines.append(json.dumps({
            "patient_key": case.patient_key,
            "case_key": case.case_key,
            "study_instance_uid": case.study_instance_uid,
            "series_instance_uid": case.series_instance_uid,
            "partition": case.partition,
            "fold": case.fold,
            "image": f"{relative}/{image}",
            "label": None if label is None else f"{relative}/{label}",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _request_document(row: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    """`request.json`. Every member comes off the SEALED row or the declaration.

    Nothing here is read from a client and nothing is defaulted. `MOS-TRAIN-124`'s
    binding was sealed at submit and `0013_training.up.sql` refuses to let it change; this
    is a projection of it into the exchange, and a member the trainer could influence
    would be a member the binding does not describe.
    """
    backend = dict(row["training_backend"] or {})
    capability = str(row["capability_id"])
    return {
        "contract_version": CONTRACT_VERSION,
        "training_run_id": str(row["public_id"]),
        "capability_id": capability,
        "training_backend": {
            "kind": str(backend.get("kind")),
            "version": str(backend.get("version")),
        },
        "partitions": {
            "fit": str(row["fit_partition"]),
            "select": str(row["select_partition"]),
        },
        "seeds": dict(environment["seeds"]),
        "determinism": dict(environment["determinism"]),
        "preprocessing": dict(dict(environment["preprocessing"])[capability]),
        "budget": _budget_document(),
    }


#: The epoch budget, as the SUPERVISOR's configuration, written into `request.json`.
#:
#: MEASURED, NOT ANTICIPATED, AND IT IS A PROPERTY OF THE PORT RATHER THAN A BUG.
#: `LocalProcessOrchestrator` builds the child's environment from a five-variable
#: allow-list instead of inheriting this process's -- that is `MOS-TRAIN-121` C3 working
#: exactly as written, "an environment this class CONSTRUCTS rather than inherits", so a
#: training container cannot reach the control plane by reading its own environment. The
#: first end-to-end run here set `MEDOS_TRAINER_EPOCHS=2` on the supervisor, the child
#: never saw it, and nnU-Net ran its own default of 1000 epochs: seven hours instead of
#: two minutes, with nothing in the record saying which was asked for.
#:
#: So anything the trainer needs travels in the RUN DIRECTORY, never in the environment.
#: That is also the better answer: `request.json` is read back by the phase and copied
#: into `result.json`, so the budget a run was given is recorded rather than inferred
#: from whatever the supervisor's environment happened to be at the time.
_BUDGET_VARIABLES: Final[dict[str, str]] = {
    "max_epochs": "MEDOS_TRAINER_EPOCHS",
    "iterations_per_epoch": "MEDOS_TRAINER_ITERATIONS",
    "validation_iterations_per_epoch": "MEDOS_TRAINER_VAL_ITERATIONS",
}


def _budget_document() -> dict[str, int]:
    """What this deployment asks for, or `{}` for nnU-Net's own full-run defaults.

    `MOS-UI-149` keeps the epoch budget off the console and
    `TrainingRunSubmitRequest` has no member for it, so it can only arrive from the
    deployment. An empty document means the trainer runs the real thing.
    """
    out: dict[str, int] = {}
    for member, variable in _BUDGET_VARIABLES.items():
        raw = os.environ.get(variable, "").strip()
        if raw:
            out[member] = max(1, int(raw))
    return out


def execute_run(
    conn: Any,
    *,
    public_id: str,
    root: str | os.PathLike[str],
    environment: Mapping[str, Any],
    spec_document: Mapping[str, Any],
    orchestrator: Any,
    stager: str | None = None,
    image_root: str | None = None,
    tenant_id: str | None = None,
    timeout_seconds: int | None = None,
    poll_seconds: float = 2.0,
) -> RunOutcome:
    """Drive one `PENDING` run to a terminal state. The whole of `MOS-TRAIN-122`'s port.

    The order is fixed by `MOS-TRAIN-135` and is the reason there are two phases:

        stage  ->  phase `plan`  ->  runs.start(fingerprint_digest=...)
               ->  phase `fit`   ->  runs.succeed(bundle_digest=...)

    The row does not leave `PENDING` until the fingerprint EXISTS, because
    `medos.training.runs.start` refuses an auto-configured run without one and
    `training_runs_guard()` then freezes it. A single-phase trainer would have to either
    start the row before the plan existed -- which the engine refuses -- or write the plan
    after `RUNNING`, which the database refuses.
    """

    from medos.training import cohort as co
    from medos.training import runs as tr
    from medos.training.orchestrator import TERMINAL_STATES, TrainingRunSpec, trainer_argv

    stager = stager or os.environ.get("MEDOS_TRAINER_STAGER", "gateway")
    if stager not in STAGERS:
        raise ExecutionError(f"MEDOS_TRAINER_STAGER={stager!r} is not one of {sorted(STAGERS)}")
    image_root = image_root or os.environ.get("MEDOS_TRAINER_IMAGE_ROOT")
    timeout = int(
        timeout_seconds or os.environ.get("MEDOS_TRAINER_TIMEOUT_S") or _DEFAULT_TIMEOUT
    )

    row = tr.get(conn, public_id, tenant_id=tenant_id)
    if row.state != "PENDING":
        raise ExecutionError(f"training run {row.public_id} is {row.state}, not PENDING")

    directory = Path(root) / row.public_id
    (directory / "cohort").mkdir(parents=True, exist_ok=True)
    (directory / RUN_DIRECTORY["images"]).mkdir(parents=True, exist_ok=True)
    run_directory = RunDirectory(directory)

    try:
        request = _request_document(row.as_dict(), environment)
        run_directory.write("request", request)
        run_directory.write("spec", spec_document)

        # `cohort.resolve` is the ONLY way to obtain cases, and it answers 403 on `test`
        # before it touches the database (MOS-TRAIN-141). The principal is named in the
        # refusal because MOS-TRAIN-214 makes the rule about WHO is asking.
        for role, partition in (
            ("cohort_fit", row["fit_partition"]),
            ("cohort_select", row["select_partition"]),
        ):
            cases = co.resolve(
                conn,
                co.CohortRequest(split_id=str(row["split_id"]), partition=str(partition)),
                principal="medos-trainer",
                tenant_id=tenant_id,
            )
            if not cases:
                raise ExecutionError(
                    f"partition {partition!r} of split {row['split_id']} resolved to no "
                    "cases. A fit over zero cases returns the initialisation and records "
                    "it as a trained model"
                )
            _cohort_lines(
                run_directory, cases, role=role, stager=stager, image_root=image_root
            )
    except Exception as exc:  # noqa: BLE001 - re-raised as a recorded failure below
        reason = f"staging failed: {exc}"
        tr.fail(conn, run_id=row.public_id, reason=reason[:2000], tenant_id=tenant_id)
        conn.commit()
        return RunOutcome(row.public_id, "FAILED", reason, run_directory=str(directory))

    # ---- phase 1: derive the plan ------------------------------------------------- #
    plan_state = _run_phase(
        orchestrator, "plan", row.public_id, str(row["capability_id"]),
        directory, timeout=timeout, poll_seconds=poll_seconds,
        terminal=TERMINAL_STATES, spec_class=TrainingRunSpec, argv=trainer_argv,
    )
    result = _read_result(run_directory)
    if plan_state != "SUCCEEDED" or result.get("status") != "SUCCEEDED":
        reason = str(result.get("reason") or f"the plan phase ended {plan_state}")
        tr.fail(conn, run_id=row.public_id, reason=reason[:2000], tenant_id=tenant_id)
        conn.commit()
        return RunOutcome(row.public_id, "FAILED", reason, run_directory=str(directory))

    plan = json.loads(run_directory.path("plan").read_text(encoding="utf-8"))
    fingerprint_digest = str(plan["fingerprint_digest"])
    tr.start(
        conn,
        run_id=row.public_id,
        runner="medos-trainer/nnunet",
        orchestrator_run_id=None,
        fingerprint_digest=fingerprint_digest,
        tenant_id=tenant_id,
    )
    conn.commit()

    # ---- phase 2: fit -------------------------------------------------------------- #
    fit_state = _run_phase(
        orchestrator, "fit", row.public_id, str(row["capability_id"]),
        directory, timeout=timeout, poll_seconds=poll_seconds,
        terminal=TERMINAL_STATES, spec_class=TrainingRunSpec, argv=trainer_argv,
    )
    result = _read_result(run_directory)
    if fit_state != "SUCCEEDED" or result.get("status") != "SUCCEEDED":
        reason = str(result.get("reason") or f"the fit phase ended {fit_state}")
        tr.fail(conn, run_id=row.public_id, reason=reason[:2000], tenant_id=tenant_id)
        conn.commit()
        return RunOutcome(
            row.public_id, "FAILED", reason,
            fingerprint_digest=fingerprint_digest, run_directory=str(directory),
        )

    bundle_digest = str(dict(result["bundle"])["bundle_digest"])
    tr.succeed(
        conn,
        run_id=row.public_id,
        bundle_digest=bundle_digest,
        bundle_location=("medos-training-bundles", f"{row.public_id}/bundle"),
        tenant_id=tenant_id,
    )
    conn.commit()
    return RunOutcome(
        row.public_id, "SUCCEEDED", "the bundle was written and verified",
        bundle_digest=bundle_digest, fingerprint_digest=fingerprint_digest,
        run_directory=str(directory),
    )


def _run_phase(
    orchestrator: Any,
    phase: str,
    public_id: str,
    capability_id: str,
    directory: Path,
    *,
    timeout: int,
    poll_seconds: float,
    terminal: frozenset[str],
    spec_class: Any,
    argv: Any,
) -> str:
    import time

    spec = spec_class(
        run_public_id=f"{public_id}:{phase}",
        capability_id=capability_id,
        argv=argv(phase, str(directory)),
        workdir=str(directory),
        # MOS-TRAIN-121 C3: nothing here is a credential. The child is told which run it
        # is and which phase it is in, and `LocalProcessOrchestrator` adds the run id and
        # the capability id itself; everything else it needs is in the run directory.
        env={"MEDOS_TRAINING_PHASE": phase},
        gpu_pool="training",
        timeout_seconds=timeout,
        labels={"phase": phase, "training_run_id": public_id},
    )
    handle = orchestrator.submit(spec)
    state = orchestrator.poll(handle)
    while state.state not in terminal:
        time.sleep(poll_seconds)
        state = orchestrator.poll(handle)
    log.info(
        "phase=%s training_run_id=%s state=%s exit_code=%s",
        phase, public_id, state.state, state.exit_code,
    )
    return state.state


def _read_result(run_directory: RunDirectory) -> dict[str, Any]:
    path = run_directory.path("result")
    if not path.is_file():
        return {
            "status": "FAILED",
            "reason": (
                f"the phase exited and wrote no {RUN_DIRECTORY['result']}. Its output is "
                f"in {RUN_DIRECTORY['log']}; a phase that fails before it can write a "
                "reason is a crash, and MOS-REL-051 wants it reported as one"
            ),
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def execute_pending(
    conn: Any,
    *,
    root: str | os.PathLike[str],
    environment: Mapping[str, Any],
    spec_documents: Mapping[str, Mapping[str, Any]],
    orchestrator: Any,
    limit: int = 1,
    tenant_id: str | None = None,
    **kwargs: Any,
) -> tuple[RunOutcome, ...]:
    """Execute up to `limit` runs that are already `PENDING`. Oldest first.

    `limit` defaults to ONE. A training run holds the GPU for hours and
    `MOS-TRAIN-123` puts the training pool outside the serving pool's reach rather than
    giving this process a way to reserve; running two at once on one card would make the
    second fail on memory after the first had already frozen its plan.
    """
    from medos.training import runs as tr

    rows = tr.list_runs(conn, state_in=["PENDING"], limit=200, tenant_id=tenant_id)
    outcomes: list[RunOutcome] = []
    for row in sorted(rows, key=lambda r: str(r["created_at"]))[:limit]:
        capability = str(row["capability_id"])
        spec_document = spec_documents.get(capability)
        if spec_document is None:
            outcomes.append(
                RunOutcome(row.public_id, "SKIPPED", f"no spec document for {capability}")
            )
            continue
        outcomes.append(
            execute_run(
                conn,
                public_id=row.public_id,
                root=root,
                environment=environment,
                spec_document=spec_document,
                orchestrator=orchestrator,
                tenant_id=tenant_id,
                **kwargs,
            )
        )
    return tuple(outcomes)
