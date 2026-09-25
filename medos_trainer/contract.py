# SPDX-License-Identifier: Apache-2.0
"""The run directory, from the trainer's side. The platform's side is `orchestrator.py`.

WHAT IS EXCHANGED, AND WHY IT IS FILES
---------------------------------------
`MOS-TRAIN-121` C1 puts the trainer in "a separate deployable, a separate service account,
a separate network namespace from every serving component", and `MOS-REL-039` allows no
driver that calls an orchestrator's API from application code. What is left is a directory
and an exit code, and that is sufficient -- it is also the whole of the trainer's
authority, because `LocalProcessOrchestrator` hands the child an environment carrying no
database URL, no object-store credential and no API key.

    request.json        in   the run's identity and the half of MOS-TRAIN-124's binding
                             the trainer needs in order to act: the backend, the seeds,
                             the determinism settings, the bound PreprocessingSpec.
    preprocessing.json  in   the bound spec document itself, byte for byte.
    cohort/fit.jsonl    in   the FIT partition's cases, resolved by the platform.
    cohort/select.jsonl in   the SELECT partition's cases.
    images/             in   the staged volumes the two files above point into.
    fingerprint.json    out  phase 1: the backend's own fingerprint document, verbatim.
    plan.json           out  phase 1: what MOS-TRAIN-223's exporter made of it.
    bundle/             out  phase 2: MOS-TRAIN-129's layout.
    result.json         out  both phases: the terminal record, success or refusal.

THE TRAINER CANNOT NAME ITS OWN DATA, AND THAT IS THE POINT
-------------------------------------------------------------
`MOS-TRAIN-141`: "The training and selection jobs MUST NOT be able to read the `test`
partition at all. The cohort resolver MUST be handed a split view filtered to
`{train, tune}` and MUST return 403, not an empty set, on a `test` request." Chapter 17
acceptance check 15 adds: "the training container's resolver rejects a filesystem path, a
bucket prefix and a glob as a cohort argument".

This module is the second sentence. There is no `--split`, no `--partition`, no `--bucket`
and no `--glob` on any subcommand: the only argument is a run directory the PLATFORM
created, the partitions were resolved by `medos.training.cohort.resolve` before the
process started, and `staged_path()` below refuses any `image` or `label` member that is
absolute, contains `..`, or resolves outside `images/`. A trainer that could read the test
partition would first have to be given a way to say the word, and there is none.

WHY `result.json` EXISTS WHEN THERE IS ALREADY AN EXIT CODE
-------------------------------------------------------------
The exit code carries one bit and `medos.training.runs` needs three things: a
`fingerprint_digest` for `start()`, a `bundle_digest` for `succeed()`, and a REASON for
`fail()` -- "a FAILED training run MUST name a reason (`MOS-REL-051`)". A run that failed
with exit 1 and no reason is a run nobody can act on, and the reason has to survive the
process. So every exit writes `result.json` first, including the failing ones, and the
exit code is a summary of what is in it rather than the record itself.

Spec: MOS-TRAIN-121, MOS-TRAIN-124, MOS-TRAIN-135, MOS-TRAIN-141, MOS-TRAIN-214,
MOS-TRAIN-223, MOS-REL-039, MOS-REL-046, MOS-REL-051, chapter 17 acceptance check 15.
Pure: reads and writes one directory, opens no socket and no database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CONTRACT_VERSION",
    "RUN_DIRECTORY",
    "CohortEntry",
    "ContractViolation",
    "RunDirectory",
    "RunRequest",
    "failure_document",
    "success_document",
]

#: Bumped when the exchange changes shape. Both sides assert it, because a platform
#: talking to an older trainer image is exactly the case where a silently-ignored member
#: becomes a run that trained on something nobody asked for.
CONTRACT_VERSION: Final[str] = "1"

#: `medos.training.orchestrator.RUN_DIRECTORY`, restated. The two MUST agree and
#: `tests/unit/test_trainer_contract.py` asserts that they do, by importing both -- a
#: constant copied into two files with no test between them is not one constant.
RUN_DIRECTORY: Final[dict[str, str]] = {
    "request": "request.json",
    "spec": "preprocessing.json",
    "cohort_fit": "cohort/fit.jsonl",
    "cohort_select": "cohort/select.jsonl",
    "images": "images",
    "fingerprint": "fingerprint.json",
    "plan": "plan.json",
    "bundle": "bundle",
    "result": "result.json",
    "log": "run.log",
}

#: `MOS-TRAIN-141`'s readable set, restated on this side of the boundary. The trainer
#: never resolves a partition, but it does CHECK the one it was handed: a `request.json`
#: naming `test` as the fit partition is a platform defect, and a trainer that trained on
#: it anyway would make the platform's 403 decorative.
_READABLE_PARTITIONS: Final[frozenset[str]] = frozenset({"train", "tune"})


class ContractViolation(ValueError):
    """The run directory is not one. Always fatal, always with the member named."""


def _require(document: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in document:
        raise ContractViolation(f"{where} is missing {key!r}")
    return document[key]


@dataclass(frozen=True)
class CohortEntry:
    """One line of `cohort/*.jsonl`: a case, and where its two volumes were staged.

    `image` and `label` are paths RELATIVE to the run directory and are validated by
    `RunDirectory.staged_path`. They are members rather than a naming convention because
    the platform stages them and the trainer must not be in the business of guessing a
    filename -- a guess that misses is a case silently dropped from a cohort somebody
    sealed.
    """

    patient_key: str
    case_key: str
    study_instance_uid: str
    series_instance_uid: str
    partition: str
    image: str
    label: str | None = None
    fold: int | None = None

    @classmethod
    def from_document(cls, document: Mapping[str, Any], *, where: str) -> CohortEntry:
        partition = str(_require(document, "partition", where))
        if partition not in _READABLE_PARTITIONS:
            raise ContractViolation(
                f"{where} carries a case in partition {partition!r}. MOS-TRAIN-141 hands "
                "the pipeline a split view filtered to {train, tune}; a case outside it "
                "reaching this file means the platform resolved a partition it may not "
                "read, and training on it would make that 403 decorative"
            )
        label = document.get("label")
        fold = document.get("fold")
        return cls(
            patient_key=str(_require(document, "patient_key", where)),
            case_key=str(_require(document, "case_key", where)),
            study_instance_uid=str(_require(document, "study_instance_uid", where)),
            series_instance_uid=str(_require(document, "series_instance_uid", where)),
            partition=partition,
            image=str(_require(document, "image", where)),
            label=None if label is None else str(label),
            fold=None if fold is None else int(fold),
        )


@dataclass(frozen=True)
class RunRequest:
    """`request.json`. The half of `MOS-TRAIN-124`'s binding the trainer has to act on.

    It deliberately does NOT carry the whole binding. The digests of the dataset version,
    the split and the annotation set are the platform's record and the trainer can neither
    check nor change them; carrying them here would invite a trainer that recomputed one.
    What is here is what changes the trainer's behaviour: which backend, which seeds,
    which determinism settings, which spec, and which partitions it was handed.
    """

    contract_version: str
    training_run_id: str
    capability_id: str
    backend_kind: str
    backend_version: str
    fit_partition: str
    select_partition: str
    seeds: Mapping[str, Any]
    determinism: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    budget: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> RunRequest:
        where = RUN_DIRECTORY["request"]
        version = str(_require(document, "contract_version", where))
        if version != CONTRACT_VERSION:
            raise ContractViolation(
                f"{where} declares contract_version {version!r}; this image implements "
                f"{CONTRACT_VERSION!r}. A platform and a trainer that disagree about the "
                "exchange must refuse, not negotiate: a member one side ignores is a run "
                "fitted against inputs the other side thinks it sent"
            )
        backend = dict(_require(document, "training_backend", where))
        partitions = dict(_require(document, "partitions", where))
        fit = str(_require(partitions, "fit", f"{where}.partitions"))
        select = str(_require(partitions, "select", f"{where}.partitions"))
        for name, value in (("fit", fit), ("select", select)):
            if value not in _READABLE_PARTITIONS:
                raise ContractViolation(
                    f"{where}.partitions.{name} is {value!r}. MOS-TRAIN-141 permits "
                    f"{sorted(_READABLE_PARTITIONS)} and nothing else"
                )
        if fit == select:
            raise ContractViolation(
                f"{where}.partitions names {fit!r} for both fit and select. MOS-TRAIN-116 "
                "makes selection a read of a partition the fit did not see; one partition "
                "doing both is model selection on the training set"
            )
        return cls(
            contract_version=version,
            training_run_id=str(_require(document, "training_run_id", where)),
            capability_id=str(_require(document, "capability_id", where)),
            backend_kind=str(_require(backend, "kind", f"{where}.training_backend")),
            backend_version=str(_require(backend, "version", f"{where}.training_backend")),
            fit_partition=fit,
            select_partition=select,
            seeds=dict(_require(document, "seeds", where)),
            determinism=dict(_require(document, "determinism", where)),
            preprocessing=dict(_require(document, "preprocessing", where)),
            budget=dict(document.get("budget") or {}),
        )


class RunDirectory:
    """One run's directory, with every path spelled by `RUN_DIRECTORY` and never inline."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ContractViolation(f"run directory {self.root} does not exist")

    # ---- paths -------------------------------------------------------------------- #
    def path(self, role: str) -> Path:
        try:
            return self.root / RUN_DIRECTORY[role]
        except KeyError:
            raise ContractViolation(f"{role!r} is not a member of the run directory") from None

    def staged_path(self, relative: str, *, where: str) -> Path:
        """Resolve a staged volume path, or refuse. Acceptance check 15, enforced.

        Refuses an absolute path, a path that leaves the run directory, and a path outside
        `images/`. The check is on the RESOLVED path and not on the spelling, because
        `images/../../etc/passwd` spells nothing forbidden and resolves somewhere it must
        not: a container that can name its own data can name the test partition.
        """
        if not relative or Path(relative).is_absolute():
            raise ContractViolation(
                f"{where}: {relative!r} is absolute. The staged volumes are named "
                "relative to the run directory; an absolute path is the container naming "
                "its own data (chapter 17 acceptance check 15)"
            )
        resolved = (self.root / relative).resolve()
        images = self.path("images").resolve()
        if not resolved.is_relative_to(images):
            raise ContractViolation(
                f"{where}: {relative!r} resolves to {resolved}, which is outside "
                f"{images}. MOS-TRAIN-141: the pipeline reads the partition it was handed "
                "or it reads nothing"
            )
        if not resolved.is_file():
            raise ContractViolation(
                f"{where}: {relative!r} was named by the cohort and is not staged at "
                f"{resolved}. A case the platform listed and did not stage is a case "
                "silently dropped from a sealed cohort, and the run digest would describe "
                "a dataset version this fit did not see"
            )
        return resolved

    # ---- reads -------------------------------------------------------------------- #
    def request(self) -> RunRequest:
        return RunRequest.from_document(self._json("request"))

    def spec_document(self) -> dict[str, Any]:
        return dict(self._json("spec"))

    def cohort(self, role: str) -> tuple[CohortEntry, ...]:
        """`cohort/fit.jsonl` or `cohort/select.jsonl`, as typed entries. Never empty.

        An empty cohort file is refused rather than trained on. `MOS-TRAIN-141`'s reason
        applies here in its general form: "an empty set is indistinguishable from 'no such
        patient', and a retry loop around an empty set is how a silent read becomes a
        routine one" -- and a fit over zero cases produces weights that are the
        initialisation, which is an artifact that passes every structural check.
        """
        path = self.path(role)
        if not path.is_file():
            raise ContractViolation(f"{path} is missing; the platform stages it before submit")
        out = tuple(
            CohortEntry.from_document(document, where=RUN_DIRECTORY[role])
            for document in self._lines(path)
        )
        if not out:
            raise ContractViolation(
                f"{RUN_DIRECTORY[role]} is empty. A fit over zero cases returns the "
                "initialisation and records it as a trained model"
            )
        return out

    def _json(self, role: str) -> dict[str, Any]:
        path = self.path(role)
        if not path.is_file():
            raise ContractViolation(
                f"{path} is missing. The platform writes it before it starts this process"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ContractViolation(f"{path} is not a JSON object")
        return document

    @staticmethod
    def _lines(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    document = json.loads(line)
                except ValueError as exc:
                    raise ContractViolation(f"{path}:{number} is not JSON: {exc}") from exc
                if not isinstance(document, dict):
                    raise ContractViolation(f"{path}:{number} is not a JSON object")
                yield document

    # ---- writes ------------------------------------------------------------------- #
    def write(self, role: str, document: Mapping[str, Any]) -> Path:
        path = self.path(role)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def success_document(phase: str, **members: Any) -> dict[str, Any]:
    """`result.json` for a phase that finished. `status` is the first key a reader needs."""
    return {"status": "SUCCEEDED", "phase": phase, **members}


def failure_document(
    phase: str, *, reason: str, refusals: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """`result.json` for a phase that did not.

    `reason` is mandatory and is passed straight to `medos.training.runs.fail`, which
    refuses an empty one: "a FAILED training run MUST name a reason (MOS-REL-051)".
    `refusals` carries the engine's own refusal vocabulary when the failure came from one
    of `medos.training`'s checks, so the console renders the same document it renders for
    a refusal at submit rather than a second kind of error.
    """
    if not reason:
        raise ValueError("a failed phase MUST name a reason (MOS-REL-051)")
    return {
        "status": "FAILED",
        "phase": phase,
        "reason": reason,
        "refusals": [dict(r) for r in refusals],
    }
