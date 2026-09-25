# SPDX-License-Identifier: Apache-2.0
"""`medos-trainer`: the process that fits the model, and the only thing that imports torch.

WHAT THIS PACKAGE IS
---------------------
The other side of `medos/training/orchestrator.py`'s port. The platform writes a run
directory, starts `/usr/local/bin/medos-trainer` as a child process with a scrubbed
environment, and reads back files and an exit code. There is no socket, no RPC and no
shared interpreter, which is what lets `medicalos/medos` carry no torch, no MONAI and no
nnU-Net at all (`MOS-TRAIN-225`, `MOS-REL-108`).

THE MODULES, AND THE ONE RULE EACH CARRIES

    stamp        entry 82's two build facts, recorded at `docker build` and read back
                 here. Stdlib only, because it runs inside the build.
    contract     the run directory: what the platform writes, what the trainer writes
                 back, and the two documents that carry a run's terminal state.
    environment  `MEDOS_TRAINING_ENVIRONMENT`'s nine keys, OBSERVED rather than declared
                 wherever observing them is possible (`MOS-TRAIN-126`).
    backend      nnU-Net v2, in two phases: `plan` derives the fingerprint from the fit
                 partition and nothing else (`MOS-TRAIN-135`), `fit` trains against the
                 frozen plan and never re-derives (`MOS-TRAIN-225`).
    packaging    the derived plan -> `PreprocessingSpec` -> MONAI Bundle, through
                 `medicalos_preprocessing.autoconfig` and `medicalos_preprocessing.bundle` rather than
                 through a second implementation of either.
    executor     the trainer deployable's own supervisor: it picks up a run a PERSON
                 submitted and drives it. It never submits one.

WHY IT IMPORTS `medos` INSTEAD OF REIMPLEMENTING IT
----------------------------------------------------
`MOS-TRAIN-129` fixes the bundle layout, `MOS-TRAIN-131` requires `configs/inference.json`
to be "byte-reproducible from the spec alone" by "a single generator", and
`MOS-TRAIN-223`'s exporter "MUST refuse to emit for any derived quantity it cannot map
exactly". All three already exist in `medos.training`, are pure Python, and are what the
platform reads the result back with. A trainer with its own copy would be a second
implementation of each, and the failure mode is the quiet one: the platform verifying a
bundle against a layout the trainer stopped emitting.

The dependency is one-way and stays one-way. `medos` does not import this package and
cannot: `tests/integration/test_trainer_boundary.py` asserts that `medicalos/medos`
contains no torch, no MONAI and no nnU-Net, and `pyproject.toml` names none of them.

Spec: docs/spec/17-training-pipeline.md 17.7.2, 17.7.3, 17.7.6; MOS-TRAIN-121 to
MOS-TRAIN-137, MOS-TRAIN-190, MOS-TRAIN-211, MOS-TRAIN-223 to MOS-TRAIN-226.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()

#: The backend this image ships, and the only value of `training_backend.kind` it will
#: accept. `MOS-UI-148` requires a dataset-fingerprint auto-configuring backend --
#: `nnunet` or `auto3dseg` -- for the no-code console, and `MOS-TRAIN-135` names nnU-Net's
#: plan freeze explicitly. `trainer/README.md` records why this image chose nnU-Net
#: over Auto3DSeg; an `auto3dseg` image is a sibling of this one, not a mode of it, because
#: two backends in one image means two fingerprint documents and one `pip freeze`.
BACKEND_KIND: str = "nnunet"
