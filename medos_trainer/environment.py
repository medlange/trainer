# SPDX-License-Identifier: Apache-2.0
"""`MEDOS_TRAINING_ENVIRONMENT`'s nine keys, observed by the installation.

WHY THE TRAINER DECLARES THIS
------------------------------
A deployment-level declaration is an assertion by the operator, not an
observation by the runner. This module is the observation point: it reads the
facts off the software and the hardware that will do the work, so the answer
exists before a fit is attempted rather than after one has started.

THE NINE KEYS, AND WHERE EACH COMES FROM
-----------------------------------------
    code_commit        the build stamp (entry 82), recorded at `docker build`
    code_dirty         the build stamp (entry 82)
    image_digest       the build stamp (entry 82)
    backend_versions   EMPTY, and that is the fact: the vanilla stack is its own
                       backend. There is no pluggable backend registry and no
                       external training framework to pin
    framework_versions `importlib.metadata`, from the installed torch / numpy
    hardware           `torch.cuda` and `nvidia-smi`, on the device granted
    preprocessing      EMPTY, and that is the fact: a standalone run binds no
                       deployment-level preprocessing spec. The plan carries the
                       preprocessing decisions (spacing, normalisation, patch)
                       with the reasons attached
    seeds              what the trainer seeds with, applied by the fit
    determinism        what the trainer sets, applied by the fit

The last two are a POLICY rather than an observation, and they are marked as
such: they are recorded here so a run's provenance can name the settings
actually used, and the fit reads THESE values rather than re-declaring its
own.

WHAT THIS MODULE REFUSES TO DECLARE
-------------------------------------
`hardware` is REFUSED rather than defaulted when no CUDA device is visible. A
trainer that silently falls back to CPU and takes four days is worse than one
that refuses, and a `hardware` block claiming a GPU that is not attached is
worse than both. `MEDOS_TRAINER_ALLOW_CPU=1` is the explicit opt-in, and what
it buys is a block that records `gpu_count: 0` and says so.

IMPORTABLE WITH ONLY STDLIB + TORCH/NUMPY. That is a property of this module:
`declare-environment` and `doctor` run where the full fit never will.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess  # noqa: S404 - nvidia-smi, argv vector, no shell
from typing import Any, Final

from medos_trainer.stamp import read_stamp

__all__ = [
    "DETERMINISM",
    "ENVIRONMENT_KEYS",
    "SEEDS",
    "HardwareUnavailable",
    "declare",
    "observe_hardware",
]

#: The declaration's vocabulary. Nine keys, named once, so the document a
#: deployment emits and the document a reviewer reads are one shape.
ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "code_commit",
    "code_dirty",
    "image_digest",
    "backend_versions",
    "seeds",
    "determinism",
    "hardware",
    "framework_versions",
    "preprocessing",
)

#: One fixed value per key, and it is a CHOICE this installation makes and
#: records -- a seed-variance study is what a site runs to find out how much
#: that choice was worth, and it needs a seed it can vary against.
SEEDS: Final[dict[str, int]] = {
    "python": 20260311,
    "numpy": 20260311,
    "torch": 20260311,
    "dataloader_worker_base": 900,
}

#: The settings this trainer APPLIES.
#:
#: `torch_use_deterministic_algorithms` is FALSE and that is the honest
#: setting, not a shortcut. Bit-exact reproducibility of a training run is not
#: required and must not be claimed: forcing every deterministic flag costs
#: throughput while still not covering every kernel, and several of the 3D
#: backward kernels have no deterministic CUDA implementation at all, so the
#: flag would make the fit RAISE rather than become reproducible. Recording
#: `false` is the honest position; recording `true` and catching the exception
#: would be an assertion without a fact behind it.
#:
#: `cudnn_benchmark` is TRUE for the opposite reason and the cost is stated:
#: the patch size is fixed by the plan, so the autotuner converges once and
#: the throughput is worth it over a full run. It is the single largest
#: contributor to run-to-run kernel variation and it is recorded, which is all
#: a reproducibility record can honestly do.
DETERMINISM: Final[dict[str, Any]] = {
    "torch_use_deterministic_algorithms": False,
    "cudnn_benchmark": True,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    "tf32_allowed": True,
}

#: The distributions the framework versions are read from. The framework is
#: vanilla PyTorch: torch and numpy are the whole list. There is deliberately
#: no row for an external segmentation framework -- the framework is this
#: source tree, and its version is the build stamp's `code_commit`.
_FRAMEWORK_DISTRIBUTIONS: Final[dict[str, str]] = {
    "torch": "torch",
    "numpy": "numpy",
}

_ABSENT: Final[str] = "absent"


class HardwareUnavailable(RuntimeError):
    """No CUDA device. Raised rather than degraded; see the module docstring."""


def _version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version(distribution))
    except PackageNotFoundError:
        return _ABSENT


def _nvidia_smi(query: str) -> str | None:
    """One `nvidia-smi` field, or `None` when the tool or the device is not there.

    `nvidia-smi` is injected by the container runtime, not installed by the
    Dockerfile, so its absence means "this container was started without a
    GPU" and is a normal branch rather than an error.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - argv vector, no shell
            [binary, f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except OSError:
        # The binary is on PATH and will not execute: the runtime mounted the
        # tool without the driver. Reported as "no device", which is what it
        # means here.
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def observe_hardware(*, allow_cpu: bool | None = None) -> dict[str, Any]:
    """The six hardware keys, read off the device. Refuses when there is none.

    `allow_cpu` defaults to `MEDOS_TRAINER_ALLOW_CPU`, and the default of
    THAT is off. The refusal is the product: a run that took four days on a
    CPU and produced a model is indistinguishable, in the record, from one
    that took four hours on the GPU the deployment believes it has.
    """
    if allow_cpu is None:
        allow_cpu = os.environ.get("MEDOS_TRAINER_ALLOW_CPU", "").strip() in ("1", "true")

    torch = importlib.import_module("torch")
    count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    driver = _nvidia_smi("driver_version") or _ABSENT

    if count == 0:
        if not allow_cpu:
            raise HardwareUnavailable(
                "no CUDA device is visible to this process. The trainer REFUSES "
                "rather than falling back to CPU: a 3D segmentation fit that takes "
                "four days on CPU is not the same experiment, and the declaration "
                "would record a GPU this run did not use. Grant the device "
                "(--gpus all, or deploy.resources.reservations.devices in compose), "
                "or set MEDOS_TRAINER_ALLOW_CPU=1 to record a CPU run AS a CPU run"
            )
        return {
            "gpu_model": _ABSENT,
            "gpu_count": 0,
            "driver": driver,
            "cuda": _ABSENT,
            "cudnn": str(torch.backends.cudnn.version() or _ABSENT),
            "nccl": _ABSENT,
            "accelerator": "cpu",
            "cpu_run_explicitly_permitted": True,
        }

    try:
        nccl = ".".join(str(p) for p in torch.cuda.nccl.version())
    except (AttributeError, RuntimeError):  # pragma: no cover - platform dependent
        nccl = _ABSENT
    return {
        "gpu_model": str(torch.cuda.get_device_name(0)),
        "gpu_count": count,
        "driver": driver,
        "cuda": str(torch.version.cuda or _ABSENT),
        "cudnn": str(torch.backends.cudnn.version() or _ABSENT),
        "nccl": nccl,
        "accelerator": "cuda",
        "vram_mib": int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)),
    }


def declare(
    *,
    stamp_path: str | os.PathLike[str] | None = None,
    allow_cpu: bool | None = None,
) -> dict[str, Any]:
    """The nine-key document. Raises rather than emitting an incomplete one.

    `backend_versions` and `preprocessing` are EMPTY DICTS, recorded rather
    than omitted: the keys stay because the nine-key shape is the vocabulary
    this trainer has always emitted, and the empties are the true statement --
    a standalone vanilla stack binds no external backend and no deployment
    spec. An absent key and a key that says "nothing is bound" are different
    facts, and only the second is true here.
    """
    stamp = read_stamp(stamp_path)
    document: dict[str, Any] = {
        "code_commit": stamp["code_commit"],
        "code_dirty": bool(stamp["code_dirty"]),
        "image_digest": stamp["image_digest"],
        "backend_versions": {},
        "seeds": dict(SEEDS),
        "determinism": dict(DETERMINISM),
        "hardware": observe_hardware(allow_cpu=allow_cpu),
        "framework_versions": {
            key: _version(distribution)
            for key, distribution in _FRAMEWORK_DISTRIBUTIONS.items()
        },
        "preprocessing": {},
    }

    missing = [k for k in ENVIRONMENT_KEYS if k not in document]
    if missing:  # pragma: no cover - the literal above is the whole set
        raise RuntimeError(f"the declaration is missing {missing}")
    return document
