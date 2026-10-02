# SPDX-License-Identifier: Apache-2.0
"""`MEDOS_TRAINING_ENVIRONMENT`'s nine keys, observed by the image that will train.

WHY THE TRAINER DECLARES THIS AND NOT THE COMPOSE FILE
--------------------------------------------------------
`medos/medos/api/routes_training.py` states the honest gap in its own docstring: "A
deployment-level declaration is an assertion by the operator, not an observation by the
runner. The observation point is `medos.training.runs.start()`, inside the process that
actually holds the GPU, and this surface does not own it."

This module is that observation point, moved one step earlier so the answer exists before
a run is submitted rather than after one has started. Seven of the nine keys are read off
the software and the hardware that will do the work:

    code_commit        the build stamp (entry 82)
    code_dirty         the build stamp (entry 82)
    image_digest       the build stamp (entry 82)
    backend_versions   `importlib.metadata`, from the installed nnU-Net
    framework_versions `importlib.metadata`, from the installed torch / numpy / SimpleITK
    hardware           `torch.cuda` and `nvidia-smi`, on the device compose granted
    preprocessing      the deployment's capability -> spec binding, with the DIGEST
                       computed from the spec document rather than typed beside it

Two are a POLICY rather than an observation, and they are marked as such here because
`MOS-TRAIN-126` turns on exactly that distinction -- "the determinism settings actually
used are recorded rather than asserted":

    seeds              what `backend.apply_determinism` will seed with
    determinism        what `backend.apply_determinism` will set

They are not observations at declaration time, and they would be a lie if the trainer then
did something else. What makes them true is that `backend.apply_determinism` reads THESE
VALUES out of `request.json` and applies them, and `trainer/tests/test_determinism.py`
asserts the two agree. A constant declared in one file and applied from another is the
shape of every drift this chapter is written about.

WHAT THIS MODULE REFUSES TO DECLARE
-------------------------------------
`backend_versions` carries `nnunet` and nothing else, because this image installs nothing
else. A submit naming `auto3dseg` then gets `503` from `TrainingEnvironment.version_of`
naming `backend_versions.auto3dseg`, which is the truth: this deployment cannot train that
backend. Declaring a version for a package that is not installed would be the entry-82
defect in a third place.

`hardware` is REFUSED rather than defaulted when no CUDA device is visible. A trainer that
silently falls back to CPU and takes four days is worse than one that refuses, and a
`hardware` block claiming a GPU that is not attached is worse than both -- it is false
inside `MOS-TRAIN-124`'s binding. `MEDOS_TRAINER_ALLOW_CPU=1` is the explicit opt-in, and
what it buys is a block that records `gpu_count: 0` and says so.

Spec: MOS-TRAIN-124, MOS-TRAIN-126, MOS-IMG-045, MOS-IMG-049, MOS-REL-037, MOS-REL-046,
MOS-REL-051, docs/spec/99-known-inconsistencies.md entry 82.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess  # noqa: S404 - nvidia-smi, argv vector, no shell
from pathlib import Path
from typing import Any, Final

from medos_trainer import BACKEND_KIND
from medos_trainer.stamp import read_stamp

__all__ = [
    "DEFAULT_BINDINGS_PATH",
    "DETERMINISM",
    "ENVIRONMENT_KEYS",
    "SEEDS",
    "HardwareUnavailable",
    "declare",
    "observe_hardware",
    "preprocessing_bindings",
]

#: `medos/medos/api/routes_training.py::_ENVIRONMENT_KEYS`, restated. Both sides assert it and
#: `tests/unit/test_trainer_environment.py` imports both: nine keys named twice with no
#: test between them is how a deployment discovers a missing one at the first submit.
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

#: `medos.training.runs._SEED_KEYS`. One fixed value per key, and it is a CHOICE this
#: image makes and records -- `MOS-TRAIN-127`'s seed-variance study is what a site runs to
#: find out how much that choice was worth, and it needs a seed it can vary against.
SEEDS: Final[dict[str, int]] = {
    "python": 20260311,
    "numpy": 20260311,
    "torch": 20260311,
    "dataloader_worker_base": 900,
}

#: `medos.training.runs._DETERMINISM_KEYS`, as the settings this trainer APPLIES.
#:
#: `torch_use_deterministic_algorithms` is FALSE and that is the honest setting, not a
#: shortcut. `MOS-TRAIN-126` is explicit: "Bit-exact reproducibility of a training run MUST
#: NOT be required and MUST NOT be claimed ... forcing every deterministic flag costs
#: throughput while still not covering every kernel." nnU-Net's 3D convolutions have no
#: deterministic CUDA implementation for several of the backward kernels, so
#: `use_deterministic_algorithms(True)` makes the fit RAISE rather than become
#: reproducible. Recording `false` is the requirement's own position; recording `true` and
#: catching the exception would be the assertion `MOS-TRAIN-126` forbids.
#:
#: `cudnn_benchmark` is TRUE for the opposite reason and the cost is stated: nnU-Net's
#: patch size is fixed by the plan, so the autotuner converges once and the throughput is
#: worth several hours over a full run. It is the single largest contributor to run-to-run
#: kernel variation and it is recorded, which is all `MOS-TRAIN-126` asks.
DETERMINISM: Final[dict[str, Any]] = {
    "torch_use_deterministic_algorithms": False,
    "cudnn_benchmark": True,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    "tf32_allowed": True,
}

#: `MOS-TRAIN-124`'s `framework_versions` keys. `monai` is among them and this image does
#: not install MONAI: the platform GENERATES MONAI Bundle configs as data and never
#: imports it, and nnU-Net does not need it. The key is present and the value says
#: `absent`, which is a record; omitting the key would trip
#: `reproducibility_block_incomplete` and declaring a version would be a claim about a
#: package `pip freeze` cannot find.
_FRAMEWORK_DISTRIBUTIONS: Final[dict[str, str]] = {
    "torch": "torch",
    "monai": "monai",
    "numpy": "numpy",
    "simpleitk": "SimpleITK",
}

#: The backend distribution behind each `training_backend.kind` this image can serve.
_BACKEND_DISTRIBUTIONS: Final[dict[str, str]] = {"nnunet": "nnunetv2"}

DEFAULT_BINDINGS_PATH: Final[str] = "/etc/medos/preprocessing-bindings.json"

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

    `nvidia-smi` is injected by the container runtime, not installed by the Dockerfile, so
    its absence means "this container was started without a GPU" and is a normal branch
    rather than an error. `MOS-OPS-090`'s reconciliation is the platform's; this is one
    field for one record.
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
        # The binary is on PATH and will not execute: the runtime mounted the tool
        # without the driver. Reported as "no device", which is what it means here.
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def observe_hardware(*, allow_cpu: bool | None = None) -> dict[str, Any]:
    """`MOS-TRAIN-124`'s six hardware keys, read off the device. Refuses when there is none.

    `allow_cpu` defaults to `MEDOS_TRAINER_ALLOW_CPU`, and the default of THAT is off. The
    refusal is the product: a run that took four days on a CPU and produced a model is
    indistinguishable, in the record, from one that took four hours on the GPU the
    deployment believes it has.
    """
    if allow_cpu is None:
        allow_cpu = os.environ.get("MEDOS_TRAINER_ALLOW_CPU", "").strip() in ("1", "true")

    torch = importlib.import_module("torch")
    count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    driver = _nvidia_smi("driver_version") or _ABSENT

    if count == 0:
        if not allow_cpu:
            raise HardwareUnavailable(
                "no CUDA device is visible to this container. The trainer REFUSES rather "
                "than falling back to CPU: a 3D segmentation fit that takes four days on "
                "CPU is not the same experiment, and MOS-TRAIN-124's `hardware` block "
                "would record a GPU this run did not use. Grant the device "
                "(deploy.resources.reservations.devices in docker-compose.yml, or "
                "`--gpus all`), or set MEDOS_TRAINER_ALLOW_CPU=1 to record a CPU run AS "
                "a CPU run"
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


def _load_spec_document(reference: str) -> dict[str, Any]:
    """`module:callable` or a path to a JSON document. Never a guess.

    The `module:callable` form exists so a deployment can name the spec the platform
    itself ships -- `medos.sdk.fixtures:selftest_spec_document` -- and get the exact
    document `MOS-IMG-054`'s self-test executes, rather than a copy of it that has drifted.
    """
    if ":" in reference and not Path(reference).exists():
        module_name, _, attribute = reference.partition(":")
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        document = value() if callable(value) else value
        if not isinstance(document, dict):
            raise ValueError(f"{reference} did not produce a JSON object")
        return dict(document)
    path = Path(reference)
    if not path.is_file():
        raise FileNotFoundError(
            f"the preprocessing binding names {reference!r}, which is neither an "
            "importable `module:callable` nor a file on this image"
        )
    return dict(json.loads(path.read_text(encoding="utf-8")))


def preprocessing_bindings(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """`{capability_id: {id, version, digest, output_kind}}`, with the digest COMPUTED.

    `medos/medos/api/routes_training.py::_SPEC_KEYS` requires all four per capability, and
    `MOS-IMG-045` makes the `PreprocessingSpec` the registered artifact a run is fitted
    against. Which spec a deployment uses is a deployment decision and is declared in
    `preprocessing-bindings.json`; the DIGEST is not a decision and is not declared. It is
    taken here, over the spec document itself, under the same canonical form the platform
    digests everything else with -- so a spec that was edited produces a different digest
    on the next declaration instead of a stale one that matches nothing.
    """
    from medos.sdk.canonical import canonical_bytes, sha256_hex

    source = Path(path or os.environ.get("MEDOS_TRAINER_BINDINGS", "")
                  or DEFAULT_BINDINGS_PATH)
    if not source.is_file():
        raise FileNotFoundError(
            f"no preprocessing bindings at {source}. MOS-TRAIN-124 binds a "
            "PreprocessingSpec id, version and digest into every run and MOS-IMG-045 "
            "makes it the registered artifact the run is fitted against; there is no "
            "default for which spec a deployment trains against"
        )
    declared = json.loads(source.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for capability_id, binding in dict(declared.get("capabilities") or {}).items():
        document = _load_spec_document(str(binding["spec_document"]))
        out[str(capability_id)] = {
            "id": str(document["id"]),
            "version": int(str(document["version"]).split(".")[0]),
            "digest": "sha256:" + sha256_hex(canonical_bytes(document)),
            "output_kind": str(binding["output_kind"]),
            # Not one of `_SPEC_KEYS`; carried so a reviewer can find the document the
            # digest was taken over without reading this module.
            "spec_document": str(binding["spec_document"]),
        }
    if not out:
        raise ValueError(f"{source} binds no capability; every submit would answer 503")
    return out


def declare(
    *,
    bindings_path: str | os.PathLike[str] | None = None,
    stamp_path: str | os.PathLike[str] | None = None,
    allow_cpu: bool | None = None,
) -> dict[str, Any]:
    """The nine-key document. Raises rather than emitting an incomplete one.

    `medos/medos/api/routes_training.py::load_training_environment` reads the result -- from
    `MEDOS_TRAINING_ENVIRONMENT` directly, or from `@<path>` naming a file holding it. The
    compose stack uses the `@` form and a one-shot service that runs this function, so the
    platform image never has to know how any of it was derived.
    """
    stamp = read_stamp(stamp_path)
    document: dict[str, Any] = {
        "code_commit": stamp["code_commit"],
        "code_dirty": bool(stamp["code_dirty"]),
        "image_digest": stamp["image_digest"],
        "backend_versions": {
            kind: _version(distribution)
            for kind, distribution in _BACKEND_DISTRIBUTIONS.items()
        },
        "seeds": dict(SEEDS),
        "determinism": dict(DETERMINISM),
        "hardware": observe_hardware(allow_cpu=allow_cpu),
        "framework_versions": {
            key: _version(distribution)
            for key, distribution in _FRAMEWORK_DISTRIBUTIONS.items()
        },
        "preprocessing": preprocessing_bindings(bindings_path),
    }

    absent = [
        f"backend_versions.{k}"
        for k, v in document["backend_versions"].items()
        if v == _ABSENT
    ]
    if absent:
        raise RuntimeError(
            f"{absent} -- this image claims to serve a backend it did not install. "
            "MOS-REL-037 requires an exact pin on a recorded version and there is "
            "nothing to pin"
        )
    if document["backend_versions"].get(BACKEND_KIND) in (None, _ABSENT):
        raise RuntimeError(f"the declared backend {BACKEND_KIND!r} is not installed")

    missing = [k for k in ENVIRONMENT_KEYS if k not in document]
    if missing:  # pragma: no cover - the literal above is the whole set
        raise RuntimeError(f"the declaration is missing {missing}")
    return document
