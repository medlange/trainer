# SPDX-License-Identifier: Apache-2.0
"""The derived plan -> a `PreprocessingSpec` -> a MONAI Bundle the platform already reads.

WHAT THIS MODULE DOES NOT DO, AND WHY THAT IS THE POINT
---------------------------------------------------------
It does not write a bundle. `medos.sdk.bundle.write_bundle` writes it and
`medos.sdk.bundle.verify` reads it back, and this module calls both -- the same two
functions the platform calls. `MOS-TRAIN-129` fixes the layout, `MOS-TRAIN-130` fixes the
metadata contract and `MOS-TRAIN-131` requires `configs/inference.json` to be "generated
from the registered `PreprocessingSpec` by a single generator ... byte-reproducible from
the spec alone". A trainer with its own packager would be a SECOND implementation of all
three, and the failure mode is the quiet one: a platform verifying a bundle against a
layout the trainer stopped emitting, which passes until the day it does not.

It does not transcribe the fingerprint either. `medos.sdk.autoconfig.
export_spec_fields` does that, under `MOS-TRAIN-223`'s fixed mapping table, and it
"MUST refuse to emit for any derived quantity it cannot map exactly". What is here is the
MERGE -- putting the exporter's dotted field paths into the bound spec document -- and the
two consequences of the merge that the exporter cannot see: the crop floor and the golden
fixture.

THE GOLDEN FIXTURE HAS TO BE RE-RECORDED AND `MOS-TRAIN-226` IS WHY
---------------------------------------------------------------------
"`MOS-TRAIN-060` already requires the fixture hash to be recorded as the last step before
signing -- which, for an auto-configured model, means *after* the derived constants are
frozen into the spec." The bound spec carries the hash recorded against ITS constants;
the derived spec has different constants -- a different target spacing, a different
intensity window, a different patch -- so the same phantom through the same chain produces
a different tensor. Carrying the old hash forward would make `MOS-IMG-054`'s startup
self-test fail on a correct model, and, worse, would make it PASS on an incorrect one if
the constants happened to round back.

`medos.sdk.preprocess.record_golden` is the ONE recorder (`MOS-TRAIN-134`,
`MOS-IMG-058`) and is what is called here. There is no second hashing function in this
file and no comparison: `MOS-TRAIN-065` forbids an operation that RE-records an existing
fixture, and this is the first recording for a spec that did not exist until the planner
ran.

`MOS-TRAIN-224`, AND THE COLUMN THAT CANNOT CARRY IT
------------------------------------------------------
"The derived *training* batch size MUST be recorded in `TrainingRun.hyperparameters` and
MUST NOT be written into `PreprocessingSpec.patch.batch_size`." The second half is
enforced here -- `_merge` writes only the fields the exporter returned, and the exporter
puts the training batch size in a different dict. The first half CANNOT be satisfied on
this deployment and the reason is recorded in `trainer/README.md`: `0013_training.
up.sql`'s `training_runs_guard()` seals `hyperparameters` at insert, and
`medos/medos/api/routes_training.py` inserts `{}`. The derived value is written into
`plan.json` and into the bundle's own record instead, and the gap is REPORTED rather than
worked around by putting it somewhere `MOS-TRAIN-224` did not ask for.

Spec: MOS-TRAIN-060, MOS-TRAIN-065, MOS-TRAIN-129 to MOS-TRAIN-134, MOS-TRAIN-136,
MOS-TRAIN-154, MOS-TRAIN-223, MOS-TRAIN-224, MOS-TRAIN-226, MOS-IMG-049, MOS-IMG-054,
MOS-IMG-058.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from medos.sdk.contract import ContractViolation

__all__ = [
    "MONAI_BUNDLE_TARGET",
    "derive_spec_document",
    "golden_fixture_bytes",
    "torchscript_bytes",
    "write_candidate_bundle",
    "write_modelcard",
]

#: The MONAI version `medos/medos/training/chain.py`'s generated `configs/inference.json`
#: targets. It is a statement about a FILE FORMAT and not about an installed package:
#: MONAI is deliberately absent from this image (`framework_versions.monai` declares
#: `absent`), because the platform "GENERATES MONAI Bundle configs as data and never
#: imports MONAI". `MOS-TRAIN-130` requires `monai_version` in `configs/metadata.json`,
#: and what it means there is which MONAI can load this bundle -- `Orientationd`,
#: `Spacingd`, `ScaleIntensityRanged`, `NormalizeIntensityd`, `EnsureTyped` and
#: `sliding_window_inference` have had these signatures since 1.3.
MONAI_BUNDLE_TARGET: Final[str] = "1.4.0"

#: `medos.sdk.bundle.write_bundle`'s three serialised forms. TorchScript is chosen
#: here because `MOS-TRAIN-154`'s table admits it, because it is the form a torch trainer
#: can emit without a second conversion tool, and because `MOS-TRAIN-159` to
#: `MOS-TRAIN-169`'s conversion-equivalence checks are a `ConversionRun`'s job and not a
#: training run's -- exporting ONNX here would be performing a conversion whose equivalence
#: nobody measured.
_WEIGHTS_FORMAT: Final[str] = "torchscript"


#: `medos/medos/training/autoconfig.py` spells the three volume axes `z`, `y`, `x`;
#: `medos/medos/training/spec.py` spells the same three `k`, `j`, `i` and REFUSES anything
#: else -- "axis_order: must be a permutation of ['k','j','i']".
#:
#: A DEFECT IN THE PLATFORM, FOUND BY RUNNING IT, REPORTED RATHER THAN ARGUED WITH. The
#: exporter `MOS-TRAIN-223` makes authoritative emits a value the parser
#: `MOS-IMG-049` makes authoritative cannot accept, so no real nnU-Net plan can be
#: transcribed into a spec on the shipped code. The fix belongs in
#: `medos/medos/training/autoconfig.py::_coerce`, which this change does not own.
#:
#: The translation below is EXACT and is not a substitution: both alphabets name the
#: same three axes in the same order -- `k`/`z` the slice axis, `j`/`y` the row,
#: `i`/`x` the column, which is `medos.sdk.preprocess.ChainInput`'s documented
#: `(C, K, J, I)` and `CanonicalVolume`'s `[k, j, i]`. Anything not in this map is left
#: alone so that a genuinely unmappable value still reaches the parser and is refused
#: there, rather than being quietly renamed here.
_AXIS_ALPHABET: Final[dict[str, str]] = {"z": "k", "y": "j", "x": "i"}


def _spelled_for_the_spec(path: str, value: Any) -> Any:
    if path != "axis_order" or not isinstance(value, list):
        return value
    return [_AXIS_ALPHABET.get(str(axis), str(axis)) for axis in value]


def _set_dotted(document: dict[str, Any], path: str, value: Any) -> None:
    node = document
    parts = path.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def derive_spec_document(
    bound: Mapping[str, Any], plan: Mapping[str, Any], *, code_commit: str
) -> dict[str, Any]:
    """The bound spec with `MOS-TRAIN-223`'s derived fields merged in, and re-recorded.

    `bound` is the deployment's registered spec -- the template. Every field the exporter
    returned overwrites its counterpart; everything the fingerprint does not determine
    (the orientation target, the interpolators, the inverse block, `io.label_set`) is the
    deployment's and is carried through unchanged, because the fingerprint has nothing to
    say about it and inventing a value would be the substitution `MOS-TRAIN-223` forbids.
    """
    from medos.sdk.fixtures import phantom, phantom_input
    from medos.sdk.preprocess import record_golden
    from medos.sdk.spec import parse_spec

    document: dict[str, Any] = _deep_copy(bound)
    for path, value in dict(plan["spec_fields"]).items():
        _set_dotted(document, path, _spelled_for_the_spec(path, value))

    patch = [int(n) for n in document["patch"]["size_voxels"]]
    crop = dict(document.get("foreground_crop") or {})
    if crop.get("mode") != "none":
        # `MOS-TRAIN-051`: the crop is at least the patch on every axis, which is what
        # makes `first_patch`'s "the first window is the origin corner" true. The derived
        # patch is new, so the floor derived from the old one no longer holds. Set to the
        # patch exactly -- the tightest form the requirement permits, so a later change to
        # either shows up rather than being absorbed by slack.
        crop["min_size_voxels"] = list(patch)
        document["foreground_crop"] = crop

    # MOS-TRAIN-224, the half that IS enforceable here: the derived TRAINING batch size
    # never reaches `patch.batch_size`. The sliding-window batch stays the deployment's
    # serving knob (MOS-OPS-078), as the requirement requires.
    document["patch"]["batch_size"] = int(dict(bound["patch"]).get("batch_size", 1))

    import numpy as np

    document["backend"] = {
        "resampler": str(dict(bound["backend"])["resampler"]),
        "numpy": f"numpy=={np.__version__}",
    }

    # MOS-TRAIN-226 / MOS-TRAIN-060: record the fixture hash AFTER the derived constants
    # are frozen into the spec, with the ONE recorder.
    shape = _phantom_shape_for(document)
    data = phantom_input() if shape is None else _phantom_input(phantom(shape))
    document["golden_fixture"] = {
        "path": "docs/golden_fixture.nii.gz",
        "sha256": _array_digest(data.arrays["image"]),
        "output_tensor_sha256": "sha256:" + "0" * 64,
        "output_shape": [1, 1, *patch],
        "recorded_at": _now(),
        "recorded_by_commit": code_commit,
    }
    probe = parse_spec(document)
    digest, recorded_shape = record_golden(probe, data)
    document["golden_fixture"]["output_tensor_sha256"] = digest
    document["golden_fixture"]["output_shape"] = list(recorded_shape)

    # Parsed once more so a caller can never be handed a document the platform's own
    # parser would refuse. `parse_spec` is the validator; there is no second
    # one in this file.
    parse_spec(document)
    return document


def _deep_copy(document: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return dict(json.loads(json.dumps(dict(document))))


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _array_digest(array: Any) -> str:
    import hashlib

    import numpy as np

    buf = np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")
    return "sha256:" + hashlib.sha256(buf).hexdigest()


def _phantom_input(array: Any) -> Any:
    from medos.sdk.fixtures import PHANTOM_AXCODES, PHANTOM_SPACING_MM
    from medos.sdk.preprocess import ChainInput

    return ChainInput(
        arrays={"image": array}, spacing_mm=PHANTOM_SPACING_MM, axcodes=PHANTOM_AXCODES
    )


def _phantom_shape_for(document: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """A phantom big enough that the derived patch is a window and not a pad.

    `first_patch` REFUSES a model-space tensor smaller than `patch.size_voxels`, citing
    `MOS-TRAIN-051`: "the sliding window's own padding would be reached". The shipped
    phantom is 48x64x64 at 2.0/1.0/1.0 mm and the planner's patch for a real cohort is
    routinely larger, so the fixture is grown to match rather than the check relaxed.
    Returns `None` when the shipped shape already suffices, so the common case uses the
    fixture `MOS-TRAIN-133` pins byte-for-byte.
    """
    from medos.sdk.fixtures import PHANTOM_SHAPE, PHANTOM_SPACING_MM

    patch = [int(n) for n in document["patch"]["size_voxels"]]
    target = [float(v) for v in document["target_spacing_mm"]]
    # Model-space voxels along each axis, if the whole phantom survived the crop.
    needed = [
        int(math.ceil(p * t / s)) + 8
        for p, t, s in zip(patch, target, PHANTOM_SPACING_MM)
    ]
    if all(n <= m for n, m in zip(needed, PHANTOM_SHAPE)):
        return None
    return (
        max(needed[0], PHANTOM_SHAPE[0]),
        max(needed[1], PHANTOM_SHAPE[1]),
        max(needed[2], PHANTOM_SHAPE[2]),
    )


def golden_fixture_bytes(spec_document: Mapping[str, Any]) -> tuple[bytes, bytes]:
    """`(docs/golden_fixture.nii.gz, docs/golden_fixture_tensor.f32)`, from the spec.

    Both are DERIVED from the spec that was just recorded, so the bundle cannot carry a
    fixture the spec's hash was not taken over. `MOS-REG-036` requires the fixture in the
    artifact and `MOS-IMG-054` runs it at worker startup; a bundle whose fixture and hash
    came from two places is a self-test that tests nothing.
    """
    import nibabel as nib
    import numpy as np
    from medos.sdk.fixtures import PHANTOM_SPACING_MM, phantom
    from medos.sdk.preprocess import first_patch, model_space_tensor
    from medos.sdk.spec import parse_spec

    spec = parse_spec(dict(spec_document))
    shape = _phantom_shape_for(spec_document)
    array = phantom() if shape is None else phantom(shape)
    data = _phantom_input(array)

    tensor = first_patch(spec, model_space_tensor(spec, data))
    tensor_bytes = np.ascontiguousarray(tensor, dtype="<f4").tobytes(order="C")

    # The volume as NIfTI, in the fixture's own (K, J, I) order with its own spacing. The
    # channel axis is dropped: a NIfTI of a single-channel CT is 3D, and the channel is
    # the bundle's `network_data_format` concern, not the file's.
    volume = np.ascontiguousarray(array[0], dtype=np.float32)
    affine = np.diag([*PHANTOM_SPACING_MM, 1.0]).astype(np.float64)
    image = nib.Nifti1Image(volume, affine)
    buffer = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = buffer
    image.to_file_map(file_map)
    return buffer.getvalue(), tensor_bytes


def torchscript_bytes(network: Any, *, patch: list[int], channels: int) -> bytes:
    """The trained network as TorchScript, traced on one patch-shaped input.

    DEEP SUPERVISION IS TURNED OFF BEFORE TRACING, and it is not cosmetic. nnU-Net's
    decoder returns a LIST of logits at descending resolutions while training; the served
    artifact returns one tensor at the input resolution. Tracing with it on produces a
    module whose output is a tuple, which `MOS-TRAIN-130`'s `network_data_format.outputs`
    cannot describe and which the serving runner would index by position -- the classic
    way a model comes to be served at the wrong scale with every structural check passing.
    """
    import torch

    # nnU-Net wraps the network in `torch.compile` during `initialize()`, so what the
    # trainer holds is an `OptimizedModule`. Tracing that traces the dynamo wrapper;
    # `_orig_mod` is the real network and is what the served artifact has to be.
    module = getattr(network, "_orig_mod", network)
    decoder = getattr(module, "decoder", None)
    if decoder is not None and hasattr(decoder, "deep_supervision"):
        decoder.deep_supervision = False
    module = module.eval()

    device = next(module.parameters()).device
    example = torch.zeros((1, channels, *patch), dtype=torch.float32, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
        traced = torch.jit.freeze(traced)
        out = traced(example)
    if not isinstance(out, torch.Tensor):  # pragma: no cover - the guard above prevents it
        raise ContractViolation(
            f"the traced network returns {type(out).__name__} and not one tensor; deep "
            "supervision is still on and the served artifact would be a tuple"
        )
    buffer = io.BytesIO()
    torch.jit.save(traced.cpu(), buffer)
    return buffer.getvalue()


def write_candidate_bundle(
    root: str | Path,
    *,
    spec_document: Mapping[str, Any],
    weights: bytes,
    checkpoint: bytes | None,
    patch: list[int],
    channels: int,
    classes: int,
    versions: Mapping[str, str],
) -> Any:
    """Write `MOS-TRAIN-129`'s layout and return `medos.sdk.bundle`'s own report.

    The return value is a `BundleReport` from `verify()`, not a dict this module built:
    `bundle_digest` and `weights_digest` are what the PLATFORM will read back, and a
    trainer that reported its own numbers would be reporting a bundle it had not verified.
    """
    from medos.sdk.bundle import metadata_document, write_bundle
    from medos.sdk.spec import parse_spec

    spec = parse_spec(dict(spec_document))
    label_set = list((dict(spec_document).get("io") or {}).get("label_set") or [])

    metadata = metadata_document(
        version=str(dict(spec_document)["model_version"]),
        monai_version=MONAI_BUNDLE_TARGET,
        pytorch_version=str(versions["torch"]),
        numpy_version=str(versions["numpy"]),
        inputs={
            "image": {
                "spatial_shape": list(patch),
                "dtype": "float32",
                # `MOS-IMG-037`'s failure mode is a mirrored segmentation that passes
                # every structural check, so the channel definition names the modality
                # and the spec's orientation target rather than saying "0: input".
                "channel_def": {str(c): "CT" for c in range(channels)},
                "num_channels": channels,
                "orientation": str(dict(spec_document)["orientation_target"]),
                "spacing_mm": list(dict(spec_document)["target_spacing_mm"]),
            }
        },
        outputs={
            "pred": {
                "spatial_shape": list(patch),
                "dtype": "float32",
                # The label set from the spec, not from nnU-Net's dataset.json: the
                # registered spec is what MOS-TRAIN-136 makes authoritative.
                "channel_def": {
                    str(entry["value"]): str(entry["name"]) for entry in label_set
                } or {str(c): f"class_{c}" for c in range(classes)},
                "num_channels": classes,
            }
        },
    )
    volume, tensor = golden_fixture_bytes(spec_document)
    return write_bundle(
        root,
        spec=spec,
        metadata=metadata,
        weights=weights,
        weights_format=_WEIGHTS_FORMAT,
        golden_volume=volume,
        golden_tensor=tensor,
        checkpoint=checkpoint,
    )


def write_modelcard(
    root: str | Path,
    *,
    spec_document: Mapping[str, Any],
    bundle_dir: str,
    weights_file: str,
    weights_digest: str,
    versions: Mapping[str, str],
    stamp: Mapping[str, Any],
) -> Path:
    """Write `medlange.modelcard/1` next to the bundle: the SDK-readable declaration.

    The card is how a consumer of the SDK reconstructs what inference needs WITHOUT the
    training run's database: the spec (which is the record of what training did,
    `MOS-IMG-046`), the weights' path and digest, the framework versions that produced
    them, and the outputs the model claims. `medos.sdk.modelcard.ModelCard.load` is the
    reader and `ModelCard.chain()` the preprocessing pipeline.

    The outputs descriptor is derived from the REGISTERED spec's label set (`io.label_set`),
    not from nnU-Net's dataset.json -- the registered spec is what `MOS-TRAIN-136` makes
    authoritative, and the card is a promise about the registered thing.
    """
    from medos.sdk.modelcard import CARD_FILENAME, document_for
    from medos.sdk.spec import parse_spec

    spec = parse_spec(dict(spec_document))
    label_set = list((dict(spec_document).get("io") or {}).get("label_set") or [])
    document = document_for(
        model_id=spec.model_id,
        model_version=spec.model_version,
        spec_document=spec_document,
        weights={
            "path": bundle_dir,
            "weights_file": weights_file,
            "format": _WEIGHTS_FORMAT,
            "digest": weights_digest,
        },
        frameworks={
            "monai_bundle": MONAI_BUNDLE_TARGET,
            "torch": str(versions["torch"]),
            "numpy": str(versions["numpy"]),
        },
        outputs=tuple(
            {"kind": "segmentation", "value": entry["value"], "name": entry["name"]}
            for entry in label_set
        ),
        stamp=dict(stamp),
    )
    path = Path(root) / CARD_FILENAME
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
