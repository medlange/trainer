# SPDX-License-Identifier: Apache-2.0
"""Preprocessing: the two things the planner exists to produce.

THE PULMOAI BENCHMARK measured the cost of their absence: nnU-Net 0.764 vs
0.000 foreground Dice at five epochs on 20 hydrothorax cases
(trainer/docs/benchmark-pulmo-2026-10-07.md). Both steps are decided BY THE
PLAN, never by the caller:

  * RESAMPLING to the corpus's median spacing, so one voxel means the same
    physical distance in every case. On the FIT PATH this runs FORWARD:
    `standalone.fit_command` resamples every training case whose spacing
    differs from the plan's target onto the target grid. On the PREDICT
    PATH it runs BOTH WAYS, and that is the subtle half: the incoming
    image goes UP to the target grid before the windows run, and the
    probability map comes BACK to the image's own spacing afterwards —
    because the volume a caller hands over is defined on ITS grid, and the
    label the predictor returns must live on the same grid the caller's
    ground truth lives on. `SlidingWindowPredictor.predict` does both
    transparently when the bundle carries a `Preprocessing`.
  * FOREGROUND Z-SCORE normalization: `image = (image - mean) / std` with
    the mean/std the fingerprint collected over the corpus's LABELLED
    FOREGROUND voxels (channel 0 — single-channel CT is the supported
    modality; multi-channel corpora must say their normalization some
    other way before this module speaks for them). Applied on the fit
    path after resampling, and on the predict path before the windows
    run. The operation is linear, so it commutes with resampling.

WHERE EACH PIECE RUNS, in one place: `preprocess_cases` is the FIT PATH
(resample the cases that differ, then z-score all of them);
`Preprocessing` is the record a bundle carries (`preprocess.json`, written
by `save_inference_bundle`, read by `load_predictor`); and
`SlidingWindowPredictor.predict(image, spacing_mm=...)` is the PREDICT
PATH (normalize, resample to target, infer, resample the probabilities
back). A bundle without `preprocess.json` loads with preprocessing=None
and predicts exactly as it always did — old bundles keep working.

THE INTERPOLATION CONTRACT is the stack's standing one: the image is
continuous and warps cubic (order 3); the label and mask are discrete and
warp nearest (order 0) — a fractional class or a fractional "labelled"
flag would both be lies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from medos_trainer.vanilla.data import Case

#: A case whose spacing is closer to the target than this (numpy allclose,
#: rtol) is ON the target grid already and is not resampled: resampling is
#: an interpolation, and interpolating a volume onto its own grid would be
#: a no-op in exact arithmetic but a tiny numerical drift in floating point.
_SPACING_RTOL = 1e-5


def resample(
    array: np.ndarray,
    from_spacing: tuple[float, float, float],
    to_spacing: tuple[float, float, float],
    order: int,
) -> np.ndarray:
    """Spacing-aware zoom: factor = from/to per spatial axis.

    One function serves both shapes: a (C,K,J,I) image resamples with the
    channel axis riding along at factor 1.0, a (K,J,I) volume resamples on
    all three axes — the branch is on ndim, and order is the caller's
    contract (3 for the continuous image, 0 for the discrete label/mask).
    Applied on the fit path by `resample_case`; applied on the predict path
    by `SlidingWindowPredictor.predict`, FORWARD for the image and BACK for
    the probability map.
    """
    # TORCH, NOT SCIPY — and this is a measured fix, the third instance of
    # the same disease: scipy.ndimage.zoom at order 3 took tens of seconds
    # per volume on the benchmark machines (the elastic augmentation was the
    # first catch, selection's per-epoch preprocessing the third), during
    # which the run looked hung. torch.interpolate does the same resampling
    # family in ~100 ms. order 3 (cubic spline) -> trilinear, order 0
    # (nearest) -> nearest; a different kernel, the same interpolation
    # contract (the image continuous, the discrete arrays exact).
    import torch
    import torch.nn.functional as F

    factors = tuple(f / t for f, t in zip(from_spacing, to_spacing))
    target = tuple(max(int(round(n * f)), 1)
                   for n, f in zip(array.shape[-3:], factors))
    mode = "trilinear" if order >= 2 else "nearest"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    source_dtype = array.dtype
    work = array if array.dtype == np.float32 else array.astype(np.float32)
    if array.ndim == 4:
        t_arr = torch.from_numpy(np.ascontiguousarray(work)).unsqueeze(0)
        out = F.interpolate(t_arr, size=target, mode=mode, **kwargs)
        out = out.squeeze(0).numpy()
    else:
        t_arr = torch.from_numpy(np.ascontiguousarray(work)).unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t_arr, size=target, mode=mode, **kwargs)
        out = out.squeeze(0).squeeze(0).numpy()
    if out.dtype != source_dtype:
        out = out.astype(source_dtype)
    return out


def resample_case(case: Case, to_spacing: tuple[float, float, float]) -> Case:
    """The case on the target grid: image cubic, label and mask nearest.

    THE FIT PATH calls this (through `preprocess_cases`) on every training
    case whose spacing differs from the plan's target. The predict path
    never calls it on a case — there only the IMAGE goes up to the target
    and the PROBABILITIES come back, because a prediction must be returned
    on the grid the caller's volume is defined on.
    """
    image = resample(case.image, case.spacing_mm, to_spacing, order=3)
    label = resample(case.label, case.spacing_mm, to_spacing, order=0)
    mask = (
        None
        if case.mask is None
        else resample(case.mask, case.spacing_mm, to_spacing, order=0)
    )
    return Case(
        image=np.ascontiguousarray(image, dtype=np.float32),
        label=np.ascontiguousarray(label),
        mask=None if mask is None else np.ascontiguousarray(mask),
        spacing_mm=tuple(float(v) for v in to_spacing),
        case_id=case.case_id,
    )


@dataclass(frozen=True)
class IntensityNorm:
    """The corpus-level foreground z-score: `image = (image - mean) / std`.

    The parameters are the fingerprint's global statistics over LABELLED
    FOREGROUND voxels of channel 0 — computed once over the training
    partition, then frozen into every bundle the fit writes. Applied to
    every case on the fit path, and to every incoming image on the predict
    path, by exactly the same record, so the net always sees the numbers
    it was trained on.
    """

    mean: float
    std: float

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """The z-score itself, for a bare (C,K,J,I) image array."""
        if not self.std > 1e-8:
            # `not ... >` also refuses NaN, whose comparisons are all False.
            raise ValueError(
                f"degenerate intensity normalization: std={self.std!r} is not "
                "positive — a corpus whose foreground has no intensity "
                "spread has no z-score to train with"
            )
        return ((image - self.mean) / self.std).astype(np.float32)

    def apply(self, case: Case) -> Case:
        """The case with its image z-scored; label, mask, spacing untouched."""
        return Case(
            image=self.normalize(case.image),
            label=case.label,
            mask=case.mask,
            spacing_mm=case.spacing_mm,
            case_id=case.case_id,
        )

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntensityNorm:
        return cls(mean=float(data["mean"]), std=float(data["std"]))


@dataclass(frozen=True)
class Preprocessing:
    """What a bundle knows about making its input look like its training
    data: the spacing everything was resampled to, and the z-score that
    was applied after. Written to `preprocess.json` by
    `save_inference_bundle`, read back by `load_predictor`, and applied
    transparently by `SlidingWindowPredictor.predict` when present."""

    target_spacing: tuple[float, float, float]
    normalization: IntensityNorm

    @classmethod
    def identity(cls) -> Preprocessing:
        """The do-nothing record: subtracting 0 and dividing by 1 changes
        no pixel. Only safe where the caller never passes a spacing (or the
        input grid IS the target): the target_spacing half of this record
        is (1, 1, 1) and a case on any other spacing WOULD be resampled."""
        return cls(
            target_spacing=(1.0, 1.0, 1.0),
            normalization=IntensityNorm(mean=0.0, std=1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_spacing": [float(v) for v in self.target_spacing],
            "normalization": self.normalization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preprocessing:
        return cls(
            target_spacing=tuple(float(v) for v in data["target_spacing"]),
            normalization=IntensityNorm.from_dict(data["normalization"]),
        )


def spacing_matches(
    spacing: tuple[float, float, float],
    target: tuple[float, float, float],
) -> bool:
    """Whether a case already sits on the target grid (see `_SPACING_RTOL`)."""
    return bool(
        np.allclose(
            np.asarray(spacing, dtype=np.float64),
            np.asarray(target, dtype=np.float64),
            rtol=_SPACING_RTOL,
            atol=1e-8,
        )
    )


def match_spatial_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """The last three axes brought to exactly `shape`: central crop when
    larger, border pad when smaller.

    THE PREDICT PATH uses this after resampling probabilities BACK to the
    caller's grid: scipy's zoom sizes its output by rounding, so a round
    trip through a non-integer factor ratio can land one voxel off the true
    input extent — and the contract is that the returned label lives on the
    input volume's EXACT grid, not within a voxel of it. Float arrays pad by
    border reflection (a probability that fades to a number would be an
    invention); discrete arrays replicate the edge value.
    """
    out = array
    spatial = range(out.ndim - 3, out.ndim)
    for axis, t in zip(spatial, shape):
        n = out.shape[axis]
        if n > t:
            start = (n - t) // 2
            out = np.take(out, indices=range(start, start + t), axis=axis)
        elif n < t:
            left = (t - n) // 2
            right = t - n - left
            pads = [(0, 0)] * out.ndim
            pads[axis] = (left, right)
            out = np.pad(
                out, pads,
                mode="reflect" if out.dtype.kind == "f" else "edge",
            )
    return out


def preprocess_cases(cases: list[Case], preprocessing: Preprocessing) -> list[Case]:
    """THE FIT PATH: the cases exactly as the net will be trained on them.

    Cases already on the target grid skip resampling (allclose, see
    `spacing_matches` — resampling onto one's own grid is a numerical
    drift, not a transform); every case is z-scored with the corpus's
    foreground statistics, the ones the fingerprint froze. `fit_command`
    calls this between planning and fitting so the fingerprint is
    collected over the corpus AS IT IS and the fit runs on the corpus AS
    THE NET WILL SEE IT.
    """
    out = []
    for case in cases:
        if not spacing_matches(case.spacing_mm, preprocessing.target_spacing):
            case = resample_case(case, preprocessing.target_spacing)
        out.append(preprocessing.normalization.apply(case))
    return out
