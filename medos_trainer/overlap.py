# SPDX-License-Identifier: Apache-2.0
"""Per-case overlap and volume, for the findings whose endpoint is a shape and not a count.

WHY THIS EXISTS BESIDE `detection.py`
--------------------------------------
Eight of this cohort's ten channels were being measured by detection, and for six of them
that is
the wrong endpoint. What the published precedent actually asks for, per finding:

  * the aorta and the pulmonary trunk -- SEGMENTATION OVERLAP plus a diameter tolerance. The
    only
    cleared chest-CT precedent for an aortic claim is Dice 0.924 +/- 0.046 on 315 cases, with
    diameter bias within +/-1.5 mm and MAE <= 2.2 mm at nine landmarks (Siemens K222360). Our
    lesion-level "sensitivity 1.000" on those channels measures only whether the single
    component
    was found, which is trivially true for a single structure and says nothing about whether the
    boundary between arch and ascending is in the right place;
  * the pleural effusion -- a VOLUME in mL, or a three-level grade. No source anywhere
    counts
    effusion per connected component, and on this cohort the reference for one effusion is split
    into 5 to 84 pieces, so a per-component sensitivity measures the label's connectivity;
  * the lung lobes, as the nearest accepted CT segmentation claim, were cleared at Dice
    0.95-0.98,
    mean surface distance 0.5-1.0 mm, HD95 2.6-5.2 mm and volume error 1.5-3.5% on >4,500 CTs
    (Siemens K183271) -- which is the shape of claim this module produces.

WHAT IT DOES NOT PRODUCE, AND WHY THAT IS SAID HERE
----------------------------------------------------
The aortic precedent's OTHER half -- a diameter MAE at named AHA landmarks -- is not computable
from a mask alone. It needs a centreline, the nine landmarks located on it, and a
cross-sectional diameter at each. That is a separate piece of work and pretending Dice stands in
for it would substitute the metric we can compute for the one the bar is written in. Stated here
so a reader of the numbers knows which half of the bar they are looking at.

THE UNITS AND THE SPACE. Volumes use the spacing passed in, and everything is computed in the
space the arrays arrive in -- which for this trainer is nnU-Net's preprocessed geometry.
`MOS-EVID-107` requires plausibility geometry in SOURCE space and refuses model space; the same
argument reaches a surface distance, because a Hausdorff distance in millimetres depends on the
voxel grid it was measured on. So HD95 here is a preprocessed-space figure and is labelled as
one; comparing it against a cleared device's 2.6-5.2 mm needs the resampling closed first.

Pure: numpy and scipy.ndimage, both already in the image. No torch, no nnU-Net, no I/O.

Spec: MOS-EVID-047, MOS-EVID-048, MOS-EVID-054, MOS-EVID-107.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["CaseOverlap", "OVERLAP_METRICS", "overlap_for_case", "surface_distances"]

#: The registry ids this module can produce, so a caller cannot ask it for one it does not
#: compute and get silence. `iou_mean_per_case` and `assd_mm` are here because the registry
#: carries them and they cost nothing extra once the confusion and the surfaces exist.
OVERLAP_METRICS: Final[tuple[str, ...]] = (
    "dice_mean_per_case", "iou_mean_per_case", "volume_error_ml", "volume_ape",
    "hd95_mm", "assd_mm",
)


@dataclass(frozen=True)
class CaseOverlap:
    """One case, one channel: the shape metrics, or `None` where a metric is undefined.

    `None` IS NOT ZERO, for the reason `MOS-EVID-051` gives: a case whose reference is empty has
    nothing to overlap, and a Dice of 0.0 there would read as a failed segmentation rather than
    an absent one. Such a case is `eligible=False` and belongs in the empty-reference block.
    """

    case: str
    channel: str
    eligible: bool
    undefined_reason: str | None = None
    dice: float | None = None
    iou: float | None = None
    reference_ml: float | None = None
    predicted_ml: float | None = None
    volume_error_ml: float | None = None
    volume_ape: float | None = None
    hd95_mm: float | None = None
    assd_mm: float | None = None


def _voxel_ml(spacing_mm: Sequence[float]) -> float:
    volume = 1.0
    for step in spacing_mm:
        volume *= float(step)
    return volume / 1000.0


def surface_distances(
    predicted: Any, reference: Any, *, spacing_mm: Sequence[float]
) -> tuple[Any, Any]:
    """Distances in millimetres from each surface to the other, both directions.

    THE SURFACE IS THE BOUNDARY AND NOT THE MASK. A distance transform over the whole mask would
    measure from every interior voxel too, which drags a Hausdorff percentile toward zero
    for a
    thick structure and toward the mask's radius for a thin one -- in both cases reporting
    something that is not a boundary agreement. The boundary is the set of
    foreground voxels with at least one background face neighbour, under the same
    6-connectivity the detection counting
    uses.

    `scipy.ndimage.distance_transform_edt` takes the voxel spacing, so the result is already in
    millimetres and anisotropy is honoured -- which matters here: this cohort's spacing is
    1.0 x 0.78 x 0.78 mm, so a distance computed in voxels would be wrong by up to 28%.
    """
    import numpy as np
    from scipy import ndimage

    # BOTH ARGUMENTS MUST BE BOOLEAN, and this is a refusal rather than a coercion.
    #
    # An integer label map passed here does not fail: `mask & ~eroded` stays integer, and
    # `distances[integer_array]` is then FANCY INDEXING by those integers instead of boolean
    # masking. The result has the wrong shape and wrong contents, and the first sign of it is a
    # concatenate complaining about dimensions somewhere else entirely -- which is how this
    # guard
    # came to be written. A caller holding a label map wants `labels == value`, and only the
    # caller knows which value.
    for name, mask in (("predicted", predicted), ("reference", reference)):
        if np.asarray(mask).dtype != bool:
            raise ValueError(
                f"{name} has dtype {np.asarray(mask).dtype}, and a surface "
                "distance needs a boolean mask. An integer label map indexes as positions "
                "which produces a wrongly shaped array of wrong numbers instead of an error"
            )

    structure = ndimage.generate_binary_structure(np.asarray(reference).ndim, 1)

    def border(mask: Any) -> Any:
        if not mask.any():
            return mask
        eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
        return mask & ~eroded

    predicted_border = border(np.asarray(predicted))
    reference_border = border(np.asarray(reference))
    if not predicted_border.any() or not reference_border.any():
        return np.asarray([]), np.asarray([])

    sampling = tuple(float(s) for s in spacing_mm)
    to_reference = ndimage.distance_transform_edt(~reference_border, sampling=sampling)
    to_predicted = ndimage.distance_transform_edt(~predicted_border, sampling=sampling)
    return to_reference[predicted_border], to_predicted[reference_border]


def overlap_for_case(
    probability: Any,
    segmentation: Any,
    *,
    label_of: Mapping[str, Any],
    channels: Sequence[str],
    supervised: Iterable[str],
    spacing_mm: Sequence[float],
    threshold: float,
    case: str = "",
    with_distances: bool = True,
) -> list[CaseOverlap]:
    """Overlap and volume per channel for ONE case, for the channels it supervises.

    Only supervised channels appear, and for the same reason they do in `detection.counts_at`: a
    channel this case does not annotate is not evidence about it, and scoring a Dice of 0 there
    would make the mask's own rule into a measured failure.

    `with_distances=False` skips HD95 and ASSD. They are the expensive part -- two distance
    transforms over the whole volume per channel -- and a run that only needs Dice and a volume
    should not pay for them.
    """
    import numpy as np
    from medos_trainer.detection import label_value_of

    marked = set(supervised)
    unknown = sorted(marked - set(channels))
    if unknown:
        raise ValueError(
            f"case {case!r} supervises {unknown}, which the channel list does not contain"
        )
    scores = np.asarray(probability)
    truth_map = np.asarray(segmentation)
    if scores.shape[0] != len(channels):
        raise ValueError(
            f"the network emitted {scores.shape[0]} channels and the cohort declares "
            f"{len(channels)}: a per-channel number would be attributed to the wrong finding"
        )
    if scores.shape[1:] != truth_map.shape:
        raise ValueError(
            f"prediction {scores.shape[1:]} and reference {truth_map.shape} differ in shape"
        )
    if len(spacing_mm) != truth_map.ndim:
        raise ValueError(
            f"spacing {tuple(spacing_mm)} has {len(spacing_mm)} axes and the volume has "
            f"{truth_map.ndim}"
        )

    per_voxel_ml = _voxel_ml(spacing_mm)
    rows: list[CaseOverlap] = []
    for column, channel in enumerate(channels):
        if channel not in marked:
            continue
        value = label_value_of(channel, label_of)
        reference = truth_map == value
        predicted = scores[column] > float(threshold)

        reference_voxels = int(np.count_nonzero(reference))
        predicted_voxels = int(np.count_nonzero(predicted))
        reference_ml = reference_voxels * per_voxel_ml
        predicted_ml = predicted_voxels * per_voxel_ml

        if reference_voxels == 0:
            # `MOS-EVID-051`'s empty-reference policy: nothing to overlap, so no Dice. What was
            # predicted anyway is still reported, because that is the false-positive volume the
            # five empty-reference metrics are made of.
            rows.append(CaseOverlap(
                case=case, channel=channel, eligible=False,
                undefined_reason="empty_ground_truth",
                reference_ml=0.0, predicted_ml=predicted_ml,
            ))
            continue

        intersection = int(np.count_nonzero(predicted & reference))
        union = predicted_voxels + reference_voxels - intersection
        dice = (2.0 * intersection) / (predicted_voxels + reference_voxels)
        iou = (intersection / union) if union else None

        hd95 = assd = None
        if with_distances:
            forward, backward = surface_distances(predicted, reference, spacing_mm=spacing_mm)
            if forward.size and backward.size:
                # HD95 IS THE MAXIMUM OF THE TWO PERCENTILES, not the percentile of the two
                # concatenated. The first version concatenated them, and that is not the
                # standard
                # definition and is not conservative: a percentile over the union is weighted by
                # how many surface voxels each side has, so a prediction that misses a whole
                # region -- whose reference surface is then large and mostly far away -- can
                # score
                # BETTER than a one-directional figure, which is the opposite of what a
                # Hausdorff
                # percentile is for. Caught by a test written against the symmetry property.
                hd95 = float(max(np.percentile(forward, 95), np.percentile(backward, 95)))
                # ASSD is the mean over all surface points of both directions, which IS the
                # concatenated form: it is an average distance, not a worst case, and weighting
                # by surface size is what "average symmetric" means.
                assd = float(np.concatenate([forward, backward]).mean())

        rows.append(CaseOverlap(
            case=case, channel=channel, eligible=True,
            dice=float(dice), iou=None if iou is None else float(iou),
            reference_ml=reference_ml, predicted_ml=predicted_ml,
            # SIGNED, and the registry says so: `volume_error_ml` aggregates as a mean of the
            # signed value reported beside its mean absolute, because a model that
            # over-segments half the cases and under-segments the other half has a mean near
            # zero and an absolute mean that says what is happening.
            volume_error_ml=predicted_ml - reference_ml,
            volume_ape=abs(predicted_ml - reference_ml) / reference_ml,
            hd95_mm=hd95, assd_mm=assd,
        ))
    return rows
