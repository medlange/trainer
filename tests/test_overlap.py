# SPDX-License-Identifier: Apache-2.0
"""Overlap and volume on shapes small enough to count by hand.

WHY THESE GATES. The cleared precedent for a chest-CT segmentation claim is Dice 0.95-0.98 with
HD95 2.6-5.2 mm and volume error 1.5-3.5 percent. Three of those four numbers are wrong in ways
that do not look wrong:

  * a Dice computed over the whole mask rather than symmetric intersection is off by a factor
    that depends on the shapes;
  * a surface distance measured from every voxel instead of from the BOUNDARY is small for a
    thick structure and large for a thin one, and either way is not a boundary agreement;
  * a distance in voxels rather than millimetres is wrong by the anisotropy -- on this cohort's
    1.0 x 0.78 x 0.78 mm spacing, by up to 28 percent.

So every one is asserted against a figure worked out by hand, on a cube or a slab whose surface
you can enumerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.overlap import (  # noqa: E402
    OVERLAP_METRICS,
    overlap_for_case,
    surface_distances,
)

LABEL_OF = {"aorta": [1], "effusion": 2}
CHANNELS = ["aorta", "effusion"]
#: 2 x 2 x 2 mm, so one voxel is 8 mm^3 and 125 voxels make 1 mL.
ISOTROPIC = (2.0, 2.0, 2.0)
VOXEL_ML = 8.0 / 1000.0


def _case(predicted_mask, reference_values, *, spacing=ISOTROPIC, supervised=("aorta",),
          with_distances=True):
    probability = np.zeros((2,) + reference_values.shape)
    probability[0][predicted_mask] = 0.9
    return overlap_for_case(
        probability, reference_values, label_of=LABEL_OF, channels=CHANNELS,
        supervised=supervised, spacing_mm=spacing, threshold=0.5, case="c1",
        with_distances=with_distances,
    )


# =====================================================================================
# Dice, IoU and the volumes
# =====================================================================================
def test_a_perfect_overlap_is_one_and_the_volumes_agree() -> None:
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[2:6, 2:6, 2:6] = 1                       # 64 voxels
    row = _case(reference == 1, reference)[0]
    assert row.dice == pytest.approx(1.0)
    assert row.iou == pytest.approx(1.0)
    assert row.reference_ml == pytest.approx(64 * VOXEL_ML)
    assert row.predicted_ml == pytest.approx(64 * VOXEL_ML)
    assert row.volume_error_ml == pytest.approx(0.0)
    assert row.volume_ape == pytest.approx(0.0)


def test_dice_and_iou_are_the_definitions_on_a_half_overlap() -> None:
    """Reference 64 voxels, prediction 64 voxels, sharing 32.
    Dice = 2*32/128 = 0.5; IoU = 32/96 = 0.3333."""
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[0:4, 0:8, 0:2] = 1                       # 4*8*2 = 64
    predicted = np.zeros((8, 8, 8), dtype=bool)
    predicted[2:6, 0:8, 0:2] = True                    # overlaps rows 2-3: 2*8*2 = 32
    row = _case(predicted, reference)[0]
    assert row.dice == pytest.approx(0.5)
    assert row.iou == pytest.approx(32 / 96)


def test_the_volume_error_is_signed_and_the_percentage_is_not() -> None:
    """A model that over-segments half the cases and under-segments the other half has a mean
    signed error near zero; the registry reports the signed mean BESIDE the absolute one for
    exactly that reason, so the sign has to survive here."""
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[0:2, 0:2, 0:2] = 1                       # 8 voxels
    over = np.zeros((8, 8, 8), dtype=bool)
    over[0:2, 0:2, 0:4] = True                         # 16 voxels
    row = _case(over, reference)[0]
    assert row.volume_error_ml == pytest.approx(8 * VOXEL_ML)
    assert row.volume_ape == pytest.approx(1.0)

    under = np.zeros((8, 8, 8), dtype=bool)
    under[0:1, 0:2, 0:2] = True                        # 4 voxels
    row = _case(under, reference)[0]
    assert row.volume_error_ml == pytest.approx(-4 * VOXEL_ML), "the sign was dropped"
    assert row.volume_ape == pytest.approx(0.5)


def test_an_empty_reference_is_ineligible_and_still_reports_what_was_predicted() -> None:
    """Both halves of `MOS-EVID-051`: it leaves the aggregate, and the volume it predicted anyway
    is still measured, because that is the false-positive volume the empty-reference metrics are
    made of. A Dice of 0.0 here would read as a failed segmentation rather than an absent one."""
    reference = np.zeros((8, 8, 8), dtype=int)
    predicted = np.zeros((8, 8, 8), dtype=bool)
    predicted[0:2, 0:2, 0:2] = True
    row = _case(predicted, reference)[0]
    assert row.eligible is False
    assert row.undefined_reason == "empty_ground_truth"
    assert row.dice is None and row.volume_ape is None
    assert row.predicted_ml == pytest.approx(8 * VOXEL_ML)
    assert row.reference_ml == pytest.approx(0.0)


def test_a_channel_the_case_does_not_annotate_produces_no_row() -> None:
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[0:2, 0:2, 0:2] = 2                       # effusion present
    rows = _case(np.zeros((8, 8, 8), dtype=bool), reference, supervised=["aorta"])
    assert [r.channel for r in rows] == ["aorta"]


# =====================================================================================
# The surface distances
# =====================================================================================
def test_the_distance_is_measured_from_the_boundary_and_not_from_the_whole_mask() -> None:
    """A SMALL PREDICTION DEEP INSIDE A LARGE REFERENCE, which is the only shape that separates
    the two.

    From the BOUNDARY: the prediction's surface is far from the reference's surface, so every
    forward distance is large -- 16 mm here -- and that is the honest reading of a prediction that
    covers a fraction of the structure.

    From the WHOLE MASK: the nearest reference VOXEL is the prediction's own voxel, so every
    forward distance is 0 and a prediction covering 1% of the structure scores a perfect boundary
    agreement.

    An earlier fixture used two overlapping slabs offset by one voxel; there both readings give
    the same 0-to-2 mm and the fixture proved nothing. Proved by breaking.
    """
    reference = np.zeros((24, 24, 24), dtype=int)
    reference[2:22, 2:22, 2:22] = 1                    # a large cube
    predicted = np.zeros((24, 24, 24), dtype=bool)
    predicted[10:14, 10:14, 10:14] = True              # a small cube at its centre

    forward, _backward = surface_distances(predicted, reference == 1, spacing_mm=ISOTROPIC)
    assert forward.size
    assert forward.min() > 4.0, (
        f"the nearest forward distance is {forward.min():.2f} mm for a prediction sitting deep "
        "inside the reference, so the distance is being measured to the nearest reference VOXEL "
        "rather than to its surface"
    )


def test_the_distance_honours_anisotropic_spacing() -> None:
    """A one-voxel offset along an axis of spacing 4 mm is 4 mm, not 1 and not 2. This cohort's
    spacing is 1.0 x 0.78 x 0.78 mm, so a distance in voxels would be wrong by up to 28%."""
    reference = np.zeros((10, 10, 10), dtype=int)
    reference[3:7, 3:7, 3:7] = 1
    predicted = np.zeros((10, 10, 10), dtype=bool)
    predicted[3:7, 3:7, 4:8] = True                    # one voxel along the last axis

    _forward, backward = surface_distances(predicted, reference == 1,
                                          spacing_mm=(1.0, 1.0, 4.0))
    assert backward.max() == pytest.approx(4.0), backward.max()


def test_hd95_is_the_larger_of_the_two_percentiles_and_not_the_percentile_of_both() -> None:
    """THE DEFINITION, ON THE ONE SHAPE THAT SEPARATES THE TWO FORMS.

    The prediction here is the reference PLUS a small blob far away. So:

      backward (reference surface to prediction surface) is 0 everywhere -- the prediction
        contains the reference;
      forward (prediction surface to reference surface) is 0 over the shared part and very large
        over the blob, and the blob is about 6% of the prediction's surface.

    Six percent is above the 95th percentile of FORWARD and below the 95th percentile of the two
    CONCATENATED, because concatenating adds the reference's 376 zero distances and dilutes it. So
    the correct form reports the blob and the concatenated form hides it -- a spurious structure
    the width of the chest, reported as sub-millimetre boundary agreement.

    The first implementation concatenated. The first fixture could not tell: its far side was both
    numerous and large, so both forms returned 12.00.
    """
    reference = np.zeros((40, 40, 40), dtype=int)
    reference[2:14, 2:14, 2:6] = 1
    predicted = np.zeros((40, 40, 40), dtype=bool)
    predicted[2:14, 2:14, 2:6] = True                  # the reference exactly
    predicted[34:37, 34:37, 34:37] = True              # plus a 3x3x3 blob, far away

    row = _case(predicted, reference)[0]
    forward, backward = surface_distances(predicted, reference == 1, spacing_mm=ISOTROPIC)
    correct = max(float(np.percentile(forward, 95)), float(np.percentile(backward, 95)))
    concatenated = float(np.percentile(np.concatenate([forward, backward]), 95))

    assert correct > concatenated * 4, (
        f"the fixture no longer separates the two forms: correct {correct:.2f} against "
        f"concatenated {concatenated:.2f}"
    )
    assert row.hd95_mm == pytest.approx(correct), (
        f"HD95 is {row.hd95_mm:.2f}; the larger of the two percentiles is {correct:.2f} and the "
        f"percentile of both concatenated is {concatenated:.2f}"
    )


def test_a_prediction_of_nothing_leaves_the_distances_undefined_not_zero() -> None:
    """There is no surface to measure from. Zero would be a perfect boundary agreement."""
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[2:6, 2:6, 2:6] = 1
    row = _case(np.zeros((8, 8, 8), dtype=bool), reference)[0]
    assert row.dice == pytest.approx(0.0), "a prediction of nothing against a present reference"
    assert row.hd95_mm is None and row.assd_mm is None


def test_skipping_the_distances_still_gives_the_overlap() -> None:
    """Two distance transforms over a whole volume per channel is the expensive part; a run that
    only needs Dice and a volume should not pay for them."""
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[2:6, 2:6, 2:6] = 1
    row = _case(reference == 1, reference, with_distances=False)[0]
    assert row.dice == pytest.approx(1.0)
    assert row.hd95_mm is None and row.assd_mm is None


# =====================================================================================
# The registry, and what this module does not claim
# =====================================================================================
def test_every_metric_this_module_names_is_in_the_registry() -> None:
    """`MOS-EVID-070` invalidates a run that persists an unregistered id, so a name invented here
    would invalidate the run it was computed for."""
    from medos_trainer.evidence import REGISTRY

    for metric in OVERLAP_METRICS:
        assert metric in REGISTRY, metric


def test_the_module_does_not_pretend_to_produce_a_diameter() -> None:
    """The aortic precedent has two halves: Dice 0.924 AND a diameter MAE of at most 2.2 mm at
    nine named landmarks. The second needs a centreline and the landmarks located on it, and is
    not computable from a mask. Substituting the metric we can compute for the one the bar is
    written in is the failure this asserts against."""
    assert not any("diameter" in metric for metric in OVERLAP_METRICS)
    import medos_trainer.overlap as module

    assert "diameter" in (module.__doc__ or ""), (
        "the module no longer says which half of the aortic bar it cannot produce"
    )


def test_a_shape_mismatch_is_refused_rather_than_broadcast() -> None:
    reference = np.zeros((6, 6, 6), dtype=int)
    with pytest.raises(ValueError, match="differ in shape"):
        overlap_for_case(np.zeros((2, 8, 8, 8)), reference, label_of=LABEL_OF,
                         channels=CHANNELS, supervised=["aorta"], spacing_mm=ISOTROPIC,
                         threshold=0.5)


def test_an_integer_label_map_passed_as_a_mask_is_refused() -> None:
    """THE GUARD THAT CAME OUT OF A TEST BUG. An integer array does not fail here: it indexes the
    distance transform as POSITIONS rather than as a mask, producing a wrongly shaped array of
    wrong numbers. The first symptom was a concatenate complaining about dimensions in a caller,
    which points nowhere near the cause."""
    reference = np.zeros((8, 8, 8), dtype=int)
    reference[2:6, 2:6, 2:6] = 1
    predicted = np.zeros((8, 8, 8), dtype=bool)
    predicted[2:6, 2:6, 2:6] = True
    with pytest.raises(ValueError, match="needs a boolean mask"):
        surface_distances(predicted, reference, spacing_mm=ISOTROPIC)
    with pytest.raises(ValueError, match="needs a boolean mask"):
        surface_distances(reference, reference == 1, spacing_mm=ISOTROPIC)
