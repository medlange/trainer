# SPDX-License-Identifier: Apache-2.0
"""Lesion-level candidate extraction and matching, on arrays small enough to count by hand.

WHY EVERY TEST HERE USES A VOLUME YOU CAN READ. A detection count is a count of lesions, and
every rule that decides what counts as one lesion -- connectivity, the extraction threshold,
the overlap match -- changes the answer by whole lesions rather than by a fraction. So the
fixtures are 1x8x8 or 1x6x6 arrays with the blobs written out, and the expected counts are
stated in the test rather than derived.

WHAT IS AT STAKE. `MOS-EVID-054` makes `sensitivity` a count over cases, and `froc_sensitivity`
the detection metric. Both are built entirely out of this module's output, so a defect here is
a clinical claim that is wrong in units nobody can see from the number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.detection import (  # noqa: E402
    candidates_for_case,
    label_value_of,
)

LABEL_OF = {"neo": [1], "effusion": 2}
CHANNELS = ["neo", "effusion"]
#: 2 x 2 x 2 mm: one voxel is 8 mm^3, so 125 voxels make 1 mL. Chosen so the millilitres in
#: these tests are round numbers rather than something a reader has to trust.
SPACING = (2.0, 2.0, 2.0)
VOXEL_ML = 8.0 / 1000.0


def _volume(shape=(1, 8, 8)):
    return np.zeros(shape, dtype=float)


def _case(probability, truth, *, supervised=("neo",), threshold=0.5, case="c1"):
    return candidates_for_case(
        probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=supervised,
        spacing_mm=SPACING, extraction_threshold=threshold, case=case,
    )


# =====================================================================================
# Counting lesions
# =====================================================================================
def test_two_separate_blobs_are_two_candidates_and_two_references() -> None:
    """The count is the point. A rule that merged these would report one detection where a
    reader sees two findings."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    probability[0, 0, 1:3, 1:3] = 0.9      # blob A
    probability[0, 0, 5:7, 5:7] = 0.8      # blob B, nowhere near A
    truth[0, 1:3, 1:3] = 1
    truth[0, 5:7, 5:7] = 1

    found = _case(probability, truth)
    assert len(found.candidates["neo"]) == 2
    assert len(found.references["neo"]) == 2
    assert {c.score for c in found.candidates["neo"]} == {0.9, 0.8}
    assert all(c.matched_reference_id is not None for c in found.candidates["neo"])


def test_diagonal_contact_is_two_lesions_under_six_connectivity() -> None:
    """CONNECTIVITY IS A DECLARED CHOICE AND IT CHANGES THE COUNT. Two blobs touching only at
    a corner are ONE component under 26-connectivity and TWO under 6. `MOS-EVID-108` fixes
    6-connectivity for the body contour on this same argument, so this module follows it rather
    than picking differently -- and the test says which, because the number depends on it."""
    probability = np.zeros((2, 1, 6, 6))
    truth = np.zeros((1, 6, 6), dtype=int)
    probability[0, 0, 1, 1] = 0.9
    probability[0, 0, 2, 2] = 0.9          # touches (1,1) only at a corner
    truth[0, 1, 1] = 1
    truth[0, 2, 2] = 1

    six = _case(probability, truth)
    assert len(six.candidates["neo"]) == 2, "six-connectivity merged a diagonal contact"

    twentysix = candidates_for_case(
        probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=["neo"],
        spacing_mm=SPACING, extraction_threshold=0.5, connectivity=26, case="c1",
    )
    assert len(twentysix.candidates["neo"]) == 1, (
        "the fixture no longer distinguishes the two connectivities, so it cannot show that "
        "the choice matters"
    )


def test_an_unusable_connectivity_is_refused() -> None:
    probability = np.zeros((2, 1, 4, 4))
    with pytest.raises(ValueError, match="neither 6 nor 26"):
        candidates_for_case(
            probability, np.zeros((1, 4, 4), dtype=int), label_of=LABEL_OF, channels=CHANNELS,
            supervised=["neo"], spacing_mm=SPACING, extraction_threshold=0.5, connectivity=18,
        )


# =====================================================================================
# Volume and the centroid
# =====================================================================================
def test_the_volume_is_the_voxel_count_times_the_voxel_volume() -> None:
    """In millilitres, from the spacing passed in. A volume computed at the wrong spacing is
    wrong by a factor nobody notices, because a lesion volume has no obvious right answer."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    probability[0, 0, 0:5, 0:5] = 0.9      # 25 voxels
    truth[0, 0:5, 0:5] = 1

    found = _case(probability, truth)
    assert found.candidates["neo"][0].volume_ml == pytest.approx(25 * VOXEL_ML)
    assert found.references["neo"][0].volume_ml == pytest.approx(25 * VOXEL_ML)


def test_the_centroid_is_in_voxels_and_says_so() -> None:
    """`MOS-EVID-069` wants LPS millimetres, which need the case's original origin and
    direction -- so this returns voxels under a name that cannot be mistaken for LPS. A field
    called `centroid_lps_mm` holding voxel indices would be a coordinate in the wrong frame
    with the right label, and every downstream distance would be silently wrong."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    probability[0, 0, 2:4, 2:4] = 0.9
    truth[0, 2:4, 2:4] = 1

    candidate = _case(probability, truth).candidates["neo"][0]
    assert candidate.centroid_voxel == pytest.approx((0.0, 2.5, 2.5))
    assert not hasattr(candidate, "centroid_lps_mm")


def test_the_match_distance_is_in_millimetres_and_uses_the_spacing() -> None:
    """A separation in voxels and a separation in millimetres differ by the spacing, and on
    this cohort the axes are not isotropic. A distance a reader uses to judge a doubtful match
    has to be in the unit they think in."""
    probability = np.zeros((2, 1, 10, 10))
    truth = np.zeros((1, 10, 10), dtype=int)
    truth[0, 2, 2] = 1                      # reference at (0, 2, 2)
    probability[0, 0, 2, 2] = 0.9           # a candidate covering it
    probability[0, 0, 2, 3] = 0.9           # and one voxel further along the last axis

    found = candidates_for_case(
        probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=["neo"],
        spacing_mm=(1.0, 1.0, 4.0), extraction_threshold=0.5, case="c1",
    )
    candidate = found.candidates["neo"][0]
    # Centroid at (0, 2, 2.5) against a reference at (0, 2, 2): half a voxel on an axis whose
    # spacing is 4 mm.
    assert candidate.match_distance_mm == pytest.approx(2.0)


# =====================================================================================
# Matching, and what counts as found
# =====================================================================================
def test_a_candidate_overlapping_nothing_is_a_false_positive_with_no_distance() -> None:
    """A distance to nothing is not a large distance, so it is `None` rather than a number a
    later average would absorb."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1:3, 1:3] = 1
    probability[0, 0, 6:8, 6:8] = 0.7       # far from the reference

    found = _case(probability, truth)
    candidate = found.candidates["neo"][0]
    assert candidate.matched_reference_id is None
    assert candidate.match_distance_mm is None
    assert candidate.overlap_voxels == 0
    assert found.references["neo"][0].matched_scores == (), "an unfound lesion looks found"


def test_a_candidate_spanning_two_references_matches_the_one_it_overlaps_most() -> None:
    """TWO RULES, TWO QUESTIONS, AND THE FIXTURE HAS TO SEPARATE THEM.

    A candidate is ATTRIBUTED to the reference it overlaps most -- one claim belongs to one
    thing, or the false-positive count means nothing. Every reference it touches is FOUND --
    a lesion a prediction covers has been found whatever else that prediction covers.
    Answering the second question with the first understates sensitivity exactly on clustered
    findings, which is the defect this test was written against and caught.

    THE SMALL REFERENCE COMES FIRST IN SCAN ORDER ON PURPOSE. With the large one first, the
    largest overlap and the first overlap are the same component and the fixture cannot tell
    "largest" from "whichever numbering happened to return first" -- proved by breaking: that
    version stayed green when the rule was changed to take the first.
    """
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1, 1] = 1                      # reference A: 1 voxel, component 1
    truth[0, 3:6, 3:6] = 1                  # reference B: 9 voxels, component 2
    probability[0, 0, 1:6, 1:6] = 0.9       # one component covering both
    # NOT UNIFORM, and this is the correction. With 0.9 everywhere the component's maximum and the
    # maximum on each reference are the same number, so this fixture could not tell them apart --
    # and a real defect lived in that gap: the component's 0.93 was recorded against a reference
    # whose every predicted voxel was 0.2. Reference A is weakened here so the two readings differ,
    # and the assertion below is per reference rather than "all of them".
    probability[0, 0, 1, 1] = 0.6           # reference A: predicted, but less strongly

    found = _case(probability, truth)
    assert len(found.candidates["neo"]) == 1
    assert len(found.references["neo"]) == 2
    candidate = found.candidates["neo"][0]
    references = {r.id: r for r in found.references["neo"]}

    assert candidate.overlap_voxels == 9, (
        "the candidate is attributed to the reference it overlaps LEAST: %r" % (candidate,)
    )
    assert references[candidate.matched_reference_id].volume_ml == pytest.approx(9 * VOXEL_ML)
    by_volume = sorted(found.references["neo"], key=lambda r: r.volume_ml)
    small, large = by_volume[0], by_volume[-1]
    assert large.matched_scores == pytest.approx((0.9,)), (
        "a reference the candidate overlapped is not recorded as found, so sensitivity is "
        "undercounted wherever one prediction covers a cluster"
    )
    assert small.matched_scores == pytest.approx((0.6,)), (
        "the small reference is recorded at the COMPONENT's maximum rather than at this model's "
        "confidence on its own voxel, which reports a lesion found more strongly than it was"
    )


def test_the_overlap_size_is_recorded_so_a_one_voxel_match_is_visible() -> None:
    """A match covering the lesion and a match touching one of its voxels are both "matched",
    and a reader has to be able to tell them apart without re-running anything."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 2:6, 2:6] = 1                  # 16 voxels
    probability[0, 0, 5, 5] = 0.9           # one voxel inside it

    candidate = _case(probability, truth).candidates["neo"][0]
    assert candidate.overlap_voxels == 1
    assert candidate.matched_reference_id is not None


def test_every_score_that_found_a_reference_is_kept_highest_first() -> None:
    """THE PROPERTY THAT MAKES AN OPERATING POINT FREE. A reference is found at `t` when any
    of these is at least `t`, so this list is sensitivity at every threshold, computed once."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1:7, 1:7] = 1
    probability[0, 0, 2, 2] = 0.55          # a weak candidate inside the lesion
    probability[0, 0, 5, 5] = 0.95          # and a strong one, not touching the first

    reference = _case(probability, truth).references["neo"][0]
    assert reference.matched_scores == (0.95, 0.55)


# =====================================================================================
# The mask, and the extraction threshold
# =====================================================================================
def test_a_channel_the_case_does_not_annotate_is_absent_not_empty() -> None:
    """Present-and-empty would read as "annotated, nothing there" -- the false negative this
    subsystem exists to remove, arriving through the evaluator instead of through the loss."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    probability[1, 0, 1:3, 1:3] = 0.99      # effusion predicted confidently
    truth[0, 1:3, 1:3] = 2                  # and present

    found = _case(probability, truth, supervised=["neo"])
    assert "effusion" not in found.candidates, (
        "an unannotated channel contributed candidates, so it can produce false positives for "
        "a finding nobody looked for"
    )
    assert "effusion" not in found.references
    assert "neo" in found.candidates


def test_the_extraction_threshold_fixes_the_component_structure() -> None:
    """A permissive threshold is what makes later operating points free -- and it is also the
    one approximation this module makes, because at a higher threshold a component can SPLIT
    and filtering by score cannot see that. Extracting at 0.2 here gives one component; at 0.6
    the same prediction gives two."""
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1:6, 1] = 1
    probability[0, 0, 1, 1] = 0.9
    probability[0, 0, 2, 1] = 0.3           # a weak bridge
    probability[0, 0, 3, 1] = 0.9

    permissive = _case(probability, truth, threshold=0.2)
    strict = _case(probability, truth, threshold=0.6)
    assert len(permissive.candidates["neo"]) == 1
    assert len(strict.candidates["neo"]) == 2, (
        "the fixture does not show the split, so it cannot demonstrate the approximation"
    )


def test_an_extraction_threshold_of_one_is_refused() -> None:
    """It would extract nothing, and a curve over no candidates is a flat line that looks like
    a measurement."""
    probability = np.zeros((2, 1, 4, 4))
    with pytest.raises(ValueError, match=r"outside \[0, 1\)"):
        _case(probability, np.zeros((1, 4, 4), dtype=int), threshold=1.0)


# =====================================================================================
# Refusals that stop a number being attributed to the wrong thing
# =====================================================================================
def test_a_head_count_that_disagrees_with_the_label_set_is_refused() -> None:
    with pytest.raises(ValueError, match="emitted 1 channels"):
        _case(np.zeros((1, 1, 4, 4)), np.zeros((1, 4, 4), dtype=int))


def test_a_reference_of_a_different_shape_is_refused() -> None:
    """An overlap between arrays of different shapes would be an accident of broadcasting."""
    with pytest.raises(ValueError, match="differ in shape"):
        _case(np.zeros((2, 1, 8, 8)), np.zeros((1, 6, 6), dtype=int))


def test_a_spacing_with_the_wrong_number_of_axes_is_refused() -> None:
    with pytest.raises(ValueError, match="axes and the volume has"):
        candidates_for_case(
            np.zeros((2, 1, 4, 4)), np.zeros((1, 4, 4), dtype=int), label_of=LABEL_OF,
            channels=CHANNELS, supervised=["neo"], spacing_mm=(1.0, 1.0),
            extraction_threshold=0.5,
        )


def test_a_case_supervising_an_undeclared_channel_is_refused() -> None:
    with pytest.raises(ValueError, match=r"supervises \['ghost'\]"):
        _case(np.zeros((2, 1, 4, 4)), np.zeros((1, 4, 4), dtype=int), supervised=["ghost"])


def test_both_spellings_of_a_label_value_are_accepted_and_a_region_is_not() -> None:
    assert label_value_of("neo", {"neo": [1]}) == 1
    assert label_value_of("neo", {"neo": 1}) == 1
    with pytest.raises(ValueError, match="multi-value region"):
        label_value_of("neo", {"neo": [1, 2]})


# =====================================================================================
# The bridge: detections -> per-case counts, at one operating point
#
# WHY THE FIXTURES ARE HAND-BUILT DETECTION OBJECTS rather than volumes run through
# `candidates_for_case`. The rules under test here are about ELIGIBILITY and about which
# candidate survives a threshold; building them from arrays would make every test also a test
# of connected components, and a failure would not say which half broke.
# =====================================================================================
from medos_trainer.detection import (  # noqa: E402
    Candidate,
    CaseDetections,
    Reference,
    counts_at,
    froc_at,
    froc_curve,
)

PATIENTS = {"a1": "p1", "a2": "p1", "b1": "p2", "b2": "p2", "empty1": "p3"}


def _detected(case, *, references=(), candidates=(), channel="neo"):
    return CaseDetections(case=case, references={channel: tuple(references)},
                          candidates={channel: tuple(candidates)})


def _candidate(score, matched=None, volume_ml=0.1, overlaps=None, overlap_scores=None):
    """A hand-built candidate. `overlaps` defaults to the single attributed reference.

    THE DEFAULT IS THE HONEST ONE: a candidate attributed to `r1` does overlap `r1`. `overlaps`
    is spelled out only where a candidate covers SEVERAL references, which is the case the
    bridge got wrong twice.

    `overlap_scores` IS THE PER-REFERENCE CONFIDENCE and defaults to this candidate's own `score`
    against every reference it overlaps -- which is what a UNIFORM component gives, and is the
    fixture shape that hid a real defect: with one score everywhere, "the component's maximum" and
    "the maximum on this reference" are the same number, so a test cannot tell them apart. Pass it
    explicitly wherever that distinction is the point.
    """
    if overlaps is None:
        overlaps = () if matched is None else (matched,)
    if overlap_scores is None:
        overlap_scores = tuple((reference_id, score) for reference_id in overlaps)
    return Candidate(score=score, volume_ml=volume_ml, centroid_voxel=(0.0, 0.0, 0.0),
                     matched_reference_id=matched,
                     match_distance_mm=None if matched is None else 1.0,
                     overlap_voxels=0 if matched is None else 5,
                     overlap_scores=tuple(overlap_scores))


def _reference(id, *scores, volume_ml=0.2):
    return Reference(id=id, volume_ml=volume_ml, centroid_voxel=(0.0, 0.0, 0.0),
                     matched_scores=tuple(sorted(scores, reverse=True)))


def test_a_case_with_a_lesion_and_no_prediction_is_a_measured_miss() -> None:
    """Eligible, and it scores zero. Excluding it would be the false negative this trainer
    exists to remove, arriving through the eligibility rule."""
    rows = counts_at([_detected("a1", references=[_reference("r1")])], "neo",
                     threshold=0.5, patient_of=PATIENTS)
    row = rows["sensitivity"][0]
    assert (row.eligible, row.numerator, row.denominator) == (True, 0.0, 1.0)


def test_a_case_with_no_lesion_is_excluded_from_sensitivity_and_measured_as_a_volume() -> None:
    """`MOS-EVID-051`'s `exclude_and_report_separately`, both halves: it leaves the aggregate
    AND what it predicted anyway is measured, because a false alarm on a case with nothing to
    find is exactly the false alarm a reader cares about."""
    detections = [_detected("empty1", candidates=[_candidate(0.8, volume_ml=0.3),
                                                  _candidate(0.2, volume_ml=9.9)])]
    rows = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS)
    sensitivity = rows["sensitivity"][0]
    assert sensitivity.eligible is False
    assert sensitivity.undefined_reason == "empty_ground_truth"

    empty = rows["empty_reference"][0]
    assert empty.eligible is True
    # Only the SURVIVING candidate's volume: the 0.2-scoring one is below the operating point
    # and the service would not report it.
    assert empty.numerator == pytest.approx(0.3)


def test_a_case_with_no_prediction_has_no_precision_rather_than_zero() -> None:
    """Scoring it zero punishes silence and scoring it one rewards it; both are claims about a
    case where nothing was claimed. A model that predicts nothing anywhere then gets `n=0` and
    `value=None` for ppv -- the same statement as the dash printed for a channel the control
    arm never predicts."""
    rows = counts_at([_detected("a1", references=[_reference("r1", 0.9)])], "neo",
                     threshold=0.95, patient_of=PATIENTS)
    ppv = rows["ppv"][0]
    assert ppv.eligible is False
    assert ppv.undefined_reason == "no_prediction"


def test_the_threshold_decides_which_lesion_counts_as_found() -> None:
    """One lesion found by a 0.6 candidate. At 0.5 it is found; at 0.7 it is not, and the case
    becomes a measured miss rather than disappearing."""
    detections = [_detected("a1", references=[_reference("r1", 0.6)],
                            candidates=[_candidate(0.6, matched="r1")])]
    low = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS)["sensitivity"][0]
    high = counts_at(detections, "neo", threshold=0.7, patient_of=PATIENTS)["sensitivity"][0]
    assert (low.numerator, low.denominator) == (1.0, 1.0)
    assert (high.numerator, high.denominator) == (0.0, 1.0)
    assert high.eligible is True, "raising the threshold turned a miss into a non-measurement"


def test_a_channel_the_case_does_not_annotate_produces_no_rows_at_all() -> None:
    """Not a zero row: a case that never annotated the channel is not evidence about it."""
    rows = counts_at([CaseDetections(case="a1", references={}, candidates={})], "neo",
                     threshold=0.5, patient_of=PATIENTS)
    assert rows["sensitivity"] == [] and rows["ppv"] == []


def test_a_case_with_no_patient_key_is_refused() -> None:
    """`MOS-EVID-057` resamples PATIENTS. A case whose patient is unknown cannot be placed in a
    cluster, and its contribution to the interval would be wrong in a direction nobody sees."""
    with pytest.raises(ValueError, match="no patient key"):
        counts_at([_detected("stranger", references=[_reference("r1")])], "neo",
                  threshold=0.5, patient_of=PATIENTS)


def test_the_precision_numerator_counts_candidates_that_overlapped_something() -> None:
    detections = [_detected("a1", references=[_reference("r1", 0.9)],
                            candidates=[_candidate(0.9, matched="r1"), _candidate(0.8),
                                        _candidate(0.1)])]
    ppv = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS)["ppv"][0]
    assert (ppv.numerator, ppv.denominator) == (1.0, 2.0), (
        "the candidate below the operating point was counted, or the matched one was not"
    )


# =====================================================================================
# The FROC curve
# =====================================================================================
def _cohort():
    """Two patients, two cases each, one lesion per case, plus one empty-reference case.

    Case a1: lesion found at 0.9, one false positive at 0.7.
    Case a2: lesion found at 0.4.
    Case b1: lesion not found at all, one false positive at 0.95.
    Case b2: lesion found at 0.6.
    Case empty1: no lesion, one false positive at 0.3.
    """
    return [
        _detected("a1", references=[_reference("a1_1", 0.9)],
                  candidates=[_candidate(0.9, matched="a1_1"), _candidate(0.7)]),
        _detected("a2", references=[_reference("a2_1", 0.4)],
                  candidates=[_candidate(0.4, matched="a2_1")]),
        _detected("b1", references=[_reference("b1_1")], candidates=[_candidate(0.95)]),
        _detected("b2", references=[_reference("b2_1", 0.6)],
                  candidates=[_candidate(0.6, matched="b2_1")]),
        _detected("empty1", candidates=[_candidate(0.3)]),
    ]


def test_the_curve_is_monotone_in_both_axes() -> None:
    """A property of thresholding and not of the data: raising the threshold can only remove
    candidates, so sensitivity and the false-positive rate both fall. A violation means the
    match or the survival rule moved with the threshold, which it must not."""
    curve = froc_curve(_cohort(), "neo")
    assert curve, "the curve is empty for a cohort that has candidates"
    ordered = sorted(curve, key=lambda r: -r["score_threshold"])
    sensitivities = [r["sensitivity"] for r in ordered]
    rates = [r["false_positives_per_case"] for r in ordered]
    assert sensitivities == sorted(sensitivities), sensitivities
    assert rates == sorted(rates), rates


def test_the_false_positive_rate_divides_by_every_annotating_case() -> None:
    """Including the cases with nothing to find. Excluding them would flatter the rate, and a
    false alarm on a case with no finding is exactly the one a reader cares about."""
    curve = froc_curve(_cohort(), "neo")
    lowest = min(curve, key=lambda r: r["score_threshold"])
    assert lowest["cases"] == 5.0, "the empty-reference case was dropped from the denominator"
    # At the lowest threshold all three unmatched candidates survive: 0.7, 0.95 and 0.3.
    assert lowest["false_positives_per_case"] == pytest.approx(3 / 5)
    assert lowest["sensitivity"] == pytest.approx(3 / 4), "three of four lesions are findable"


def test_the_thresholds_are_the_candidate_scores_and_not_a_grid() -> None:
    """A grid either misses a point the data distinguishes or invents points it does not. Every
    distinct candidate score is an operating point the cohort can actually be placed at."""
    curve = froc_curve(_cohort(), "neo")
    thresholds = {r["score_threshold"] for r in curve}
    for score in (0.95, 0.9, 0.7, 0.6, 0.4, 0.3):
        assert any(abs(t - score) < 1e-9 for t in thresholds), score


def test_interpolation_at_a_declared_point_lands_between_the_neighbours() -> None:
    curve = froc_curve(_cohort(), "neo")
    at_half = froc_at(curve, 0.5)
    assert at_half is not None
    assert 0.0 <= at_half <= 1.0


def test_a_rate_the_model_never_reaches_is_none_and_not_the_nearest_value() -> None:
    """Reporting the closest point would claim a measurement at an operating point the model
    cannot be placed at, and `MOS-EVID-055`'s whole argument is that a number and its operating
    point travel together."""
    curve = froc_curve(_cohort(), "neo")
    assert froc_at(curve, 8.0) is None, (
        "a cohort with 0.6 false positives per case reported a sensitivity at 8 per case"
    )
    assert froc_at([], 1.0) is None


def test_a_rate_reached_by_several_thresholds_takes_the_most_sensitive_one() -> None:
    """Several thresholds can share a false-positive rate. The operating point a deployment
    would choose there is the one that finds the most, so that is the one reported."""
    detections = [
        _detected("a1", references=[_reference("a1_1", 0.9), _reference("a1_2", 0.5)],
                  candidates=[_candidate(0.9, matched="a1_1"),
                              _candidate(0.5, matched="a1_2")]),
    ]
    curve = froc_curve(detections, "neo")
    # No false positives at any threshold, so every row sits at rate 0.0 with a different
    # sensitivity; the best of them is what a deployment would get.
    assert all(r["false_positives_per_case"] == 0.0 for r in curve)
    assert froc_at(curve, 0.0) == pytest.approx(1.0)


def test_the_declared_points_are_the_published_set_and_are_sorted() -> None:
    """A curve read at different points is a different number under one name, so the points are
    declared once and are the set the detection literature reports."""
    from medos_trainer.detection import FROC_POINTS

    assert FROC_POINTS == (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert list(FROC_POINTS) == sorted(FROC_POINTS)


def test_a_candidate_exactly_at_the_operating_point_survives() -> None:
    """`MOS-SVC-089` fixes the platform's rule: `present = score >= score_threshold`. A `>`
    here would silently drop the candidate sitting exactly on the threshold, so a finding the
    service WOULD report and a finding this counts would be different sets -- and the
    disagreement would appear only at one score, which is where nobody looks."""
    detections = [_detected("a1", references=[_reference("r1", 0.5)],
                            candidates=[_candidate(0.5, matched="r1")])]
    rows = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS)
    assert rows["sensitivity"][0].numerator == 1.0, "the lesion at the threshold was missed"
    assert rows["ppv"][0].denominator == 1.0, "the candidate at the threshold was dropped"


def test_one_candidate_covering_two_lesions_finds_both_through_the_bridge() -> None:
    """THE GATE THAT WAS MISSING, AND THE DEFECT IT WAS MISSING WAS MADE TWICE.

    Counting found lesions through a candidate's single `matched_reference_id` -- the reference
    it is ATTRIBUTED to -- undercounts sensitivity wherever one prediction covers a cluster.
    That was fixed inside `candidates_for_case` and then reintroduced in `counts_at` under a
    different spelling, with all 240 tests still green, because every bridge fixture had one
    overlap per candidate.

    So: one candidate, two references, and both must count. The clustered findings on this
    cohort are coronary calcifications and nodules, which is where the undercount is largest and
    least visible.
    """
    candidate = _candidate(0.9, matched="r1", overlaps=("r1", "r2"))
    detections = [_detected("a1", references=[_reference("r1", 0.9), _reference("r2", 0.9)],
                            candidates=[candidate])]
    rows = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS)
    row = rows["sensitivity"][0]
    assert (row.numerator, row.denominator) == (2.0, 2.0), (
        "a lesion the surviving candidate covered was not counted as found: %r" % (row,)
    )
    # And it is ONE claim, so precision sees one candidate and not two.
    assert rows["ppv"][0].denominator == 1.0


def test_the_largest_component_policy_keeps_the_structure_and_not_the_confident_speck() -> None:
    """TWO THINGS AT ONCE, because one fixture separates both rules only if it is built to.

    The cohort here is one case with two real lesions and one confident speck that overlaps
    nothing -- which is what the model actually emits: about 28 spurious components per channel,
    some of them the ones it is most sure about.

      all_components      2 of 2 lesions found, 2 of 3 candidates real
      largest_component   1 of 2 found, 1 of 1 real -- the trade the reader must see
      by SCORE (a defect) 0 of 2 found, 0 of 1 real -- it keeps the speck

    The earlier fixture gave the biggest component the highest score, so "largest by volume" and
    "most confident" picked the same one; the version after that gave both rules the same COUNTS
    while differing in WHICH lesion was found, which the assertions could not see either. Proved
    by breaking twice.
    """
    big = _candidate(0.6, matched="r1", volume_ml=5.0)
    small = _candidate(0.7, matched="r2", volume_ml=0.2)
    speck = _candidate(0.95, volume_ml=0.05)          # confident and spurious
    detections = [_detected("a1", references=[_reference("r1", 0.6), _reference("r2", 0.7)],
                            candidates=[big, small, speck])]

    everything = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS,
                           policy="all_components")
    largest = counts_at(detections, "neo", threshold=0.5, patient_of=PATIENTS,
                        policy="largest_component")

    assert (everything["sensitivity"][0].numerator,
            everything["sensitivity"][0].denominator) == (2.0, 2.0)
    assert (everything["ppv"][0].numerator, everything["ppv"][0].denominator) == (2.0, 3.0)

    assert largest["sensitivity"][0].numerator == 1.0, (
        "the policy either kept a candidate it should have discarded, or discarded the one "
        "covering a lesion: %r" % (largest["sensitivity"][0],)
    )
    assert (largest["ppv"][0].numerator, largest["ppv"][0].denominator) == (1.0, 1.0), (
        "the surviving candidate is not the real one, so the policy chose by score rather than "
        "by volume: %r" % (largest["ppv"][0],)
    )


def test_an_unknown_component_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        counts_at([_detected("a1", references=[_reference("r1")])], "neo", threshold=0.5,
                  patient_of=PATIENTS, policy="keep_the_best_one")


# =====================================================================================
# A LESION IS FOUND BY THE CONFIDENCE ON ITS OWN VOXELS, not by the component's maximum
# =====================================================================================
#
# THE DEFECT THESE WERE WRITTEN AGAINST, and it survived 36 green tests. `Candidate.score` is the
# maximum over the whole component extracted at 0.1, and that single number was recorded as the
# confidence against EVERY reference the component touched. So a reference with no predicted voxel
# at the operating point was reported found, along two independent paths: `froc_curve` through
# `matched_scores`, and `counts_at` through the overlap SET with no score consulted at all.
#
# WHY NOTHING CAUGHT IT. The fixture that exercised a component spanning two references painted the
# whole component 0.9, so "the component's maximum" and "the maximum on this reference" were the
# same number and no assertion could separate them. That fixture is rebuilt below rather than
# supplemented: a blind test left in place is a test that will be trusted again.

def _spanning_case(near=0.93, far=0.2, bridge=0.2, extraction=0.1):
    """ONE component at the extraction threshold, spanning two references of different strength.

    Reference A is a single voxel the model barely predicts; reference B is nine voxels it predicts
    strongly; a bridge of weak voxels joins them so `scipy.ndimage.label` sees one component. This
    is the shape of a real pleural effusion, whose reference on this cohort is split into 5 to 84
    pieces with one predicted sheet across them.
    """
    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1, 1] = 1                       # reference A: 1 voxel, weakly predicted
    truth[0, 3:6, 3:6] = 1                   # reference B: 9 voxels, strongly predicted
    probability[0, 0, 1:6, 1:6] = bridge     # the connecting sheet
    probability[0, 0, 3:6, 3:6] = near       # strong over B
    probability[0, 0, 1, 1] = far            # weak over A
    return _case(probability, truth, threshold=extraction)


def test_the_confidence_recorded_against_a_reference_is_the_one_on_its_own_voxels() -> None:
    """THE DEFECT ITSELF, at the point where the number is written down.

    Reference A's every predicted voxel is 0.2 and reference B's are 0.93. The component maximum is
    0.93. Recording 0.93 against A -- which is what this did -- says the model found A with 93%
    confidence when it has no voxel over A above 0.2.
    """
    found = _spanning_case()
    assert len(found.candidates["neo"]) == 1, "the fixture no longer produces ONE component"
    references = {r.id: r for r in found.references["neo"]}
    weak = min(references.values(), key=lambda r: r.volume_ml)
    strong = max(references.values(), key=lambda r: r.volume_ml)

    assert strong.matched_scores == pytest.approx((0.93,))
    assert weak.matched_scores == pytest.approx((0.2,)), (
        f"the weakly-predicted reference was recorded at {weak.matched_scores}, which is the "
        "component's maximum and not this model's confidence anywhere on that lesion"
    )


def test_a_reference_with_no_voxel_above_the_operating_point_is_not_counted_found() -> None:
    """THE CONSEQUENCE, through `counts_at`, which is the number a report prints.

    At 0.5 the model has predicted nothing at all over reference A, so sensitivity is 1 of 2. The
    defect reported 2 of 2 -- and it reached that number without consulting any per-reference score,
    by taking the whole overlap set of a candidate that survived on its OTHER end.

    At 0.15 both are found, which is the half that shows the gate is about the THRESHOLD and not
    about refusing weak overlaps outright.
    """
    found = _spanning_case()
    at_half = counts_at([found], "neo", threshold=0.5, patient_of={"c1": "p1"})
    assert at_half["sensitivity"][0].numerator == 1.0, (
        f"sensitivity numerator is {at_half['sensitivity'][0].numerator} of "
        f"{at_half['sensitivity'][0].denominator}: a lesion with no predicted voxel at 0.5 is "
        "being counted as found"
    )
    assert at_half["sensitivity"][0].denominator == 2.0

    at_low = counts_at([found], "neo", threshold=0.15, patient_of={"c1": "p1"})
    assert at_low["sensitivity"][0].numerator == 2.0, (
        "at 0.15 both lesions ARE predicted, so refusing the weak one here would be the opposite "
        "error -- undercounting a lesion the model did find"
    )


def test_the_froc_curve_cannot_report_a_lesion_found_above_its_own_confidence() -> None:
    """The curve is built from `matched_scores`, so the same defect made it flat where it should
    fall.

    With 0.93 recorded against both references the curve reports sensitivity 1.0 at every threshold
    up to 0.93 -- a spurious plateau, and exactly over the range a clinical operating point would be
    chosen from.
    """
    found = _spanning_case()
    curve = froc_curve([found], "neo")
    above = [point for point in curve if point["score_threshold"] > 0.5]
    assert above, "the fixture produces no curve points above 0.5"
    assert all(point["sensitivity"] <= 0.5 + 1e-9 for point in above), (
        "the curve reports more than half the lesions found at a threshold above the weak "
        f"lesion's own confidence: "
        f"{[(p['score_threshold'], p['sensitivity']) for p in above]}"
    )

    # AND THE OTHER END, which is the half that needs the threshold SET to be right.
    #
    # Proved by breaking: dropping the per-reference scores from the threshold set left the
    # assertion above green, because it only removes points BELOW 0.5 -- the curve then jumps from
    # 0.5 straight to nothing and never reports the operating point at which both lesions ARE
    # found. A FROC curve that cannot express its own best sensitivity understates the model at
    # every low operating point, and `froc_at` interpolates between the points it has.
    thresholds = [point["score_threshold"] for point in curve]
    assert any(t == pytest.approx(0.2) for t in thresholds), (
        f"the weak lesion's own confidence 0.2 is not a point on the curve: {thresholds}. The "
        "thresholds are built from the values at which an answer changes, and that is one of them"
    )
    complete = [point for point in curve if point["sensitivity"] >= 1.0 - 1e-9]
    assert complete, (
        f"the curve never reaches sensitivity 1.0, though at 0.2 both lesions are predicted: "
        f"{[(p['score_threshold'], p['sensitivity']) for p in curve]}"
    )
    assert max(point["score_threshold"] for point in complete) == pytest.approx(0.2)


def test_precision_still_treats_the_whole_component_as_one_detection() -> None:
    """THE OTHER HALF MUST NOT MOVE, and this is the gate that says so.

    A detection is one predicted object with one confidence, and that confidence IS the component's
    maximum -- which is why `Candidate.score` was right and is unchanged. The fix is only about the
    found/not-found question. If it had also changed the false-positive side, precision would have
    started counting a single predicted sheet as several detections.
    """
    found = _spanning_case()
    at_half = counts_at([found], "neo", threshold=0.5, patient_of={"c1": "p1"})
    assert at_half["ppv"][0].denominator == 1.0, "one component became several detections"
    assert at_half["ppv"][0].numerator == 1.0, (
        "the component overlaps a reference, so it is a true positive whichever end of it "
        "cleared the threshold"
    )
    assert found.candidates["neo"][0].score == pytest.approx(0.93)


def test_the_overlap_set_is_derived_from_the_scores_so_the_two_cannot_drift() -> None:
    """One source of truth, asserted rather than trusted.

    The overlap SET and the per-reference SCORES describe the same overlaps. Two fields would let a
    future edit update one and not the other, and the direction that fails silently is the set
    staying wide while the scores narrow -- which is the defect this whole section is about.
    """
    found = _spanning_case()
    candidate = found.candidates["neo"][0]
    assert candidate.overlapped_reference_ids == tuple(
        reference_id for reference_id, _score in candidate.overlap_scores)
    assert len(candidate.overlap_scores) == 2
    assert "overlapped_reference_ids" not in Candidate.__dataclass_fields__, (
        "the overlap set is a stored field again, so it can disagree with the scores"
    )
