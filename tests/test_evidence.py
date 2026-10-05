# SPDX-License-Identifier: Apache-2.0
"""Chapter 7's aggregation contract: the registry, the four companions, and the bootstrap.

WHY THESE GATES AND NOT OTHERS. Three of the rules here would have been got wrong by anyone
writing from intuition, and each is wrong in a way the number does not show:

  * a case-level bootstrap instead of a patient-level cluster one narrows the interval, and a
    narrow interval looks like a confident result;
  * Wilson or Clopper-Pearson on a proportion is the textbook choice and is FORBIDDEN here,
    because clustered cases are not independent Bernoulli trials;
  * an unregistered metric id does not warn, it invalidates the whole run (`MOS-EVID-070`).

So the tests below are written against the requirements rather than against the implementation,
and the bootstrap one constructs a cohort where the two resampling schemes MUST disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer import evidence  # noqa: E402


def _conventions(**over):
    base = {"metric_registry_version": 1, "dice_aggregation": "dice_mean_per_case",
            "fp_volume_threshold_ml": 0.5, "bootstrap_seed": 20260926,
            "min_candidate_volume_ml": 0.02}
    base.update(over)
    return evidence.conventions(**base)


def _rows(values, patients=None, *, eligible=True):
    patients = patients or [f"p{i}" for i in range(len(values))]
    return [
        evidence.CaseValue(case=f"c{i}", patient_key=patients[i], value=v, eligible=eligible)
        for i, v in enumerate(values)
    ]


# =====================================================================================
# The registry is closed
# =====================================================================================
def test_an_unregistered_metric_id_is_refused_and_the_refusal_says_why() -> None:
    """`MOS-EVID-070` does not warn about an unregistered id -- it invalidates the run. So the
    refusal happens here, where it costs nothing, rather than at the schema after a run."""
    with pytest.raises(ValueError, match="not in the MOS-EVID-054 registry"):
        evidence.aggregate("recall", _rows([0.8]), conventions_block=_conventions())


@pytest.mark.parametrize("metric", ["recall", "precision", "dice", "recall_micro", "f1", "nsd"])
def test_the_names_this_trainer_used_before_are_not_registry_ids(metric) -> None:
    """EVERY NUMBER THIS TRAINER PRODUCED BEFORE THIS FILE carried one of these names. The
    registry's `sensitivity` is count-based over CASES and the `recall_micro` measured so far
    is a ratio of VOXELS -- one English word, two quantities, and only one of them can support
    a clinical claim."""
    assert metric not in evidence.REGISTRY


def test_the_five_empty_reference_ids_are_all_registered() -> None:
    """`MOS-EVID-052` requires five and the `MOS-EVID-054` table prints one. Obeying the table
    literally would make every conforming run INVALIDATED, so the registered resolution is that
    all five belong to version 1 -- and this asserts which side the code took, rather than
    leaving a reader to infer it."""
    for name in ("empty_gt_case_count", "empty_gt_false_positive_rate",
                 "empty_gt_mean_fp_volume_ml", "empty_gt_p95_fp_volume_ml",
                 "empty_gt_max_fp_volume_ml"):
        assert name in evidence.REGISTRY, name


def test_exactly_the_four_threshold_needing_ids_need_a_threshold() -> None:
    """`MOS-EVID-055` names four. A fifth would let a number be persisted without the
    operating point that produced it; a missing one would demand a threshold for a
    metric that has none."""
    needs = {k for k, v in evidence.REGISTRY.items() if v["needs_threshold"] is True}
    assert needs == {"sensitivity", "specificity", "ppv", "froc_sensitivity"}


def test_a_metric_that_needs_a_threshold_is_refused_without_an_operating_point() -> None:
    """On the path that metric belongs to: `sensitivity` is count-based, so the threshold
    refusal has to fire there and not only on the per-case path."""
    rows = [evidence.CaseCounts(case="c0", patient_key="p0", numerator=1.0, denominator=2.0,
                                eligible=True)]
    with pytest.raises(ValueError, match="needs an operating point"):
        evidence.aggregate_counted("sensitivity", rows, conventions_block=_conventions())


def test_a_metric_needing_a_threshold_is_accepted_with_one_on_the_path_that_fits_it() -> None:
    """THIS TEST USED THE WRONG PATH AND THE SYMMETRIC GUARD CAUGHT IT. `sensitivity` is
    `per_case: no` in the registry -- count-based over cases -- so reducing per-case values for
    it was producing a mean of ratios under a name the registry defines as a pooled count."""
    got = evidence.aggregate_counted(
        "sensitivity",
        [evidence.CaseCounts(case="c0", patient_key="p0", numerator=2.0, denominator=3.0,
                             eligible=True)],
        conventions_block=_conventions(),
        operating_point={"name": "neo_probability", "value": 0.5, "selected_on": "tune"},
    )
    assert got.value == pytest.approx(2 / 3)
    assert got.as_document()["operating_point"]["selected_on"] == "tune"


def test_a_count_based_metric_cannot_be_reduced_as_per_case_values() -> None:
    """The mirror of the guard on the other path, and the reason this file has both: a mistake
    in either direction produces a number under a registered name that the registry defines
    differently, and nothing about the number shows which happened."""
    with pytest.raises(ValueError, match="not a per-case metric"):
        evidence.aggregate("sensitivity", _rows([1.0, 0.0]), conventions_block=_conventions(),
                           operating_point={"name": "n", "value": 0.5, "selected_on": "tune"})


# =====================================================================================
# The four companions
# =====================================================================================
def test_an_aggregate_carries_n_patients_and_the_interval_and_the_conventions() -> None:
    """`MOS-EVID-056`: "a bare scalar metric MUST fail schema validation". The document is the
    shape that cannot be a bare scalar."""
    got = evidence.aggregate("dice_mean_per_case", _rows([0.8, 0.9, 0.7]),
                             conventions_block=_conventions())
    document = got.as_document()
    assert set(document) == {"metric", "value", "n", "n_patients", "ci_low", "ci_high",
                             "conventions"}
    assert document["n"] == 3
    assert document["n_patients"] == 3
    assert document["ci_low"] is not None and document["ci_high"] is not None
    assert document["ci_low"] <= document["value"] <= document["ci_high"]


def test_the_convention_block_has_section_7_12_1_s_seven_members_and_one_more() -> None:
    """Section 7.12.1 fixes seven. The eighth, `min_candidate_volume_ml`, is here on the same
    argument the seven are: it changes what the number means, and without it a detection
    precision is 0.058 where it should be usable -- measured, not guessed. The registries
    already carry `connected_component_filter` as rule-set data, so this is that value
    travelling with the metric rather than an invention.

    The seven are asserted by name so that an eighth cannot quietly become a ninth.
    """
    block = _conventions()
    assert {
        "dice_aggregation", "empty_gt_policy", "fp_volume_threshold_ml", "ci_method",
        "bootstrap_b", "bootstrap_seed", "metric_registry_version",
    } <= set(block)
    assert set(block) - {
        "dice_aggregation", "empty_gt_policy", "fp_volume_threshold_ml", "ci_method",
        "bootstrap_b", "bootstrap_seed", "metric_registry_version",
    } == {"min_candidate_volume_ml"}


def test_an_incomplete_convention_block_is_refused() -> None:
    block = dict(_conventions())
    del block["bootstrap_seed"]
    with pytest.raises(ValueError, match=r"missing \['bootstrap_seed'\]"):
        evidence.aggregate("dice_mean_per_case", _rows([0.8]), conventions_block=block)


def test_a_dice_aggregation_that_names_neither_convention_is_refused() -> None:
    """`MOS-EVID-050`: a criterion MUST name one of the two explicitly and there is no
    default. `dice: 0.91` with no convention is the exact string the spec says this rule makes
    structurally impossible."""
    with pytest.raises(ValueError, match="no default"):
        _conventions(dice_aggregation="mean")


def test_the_fixed_conventions_are_the_spec_s_values_and_not_a_local_choice() -> None:
    block = _conventions()
    assert block["empty_gt_policy"] == "exclude_and_report_separately"
    assert block["bootstrap_b"] == 2000, "MOS-EVID-058 fixes B"
    assert "percentile" in block["ci_method"] and "cluster" in block["ci_method"]


# =====================================================================================
# Eligibility: nothing to measure against found nothing
# =====================================================================================
def test_an_ineligible_case_is_excluded_from_n_and_from_the_value() -> None:
    """`exclude_and_report_separately`: a zero-reference case does not score zero, it leaves
    the aggregate and joins the empty-reference block."""
    rows = _rows([0.8, 0.6]) + [
        evidence.CaseValue(case="c9", patient_key="p9", value=None, eligible=False,
                           undefined_reason="empty_ground_truth")
    ]
    got = evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())
    assert got.n == 2
    assert got.value == pytest.approx(0.7)


def test_a_measured_zero_is_eligible_and_pulls_the_mean_down() -> None:
    """A case where the finding is present and nothing was predicted is a MISS. Excluding it
    would be the false negative this whole subsystem exists to remove, arriving through the
    eligibility rule."""
    got = evidence.aggregate("dice_mean_per_case", _rows([0.9, 0.0]),
                             conventions_block=_conventions())
    assert got.n == 2
    assert got.value == pytest.approx(0.45)


def test_an_eligible_case_with_no_value_is_refused() -> None:
    """Eligibility is the claim that the case was measured. A None beside it is the two halves
    of `MOS-EVID-051` contradicting each other, and silently dropping it would make `n` a
    count of rows rather than of measurements."""
    rows = [evidence.CaseValue(case="c0", patient_key="p0", value=None, eligible=True)]
    with pytest.raises(ValueError, match="carries no value"):
        evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())


def test_a_cohort_where_nothing_was_eligible_gives_none_and_not_zero() -> None:
    """Zero would read as a model that finds nothing, on a cohort where nothing could be
    measured at all."""
    rows = [evidence.CaseValue(case="c0", patient_key="p0", value=None, eligible=False,
                              undefined_reason="empty_ground_truth")]
    got = evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())
    assert got.value is None
    assert (got.n, got.n_patients) == (0, 0)
    assert got.ci_low is None and got.ci_high is None


# =====================================================================================
# The bootstrap, which is where the wrong answer looks most convincing
# =====================================================================================
def test_the_patient_count_is_patients_and_not_cases() -> None:
    """Four series from two patients is `n=4, n_patients=2`. A run that reported four patients
    would be claiming twice the evidence it has."""
    rows = _rows([0.8, 0.82, 0.4, 0.42], patients=["p1", "p1", "p2", "p2"])
    got = evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())
    assert (got.n, got.n_patients) == (4, 2)


def test_resampling_patients_gives_a_wider_interval_than_resampling_cases() -> None:
    """THE RULE THAT IS EASIEST TO GET WRONG AND HARDEST TO SEE IN THE RESULT.

    Case-level resampling treats several series of one patient as independent observations and
    narrows the interval. How much it narrows by is not a matter of taste: for a cohort of `m`
    strongly correlated cases per patient the design effect is about `sqrt(m)`, so the fixture
    fixes `m` and the assertion is derived from it rather than from a factor somebody liked.

    m = 6 here, so the honest interval should be roughly 2.4 times the dishonest one, and the
    test demands 1.5 -- comfortably inside that and still far outside noise. An earlier fixture
    used TWO patients with four series each: the cluster bootstrap there can only draw three
    distinct means, so it gave 0.80 against 0.60 -- wider, correctly, but only by 1.33, which is
    the coarseness of two clusters rather than a weak effect.

    The case-level figure is computed here from the same values, so the assertion is against the
    forbidden alternative and not against a tolerance.
    """
    per_patient = 6
    patient_means = [0.90, 0.85, 0.20, 0.15]
    values: list[float] = []
    patients: list[str] = []
    for index, mean in enumerate(patient_means):
        # A tiny within-patient spread: the correlation inside a patient is what the cluster
        # bootstrap exists to respect, and a fixture with none would not distinguish the two
        # schemes at all.
        for series in range(per_patient):
            values.append(mean + (series - per_patient / 2) * 0.002)
            patients.append(f"p{index}")
    rows = _rows(values, patients=patients)
    clustered = evidence.aggregate("dice_mean_per_case", rows,
                                   conventions_block=_conventions())
    assert clustered.n == len(values) and clustered.n_patients == len(patient_means)

    generator = np.random.default_rng(20260926)
    array = np.asarray(values)
    case_level = np.asarray([
        array[generator.integers(0, len(array), size=len(array))].mean()
        for _ in range(evidence.BOOTSTRAP_B)
    ])
    case_low, case_high = np.percentile(case_level, [2.5, 97.5])

    clustered_width = clustered.ci_high - clustered.ci_low
    case_width = float(case_high - case_low)
    assert clustered_width > case_width * 1.5, (
        f"the cluster bootstrap gave a width of {clustered_width:.4f} against the case-level "
        f"{case_width:.4f} at {per_patient} cases per patient, where the design effect should "
        f"be about {per_patient ** 0.5:.1f}: patients are not being resampled as clusters"
    )


def test_the_interval_is_reproducible_from_the_seed_alone() -> None:
    """`MOS-EVID-060`: reproducible from the persisted per-case rows. An interval nobody can
    recompute is a decoration, and the seed is what makes it not one."""
    rows = _rows([0.9, 0.5, 0.7, 0.3], patients=["p1", "p2", "p3", "p4"])
    first = evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())
    again = evidence.aggregate("dice_mean_per_case", rows, conventions_block=_conventions())
    assert (first.ci_low, first.ci_high) == (again.ci_low, again.ci_high)

    other = evidence.aggregate("dice_mean_per_case", rows,
                               conventions_block=_conventions(bootstrap_seed=1))
    assert (other.ci_low, other.ci_high) != (first.ci_low, first.ci_high), (
        "the seed does not change the interval, so it is not the seed the interval was drawn "
        "with and recording it proves nothing"
    )


def test_a_median_metric_is_bootstrapped_as_a_median() -> None:
    """`hd95_mm` and `assd_mm` aggregate by MEDIAN over eligible cases, not by mean. Taking the
    mean of a distance distribution with one catastrophic case is how an aggregate hides
    exactly the case a reader needs to see."""
    rows = _rows([2.0, 3.0, 4.0, 500.0], patients=["p1", "p2", "p3", "p4"])
    got = evidence.aggregate("hd95_mm", rows, conventions_block=_conventions())
    assert got.value == pytest.approx(3.5), "hd95 was not aggregated as a median"
    assert got.value < 10.0


# =====================================================================================
# Count-based metrics: pooled, not a mean of ratios
# =====================================================================================
def _counts(pairs, patients=None, *, eligible=True):
    patients = patients or [f"p{i}" for i in range(len(pairs))]
    return [
        evidence.CaseCounts(case=f"c{i}", patient_key=patients[i], numerator=float(n),
                            denominator=float(d), eligible=eligible)
        for i, (n, d) in enumerate(pairs)
    ]


OP = {"name": "neo_probability", "value": 0.5, "selected_on": "tune"}


def test_sensitivity_pools_the_counts_rather_than_averaging_per_case_ratios() -> None:
    """THE DISTINCTION THE REGISTRY MAKES AND THE REASON IT MATTERS HERE.

    Two cases: one holds a single lesion and it was found (1/1); the other holds twenty and one
    was found (1/20). Pooled sensitivity is 2/21 = 0.095 -- the model found two of twenty-one
    lesions. The mean of the per-case ratios is (1.0 + 0.05)/2 = 0.525, which reads as a
    passable model and is arithmetic about cases rather than about findings.

    On a cohort where the hard cases are the ones carrying many small findings, that gap is the
    whole measurement.
    """
    got = evidence.aggregate_counted("sensitivity", _counts([(1, 1), (1, 20)]),
                                     conventions_block=_conventions(), operating_point=OP)
    assert got.value == pytest.approx(2 / 21)
    mean_of_ratios = (1.0 + 1 / 20) / 2
    assert abs(got.value - mean_of_ratios) > 0.4, (
        "the fixture no longer separates the pooled count from the mean of ratios"
    )


def test_a_case_with_no_reference_lesion_is_ineligible_rather_than_zero() -> None:
    """`exclude_and_report_separately`: it has nothing to be right or wrong about, so it leaves
    the aggregate and joins the empty-reference block. Scoring it zero would be the false
    negative this trainer exists to remove, arriving through the aggregation rule."""
    rows = _counts([(2, 4), (1, 2)]) + [
        evidence.CaseCounts(case="c9", patient_key="p9", numerator=0.0, denominator=0.0,
                            eligible=False, undefined_reason="empty_ground_truth")
    ]
    got = evidence.aggregate_counted("sensitivity", rows, conventions_block=_conventions(),
                                     operating_point=OP)
    assert got.n == 2
    assert got.value == pytest.approx(3 / 6)


def test_an_eligible_case_with_a_zero_denominator_is_refused() -> None:
    rows = [evidence.CaseCounts(case="c0", patient_key="p0", numerator=0.0, denominator=0.0,
                                eligible=True)]
    with pytest.raises(ValueError, match="denominator of 0"):
        evidence.aggregate_counted("sensitivity", rows, conventions_block=_conventions(),
                                   operating_point=OP)


def test_a_per_case_metric_cannot_be_pooled_as_counts() -> None:
    """`dice_mean_per_case` aggregates by MEAN over cases. Pooling its values as a ratio would
    silently turn it into `dice_pooled`, which is a different registered metric with a different
    value -- and both are mandatory, so conflating them loses one of the two."""
    with pytest.raises(ValueError, match="is a per-case metric"):
        evidence.aggregate_counted("dice_mean_per_case", _counts([(1, 2)]),
                                   conventions_block=_conventions())


def test_the_count_bootstrap_resamples_patients_and_re_sums_both_sides() -> None:
    """A patient contributing several cases must move the interval as ONE observation. With the
    counts re-summed per draw, a patient whose cases all fail drags the whole ratio; averaging
    their per-case ratios first would blunt that."""
    rows = _counts([(9, 10), (9, 10), (9, 10), (0, 10), (0, 10), (0, 10)],
                   patients=["p1", "p1", "p1", "p2", "p2", "p2"])
    got = evidence.aggregate_counted("sensitivity", rows, conventions_block=_conventions(),
                                     operating_point=OP)
    assert got.value == pytest.approx(27 / 60)
    assert (got.n, got.n_patients) == (6, 2)
    # Only two clusters exist, so the draws can only be all-p1, all-p2 or a mix: the interval
    # must span almost the entire range rather than hugging the point estimate.
    assert got.ci_low < 0.1 and got.ci_high > 0.8, (got.ci_low, got.ci_high)


def test_an_unregistered_id_is_refused_on_the_counted_path_too() -> None:
    """The refusal cannot live on only one of the two paths: `MOS-EVID-070` does not care which
    function computed the id it invalidates the run for."""
    with pytest.raises(ValueError, match="not in the MOS-EVID-054 registry"):
        evidence.aggregate_counted("lesion_recall", _counts([(1, 2)]),
                                   conventions_block=_conventions(), operating_point=OP)


def test_a_threshold_needing_metric_is_refused_without_an_operating_point_on_both() -> None:
    with pytest.raises(ValueError, match="needs an operating point"):
        evidence.aggregate_counted("ppv", _counts([(1, 2)]), conventions_block=_conventions())
