# SPDX-License-Identifier: Apache-2.0
"""Registry metrics and the four companions, from per-case rows. Chapter 7's aggregation half.

WHY THIS FILE EXISTS
--------------------
Every number this trainer produced before it was invented here: `recall_micro`, `dice_micro`,
`precision_macro`. None of them is a metric the platform may persist. `MOS-EVID-054` closes the
set at seventeen registered ids and `MOS-EVID-070` makes a run that persists an unregistered id
`INVALIDATED` -- not warned about, invalidated. And the names collide misleadingly: the
registry's `sensitivity` is "count-based over cases", while the `recall_micro` measured so far
is a ratio of voxels. One English word, two quantities, and the voxel one cannot support a
clinical claim.

WHAT `MOS-EVID-056` DEMANDS OF EVERY AGGREGATE, AND WHY EACH PART IS HERE
-------------------------------------------------------------------------
"Every aggregate MUST be persisted and reported with four companions: `n` (eligible cases),
`n_patients`, a confidence interval, and the convention block. A bare scalar metric MUST fail
schema validation. This is the rule that makes the previous version's `dice: 0.91` -- no
cohort, no n, no CI, no run -- structurally impossible."

The confidence interval is the part nobody guesses right, and this module would have got it
wrong three ways. `MOS-EVID-057` to `MOS-EVID-060` fix it:

  * a PATIENT-LEVEL CLUSTER bootstrap over `patient_key`, resampling patients and taking all
    of a drawn patient's cases. Case-level resampling is FORBIDDEN, because two series of one
    patient are not two independent observations and treating them as such narrows the interval
    by a factor nobody can see in the number;
  * the PERCENTILE method. Wilson and Clopper-Pearson are forbidden even for a proportion --
    they assume independent Bernoulli trials, which clustered cases are not;
  * B = 2000 draws, and the SEED RECORDED, so the interval is reproducible from the persisted
    per-case rows alone. An interval nobody can recompute is a decoration.

THE EMPTY-REFERENCE POLICY IS FIXED PLATFORM-WIDE and is not this module's choice:
`exclude_and_report_separately`. A case with zero reference voxels gets `value=None`,
`undefined_reason="empty_ground_truth"`, `eligible=False`, and is counted in the five empty-GT
metrics instead. A case where the finding IS present and nothing was predicted scores 0.0 and
is ALWAYS eligible -- that is a measured miss, not an absent measurement. That distinction is
the same one `metrics_from_counts` already draws between `None` and `0.0`, and it turns out the
spec names it.

FIVE EMPTY-GT IDS, AND A REGISTERED CONTRADICTION. `MOS-EVID-052` requires five ids whenever
the cohort holds an empty-reference case, and the `MOS-EVID-054` table prints only one of them
(`empty_gt_false_positive_rate`). Obeying the table literally would make every conforming run
`INVALIDATED` under `MOS-EVID-070`. The contradiction is recorded in
`docs/spec/99-known-inconsistencies.md` (items 14, 27, 36) and its resolution is that all five
are registered in registry version 1. This module follows the resolution, and `REGISTRY` says
so at the rows in question rather than leaving a reader to wonder which side it took.

Pure: numpy only. No torch, no nnU-Net, no I/O -- so the bootstrap and the eligibility rules
are checkable without a card, which is the point of putting them here rather than in the tool.

Spec: MOS-EVID-047 to MOS-EVID-060, MOS-EVID-067, MOS-EVID-069, MOS-EVID-070.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "BOOTSTRAP_B",
    "CaseCounts",
    "aggregate_counted",
    "CI_METHOD",
    "EMPTY_GT_POLICY",
    "REGISTRY",
    "CaseValue",
    "Aggregate",
    "aggregate",
    "conventions",
]

#: `MOS-EVID-058`: the number of bootstrap draws, fixed by the spec rather than chosen.
BOOTSTRAP_B: Final[int] = 2000

#: `MOS-EVID-059`: the percentile method over the cluster bootstrap distribution. Named in the
#: convention block so a reader never has to ask which interval they are looking at.
CI_METHOD: Final[str] = "patient_cluster_bootstrap_percentile"

#: `MOS-EVID-051`/`MOS-STORE-296`: fixed platform-wide, not a per-run choice.
EMPTY_GT_POLICY: Final[str] = "exclude_and_report_separately"

#: The metric ids this module can compute, with the aggregation each one's registry row fixes.
#: A closed set on purpose: `MOS-EVID-070` invalidates a run that persists an id absent from
#: the registry version recorded on it, so an id invented here would invalidate the run it was
#: invented for.
#:
#: `needs_threshold` is `MOS-EVID-055`'s rule: those four may never be persisted or reported
#: without an operating point, and a bare number MUST be rejected by the schema.
REGISTRY: Final[dict[str, dict[str, Any]]] = {
    # -- segmentation -----------------------------------------------------------------
    "dice_mean_per_case": {"aggregation": "mean", "per_case": True, "needs_threshold": False,
                           "unit": "1"},
    #: The pooled form of `MOS-EVID-048`: `2*sum|A&B| / sum(|A|+|B|)`. Identical to the
    #: `dice_micro` this trainer already computes -- `|A&B|` is tp and `|A|+|B|` is
    #: `2tp+fp+fn` -- so that number was right and only its NAME was unregistered.
    "dice_pooled": {"aggregation": "pooled", "per_case": False, "needs_threshold": False,
                    "unit": "1"},
    "iou_mean_per_case": {"aggregation": "mean", "per_case": True, "needs_threshold": False,
                          "unit": "1"},
    "hd95_mm": {"aggregation": "median", "per_case": True, "needs_threshold": False,
                "unit": "mm"},
    "assd_mm": {"aggregation": "median", "per_case": True, "needs_threshold": False,
                "unit": "mm"},
    "volume_error_ml": {"aggregation": "mean", "per_case": True, "needs_threshold": False,
                        "unit": "mL"},
    "volume_ape": {"aggregation": "median", "per_case": True, "needs_threshold": False,
                   "unit": "1"},
    # -- classification and detection -------------------------------------------------
    "sensitivity": {"aggregation": "count", "per_case": False, "needs_threshold": True,
                    "unit": "1"},
    "specificity": {"aggregation": "count", "per_case": False, "needs_threshold": True,
                    "unit": "1"},
    "ppv": {"aggregation": "count", "per_case": False, "needs_threshold": True, "unit": "1"},
    "froc_sensitivity": {"aggregation": "interpolated", "per_case": False,
                         "needs_threshold": True, "unit": "1"},
    "auroc": {"aggregation": "over_case_scores", "per_case": False, "needs_threshold": False,
              "unit": "1"},
    "auprc": {"aggregation": "over_case_scores", "per_case": False, "needs_threshold": False,
              "unit": "1"},
    "ece_15bin": {"aggregation": "bins", "per_case": False, "needs_threshold": False,
                  "unit": "1"},
    "brier": {"aggregation": "mean", "per_case": True, "needs_threshold": False, "unit": "1"},
    # -- measurement ------------------------------------------------------------------
    "mae": {"aggregation": "mean", "per_case": True, "needs_threshold": False,
            "unit": "metric unit"},
    # -- the empty-reference block ----------------------------------------------------
    #: FOUR OF THESE FIVE ARE ABSENT FROM THE MOS-EVID-054 TABLE and required by
    #: MOS-EVID-052. See the module docstring: the contradiction is registered and its
    #: resolution is that all five belong to registry version 1.
    "empty_gt_case_count": {"aggregation": "count", "per_case": False,
                            "needs_threshold": False, "unit": "1"},
    "empty_gt_false_positive_rate": {"aggregation": "count", "per_case": False,
                                     "needs_threshold": "via volume threshold", "unit": "1"},
    "empty_gt_mean_fp_volume_ml": {"aggregation": "mean", "per_case": False,
                                   "needs_threshold": False, "unit": "mL"},
    "empty_gt_p95_fp_volume_ml": {"aggregation": "percentile", "per_case": False,
                                  "needs_threshold": False, "unit": "mL"},
    "empty_gt_max_fp_volume_ml": {"aggregation": "max", "per_case": False,
                                  "needs_threshold": False, "unit": "mL"},
}


@dataclass(frozen=True)
class CaseValue:
    """One case's contribution to one metric, with its eligibility stated.

    `patient_key` IS REQUIRED AND IS NOT THE CASE KEY. The bootstrap resamples PATIENTS, so a
    cohort where two series of one patient carry the same `patient_key` gives a wider and
    correct interval, and one where they carry different keys gives a narrower and wrong one.
    It is a separate field rather than a prefix of `case` because deriving it by string
    surgery is how a convention becomes a bug.
    """

    case: str
    patient_key: str
    #: `None` when the metric is undefined for this case. Then `undefined_reason` says why and
    #: `eligible` is False. A measured zero is `0.0` with `eligible` True -- the difference
    #: between "nothing to measure" and "found nothing", which is the whole of
    #: `MOS-EVID-051`.
    value: float | None
    eligible: bool
    undefined_reason: str | None = None


@dataclass(frozen=True)
class Aggregate:
    """A metric with everything `MOS-EVID-056` requires beside it. A bare float is refused."""

    metric: str
    value: float | None
    n: int
    n_patients: int
    ci_low: float | None
    ci_high: float | None
    conventions: Mapping[str, Any]
    operating_point: Mapping[str, Any] | None = None

    def as_document(self) -> dict[str, Any]:
        document = {
            "metric": self.metric,
            "value": self.value,
            "n": self.n,
            "n_patients": self.n_patients,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "conventions": dict(self.conventions),
        }
        if self.operating_point is not None:
            document["operating_point"] = dict(self.operating_point)
        return document


def conventions(
    *,
    metric_registry_version: int,
    dice_aggregation: str,
    fp_volume_threshold_ml: float,
    bootstrap_seed: int,
    min_candidate_volume_ml: float,
    empty_gt_policy: str = EMPTY_GT_POLICY,
    ci_method: str = CI_METHOD,
    bootstrap_b: int = BOOTSTRAP_B,
) -> dict[str, Any]:
    """The seven-member convention block of section 7.12.1, in that order and no other members.

    A FIXED OBJECT AND NOT A FREE-FORM BAG. Every member changes what a number means:
    `dice_aggregation` decides whether 0.91 is a mean over cases or a pooled ratio,
    `empty_gt_policy` decides whether the cases with nothing to find are in it,
    `fp_volume_threshold_ml` decides what counts as a false positive on such a case, and the
    three bootstrap members decide whether the interval beside it can be recomputed at all.
    """
    if dice_aggregation not in {"dice_mean_per_case", "dice_pooled"}:
        raise ValueError(
            f"dice_aggregation is {dice_aggregation!r}; MOS-EVID-050 requires a criterion to "
            "name one of the two registered conventions explicitly, and there is no default"
        )
    if not fp_volume_threshold_ml > 0:
        raise ValueError(
            f"fp_volume_threshold_ml is {fp_volume_threshold_ml!r}. It decides what counts as "
            "a false positive on a case with nothing to find, and at zero every stray voxel "
            "is one"
        )
    return {
        "dice_aggregation": dice_aggregation,
        "empty_gt_policy": empty_gt_policy,
        "fp_volume_threshold_ml": float(fp_volume_threshold_ml),
        #: An EIGHTH member beyond section 7.12.1's seven, and it is here on the same argument
        #: the seven are: it changes what the number means. A component filter chosen per run
        #: and not recorded makes two reports incomparable while looking identical. The
        #: registries already carry `connected_component_filter` as rule-set data, so this is
        #: that value travelling with the metric rather than a new idea.
        "min_candidate_volume_ml": float(min_candidate_volume_ml),
        "ci_method": ci_method,
        "bootstrap_b": int(bootstrap_b),
        "bootstrap_seed": int(bootstrap_seed),
        "metric_registry_version": int(metric_registry_version),
    }


#: The seven members section 7.12.1 fixes, named once so the two aggregation paths cannot
#: disagree about what a complete block is.
_CONVENTION_MEMBERS: Final[frozenset[str]] = frozenset({
    "dice_aggregation", "empty_gt_policy", "fp_volume_threshold_ml", "ci_method",
    "bootstrap_b", "bootstrap_seed", "metric_registry_version",
    "min_candidate_volume_ml",
})


@dataclass(frozen=True)
class CaseCounts:
    """One case's NUMERATOR and DENOMINATOR for a count-based metric.

    WHY COUNTS AND NOT A RATIO. The `MOS-EVID-054` registry marks `sensitivity`, `specificity`,
    `ppv` and the empty-reference rates as `count-based` with `per_case: no`. That is not a
    detail of presentation: a mean of per-case sensitivities weights a case holding one lesion
    the same as a case holding twenty, and on a cohort where the hard cases are the ones with
    many small findings that is the difference between a number that sees them and one that
    does not.

    So a case contributes `found` out of `present`, the aggregate is `sum(found) / sum(present)`
    and the bootstrap re-sums both over resampled PATIENTS. Handing this function a ratio would
    make the resampling average ratios, which is the same defect one level down.
    """

    case: str
    patient_key: str
    numerator: float
    denominator: float
    eligible: bool
    undefined_reason: str | None = None


def aggregate_counted(
    metric: str,
    rows: Sequence[CaseCounts],
    *,
    conventions_block: Mapping[str, Any],
    operating_point: Mapping[str, Any] | None = None,
) -> Aggregate:
    """A count-based registry metric: `sum(numerator) / sum(denominator)` over eligible cases.

    The bootstrap resamples PATIENTS and re-sums both sides, so a patient contributing many
    cases moves the interval as one observation. Averaging per-case ratios instead -- or
    resampling cases -- would both narrow it, and in opposite ways that are hard to tell apart
    afterwards.

    A case whose denominator is zero has nothing to be right or wrong about and is INELIGIBLE:
    for `sensitivity` that is a case with no reference lesion, which `MOS-EVID-051`'s
    `exclude_and_report_separately` moves into the empty-reference block rather than scoring.
    Counting it as zero would be the false negative this trainer exists to remove, arriving
    through the aggregation rule.
    """
    import numpy as np

    if metric not in REGISTRY:
        raise ValueError(
            f"{metric!r} is not in the MOS-EVID-054 registry, whose ids are "
            f"{sorted(REGISTRY)}. MOS-EVID-070 makes a run that persists an unregistered id "
            "INVALIDATED, so this would invalidate the run it was computed for"
        )
    if REGISTRY[metric]["needs_threshold"] and operating_point is None:
        raise ValueError(
            f"{metric!r} needs an operating point (MOS-EVID-055) and none was given. A "
            "sensitivity without the threshold that produced it is not a measurement"
        )
    # THE DISCRIMINATOR IS THE REGISTRY'S OWN `per_case` FIELD, not a hand-written list of
    # aggregation words. The first version listed the allowed words and included "mean" -- which
    # admits `dice_mean_per_case` and `brier` and so admitted everything, defeating the guard it
    # was. A list of sites has the same blind spot as the drift it guards against.
    if REGISTRY[metric]["per_case"]:
        raise ValueError(
            f"{metric!r} is a per-case metric (MOS-EVID-054), so it aggregates over its "
            "per-case VALUES and not as a count over cases. Pooling them as a ratio would "
            "silently produce a different registered metric with a different value"
        )
    missing = sorted(_CONVENTION_MEMBERS - set(conventions_block))
    if missing:
        raise ValueError(
            f"the convention block is missing {missing}. MOS-EVID-056 makes a bare scalar "
            "metric fail schema validation, and an incomplete block is a bare scalar with "
            "decoration"
        )

    eligible = [row for row in rows if row.eligible]
    for row in eligible:
        if row.denominator <= 0:
            raise ValueError(
                f"case {row.case!r} is eligible for {metric!r} with a denominator of "
                f"{row.denominator!r}. Eligibility is the claim that this case could be right "
                "or wrong, and a zero denominator says it could not"
            )
    if not eligible:
        return Aggregate(metric=metric, value=None, n=0, n_patients=0, ci_low=None,
                         ci_high=None, conventions=dict(conventions_block),
                         operating_point=operating_point)

    def ratio(sample: Sequence[CaseCounts]) -> float:
        denominator = sum(r.denominator for r in sample)
        return float(sum(r.numerator for r in sample) / denominator)

    by_patient: dict[str, list[CaseCounts]] = {}
    for row in eligible:
        by_patient.setdefault(row.patient_key, []).append(row)
    patients = sorted(by_patient)
    generator = np.random.default_rng(int(conventions_block["bootstrap_seed"]))
    draws = int(conventions_block["bootstrap_b"])

    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = generator.integers(0, len(patients), size=len(patients))
        sample: list[CaseCounts] = []
        for index in chosen:
            sample.extend(by_patient[patients[int(index)]])
        estimates[draw] = ratio(sample)

    low, high = (float(x) for x in np.percentile(estimates, [2.5, 97.5]))
    return Aggregate(
        metric=metric, value=ratio(eligible), n=len(eligible), n_patients=len(patients),
        ci_low=low, ci_high=high, conventions=dict(conventions_block),
        operating_point=operating_point,
    )


def _point_estimate(metric: str, values: Sequence[float]) -> float:
    import numpy as np

    how = REGISTRY[metric]["aggregation"]
    array = np.asarray(values, dtype=float)
    if how == "median":
        return float(np.median(array))
    if how in {"mean", "count", "pooled", "over_case_scores", "interpolated", "bins", "max",
               "percentile"}:
        # `mean` is the only one this function can compute from per-case VALUES. The others
        # are computed from counts or scores by their own callers and passed in already
        # reduced, one value per case, so the mean of one number is that number.
        return float(array.mean()) if how != "max" else float(array.max())
    raise ValueError(f"no reduction is defined for aggregation {how!r} of metric {metric!r}")


def aggregate(
    metric: str,
    rows: Sequence[CaseValue],
    *,
    conventions_block: Mapping[str, Any],
    operating_point: Mapping[str, Any] | None = None,
) -> Aggregate:
    """One registry metric over per-case rows, with `n`, `n_patients` and a cluster bootstrap.

    REFUSES an unregistered id, because `MOS-EVID-070` would invalidate the whole run for it,
    and refuses a threshold-needing metric with no operating point, because `MOS-EVID-055`
    says a number without one "is not a measurement and the schema MUST reject it".

    Returns `value=None` with `n=0` when no case was eligible. That is not a failure and it is
    not zero: on a cohort where nothing could be measured, a zero would read as a model that
    finds nothing.
    """
    import numpy as np

    if metric not in REGISTRY:
        raise ValueError(
            f"{metric!r} is not in the MOS-EVID-054 registry, whose ids are "
            f"{sorted(REGISTRY)}. MOS-EVID-070 makes a run that persists an unregistered id "
            "INVALIDATED, so this would invalidate the run it was computed for"
        )
    if REGISTRY[metric]["needs_threshold"] and operating_point is None:
        raise ValueError(
            f"{metric!r} needs an operating point (MOS-EVID-055) and none was given. A "
            "sensitivity without the threshold that produced it is not a measurement"
        )
    # THE MIRROR OF THE GUARD IN `aggregate_counted`, from the same registry field. A metric the
    # registry marks `per_case: no` has no per-case value to average: `sensitivity` is a count
    # over cases and `dice_pooled` is a ratio of sums, and reducing either as a mean of
    # whatever a caller happened to pass would produce a number under a registered name that
    # the registry does not define that way.
    if not REGISTRY[metric]["per_case"]:
        raise ValueError(
            f"{metric!r} is not a per-case metric (MOS-EVID-054), so it has no per-case values "
            "to reduce. Use `aggregate_counted` with a numerator and a denominator per case"
        )
    missing = sorted(_CONVENTION_MEMBERS - set(conventions_block))
    if missing:
        raise ValueError(
            f"the convention block is missing {missing}. MOS-EVID-056 makes a bare scalar "
            "metric fail schema validation, and an incomplete block is a bare scalar with "
            "decoration"
        )

    eligible = [row for row in rows if row.eligible]
    for row in eligible:
        if row.value is None:
            raise ValueError(
                f"case {row.case!r} is eligible for {metric!r} and carries no value. "
                "Eligibility is the claim that this case was measured; a None beside it is "
                "the two halves of MOS-EVID-051 contradicting each other"
            )
    n_patients = len({row.patient_key for row in eligible})
    if not eligible:
        return Aggregate(metric=metric, value=None, n=0, n_patients=0, ci_low=None,
                         ci_high=None, conventions=dict(conventions_block),
                         operating_point=operating_point)

    value = _point_estimate(metric, [float(row.value) for row in eligible])  # type: ignore[arg-type]

    # THE CLUSTER BOOTSTRAP. Patients are drawn with replacement and ALL of a drawn patient's
    # cases come with them, which is what makes the interval honest when one patient
    # contributed several series. Resampling cases instead would treat those as independent
    # observations and narrow the interval by a factor invisible in the result.
    by_patient: dict[str, list[float]] = {}
    for row in eligible:
        by_patient.setdefault(row.patient_key, []).append(float(row.value))  # type: ignore[arg-type]
    patients = sorted(by_patient)
    generator = np.random.default_rng(int(conventions_block["bootstrap_seed"]))
    draws = int(conventions_block["bootstrap_b"])

    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = generator.integers(0, len(patients), size=len(patients))
        sample: list[float] = []
        for index in chosen:
            sample.extend(by_patient[patients[int(index)]])
        estimates[draw] = _point_estimate(metric, sample)

    low, high = (float(x) for x in np.percentile(estimates, [2.5, 97.5]))
    return Aggregate(
        metric=metric, value=value, n=len(eligible), n_patients=n_patients,
        ci_low=low, ci_high=high, conventions=dict(conventions_block),
        operating_point=operating_point,
    )
