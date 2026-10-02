# SPDX-License-Identifier: Apache-2.0
"""Lesion-level detection: candidates, matching, and the counts a clinical metric is made of.

WHY THIS FILE EXISTS, AND WHY THE NUMBERS BEFORE IT WERE THE WRONG QUANTITY
---------------------------------------------------------------------------
Everything this trainer has measured so far is VOXEL-level: recall, precision and Dice over
voxels, per channel and micro-averaged. Those were the right numbers for the question they
answered -- whether unmasked training suppresses findings -- because the supervision mask acts
on voxels and nothing else.

They are not, and cannot become, evidence of clinical performance. `MOS-EVID-054`'s metric
registry is explicit about it:

    sensitivity        classification, detection   count-based OVER CASES     needs a threshold
    ppv                classification, detection   count-based               needs a threshold
    froc_sensitivity   detection                   interpolated at declared FP/scan points
    dice_mean_per_case segmentation                mean over eligible cases

`sensitivity` in that table is a count over CASES, not a ratio of voxels. One English word,
two different quantities. A model can score 0.5 voxel recall and find every lesion (it
under-segments what it finds), or 0.8 voxel recall and miss every small lesion entirely. The
second is the failure that matters clinically and voxel recall cannot see it.

WHY CANDIDATES AND NOT A THRESHOLD SWEEP
-----------------------------------------
`MOS-EVID-069` requires an `EvaluationRun` to persist, per predicted candidate,
`{score, centroid_lps_mm, volume_ml, matched_reference_id, match_distance_mm}`, and says why:
"Together with `case_score` this makes ROC, PR and FROC curves recomputable and **makes an
operating threshold re-selectable without re-running inference**, which is the property that
stops a threshold change from becoming a GPU project."

That is a direct verdict on the seven-point threshold sweep this trainer grew first: it works,
it costs 24 minutes of card per arm per grid, and the platform already knows the right artefact
is a candidate list from which every threshold is free. So the sweep stays as a voxel-level
instrument and this is what a clinical claim is built from.

THE ONE APPROXIMATION, DECLARED RATHER THAN HIDDEN
---------------------------------------------------
Candidates are extracted ONCE, at a permissive `extraction_threshold`, and a higher operating
point is applied afterwards by filtering on each candidate's score. That is what makes a
threshold free. It is also an approximation: at a higher threshold a single component can
SPLIT into several, and filtering cannot see that. The standard detection literature accepts
it and so does this module -- but the extraction threshold is part of the convention block, so
a reader can see which threshold the component structure was fixed at rather than inferring it.

THE MATCHING RULE, AND WHY IT NEEDS NO TOLERANCE PARAMETER
-----------------------------------------------------------
A predicted component is ATTRIBUTED to the reference lesion it overlaps MOST, which is what
makes the false-positive count meaningful: one candidate is one claim and belongs to one thing.
A reference lesion is FOUND if ANY surviving candidate overlaps it, which is a different
question with a different answer -- a lesion a prediction covers has been found whatever else
that prediction also covers. Answering the second with the first understates sensitivity
precisely on clustered findings, which is where the undercount is largest and least visible.

The convention that any overlap finds is the one detection challenges use, and its risk is the
mirror image: one blob smeared over a cluster claims every lesion in it. `overlap_voxels` on
each candidate is what lets a reader see that happening rather than take the count on faith. Overlap rather than
a centroid-distance tolerance, because these predictions are segmentations: a candidate that
covers a lesion has found it whatever its centroid does, and a tolerance in millimetres is a
parameter whose value would have to be defended per finding. `match_distance_mm` is still
recorded, because `MOS-EVID-069` requires it and because it is what a reader uses to judge a
match they doubt.

SPACE. Everything here is computed in the space the arrays arrive in, which for this trainer
is nnU-Net's PREPROCESSED geometry, and `volume_ml` uses the spacing passed in. A centroid is
returned in voxels, not LPS millimetres: `MOS-EVID-069` wants LPS, and LPS needs the case's
original origin and direction, which means resampling the prediction back. That is a declared
gap rather than a silent approximation -- see `candidates_for_case`'s `centroid_voxel`.

Pure: numpy and scipy.ndimage only, both already in the trainer image via nnU-Net. No torch, no
nnU-Net, no I/O -- so every rule here is checkable on arrays small enough to count by hand.

Spec: MOS-EVID-054, MOS-EVID-055, MOS-EVID-056, MOS-EVID-069.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

__all__ = [
    "FROC_POINTS",
    "CaseDetections",
    "counts_at",
    "froc_at",
    "froc_curve",
    "Candidate",
    "Reference",
    "candidates_for_case",
    "label_value_of",
]

#: 6-connectivity in 3D: a voxel touches its six face neighbours and not its edge or corner
#: ones. Declared because it changes what counts as ONE lesion -- under 26-connectivity two
#: blobs touching at a corner are one candidate, and a detection count is a count of lesions.
#: `MOS-EVID-108` fixes 6-connectivity for the body contour on exactly this argument, so this
#: module follows rather than choosing differently.
CONNECTIVITY: Final[int] = 6


@dataclass(frozen=True)
class Candidate:
    """One predicted component. The members `MOS-EVID-069` requires, plus what it matched."""

    #: The highest probability inside the component. A candidate survives an operating point
    #: `t` when `score >= t` -- the max and not the mean, because a component is a detection
    #: claim and its strongest voxel is the strength of the claim.
    score: float
    volume_ml: float
    #: VOXEL coordinates, not LPS millimetres. See the module docstring: LPS needs the case's
    #: original origin and direction. Named so that nobody mistakes one for the other.
    centroid_voxel: tuple[float, ...]
    #: The reference lesion this overlaps most, or `None` for a false positive.
    matched_reference_id: str | None
    #: Centroid separation from the matched reference, in millimetres. `None` when unmatched:
    #: a distance to nothing is not a large distance.
    match_distance_mm: float | None
    #: Voxels shared with the matched reference. Recorded because a match with an overlap of
    #: one voxel and a match that covers the lesion are both "matched" and a reader needs to
    #: be able to tell them apart.
    overlap_voxels: int
    #: EVERY reference this candidate overlaps, with THIS CANDIDATE'S CONFIDENCE ON THAT
    #: REFERENCE'S OWN VOXELS: `(reference_id, max probability within the shared region)`.
    #:
    #: WHY THE SCORE IS PER REFERENCE AND NOT THE COMPONENT'S. `score` above is the maximum over
    #: the whole component, which is the right number for a DETECTION -- one predicted object, one
    #: confidence. It is the wrong number for "was this lesion found", and using it there lets a
    #: reference with NO predicted voxel at the operating point be reported found: one component
    #: extracted at 0.1 can span a strong lesion at 0.93 and a weak one whose every voxel is 0.2,
    #: and the component's 0.93 was recorded for both. Reproduced on a four-voxel fixture:
    #: sensitivity 1.000 at threshold 0.5 beside a Dice, from the same probabilities, showing the
    #: second lesion entirely unpredicted -- and the FROC curve flat at 1.0 out to 0.93, because
    #: raising the threshold cannot remove a reference whose recorded score sits somewhere else.
    #:
    #: THE SAME MISTAKE, THIRD SPELLING. Attribution was first used for the found/not-found
    #: question in `candidates_for_case`, then again in `counts_at`; this is the same confusion one
    #: level down -- the right SET of references with the wrong SCORE against each of them.
    #:
    #: `overlapped_reference_ids` is a property over this field rather than a second field, so the
    #: two cannot drift: there is one place the overlap set is decided.
    overlap_scores: tuple[tuple[str, float], ...] = ()

    @property
    def overlapped_reference_ids(self) -> tuple[str, ...]:
        """Every reference this candidate touches at the extraction threshold.

        Derived, not stored. A reader asking "which lesions does this claim cover" wants this; a
        reader asking "was this lesion found at t" wants `overlap_scores` and must compare against
        t, because covering a lesion at 0.2 is not finding it at 0.5.
        """
        return tuple(reference_id for reference_id, _score in self.overlap_scores)


@dataclass(frozen=True)
class Reference:
    """One reference lesion, and whether anything found it."""

    id: str
    volume_ml: float
    centroid_voxel: tuple[float, ...]
    #: Scores of every candidate overlapping it, highest first. A reference is FOUND at
    #: operating point `t` when any of these is `>= t` -- so this list is what makes
    #: sensitivity at any threshold computable without re-running anything.
    matched_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class CaseDetections:
    """One case's candidates and references, per channel, for the channels it supervises.

    Channels the case does NOT annotate are absent from both maps rather than present and
    empty. Present-and-empty would read as "annotated, nothing there", which is the false
    negative this whole subsystem exists to remove -- arriving through the evaluator instead of
    through the loss.
    """

    case: str
    candidates: Mapping[str, tuple[Candidate, ...]] = field(default_factory=dict)
    references: Mapping[str, tuple[Reference, ...]] = field(default_factory=dict)


#: `froc_sensitivity` is "interpolated at declared FP/scan points" (`MOS-EVID-054`), and these
#: are the points. They are the LUNA16 set, which is what the detection literature reports and
#: therefore what a reader can compare against; they are DECLARED here rather than chosen per
#: run because a curve read at different points is a different number under one name.
FROC_POINTS: Final[tuple[float, ...]] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _surviving(candidates: Sequence[Candidate], threshold: float) -> list[Candidate]:
    """Candidates at an operating point. `>=`, matching `MOS-SVC-089`'s `present = score >=
    score_threshold` -- the platform's own rule, so a finding the service would report and a
    finding this counts are the same set."""
    return [c for c in candidates if c.score >= threshold]


#: The component policies this module can COUNT UNDER. Neither is applied to anything: they are
#: two ways of reading one candidate list, reported side by side so the choice between them is
#: made on numbers.
#:
#: `MOS-REG-048` is why they are reported and not chosen here: "Changing an HU threshold or a
#: component filter changes the clinical output and MUST pass the same gate as a weights
#: change." A filter is therefore not a free fix for fragmentation -- it costs a new
#: `EvaluationRun` and the promotion gate, exactly as a retrained network does. What IS free is
#: measuring what it would buy, from candidates already extracted and with no card at all.
COMPONENT_POLICIES: Final[tuple[str, ...]] = ("all_components", "largest_component")


def counts_at(
    detections: Sequence[Any],
    channel: str,
    *,
    threshold: float,
    patient_of: Mapping[str, str],
    policy: str = "all_components",
) -> dict[str, list[Any]]:
    """Per-case numerators and denominators for `sensitivity` and `ppv`, at one operating point.

    Returns `{"sensitivity": [...], "ppv": [...], "empty_reference": [...]}` of `CaseCounts`-
    shaped rows, built here rather than in `evidence.py` because what counts as found is a
    detection question and what to do with the counts is an aggregation one.

    ELIGIBILITY IS DIFFERENT FOR THE TWO METRICS, AND THAT IS THE POINT.

      * `sensitivity` needs a reference lesion. A case with none is INELIGIBLE with
        `undefined_reason="empty_ground_truth"` -- `MOS-EVID-051`'s `exclude_and_report_
        separately` -- and moves into the empty-reference rows, where what it predicted anyway
        is measured as a volume. A case WITH a lesion and no prediction is eligible and scores
        0: that is a measured miss.
      * `ppv` needs a PREDICTION. A case with none has no precision to report, so it is
        ineligible with `undefined_reason="no_prediction"`. Scoring it zero would punish
        silence, and scoring it one would reward it; both are claims about a case where nothing
        was claimed. A model that predicts nothing anywhere therefore gets `value=None` and
        `n=0` for `ppv`, which reads as "no precision could be measured" -- the same statement
        as the dash this evaluator already prints for a channel the control arm never predicts.

    `policy` READS THE SAME CANDIDATES TWO WAYS AND CHANGES NOTHING.

      * `all_components`: every surviving candidate is a detection claim. This is what the
        model actually emits, and on this cohort it emits about 29 of them per case per channel
        against one reference lesion -- sensitivity 1.000 and precision 0.077.
      * `largest_component`: only the biggest surviving candidate counts. For a channel whose
        finding is ONE anatomical structure -- an aortic arch, a pulmonary trunk -- that is the
        anatomy rather than a trick. For nodules or calcifications, which are genuinely several,
        it would throw away real findings, and the sensitivity column is what shows that
        happening.

    Reporting both is the whole point: the filter route is viable only if the second column
    reaches a clinical bar, and that is a measurement rather than an opinion.

    A CANDIDATE'S MATCH DOES NOT MOVE WITH THE THRESHOLD. Matching is overlap with the
    reference, which is fixed at extraction; raising the threshold only removes candidates. So a
    surviving candidate that overlapped a lesion is never a false positive at a higher point,
    and the false-positive count falls monotonically -- which is what makes the FROC curve below
    monotone and therefore interpolatable.
    """
    from medos_trainer.evidence import CaseCounts

    if policy not in COMPONENT_POLICIES:
        raise ValueError(
            f"policy {policy!r} is not one of {list(COMPONENT_POLICIES)}. A policy invented at "
            "the call site would make two reports incomparable while looking identical"
        )

    sensitivity: list[CaseCounts] = []
    ppv: list[CaseCounts] = []
    empty_reference: list[CaseCounts] = []

    for found in detections:
        if channel not in found.references:
            continue                      # the case does not annotate this channel
        case = found.case
        if case not in patient_of:
            raise ValueError(
                f"case {case!r} has no patient key. MOS-EVID-057 resamples PATIENTS, so a case "
                "whose patient is unknown cannot be placed in a cluster and its contribution "
                "to the interval would be wrong in a direction nobody can see"
            )
        patient = patient_of[case]
        references = found.references[channel]
        survivors = _surviving(found.candidates[channel], threshold)
        if policy == "largest_component" and survivors:
            # By VOLUME and not by score: the policy models "this finding is one structure", and
            # the biggest component is the structure. Taking the highest-scoring one instead
            # would keep whichever speck the network was most confident about.
            survivors = [max(survivors, key=lambda c: c.volume_ml)]
        kept = {id(c) for c in survivors}

        if references:
            # A REFERENCE IS FOUND BY A SURVIVING CANDIDATE THAT OVERLAPS IT -- any overlap,
            # from the candidate's full `overlapped_reference_ids` and not from the single
            # `matched_reference_id` it is attributed to.
            #
            # THIS EXACT MISTAKE WAS MADE TWICE. Using the attribution for the found/not-found
            # question undercounts sensitivity wherever one prediction covers a cluster, which
            # on this cohort is coronary calcifications and nodules -- where the undercount is
            # both largest and least visible. It was fixed in `candidates_for_case` and then
            # reintroduced here under a different spelling, with every test still green, which
            # is why the full overlap set is now a field on `Candidate`.
            #
            # Filtering by SURVIVORS rather than by `matched_scores` is still necessary: under
            # `largest_component` a lesion found only by a discarded candidate must not count,
            # or sensitivity would be read under one policy and precision under another in one
            # table.
            # AT THIS THRESHOLD, ON THE REFERENCE'S OWN VOXELS. Taking the candidate's whole
            # overlap set would count a lesion found because something ELSE in the same component
            # reached the threshold -- see `Candidate.overlap_scores`. A candidate whose overlap
            # score on a reference clears `threshold` necessarily clears it component-wide too,
            # since the overlap maximum cannot exceed the component maximum, so the survivor
            # filter above is not weakened by this: it still decides WHICH candidates count, which
            # is what `largest_component` needs.
            found_ids = {
                reference_id for candidate in survivors
                for reference_id, on_reference in candidate.overlap_scores
                if on_reference >= threshold
            }
            hits = sum(1 for reference in references if reference.id in found_ids)
            sensitivity.append(CaseCounts(case=case, patient_key=patient,
                                          numerator=float(hits),
                                          denominator=float(len(references)), eligible=True))
        else:
            sensitivity.append(CaseCounts(
                case=case, patient_key=patient, numerator=0.0, denominator=0.0,
                eligible=False, undefined_reason="empty_ground_truth",
            ))
            # What it predicted on a case with nothing to find, as a VOLUME: the five
            # `MOS-EVID-052` metrics are all about how much, not how many.
            empty_reference.append(CaseCounts(
                case=case, patient_key=patient,
                numerator=float(sum(c.volume_ml for c in survivors)),
                denominator=1.0, eligible=True,
            ))

        if survivors:
            true_positives = sum(1 for c in survivors if c.matched_reference_id is not None)
            ppv.append(CaseCounts(case=case, patient_key=patient,
                                  numerator=float(true_positives),
                                  denominator=float(len(survivors)), eligible=True))
        else:
            ppv.append(CaseCounts(
                case=case, patient_key=patient, numerator=0.0, denominator=0.0,
                eligible=False, undefined_reason="no_prediction",
            ))

    return {"sensitivity": sensitivity, "ppv": ppv, "empty_reference": empty_reference}


def froc_curve(detections: Sequence[Any], channel: str) -> list[dict[str, Any]]:
    """The full detection curve for one channel: every operating point the data can distinguish.

    COMPUTED ONCE FROM THE CANDIDATES AND FROM NOTHING ELSE, which is the property
    `MOS-EVID-069` exists for: "makes an operating threshold re-selectable without re-running
    inference". The thresholds are the candidate scores themselves -- no grid, because a grid
    either misses a point the data distinguishes or invents points it does not.

    Each row is `{score_threshold, sensitivity, false_positives_per_case, cases}`.
    `false_positives_per_case` divides by every case that ANNOTATES the channel, including the
    ones with no reference lesion: a false alarm on a case with nothing to find is exactly the
    false alarm a reader cares about, and excluding those cases would flatter the rate.
    """
    eligible_cases = [f for f in detections if channel in f.references]
    if not eligible_cases:
        return []

    references = [r for f in eligible_cases for r in f.references[channel]]
    candidates = [c for f in eligible_cases for c in f.candidates[channel]]
    total_references = len(references)
    total_cases = len(eligible_cases)

    # EVERY VALUE AT WHICH AN ANSWER CHANGES, which is two sets and not one.
    #
    # A candidate's own `score` changes the false-positive count: raising the threshold past it
    # removes that detection. A candidate's per-reference OVERLAP score changes the sensitivity:
    # raising the threshold past it stops that lesion counting as found. The two differ whenever a
    # component is not uniform -- one sheet spanning a strong lesion and a weak one -- and with only
    # the component scores in the set the curve cannot express the point where the weak lesion drops
    # out. It would then report the sensitivity of the strongest threshold below it, which is an
    # understatement at every low operating point and an overstatement at every high one.
    #
    # 1.0 beyond the highest closes the curve at zero predictions so the interpolation has a left
    # end.
    thresholds = sorted(
        {c.score for c in candidates}
        | {on_reference for c in candidates for _id, on_reference in c.overlap_scores}
        | {1.0 + 1e-9},
        reverse=True,
    )
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        survivors = [c for c in candidates if c.score >= threshold]
        false_positives = sum(1 for c in survivors if c.matched_reference_id is None)
        hits = sum(
            1 for r in references if any(s >= threshold for s in r.matched_scores)
        )
        curve.append({
            "score_threshold": float(threshold),
            # `None` AND NOT `float("nan")`, AND THIS WAS A REAL DEFECT.
            #
            # A channel every eligible case leaves unannotated has no sensitivity -- there is
            # nothing to be sensitive to. It used to be NaN, and `json.dumps` writes NaN as the
            # bare token `NaN`, which is NOT valid JSON: a strict parser refuses the WHOLE report,
            # not just that field. Found by the clinical-chain smoke test, on the case
            # `MOS-EVID-051` is about -- supervised, empty ground truth -- and reachable on this
            # cohort for a channel as rare as `benign_nodule`.
            #
            # The false-positive rate stays, because it IS defined: what the model predicted on a
            # case with nothing to find is the whole of the five empty-reference metrics.
            "sensitivity": (float(hits) / total_references) if total_references else None,
            "false_positives_per_case": float(false_positives) / total_cases,
            "cases": float(total_cases),
        })
    return curve


def froc_at(
    curve: Sequence[Mapping[str, Any]], false_positives_per_case: float
) -> float | None:
    """Sensitivity at a declared FP/scan point, by linear interpolation along the curve.

    `None` when the curve never reaches that rate -- which happens to a model that produces
    fewer false positives per scan than the point asks for. That is not zero sensitivity and it
    is not the sensitivity at the closest point either: reporting the nearest value would claim
    a measurement at an operating point the model cannot be placed at. `MOS-EVID-055`'s whole
    argument is that a number and its operating point travel together.
    """
    if not curve:
        return None
    if any(row["sensitivity"] is None for row in curve):
        # NOTHING TO INTERPOLATE. A curve whose sensitivity is undefined belongs to a channel no
        # eligible case annotates; interpolating across it would invent a number, and arithmetic on
        # `None` would raise here instead of saying so.
        return None
    ordered = sorted(curve, key=lambda row: row["false_positives_per_case"])
    target = float(false_positives_per_case)
    if target > ordered[-1]["false_positives_per_case"]:
        return None
    previous = None
    for row in ordered:
        rate = row["false_positives_per_case"]
        if rate == target:
            # Several thresholds can share a rate; the most sensitive one is the operating point
            # a deployment would pick at that rate.
            same = [r["sensitivity"] for r in ordered if r["false_positives_per_case"] == target]
            return float(max(same))
        if rate > target:
            if previous is None:
                return float(row["sensitivity"]) if target >= 0 else None
            span = rate - previous["false_positives_per_case"]
            weight = 0.0 if span == 0 else (target - previous["false_positives_per_case"]) / span
            return float(previous["sensitivity"]
                         + weight * (row["sensitivity"] - previous["sensitivity"]))
        previous = row
    return float(ordered[-1]["sensitivity"])


def label_value_of(channel: str, label_of: Mapping[str, Any]) -> int:
    """The label VALUE for a channel, from `supervision.json`'s own mapping.

    Accepts a bare `k` and a singleton region `[k]`, both of which appear in nnU-Net's
    dataset.json, and refuses a multi-value region: counting one per channel needs a decision
    about what overlap between two values means, and this module has not been given one.
    """
    value = label_of[channel]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"{channel} maps to {value!r}: a detection count joins ONE label value per "
                "channel, and a multi-value region cannot be counted per channel without "
                "deciding what overlap between its members means"
            )
        value = value[0]
    return int(value)


def _components(mask: Any, connectivity: int) -> tuple[Any, int]:
    import numpy as np
    from scipy import ndimage

    if connectivity == 6:
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
    elif connectivity == 26:
        structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    else:
        raise ValueError(
            f"connectivity {connectivity!r} is neither 6 nor 26. It decides what counts as ONE "
            "lesion, so it is declared and not inferred"
        )
    labelled, count = ndimage.label(np.asarray(mask), structure=structure)
    return labelled, int(count)


def _voxel_ml(spacing_mm: Sequence[float]) -> float:
    """One voxel in millilitres. 1 mL is 1000 mm^3."""
    volume = 1.0
    for step in spacing_mm:
        volume *= float(step)
    return volume / 1000.0


def candidates_for_case(
    probability: Any,
    segmentation: Any,
    *,
    label_of: Mapping[str, Any],
    channels: Sequence[str],
    supervised: Iterable[str],
    spacing_mm: Sequence[float],
    extraction_threshold: float,
    connectivity: int = CONNECTIVITY,
    min_candidate_volume_ml: float = 0.0,
    case: str = "",
) -> CaseDetections:
    """Extract and match the candidates and reference lesions of ONE case.

    `probability` is `[C, ...spatial]` sigmoid output; `segmentation` is the integer reference
    label map over the same spatial shape. `extraction_threshold` fixes the component
    structure once -- see the module docstring on the one declared approximation.

    `min_candidate_volume_ml` DROPS A COMPONENT THAT IS TOO SMALL TO BE A FINDING, and without
    it these metrics are meaningless rather than merely pessimistic. Measured on six cases of
    this cohort: 722 candidates against 6 reference lesions on `aorta_arch`, giving a precision
    of 0.058 and putting every declared FP/scan point out of reach. A sigmoid output thresholded
    over a 355x512x512 volume speckles -- isolated voxels and three-voxel clusters scattered
    through the lung -- and each speck became a detection claim.

    It is a DECLARED convention and not a tuning knob: the registries already carry
    `connected_component_filter` and `keep: largest_connected_component` as rule-set DATA
    (`MOS-EVID-108` fixes the connectivity for the body contour on the same argument), so the
    value travels in the convention block beside the numbers it produced. A filter chosen per
    run and not recorded would make two reports incomparable while looking identical.

    ONLY THE CHANNELS THE CASE SUPERVISES appear in the result. That is the same rule the loss
    applies, and it is what makes a masked and an unmasked arm comparable: an unannotated
    channel cannot contribute a false positive, because nobody looked.
    """
    import numpy as np

    marked = set(supervised)
    unknown = sorted(marked - set(channels))
    if unknown:
        raise ValueError(
            f"case {case!r} supervises {unknown}, which the channel list does not contain. "
            "supervision.json is inconsistent with itself and no mask can be built from it"
        )
    if not 0.0 <= float(extraction_threshold) < 1.0:
        raise ValueError(
            f"extraction_threshold {extraction_threshold!r} is outside [0, 1). It fixes the "
            "component structure every later operating point is filtered out of, so 1.0 would "
            "extract nothing and there would be no curve at all"
        )

    scores = np.asarray(probability)
    truth_map = np.asarray(segmentation)
    if scores.shape[0] != len(channels):
        raise ValueError(
            f"the network emitted {scores.shape[0]} channels and the cohort declares "
            f"{len(channels)}: a per-channel detection would be attributed to the wrong finding"
        )
    if scores.shape[1:] != truth_map.shape:
        raise ValueError(
            f"prediction {scores.shape[1:]} and reference {truth_map.shape} differ in shape, so "
            "an overlap between them would be an accident of broadcasting"
        )
    if len(spacing_mm) != truth_map.ndim:
        raise ValueError(
            f"spacing {tuple(spacing_mm)} has {len(spacing_mm)} axes and the volume has "
            f"{truth_map.ndim}: a volume in millilitres computed from it would be wrong by "
            "whatever the missing axis is"
        )

    per_voxel_ml = _voxel_ml(spacing_mm)
    spacing = np.asarray([float(s) for s in spacing_mm], dtype=float)

    candidates: dict[str, tuple[Candidate, ...]] = {}
    references: dict[str, tuple[Reference, ...]] = {}

    for column, channel in enumerate(channels):
        if channel not in marked:
            continue
        value = label_value_of(channel, label_of)
        truth = truth_map == value
        predicted = scores[column] > float(extraction_threshold)

        truth_labels, truth_count = _components(truth, connectivity)
        predicted_labels, predicted_count = _components(predicted, connectivity)

        reference_centroids: dict[int, Any] = {}
        reference_rows: list[Reference] = []
        for index in range(1, truth_count + 1):
            where = truth_labels == index
            centroid = np.asarray([float(c) for c in np.argwhere(where).mean(axis=0)])
            reference_centroids[index] = centroid
            reference_rows.append(Reference(
                id="%s_%s_%d" % (case or "case", channel, index),
                volume_ml=float(np.count_nonzero(where)) * per_voxel_ml,
                centroid_voxel=tuple(centroid),
            ))

        matched_scores: dict[int, list[float]] = {i: [] for i in range(1, truth_count + 1)}
        candidate_rows: list[Candidate] = []
        for index in range(1, predicted_count + 1):
            where = predicted_labels == index
            volume_ml = float(np.count_nonzero(where)) * per_voxel_ml
            if volume_ml < float(min_candidate_volume_ml):
                # Too small to be a finding. Dropped BEFORE matching, so it can neither claim a
                # lesion nor count as a false positive: a speck is not a detection either way.
                continue
            component_scores = scores[column][where]
            score = float(component_scores.max()) if component_scores.size else 0.0
            centroid = np.asarray([float(c) for c in np.argwhere(where).mean(axis=0)])

            # THE MATCH: the reference component sharing the most voxels with this candidate.
            # `np.bincount` over the reference labels inside the candidate gives every overlap
            # at once; index 0 is background and is dropped.
            overlaps = np.bincount(truth_labels[where].ravel(),
                                   minlength=truth_count + 1)
            overlaps[0] = 0
            best = int(overlaps.argmax())
            if overlaps[best] > 0:
                # EVERY REFERENCE THIS CANDIDATE TOUCHES IS FOUND, not only the one it belongs
                # to. These are two different questions and the first version answered the
                # second for both:
                #
                #   "which reference does this candidate belong to" -- the largest overlap,
                #     because a candidate is one detection claim and it has to be attributed
                #     to one thing for the false-positive count to mean anything;
                #   "was this reference found" -- ANY overlap, because a lesion a prediction
                #     covers has been found whatever else that prediction also covers.
                #
                # Recording only the best match understates sensitivity exactly when one blob
                # covers a CLUSTER -- coronary calcifications, nodules -- which is where the
                # undercount would have been both largest and least visible.
                touched = [int(i) for i in np.nonzero(overlaps)[0]]
                # THE CONFIDENCE ON EACH REFERENCE'S OWN VOXELS, not the component's maximum.
                # `where & (truth_labels == r)` is the shared region, which is non-empty for every
                # member of `touched` by construction, so the max is always defined.
                #
                # The inner loop variable is NOT `index`: the outer loop's `index` is the COMPONENT
                # label, and shadowing it here made two different things share one name in eleven
                # lines of code that turns on exactly that distinction.
                overlap_scores: list[tuple[str, float]] = []
                for reference_index in touched:
                    shared = where & (truth_labels == reference_index)
                    on_reference = float(scores[column][shared].max())
                    matched_scores[reference_index].append(on_reference)
                    overlap_scores.append(
                        (reference_rows[reference_index - 1].id, on_reference))
                separation = float(
                    np.sqrt((((centroid - reference_centroids[best]) * spacing) ** 2).sum())
                )
                candidate_rows.append(Candidate(
                    score=score,
                    volume_ml=volume_ml,
                    centroid_voxel=tuple(centroid),
                    matched_reference_id=reference_rows[best - 1].id,
                    match_distance_mm=separation,
                    overlap_voxels=int(overlaps[best]),
                    overlap_scores=tuple(overlap_scores),
                ))
            else:
                candidate_rows.append(Candidate(
                    score=score,
                    volume_ml=volume_ml,
                    centroid_voxel=tuple(centroid),
                    matched_reference_id=None,
                    match_distance_mm=None,
                    overlap_voxels=0,
                ))

        candidates[channel] = tuple(candidate_rows)
        references[channel] = tuple(
            Reference(
                id=row.id, volume_ml=row.volume_ml, centroid_voxel=row.centroid_voxel,
                matched_scores=tuple(sorted(matched_scores[index], reverse=True)),
            )
            for index, row in enumerate(reference_rows, start=1)
        )

    return CaseDetections(case=case, candidates=candidates, references=references)
