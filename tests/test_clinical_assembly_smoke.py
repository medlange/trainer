# SPDX-License-Identifier: Apache-2.0
"""A CPU smoke test of the whole clinical measurement chain, on shapes it will really see.

WHY THIS EXISTS, AND IT IS NOT A UNIT TEST. Every piece of the clinical pass is gated
individually: `detection.py` has 41 tests, `overlap.py` 14, `evidence.py` 30, and
`ab_evaluate.shape_blocks` 13. What NOTHING covers is the SEQUENCE -- extract candidates, measure
shapes, count at an operating point, build a curve, aggregate both halves, and serialise the result
-- because running it for real needs a card, a trained checkpoint and 170 GB of preprocessed data,
and the pass takes 45 to 85 minutes before it says whether it worked.

That is the expensive failure this file is for. Three of its four defect classes cannot be caught by
a unit test of any piece:

  * a DTYPE that only appears in the chain. `overlap.surface_distances` refuses a non-boolean mask
    because `distances[integer_array]` is fancy indexing rather than masking, and the array reaching
    it comes from a comparison two functions upstream;
  * a NON-FINITE number reaching `json.dumps`, which writes NaN and Infinity as BARE TOKENS. That
    is not valid JSON and a strict parser refuses the WHOLE report rather than the one field. This
    is the defect this file found on its first run: `froc_curve` used `float("nan")` for a channel
    every eligible case leaves unannotated, which is `MOS-EVID-051`'s excluded case and is reachable
    on this cohort for a channel as rare as `benign_nodule`. Now `None`, which serialises as `null`;
  * a NUMPY SCALAR reaching `json.dumps`, which raises `TypeError: Object of type float32 is not
    JSON serializable`. Stated honestly: every producer converts with `float(...)` -- `_shape_value`
    before the aggregate, `froc_curve` and `counts_at` at the point of construction -- so a numpy
    scalar cannot currently reach the report, and proving that by breaking showed the assertion
    staying green. It is a REGRESSION guard, not a bug-finder, and the distinction is written here
    rather than left for a reader to assume;
  * a SHAPE disagreement between the two endpoints, which take the same probability array through
    different code.

THE SHAPE IS REDUCED AND THE REASON IS STATED. The real patch is 128x224x224 over ten channels,
which is 640 MB of float32 per case and minutes per distance transform. What this test needs to
exercise is the CHAIN, and the chain does not care about the extent -- so 40x64x64 over ten
channels, with a realistic anisotropic spacing and a realistic supervision mask, and the volume
left to the lab run it is protecting.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer import detection, evidence, overlap  # noqa: E402

_TOOL = TRAINER / "tools" / "ab_evaluate.py"


def _evaluator():
    """The mounted tool, imported by path. See `test_ab_evaluate.py` for why it is not importable."""
    spec = importlib.util.spec_from_file_location("ab_evaluate_smoke", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The cohort's own ten channels and the anisotropic spacing its plan resamples to.
CHANNELS = ["aorta_arch", "aorta_ascending", "aorta_descending", "benign_nodule",
            "coronary_calcification", "lung_neoplasm", "pleural_effusion", "pneumonia",
            "pulmonary_trunk", "vertebral_body"]
LABEL_OF = {name: index + 1 for index, name in enumerate(CHANNELS)}
SPACING = (1.0, 0.782, 0.782)
SHAPE = (40, 64, 64)


def _synthetic_case(seed: int, supervised: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One case: probabilities over ten channels and an integer label map.

    THE LESIONS ARE DELIBERATELY AWKWARD, because the easy shapes are the ones already gated:
    a multi-component channel (a cluster, like coronary calcification), a channel predicted but
    not annotated, a channel annotated but not predicted, and a NON-UNIFORM component spanning two
    reference pieces -- which is the shape that hid a real defect in `detection.py`.
    """
    generator = np.random.default_rng(seed)
    probability = generator.uniform(0.0, 0.08, size=(len(CHANNELS), *SHAPE)).astype(np.float32)
    truth = np.zeros(SHAPE, dtype=np.int16)

    for name in supervised:
        value = LABEL_OF[name]
        column = CHANNELS.index(name)
        if name == "coronary_calcification":
            # A CLUSTER: many small references, one predicted sheet across them, non-uniform.
            for offset in range(6):
                z, y, x = 10 + offset, 20 + 3 * offset, 20 + 3 * offset
                truth[z, y:y + 2, x:x + 2] = value
            probability[column, 8:20, 18:42, 18:42] = 0.2          # the weak sheet
            probability[column, 10, 20:22, 20:22] = 0.93           # strong on the first piece only
        elif name == "benign_nodule":
            truth[25:28, 30:33, 30:33] = value                     # annotated, never predicted
        elif name == "pleural_effusion":
            truth[5:30, 5:25, 5:25] = value                        # large, well predicted
            probability[column, 5:30, 5:25, 5:25] = 0.88
        else:
            truth[15:22, 40:50, 40:50] = value
            probability[column, 15:22, 40:50, 40:50] = 0.7

    # A channel PREDICTED but not annotated by this case: it must not reach either endpoint,
    # because the supervision mask is what decides whether a case is evidence about a channel.
    unannotated = CHANNELS.index("lung_neoplasm")
    probability[unannotated, 30:35, 10:20, 10:20] = 0.95
    return probability, truth


def test_the_whole_clinical_chain_runs_and_serialises() -> None:
    """Extract, measure, count, curve, aggregate, serialise -- in the evaluator's own order."""
    evaluator = _evaluator()
    supervision = {
        "c1": ["aorta_arch", "coronary_calcification", "pleural_effusion", "benign_nodule"],
        "c2": ["aorta_arch", "pneumonia", "pleural_effusion"],
        "c3": ["coronary_calcification", "vertebral_body"],
    }
    patient_of = {"c1": "p1", "c2": "p1", "c3": "p2"}

    detected: list[detection.CaseDetections] = []
    shapes: list[overlap.CaseOverlap] = []
    for index, (case, marked) in enumerate(supervision.items()):
        probability, truth = _synthetic_case(2026 + index, marked)
        detected.append(detection.candidates_for_case(
            probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=marked,
            spacing_mm=SPACING, extraction_threshold=0.1, min_candidate_volume_ml=0.02,
            case=case,
        ))
        shapes.extend(overlap.overlap_for_case(
            probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=marked,
            spacing_mm=SPACING, threshold=0.5, case=case, with_distances=True,
        ))

    conventions = evidence.conventions(
        metric_registry_version=1, dice_aggregation="dice_pooled",
        fp_volume_threshold_ml=0.1, bootstrap_seed=20260926, min_candidate_volume_ml=0.02,
    )
    point = {"name": "probability", "value": 0.5, "selected_on": "tune"}

    per_channel: dict = {}
    for channel in CHANNELS:
        curve = detection.froc_curve(detected, channel)
        for policy in detection.COMPONENT_POLICIES:
            rows = detection.counts_at(detected, channel, threshold=0.5,
                                       patient_of=patient_of, policy=policy)
            if not rows["sensitivity"]:
                break
            per_channel.setdefault(channel, {})[policy] = {
                "sensitivity": evidence.aggregate_counted(
                    "sensitivity", rows["sensitivity"], conventions_block=conventions,
                    operating_point=point).as_document(),
                "ppv": evidence.aggregate_counted(
                    "ppv", rows["ppv"], conventions_block=conventions,
                    operating_point=point).as_document(),
            }
        if channel in per_channel:
            per_channel[channel]["froc"] = {
                "%g" % fp: detection.froc_at(curve, fp) for fp in detection.FROC_POINTS
            }

    blocks, overlap_conventions = evaluator.shape_blocks(
        shapes, channels=CHANNELS, patient_of=patient_of, operating_point=point,
        min_candidate_volume_ml=0.02, with_distances=True,
    )
    for channel, block in blocks.items():
        per_channel.setdefault(channel, {})["overlap"] = block

    # THE SERIALISATION IS THE GATE. A numpy scalar anywhere in that tree raises here, and on the
    # lab it would raise after the inference pass had been paid for.
    report = {"per_channel": per_channel, "conventions": conventions,
              "overlap": {"conventions": overlap_conventions}, "operating_point": point}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    assert len(text) > 2000
    assert "NaN" not in text and "Infinity" not in text, (
        "a non-finite number reached the report; `json.dumps` writes those as bare NaN/Infinity, "
        "which is not valid JSON and every strict parser downstream refuses it"
    )

    measured = {channel for channel in CHANNELS if channel in per_channel}
    assert "lung_neoplasm" not in measured, (
        "a channel no case annotates was measured, so the supervision mask did not reach one of "
        "the two endpoints"
    )
    assert measured == {"aorta_arch", "coronary_calcification", "pleural_effusion",
                        "benign_nodule", "pneumonia", "vertebral_body"}


def test_the_two_endpoints_agree_about_which_cases_they_measured() -> None:
    """ONE REPORT MUST NOT DESCRIBE TWO CASE SETS.

    Detection and shape take the same probability array through different code and apply the
    supervision mask separately. If they disagreed, a channel's sensitivity would be over one set of
    cases and its Dice over another, and the two numbers would be printed side by side.
    """
    supervision = {"c1": ["aorta_arch", "pleural_effusion"], "c2": ["aorta_arch"]}
    patient_of = {"c1": "p1", "c2": "p2"}

    detected, shapes = [], []
    for index, (case, marked) in enumerate(supervision.items()):
        probability, truth = _synthetic_case(7 + index, marked)
        detected.append(detection.candidates_for_case(
            probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=marked,
            spacing_mm=SPACING, extraction_threshold=0.1, case=case,
        ))
        shapes.extend(overlap.overlap_for_case(
            probability, truth, label_of=LABEL_OF, channels=CHANNELS, supervised=marked,
            spacing_mm=SPACING, threshold=0.5, case=case, with_distances=False,
        ))

    for channel in ("aorta_arch", "pleural_effusion"):
        counted = {found.case for found in detected if channel in found.references}
        measured = {row.case for row in shapes if row.channel == channel}
        assert counted == measured, (
            f"{channel}: detection measured {sorted(counted)} and shape measured "
            f"{sorted(measured)}"
        )
    assert not [row for row in shapes if row.channel == "lung_neoplasm"]


def test_a_reference_the_model_never_predicted_is_not_reported_found() -> None:
    """THE FIXED DEFECT, END TO END ON A REALISTIC CLUSTER.

    `coronary_calcification` here is six small references under one weak predicted sheet, with only
    the first piece predicted strongly. At the operating point the other five have no predicted
    voxel at all, so sensitivity must be 1 of 6. Before the fix the sheet's single 0.93 was recorded
    against all six and the answer was 6 of 6 -- which is the shape of the number this cohort's
    coronary channel actually reports.
    """
    probability, truth = _synthetic_case(11, ["coronary_calcification"])
    found = detection.candidates_for_case(
        probability, truth, label_of=LABEL_OF, channels=CHANNELS,
        supervised=["coronary_calcification"], spacing_mm=SPACING,
        extraction_threshold=0.1, case="c1",
    )
    references = found.references["coronary_calcification"]
    assert len(references) == 6, f"the fixture no longer builds six references: {len(references)}"

    rows = detection.counts_at([found], "coronary_calcification", threshold=0.5,
                               patient_of={"c1": "p1"}, policy="all_components")
    numerator = rows["sensitivity"][0].numerator
    assert numerator == 1.0, (
        f"{numerator} of 6 references reported found at 0.5, where only one has a predicted voxel "
        "above it. The component's maximum is being credited to every piece it touches"
    )
    at_low = detection.counts_at([found], "coronary_calcification", threshold=0.15,
                                 patient_of={"c1": "p1"}, policy="all_components")
    assert at_low["sensitivity"][0].numerator == 6.0, (
        "at 0.15 the sheet does cover all six, so refusing them there would be the opposite error"
    )


def test_a_channel_no_eligible_case_annotates_serialises_as_null_and_not_as_NaN() -> None:
    """THE DEFECT THIS FILE FOUND, kept as its own gate.

    `MOS-EVID-051`'s excluded case is a channel a reader supervised and annotated nothing for. The
    model can still predict there, so the channel has a false-positive rate and NO sensitivity. That
    used to be `float("nan")`, and `json.dumps` writes NaN as a bare token: not valid JSON, and a
    strict parser refuses the whole report rather than the one field -- so the entire clinical
    measurement of a 45-minute pass would be unreadable because one rare channel had no reference.

    `None` is the honest value and serialises as `null`. The false-positive rate stays, because it
    is defined and is what the five empty-reference metrics are made of.
    """
    probability = np.zeros((len(CHANNELS), *SHAPE), dtype=np.float32)
    truth = np.zeros(SHAPE, dtype=np.int16)
    column = CHANNELS.index("benign_nodule")
    probability[column, 2:5, 10:16, 10:16] = 0.8        # predicted; nothing annotated anywhere

    found = detection.candidates_for_case(
        probability, truth, label_of=LABEL_OF, channels=CHANNELS,
        supervised=["benign_nodule"], spacing_mm=SPACING, extraction_threshold=0.1, case="c1",
    )
    assert not found.references["benign_nodule"]
    assert found.candidates["benign_nodule"], "the fixture predicts nothing, so it proves nothing"

    curve = detection.froc_curve([found], "benign_nodule")
    assert curve, "the curve is empty, so the false-positive rate was lost with the sensitivity"
    assert all(row["sensitivity"] is None for row in curve)
    assert any(row["false_positives_per_case"] > 0 for row in curve), (
        "the false-positive rate is defined for an unannotated channel and must survive"
    )
    block = {"%g" % fp: detection.froc_at(curve, fp) for fp in detection.FROC_POINTS}
    assert set(block.values()) == {None}

    text = json.dumps({"froc": block, "curve": curve})
    assert "NaN" not in text and "Infinity" not in text
    # A STRICT PARSE, because `json.loads` accepts bare NaN by default and would hide this.
    def refuse(constant):
        raise AssertionError(f"the report carries the bare token {constant}, which is not JSON")

    json.loads(text, parse_constant=refuse)
