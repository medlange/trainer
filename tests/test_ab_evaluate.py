# SPDX-License-Identifier: Apache-2.0
"""The A/B evaluator, checked where it can be wrong without a GPU.

WHY THIS FILE EXISTS. The evaluator is the instrument a claim about the masked trainer
rests on: that unmasked training teaches suppression. An instrument that produces the
number nobody checked is worse than no instrument -- the number gets quoted. Three of its
defects were already found by running it once against a real run directory before any
report existed (the nnU-Net roots set after the import, two arms in one process, the epoch
counter reading its own log), and those were the ones a dry run could reach.

WHAT IS CHECKED HERE is what a dry run cannot: the arithmetic against numbers worked out
by hand, and the refusals -- every condition under which two measurements must NOT be
subtracted. A refusal that does not fire is the same defect as a wrong number, arriving
later and with more confidence behind it.

WHAT IS NOT CHECKED HERE is the part that needs a card and a trained network:
`measure()` loading a checkpoint and running the validation loop. That is exercised by
running it, which is how the three defects above were found.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    # THE PACKAGE, NOT THE TOOL. `_load` below explains why `trainer/tools/` must stay off
    # every import path; `trainer/` is a different matter and two other suites here already
    # add it. The evaluator imports `medos_trainer.backend` and `medos_trainer.masked` --
    # inside the image those are installed, and a test that could not resolve them would be
    # testing a tool nobody could run.
    sys.path.insert(0, str(TRAINER))

_TOOL = TRAINER / "tools" / "ab_evaluate.py"


def _load():
    """Import the tool by path.

    `trainer/tools/` is not on any import path and must not be: the trainer image does not
    ship it (`Dockerfile.dockerignore` admits `trainer/` but the tool is run by mounting
    it), and making it importable would invite the platform to depend on it.
    """
    spec = importlib.util.spec_from_file_location("ab_evaluate_under_test", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ab = _load()
CHANNELS = ["neo", "effusion", "pneumonia"]


# =====================================================================================
# The arithmetic, against numbers worked out by hand
# =====================================================================================
def test_recall_dice_and_precision_are_what_the_definitions_say() -> None:
    """One channel with every count non-zero, checked against the formulas.

    tp=6 fp=2 fn=4:  recall 6/10 = .6   precision 6/8 = .75   dice 12/(12+2+4) = .6667
    """
    got = ab.metrics_from_counts([6.0], [2.0], [4.0], ["neo"])
    assert got["recall_per_channel"][0] == pytest.approx(0.6)
    assert got["precision_per_channel"][0] == pytest.approx(0.75)
    assert got["dice_per_channel"][0] == pytest.approx(12 / 18)


def test_the_micro_average_is_over_summed_counts_not_a_mean_of_ratios() -> None:
    """THE DIFFERENCE MATTERS AND IS THE POINT OF A MICRO AVERAGE. A mean of per-channel
    Dice weights a channel with four supervised voxels the same as one with four million;
    the summed form weights them by what was measured. Two epochs' values are then
    comparable because both were computed the same way -- the argument register entry 102
    makes for the checkpoint metric, applied to the report.

    tp 9+1, fp 1+9, fn 0+0: micro recall 10/10 = 1.0, while the mean of the two channels'
    recalls is also 1.0 -- so recall cannot distinguish the two. Precision does: micro
    10/20 = .5, mean of (.9, .1) = .5 too. Dice separates them: micro 20/(20+10) = .6667
    against a mean of (.947, .182) = .565.
    """
    got = ab.metrics_from_counts([9.0, 1.0], [1.0, 9.0], [0.0, 0.0], ["a", "b"])
    assert got["dice_micro"] == pytest.approx(20 / 30)
    mean_of_ratios = sum(got["dice_per_channel"]) / 2
    assert got["dice_micro"] != pytest.approx(mean_of_ratios), (
        "the micro average equals the mean of ratios, so it is not computed from sums"
    )


def test_a_channel_with_nothing_to_measure_is_none_and_not_zero() -> None:
    """`tp + fn == 0` means no supervised pair in this measurement held the finding.
    Reporting 0.0 makes an absent measurement read as a failed one -- the same error as
    averaging a `nan` into a mean, which is what the checkpoint metric was changed to
    avoid."""
    got = ab.metrics_from_counts([0.0], [0.0], [0.0], ["neo"])
    assert got["recall_per_channel"] == [None]
    assert got["precision_per_channel"] == [None]
    assert got["dice_per_channel"] == [None]
    assert got["recall_micro"] is None


def test_found_nothing_is_distinguished_from_nothing_to_find() -> None:
    """A channel present in the reference and never predicted: recall is 0.0, a real
    measurement. Precision has no denominator, so it is None. Collapsing the two would
    report a perfect-precision network that finds nothing."""
    got = ab.metrics_from_counts([0.0], [0.0], [5.0], ["neo"])
    assert got["recall_per_channel"] == [0.0], "a missed finding is a measured zero"
    assert got["precision_per_channel"] == [None], "precision has no denominator here"


def test_counts_and_channels_must_agree_in_length() -> None:
    """A silent zip() would drop a channel off the end and report nine numbers for ten
    heads, with the names shifted under them."""
    with pytest.raises(ValueError, match="disagree in length"):
        ab.metrics_from_counts([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], ["only-one"])


# =====================================================================================
# The refusals: every condition under which two measurements must not be subtracted
# =====================================================================================
def _report(run: str, **over) -> dict:
    base = {
        "run": run, "inputs_digest": "d" * 64, "seed": 1, "iterations": 1000,
        "checkpoint": "checkpoint_final.pth",
        # Every real report carries these, so every fixture does. A helper that omitted them
        # would let a gate pass against a report shape the evaluator never writes.
        "operating_point": {"id": "balanced", "score_threshold": 0.5},
        "score_threshold": 0.5,
    }
    base.update(over)
    return base


def test_two_measurements_made_alike_are_comparable() -> None:
    ab.comparable([_report("masked-585"), _report("control-585")])


def test_diverged_preprocessed_data_is_refused() -> None:
    """Patch-based measurement is valid only if both arms see the same patch sequence,
    which holds only if their preprocessed bytes are identical. Diverged bytes mean the
    difference measured two different samples, and it would look meaningful."""
    with pytest.raises(SystemExit, match="inputs_digest differs"):
        ab.comparable([_report("masked-585"), _report("control-585", inputs_digest="e" * 64)])


@pytest.mark.parametrize("field,value", [("seed", 2), ("iterations", 50)])
def test_a_different_sampler_or_sample_size_is_refused(field, value) -> None:
    with pytest.raises(SystemExit, match=f"{field} differs"):
        ab.comparable([_report("masked-585"), _report("control-585", **{field: value})])


def test_comparing_two_best_checkpoints_is_refused() -> None:
    """THE SUBTLE ONE. `checkpoint_best` is selected by each arm's OWN metric -- the masked
    arm's micro-Dice is masked, the control's is not -- so two "best" weights are two
    selections made under different rules. That is the same confusion as comparing the
    arms' own printed numbers, hidden one level deeper: in which weights were loaded."""
    with pytest.raises(SystemExit, match="checkpoint differs"):
        ab.comparable([_report("masked-585"),
                       _report("control-585", checkpoint="checkpoint_best.pth")])


# =====================================================================================
# Which weights, and how many epochs: both stated, neither guessed
# =====================================================================================
def _fold(tmp_path: Path) -> Path:
    fold = tmp_path / "results" / "Dataset501" / "trainer__plans__3d_fullres" / "fold_0"
    fold.mkdir(parents=True)
    return fold


def test_the_caller_names_the_checkpoint_and_an_unknown_name_is_refused(tmp_path) -> None:
    fold = _fold(tmp_path)
    (fold / "checkpoint_best.pth").write_bytes(b"x")
    (fold / "checkpoint_final.pth").write_bytes(b"y")

    assert ab.checkpoint_path(tmp_path, "best").name == "checkpoint_best.pth"
    assert ab.checkpoint_path(tmp_path, "final").name == "checkpoint_final.pth"
    with pytest.raises(SystemExit, match="unknown checkpoint"):
        ab.checkpoint_path(tmp_path, "latest")


def test_asking_for_final_before_training_ends_says_so(tmp_path) -> None:
    """`checkpoint_final` is written last. The refusal must explain that rather than read
    as "this run has no weights"."""
    fold = _fold(tmp_path)
    (fold / "checkpoint_best.pth").write_bytes(b"x")
    with pytest.raises(SystemExit, match="writes checkpoint_final last"):
        ab.checkpoint_path(tmp_path, "final")


def _real_log(epochs: int) -> str:
    """The shape nnU-Net actually writes, TRAILING SPACES AND ALL.

    Every line ends in a space, and each epoch is announced by a header and closed by a
    timing line that ALSO reads `: Epoch `. A fixture without those two details cannot fail
    against either defect the real log found: the substring test that counted the timing line
    as an epoch, and an end-anchored match that the trailing space defeats.
    """
    out: list[str] = []
    for i in range(epochs):
        out += [
            " ",
            f"2026-09-25 00:{i // 60:02d}:{i % 60:02d}.123456: Epoch {i} ",
            "2026-09-25 00:00:00.123456: Current learning rate: 0.00277 ",
            "2026-09-25 00:02:07.123456: train_loss 0.1234 ",
            "2026-09-25 00:02:07.123456: Pseudo dice [0.82, 0.70, 0.0] ",
            "2026-09-25 00:02:07.123456: Epoch time: 126.8 s ",
        ]
    return "\n".join(out) + "\n"


def test_the_epoch_count_ignores_the_evaluators_own_log(tmp_path) -> None:
    """Constructing the trainer creates a NEW timestamped log. `logs[-1]` picked that one --
    empty -- and the report said "0 epochs" against thirty."""
    fold = _fold(tmp_path)
    (fold / "training_log_2026_9_25_14_14_03.txt").write_text(_real_log(30), encoding="utf-8")
    # The evaluator's own, created later and holding no epochs.
    (fold / "training_log_2026_9_25_19_58_10.txt").write_text(
        "2026-09-25 19:58:10.1: using pin_memory on device 0 \n", encoding="utf-8")

    assert ab.epochs_done(tmp_path) == 30


def test_a_timing_line_is_not_an_epoch(tmp_path) -> None:
    """`": Epoch " in line` TOOK BOTH LINES AND DOUBLED THE COUNT. It reported 304 for the
    152-epoch run this was found on, and the same substring had already turned 77 into 153 in
    a shell one-liner earlier the same day -- which is why the count is asserted here against
    a log built the way nnU-Net writes one, not against a list of bare headers.

    A doubled epoch count does not look wrong. It looks like a run that went further than it
    did, and it would have been printed in the A/B report beside the metrics as the evidence
    that both arms trained the same length.
    """
    fold = _fold(tmp_path)
    log = fold / "training_log_2026_9_25_14_14_03.txt"
    log.write_text(_real_log(7), encoding="utf-8")
    assert log.read_text(encoding="utf-8").count(": Epoch ") == 14, (
        "the fixture lacks the timing lines that caused the defect, so it cannot catch it"
    )
    assert ab.epochs_done(tmp_path) == 7


def test_the_trailing_space_nnunet_writes_does_not_hide_an_epoch(tmp_path) -> None:
    """The counter's first replacement anchored on `Epoch \\d+$` and matched NOTHING against
    the real log, because every line nnU-Net writes ends in a space. That failure was loud --
    zero against 152 -- but only because zero is obviously wrong; the same brittleness in a
    per-channel count would have been quiet."""
    fold = _fold(tmp_path)
    (fold / "training_log_2026_9_25_14_14_03.txt").write_text(
        "2026-09-25 00:00:01.1: Epoch 0 \n"        # a trailing space, as written
        "2026-09-25 00:00:02.1: Epoch 1\t\n"       # and a tab, should one ever appear
        "2026-09-25 00:00:03.1: Epoch 2\n",        # and none at all
        encoding="utf-8")
    assert ab.epochs_done(tmp_path) == 3


def test_the_digest_notices_a_changed_byte_and_a_changed_size(tmp_path) -> None:
    """The digest is what `comparable` rests on, so it must be sensitive to both what the
    files are and what is in them."""
    folder = tmp_path / ab.DATASET / f"nnUNetPlans_{ab.CONFIGURATION}"
    folder.mkdir(parents=True)
    for name in ("a.b2nd", "b.b2nd", "a.pkl"):
        (folder / name).write_bytes(b"0" * 64)
    first = ab.inputs_digest(tmp_path)

    (folder / "a.b2nd").write_bytes(b"1" * 64)          # same size, different content
    assert ab.inputs_digest(tmp_path) != first, "a changed byte went unnoticed"

    (folder / "a.b2nd").write_bytes(b"0" * 64)
    assert ab.inputs_digest(tmp_path) == first, "the digest is not reproducible"

    (folder / "c.b2nd").write_bytes(b"0" * 64)          # a file appears
    assert ab.inputs_digest(tmp_path) != first, "a new file went unnoticed"


def test_an_empty_preprocessed_folder_is_refused_rather_than_digested(tmp_path) -> None:
    """An empty folder has a perfectly stable digest, and two empty folders agree. That is
    the one case where `comparable` would pass on nothing at all."""
    (tmp_path / ab.DATASET / f"nnUNetPlans_{ab.CONFIGURATION}").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no preprocessed files"):
        ab.inputs_digest(tmp_path)


# =====================================================================================
# The report a person reads
# =====================================================================================
def test_the_table_prints_a_dash_for_an_undefined_cell_not_a_zero(capsys, tmp_path) -> None:
    """`--` and `0.0000` are different statements: nothing to measure against measured
    zero. One column reading as the other is how a channel nobody annotated comes to look
    like a channel the network failed on."""
    # `metrics_from_counts` returns `channels` itself; passing it again was a duplicate
    # keyword, which is the sort of thing a test finds about its own setup.
    left = _report("masked-585", epochs_done=200, trained_unmasked=False,
                   **ab.metrics_from_counts([6.0, 0.0, 1.0], [2.0, 0.0, 0.0],
                                            [4.0, 0.0, 0.0], CHANNELS))
    right = _report("control-585", epochs_done=200, trained_unmasked=True,
                    **ab.metrics_from_counts([1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                             [9.0, 0.0, 0.0], CHANNELS))
    paths = []
    for report in (left, right):
        p = tmp_path / f"{report['run']}.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        paths.append(p)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "RECALL (primary)" in out, "recall is not presented first"
    assert out.index("RECALL (primary)") < out.index("DICE") < out.index("PRECISION"), (
        "the metrics are not in the order the claim needs them"
    )
    effusion = [line for line in out.splitlines() if line.startswith("effusion")]
    assert effusion and all("--" in line and "0.0000" not in line for line in effusion), (
        "a channel with nothing measured is printed as a zero: %r" % effusion
    )


# =====================================================================================
# The operating point (MOS-SVC-021)
#
# "Every reported sensitivity, specificity or PPV anywhere in the platform MUST carry the
# `operating_point.id` and `score_threshold` at which it was measured. A metric without a
# threshold MUST NOT be displayed or returned by the API."
#
# The threshold was the default argument of `masked_tp_fp_fn_tn` and appeared nowhere else.
# Every pseudo-Dice the trainer logged and every recall this evaluator reported was measured
# at 0.5 and said so nowhere -- including in a comparison that was handed to a reader.
# =====================================================================================
def test_two_arms_measured_at_different_operating_points_are_refused() -> None:
    """THE CONDITION THE LIST DID NOT CARRY.

    Recall at 0.5 minus recall at 0.7 is not a difference between two networks. It was the
    only one of the five comparability conditions missing, so this subtraction would have
    been performed and printed without complaint -- and it is the subtraction most likely to
    be attempted by accident, because a threshold is the one knob a reader might sweep.
    """
    with pytest.raises(SystemExit, match="score_threshold differs"):
        ab.comparable([_report("masked-585", score_threshold=0.5),
                       _report("control-585", score_threshold=0.7)])


def test_the_comparison_header_names_the_operating_point(capsys, tmp_path) -> None:
    """A reader who sees ten channels of recall and no threshold has been handed a number
    they cannot act on, which is what `MOS-SVC-021` forbids."""
    paths = []
    for run, flip in (("masked-585", False), ("control-585", True)):
        report = _report(
            run, epochs_done=200, trained_unmasked=flip,
            operating_point={"id": "balanced", "score_threshold": 0.5},
            **ab.metrics_from_counts([6.0, 1.0, 1.0], [2.0, 0.0, 0.0], [4.0, 1.0, 0.0],
                                     CHANNELS),
        )
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "score_threshold 0.5" in out, out[:400]
    assert out.index("score_threshold") < out.index("RECALL"), (
        "the operating point is printed after the numbers it qualifies"
    )


def test_the_recorded_threshold_is_the_one_the_trainer_applies() -> None:
    """THE TWO-NUMBERS-ONE-NAME GATE, over the syntax tree because nothing else can see it.

    `measure()` records `score_threshold` in the report AND sets it on the trainer. If
    `validation_step` went back to letting `masked_tp_fp_fn_tn` default, the report would
    carry 0.7 while the counts were taken at 0.5: a wrong number wearing a correct label,
    with every structural check passing and no way to notice from the output.

    Running the trainer to check this needs plans, a staged cohort and a card. The call site
    does not.
    """
    import ast

    source = (Path(__file__).resolve().parents[1]
              / "medos_trainer" / "masked_trainer.py").read_text(encoding="utf-8")
    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "masked_tp_fp_fn_tn"
    ]
    assert calls, "the trainer no longer computes confusion counts through masked.py"
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        assert "threshold" in keywords, (
            f"masked_tp_fp_fn_tn at line {call.lineno} takes the default threshold. The "
            "evaluator records the operating point it set, so the recorded number and the "
            "one that produced the counts would disagree"
        )
        value = keywords["threshold"]
        assert (isinstance(value, ast.Attribute) and value.attr == "_score_threshold"
                and isinstance(value.value, ast.Name) and value.value.id == "self"), (
            f"the threshold at line {call.lineno} is not `self._score_threshold`, so setting "
            "that attribute would not change what is measured"
        )


def test_a_report_written_before_the_threshold_was_recorded_is_refused() -> None:
    """The two reports this evaluator produced on the night of 2026-09-25 carry no operating
    point, because it did not record one. Their numbers were handed to a reader.

    Refusing rather than printing "unrecorded" is the point: a threshold-less recall is not a
    weaker measurement, it is an uninterpretable one, and `MOS-SVC-021` says so in as many
    words. Re-measuring costs two minutes per arm.
    """
    stale = _report("masked-585")
    del stale["operating_point"]
    with pytest.raises(SystemExit, match="carry no operating_point"):
        ab.comparable([stale, _report("control-585")])


# =====================================================================================
# The evaluator is MOUNTED and the trainer is COMPILED IN, so the two can disagree
# =====================================================================================
class _Trainer:
    """Only what `_apply_operating_point` touches."""

    def __init__(self, declares: bool) -> None:
        if declares:
            self._score_threshold = 0.5


def test_an_image_that_honours_the_threshold_gets_the_one_it_was_asked_for() -> None:
    trainer = _Trainer(declares=True)
    applied, how = ab._apply_operating_point(trainer, 0.7)
    assert (applied, how) == (0.7, "set")
    assert trainer._score_threshold == 0.7


def test_an_older_image_records_the_threshold_it_compiled_in_rather_than_the_request() -> None:
    """READ OUT OF THE INSTALLED SIGNATURE, NOT ASSUMED TO BE 0.5. The number this evaluator
    reports has to be the number the code used, and on an image built before the attribute
    existed the only place that number lives is the default argument."""
    import inspect

    from medos_trainer.masked import masked_tp_fp_fn_tn

    compiled = float(inspect.signature(masked_tp_fp_fn_tn).parameters["threshold"].default)
    applied, how = ab._apply_operating_point(_Trainer(declares=False), compiled)
    assert (applied, how) == (compiled, "image default")


def test_asking_an_older_image_for_a_point_it_cannot_reach_is_refused() -> None:
    """Setting an attribute nobody reads and reporting the request is the exact defect
    `MOS-SVC-021` is about: the label would be right and the number would be a fabrication."""
    with pytest.raises(SystemExit, match="cannot be moved off it"):
        ab._apply_operating_point(_Trainer(declares=False), 0.7)


def test_the_table_states_its_own_resolution(capsys, tmp_path) -> None:
    """A per-channel delta of 0.03 rendered to four decimals reads as a finding, and at 1000
    sampled patches it is noise: two of them changed SIGN between the first and second
    measurement of the same two checkpoints.

    The number is printed with the table rather than kept in a docstring, because the reader
    who needs it is the one looking at the table.
    """
    paths = []
    for run in ("masked-585", "control-585"):
        report = _report(run, epochs_done=200, trained_unmasked=(run == "control-585"),
                         **ab.metrics_from_counts([6.0, 1.0, 1.0], [2.0, 0.0, 0.0],
                                                  [4.0, 1.0, 0.0], CHANNELS))
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "resolution:" in out, out[:600]
    assert "%.2f" % ab.SAMPLING_NOISE["per_channel"] in out
    assert out.index("resolution:") < out.index("RECALL"), (
        "the resolution is printed after the numbers it qualifies"
    )


def test_the_seed_is_not_claimed_to_fix_the_sample() -> None:
    """`comparable` requires equal seeds, which reads as a reproducibility claim. It is not
    one: the validation augmenter is non-deterministic by construction and by name. The
    docstring has to say which of the two it means, or the next reader will subtract two
    numbers believing them repeatable."""
    text = ab.comparable.__doc__ or ""
    assert "NonDetMultiThreadedAugmenter" in text
    assert "NOT the" in text and "PATCHES" in text


# =====================================================================================
# Whole-case counting, against arrays small enough to count by hand
#
# WHY IT EXISTS AT ALL. The patch-based measurement draws 1000 random patches through a
# non-deterministic augmenter, and two runs of ONE checkpoint differ by up to 0.08 per
# channel -- enough that two per-channel deltas in the first comparison this tool produced
# changed sign on the second. Whole-case inference is deterministic and complete, so the
# number stops depending on which patches happened to arrive.
#
# WHAT IS CHECKED HERE is the join: an INTEGER reference label map against one sigmoid
# channel per finding, restricted to the channels each case annotates. That join is where a
# silent defect lives, because channels shifted by one produce plausible, uniformly poor
# numbers rather than an error.
# =====================================================================================
import numpy as np  # noqa: E402

LABEL_OF = {"neo": [1], "effusion": [2], "pneumonia": 3}


def _case(predicted_channel_0, predicted_channel_1, predicted_channel_2, truth_values):
    """Three channels over four voxels, with the reference as an integer label map."""
    probability = np.array(
        [predicted_channel_0, predicted_channel_1, predicted_channel_2], dtype=float
    )
    return probability, np.array(truth_values, dtype=int)


def test_the_channel_is_joined_to_its_label_value_and_not_to_its_position() -> None:
    """THE DEFECT THIS GUARDS. `label_of` says `pneumonia` is label 3. If the counter used the
    channel's INDEX instead -- 2 -- it would compare the pneumonia head against effusion's
    voxels, and every number would be wrong in a way that reads as a weak model.

    The label map here is deliberately built so index and value disagree for every channel.
    """
    probability, truth = _case(
        [0.9, 0.1, 0.1, 0.1],   # channel 0 fires on voxel 0, which holds label 1 (neo)
        [0.1, 0.9, 0.1, 0.1],   # channel 1 fires on voxel 1, which holds label 2 (effusion)
        [0.1, 0.1, 0.9, 0.1],   # channel 2 fires on voxel 2, which holds label 3 (pneumonia)
        [1, 2, 3, 0],
    )
    tp, fp, fn = ab.masked_counts_for_case(
        probability, truth, label_of=LABEL_OF,
        channels=["neo", "effusion", "pneumonia"],
        supervised=["neo", "effusion", "pneumonia"], threshold=0.5,
    )
    assert tp == [1.0, 1.0, 1.0], f"every channel should hit its own label: {tp}"
    assert fp == [0.0, 0.0, 0.0]
    assert fn == [0.0, 0.0, 0.0]


def test_a_channel_this_case_does_not_annotate_contributes_nothing() -> None:
    """The same rule the loss applies, and the reason the two arms are comparable at all. The
    prediction on an unsupervised channel is confidently wrong here, and must still count for
    nothing -- not as a false positive."""
    probability, truth = _case(
        [0.9, 0.9, 0.9, 0.9],   # neo: predicted everywhere, and NOT supervised
        [0.9, 0.1, 0.1, 0.1],   # effusion: one hit
        [0.1, 0.1, 0.1, 0.1],
        [2, 0, 0, 0],
    )
    tp, fp, fn = ab.masked_counts_for_case(
        probability, truth, label_of=LABEL_OF,
        channels=["neo", "effusion", "pneumonia"],
        supervised=["effusion"], threshold=0.5,
    )
    assert (tp[0], fp[0], fn[0]) == (0.0, 0.0, 0.0), (
        "an unannotated channel contributed to the counts, which is the false-negative "
        "signal this whole subsystem exists to remove -- arriving through the evaluator"
    )
    assert (tp[1], fp[1], fn[1]) == (1.0, 0.0, 0.0)


def test_the_threshold_decides_and_is_not_assumed() -> None:
    probability, truth = _case([0.6, 0.4, 0.0, 0.0], [0.0] * 4, [0.0] * 4, [1, 1, 0, 0])
    low = ab.masked_counts_for_case(probability, truth, label_of=LABEL_OF,
                                   channels=["neo", "effusion", "pneumonia"],
                                   supervised=["neo"], threshold=0.5)
    high = ab.masked_counts_for_case(probability, truth, label_of=LABEL_OF,
                                    channels=["neo", "effusion", "pneumonia"],
                                    supervised=["neo"], threshold=0.7)
    assert (low[0][0], low[2][0]) == (1.0, 1.0), "at 0.5 one voxel is found and one missed"
    assert (high[0][0], high[2][0]) == (0.0, 2.0), "at 0.7 neither is found"


def test_a_bare_label_value_and_a_singleton_region_mean_the_same_thing() -> None:
    """Both spellings appear in nnU-Net's own dataset.json, so both are accepted rather than
    one being assumed and the other silently missing every voxel."""
    probability, truth = _case([0.9, 0.0, 0.0, 0.0], [0.0] * 4, [0.9, 0.0, 0.0, 0.0],
                               [1, 0, 0, 0])
    tp, _fp, _fn = ab.masked_counts_for_case(
        probability, truth, label_of={"neo": [1], "effusion": [2], "pneumonia": 3},
        channels=["neo", "effusion", "pneumonia"], supervised=["neo"], threshold=0.5,
    )
    assert tp[0] == 1.0
    tp2, _f, _n = ab.masked_counts_for_case(
        probability, truth, label_of={"neo": 1, "effusion": [2], "pneumonia": 3},
        channels=["neo", "effusion", "pneumonia"], supervised=["neo"], threshold=0.5,
    )
    assert tp2 == tp, "the two spellings of one label value gave different counts"


def test_a_multi_value_region_is_refused_rather_than_guessed() -> None:
    """Counting a two-value region per channel needs a decision about what overlap means, and
    this counter has not been given one. Refusing beats picking the first value."""
    # ONE channel, because the head-count refusal fires first and would mask this one --
    # which it should, and which this fixture originally tripped over.
    probability = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=float)
    truth = np.array([1, 2, 0, 0], dtype=int)
    with pytest.raises(SystemExit, match="multi-value region"):
        ab.masked_counts_for_case(probability, truth, label_of={"neo": [1, 2]},
                                  channels=["neo"], supervised=["neo"], threshold=0.5)


def test_a_head_count_that_disagrees_with_the_label_set_is_refused() -> None:
    """Attributing a per-channel number to the wrong finding is worse than having none."""
    probability = np.zeros((2, 4))
    with pytest.raises(SystemExit, match="emitted 2 channels"):
        ab.masked_counts_for_case(probability, np.zeros(4, dtype=int), label_of=LABEL_OF,
                                  channels=["neo", "effusion", "pneumonia"],
                                  supervised=["neo"], threshold=0.5)


def test_a_case_supervising_a_channel_the_cohort_does_not_declare_is_refused() -> None:
    with pytest.raises(SystemExit, match=r"supervises \['ghost'\]"):
        ab.masked_counts_for_case(np.zeros((1, 4)), np.zeros(4, dtype=int),
                                  label_of={"neo": 1}, channels=["neo"],
                                  supervised=["ghost"], threshold=0.5)


def test_the_first_case_initialises_the_totals_rather_than_adding_to_a_guess() -> None:
    """A zeros list whose length was guessed before the first case was read is how a channel
    count comes to be fixed by the wrong thing."""
    assert ab.accumulate(None, [1.0, 2.0]) == [1.0, 2.0]
    assert ab.accumulate([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
    with pytest.raises(SystemExit, match="channel list changed"):
        ab.accumulate([1.0, 2.0], [1.0])


def test_a_whole_case_report_cannot_be_subtracted_from_a_patch_based_one() -> None:
    """One samples 1000 patches through a non-deterministic augmenter and one sees every case.
    They are different quantities, not two measurements of one thing to a tolerance, and the
    whole reason `measure_whole_case` exists is that the difference between them is larger than
    the difference this tool is trying to report."""
    with pytest.raises(SystemExit, match="measurement differs"):
        ab.comparable([_report("masked-585", measurement="whole_case"),
                       _report("control-585", measurement="patches", iterations=1000)])


def test_two_whole_case_reports_are_comparable_without_an_iteration_count() -> None:
    """A whole-case report has no `iterations`: nothing was sampled. The check must not demand
    a field that only the other instrument has."""
    reports = []
    for run in ("masked-585", "control-585"):
        report = _report(run, measurement="whole_case", cases=119)
        # ACTUALLY ABSENT. The helper supplies `iterations` because a patch-based report has
        # one; leaving it in would have made this test pass while asserting nothing about the
        # field it names.
        del report["iterations"]
        reports.append(report)
    assert all("iterations" not in r for r in reports)
    ab.comparable(reports)


def test_a_whole_case_comparison_prints_without_an_iteration_count(capsys, tmp_path) -> None:
    """THE GATE THAT WAS ONE LEVEL TOO HIGH. The test above exercises `comparable()`, which is
    the function this file was thinking about; `compare()` is the function a person calls, and
    it printed `first["iterations"]` unconditionally. The first real whole-case comparison died
    on `KeyError: 'iterations'` after both measurements had already been paid for -- 26 minutes
    of card time each.

    So this one drives the whole path, and asserts the header names the extent the instrument
    actually has.
    """
    paths = []
    for run in ("masked-585", "control-585"):
        report = _report(run, measurement="whole_case", cases=119, epochs_done=200,
                         trained_unmasked=(run == "control-585"),
                         **ab.metrics_from_counts([6.0, 1.0, 1.0], [2.0, 0.0, 0.0],
                                                  [4.0, 1.0, 0.0], CHANNELS))
        del report["iterations"]
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "119 cases" in out, out[:500]
    assert "iterations" not in out, "a whole-case comparison claims an iteration count"
    assert "whole_case" in out


def test_a_deterministic_measurement_is_not_given_a_sampling_error(capsys, tmp_path) -> None:
    """`SAMPLING_NOISE` belongs to the patch-based instrument, where an augmenter chooses which
    patches arrive. A whole-case pass sees every case and every voxel, so printing a sampling
    error for it would invent an uncertainty the measurement does not have -- the mirror image
    of printing a metric with no threshold, and just as misleading to act on."""
    paths = []
    for run in ("masked-585", "control-585"):
        report = _report(run, measurement="whole_case", cases=119, epochs_done=200,
                         trained_unmasked=(run == "control-585"),
                         **ab.metrics_from_counts([6.0], [2.0], [4.0], ["neo"]))
        del report["iterations"]
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "deterministic" in out
    assert "+-" not in out, "a deterministic measurement was given an error bar"


# =====================================================================================
# The threshold sweep: one prediction, many operating points
# =====================================================================================
def test_the_sweep_agrees_with_the_scalar_counter_everywhere() -> None:
    """A DIFFERENTIAL GATE AGAINST AN ALREADY-GATED IMPLEMENTATION, which is the only reason
    the faster path is safe to use at all. The scalar counter has eight tests of its own; this
    asserts the sweep is that arithmetic rather than a second opinion about it.

    The grid includes 0.0 and 1.0 on purpose: those are where a saturated sigmoid actually
    lands, and where a comparison that drifted from `>` to `>=` would first show.
    """
    rng = np.random.default_rng(20260926)
    probability = rng.random((3, 64))
    truth = rng.integers(0, 4, size=64)
    thresholds = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    swept = ab.masked_counts_sweep(
        probability, truth, label_of=LABEL_OF, channels=["neo", "effusion", "pneumonia"],
        supervised=["neo", "pneumonia"], thresholds=thresholds,
    )
    for threshold in thresholds:
        scalar = ab.masked_counts_for_case(
            probability, truth, label_of=LABEL_OF,
            channels=["neo", "effusion", "pneumonia"],
            supervised=["neo", "pneumonia"], threshold=threshold,
        )
        assert swept[threshold] == scalar, (
            f"the sweep and the scalar counter disagree at {threshold}: "
            f"{swept[threshold]} against {scalar}"
        )


def test_the_sweep_keeps_an_unsupervised_channel_at_zero_at_every_threshold() -> None:
    """The mask is a fact about the CASE, not about the operating point. A channel that leaked
    at some thresholds and not others would make the curve a measurement of that leak."""
    rng = np.random.default_rng(1)
    probability = rng.random((3, 32))
    truth = rng.integers(0, 4, size=32)
    swept = ab.masked_counts_sweep(
        probability, truth, label_of=LABEL_OF, channels=["neo", "effusion", "pneumonia"],
        supervised=["neo"], thresholds=[0.0, 0.3, 0.6, 0.99],
    )
    for threshold, (tp, fp, fn) in swept.items():
        assert (tp[1], fp[1], fn[1]) == (0.0, 0.0, 0.0), f"effusion leaked at {threshold}"
        assert (tp[2], fp[2], fn[2]) == (0.0, 0.0, 0.0), f"pneumonia leaked at {threshold}"


def test_the_counts_move_monotonically_with_the_threshold() -> None:
    """Raising the threshold can only shrink what is predicted, so tp and fp must fall and fn
    must rise. That is a property of thresholding and not of the data, so a violation says the
    comparison or the split is wrong -- not that the model is unusual."""
    rng = np.random.default_rng(7)
    probability = rng.random((1, 256))
    truth = rng.integers(0, 2, size=256)
    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    swept = ab.masked_counts_sweep(
        probability, truth, label_of={"neo": 1}, channels=["neo"], supervised=["neo"],
        thresholds=thresholds,
    )
    tps = [swept[t][0][0] for t in thresholds]
    fps = [swept[t][1][0] for t in thresholds]
    fns = [swept[t][2][0] for t in thresholds]
    assert tps == sorted(tps, reverse=True), tps
    assert fps == sorted(fps, reverse=True), fps
    assert fns == sorted(fns), fns
    assert all(tp + fn == tps[0] + fns[0] for tp, fn in zip(tps, fns)), (
        "tp + fn counts the reference voxels and cannot depend on the threshold"
    )


def test_an_empty_threshold_list_is_refused_rather_than_returning_nothing() -> None:
    with pytest.raises(SystemExit, match="no thresholds"):
        ab.masked_counts_sweep(np.zeros((1, 4)), np.zeros(4, dtype=int), label_of={"neo": 1},
                               channels=["neo"], supervised=["neo"], thresholds=[])


def test_two_operating_points_cannot_share_an_id() -> None:
    """THE DEFECT THIS WAS WRITTEN AGAINST. The first spelling was
    `("%g" % threshold).lstrip("0.")`, which maps BOTH 0.1 and 0.01 to `t1`.

    An id is not decoration: `MOS-SVC-020` makes the supported operating points a property of
    the service and the applied one a property of the deployment, and `MOS-SVC-021` carries the
    id beside every reported sensitivity. Two points answering to one name is a metric
    attributed to a threshold nobody can resolve.
    """
    grid = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]
    ids = [ab.operating_point_id(t) for t in grid]
    assert len(set(ids)) == len(ids), dict(zip(grid, ids))
    assert ab.operating_point_id(0.1) != ab.operating_point_id(0.01)
    assert ab.operating_point_id(0.5) == "t0500"


def test_an_operating_point_outside_zero_to_one_is_refused() -> None:
    """A sigmoid score cannot reach it, so every count there would be the same count -- a curve
    with a flat tail that looks like a measurement."""
    for bad in (-0.1, 1.5):
        with pytest.raises(SystemExit, match=r"outside \[0, 1\]"):
            ab.operating_point_id(bad)


def test_the_default_grid_brackets_the_threshold_everything_so_far_was_measured_at() -> None:
    """0.5 is where every number in this session was taken, so the curve has to contain it or
    it could not be compared with anything already recorded."""
    assert 0.5 in ab.CURVE_THRESHOLDS
    assert list(ab.CURVE_THRESHOLDS) == sorted(ab.CURVE_THRESHOLDS)
    assert min(ab.CURVE_THRESHOLDS) < 0.5 < max(ab.CURVE_THRESHOLDS), (
        "a curve that only goes one way from 0.5 cannot show whether the precision problem is "
        "the model or the threshold"
    )


# =====================================================================================
# Macro against micro: the same counts, two questions
# =====================================================================================
def test_the_macro_average_is_not_the_micro_one_and_the_difference_is_the_point() -> None:
    """THE SITUATION THIS COHORT ACTUALLY PRESENTED, in miniature.

    One large channel found well, one tiny channel not found at all. Micro is carried by the
    large channel and reads as a good model; macro halves, because half the findings cannot be
    found. The control arm in the masked/unmasked comparison was exactly this shape: micro
    recall 0.70 while predicting nothing whatsoever for one channel.
    """
    got = ab.metrics_from_counts(
        [1_000_000.0, 0.0],      # tp: the large channel found, the small one not
        [0.0, 0.0],              # fp
        [100_000.0, 20_000.0],   # fn
        ["aorta", "calcification"],
    )
    assert got["recall_micro"] == pytest.approx(1_000_000 / 1_120_000)   # 0.893
    assert got["recall_macro"] == pytest.approx((1_000_000 / 1_100_000 + 0.0) / 2)  # 0.455
    assert got["recall_micro"] > got["recall_macro"] + 0.4, (
        "the fixture no longer shows the divergence, so it cannot demonstrate why both are "
        "reported"
    )
    assert got["channels_found"] == 1
    assert got["channels_measured"] == 2


def test_a_channel_with_nothing_to_measure_is_left_out_of_the_macro_average() -> None:
    """Counting it as zero would make an unannotated channel read as a failed one -- the same
    error `metrics_from_counts` refuses to make per channel, arriving through the average."""
    got = ab.metrics_from_counts([6.0, 0.0], [2.0, 0.0], [4.0, 0.0], ["neo", "absent"])
    assert got["recall_per_channel"] == [pytest.approx(0.6), None]
    assert got["recall_macro"] == pytest.approx(0.6), "the absent channel was averaged in"
    assert got["channels_measured"] == 1
    assert got["channels_found"] == 1


def test_coverage_counts_a_measured_zero_as_not_found() -> None:
    """A channel present in the reference and never predicted has recall 0.0, which is a
    measurement. It is also a capability that does not exist, and the count has to say so."""
    got = ab.metrics_from_counts([0.0], [0.0], [5.0], ["neo"])
    assert got["recall_per_channel"] == [0.0]
    assert got["channels_measured"] == 1
    assert got["channels_found"] == 0


def test_every_macro_is_none_when_nothing_was_measurable() -> None:
    got = ab.metrics_from_counts([0.0], [0.0], [0.0], ["neo"])
    assert got["recall_macro"] is None
    assert got["precision_macro"] is None
    assert got["dice_macro"] is None
    assert got["channels_found"] == 0


def test_the_comparison_prints_the_macro_row_and_the_coverage(capsys, tmp_path) -> None:
    """THE ROW THAT WAS MISSING FOR AN EVENING. Micro alone said the control arm was better
    everywhere on the curve; macro says the masked arm is better at every threshold, and the
    coverage count says the control predicts nothing at all for one of ten channels.

    Both are true about the same counts, and printing only the first is what let the effect the
    mask exists for hide behind a good-looking average.
    """
    paths = []
    # One large channel found, one small channel not found: the shape this cohort presented.
    counts = {
        "masked-585": ([1_000_000.0, 12_000.0], [0.0, 3_000.0], [100_000.0, 8_000.0]),
        "control-585": ([1_000_000.0, 0.0], [0.0, 0.0], [100_000.0, 20_000.0]),
    }
    for run, (tp, fp, fn) in counts.items():
        report = _report(run, measurement="whole_case", cases=119, epochs_done=200,
                         trained_unmasked=(run == "control-585"),
                         **ab.metrics_from_counts(tp, fp, fn, ["aorta", "calcification"]))
        del report["iterations"]
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)

    ab.compare(paths)
    out = capsys.readouterr().out
    assert "macro" in out, out[:800]
    assert "channels found" in out
    assert "2/2" in out and "1/2" in out, "the coverage does not distinguish the two arms"
    recall_block = out[out.index("RECALL"):out.index("DICE")]
    assert recall_block.index("micro") < recall_block.index("macro"), (
        "macro is printed before micro, so a reader meets the clinical weighting before the "
        "one that is comparable with the training logs"
    )


# --------------------------------------------------------------------------------------------
# THE SHAPE BLOCK: THE WIRING, NOT JUST THE ARITHMETIC
#
# `overlap.py` had fourteen gates of its own and every one of them passed while nothing in the
# evaluator called it. A state key nobody subscribes to passes every grep-based gate; so does a
# metric module nobody imports. What follows exercises `shape_blocks`, which is the seam between
# the module and the report, with rows whose right answer was worked out by hand.
# --------------------------------------------------------------------------------------------

from medos_trainer import overlap as overlap_module  # noqa: E402

SHAPE_POINT = {"name": "probability", "value": 0.5, "selected_on": "tune"}
SHAPE_PATIENTS = {"c1": "p1", "c2": "p2", "c3": "p3", "c4": "p4"}


def _shape_row(case, channel, **fields):
    """A `CaseOverlap` with the eligible defaults filled in, so a test states only what it means.

    Built through the real dataclass and not a stub: a stub would keep passing after a field was
    renamed, and the point of these gates is the seam between two real modules.
    """
    base = dict(eligible=True, dice=0.9, iou=0.8, reference_ml=10.0, predicted_ml=11.0,
                volume_error_ml=1.0, volume_ape=0.1, hd95_mm=4.0, assd_mm=1.0)
    base.update(fields)
    return overlap_module.CaseOverlap(case=case, channel=channel, **base)


def _blocks(rows, *, channels=("aorta",), with_distances=True):
    blocks, conventions = ab.shape_blocks(
        rows, channels=list(channels), patient_of=SHAPE_PATIENTS,
        operating_point=SHAPE_POINT, min_candidate_volume_ml=0.02,
        with_distances=with_distances,
    )
    return blocks, conventions


def test_every_metric_the_overlap_module_computes_is_read_into_a_report() -> None:
    """THE BRIDGE GATE, and it is the one that was missing when a defect came back.

    `overlap.OVERLAP_METRICS` declares what the module produces and `ab.OVERLAP_FIELDS` declares
    what the report reads. Nothing but this test makes the two agree, so a seventh metric added to
    the module -- or a field renamed on `CaseOverlap` -- would otherwise be computed on every case
    of a 45-minute pass and then silently dropped on the floor.
    """
    assert set(overlap_module.OVERLAP_METRICS) == set(ab.OVERLAP_FIELDS), (
        "the module computes "
        f"{sorted(set(overlap_module.OVERLAP_METRICS) - set(ab.OVERLAP_FIELDS))} that no report "
        "reads, and the report reads "
        f"{sorted(set(ab.OVERLAP_FIELDS) - set(overlap_module.OVERLAP_METRICS))} that the module "
        "does not declare"
    )
    fields = set(overlap_module.CaseOverlap.__dataclass_fields__)
    missing = sorted(f for f in ab.OVERLAP_FIELDS.values() if f not in fields)
    assert not missing, f"{missing} are read off a CaseOverlap that has no such field"


def test_the_shape_block_names_the_per_case_dice_convention_and_not_the_pooled_one() -> None:
    """`MOS-EVID-050` allows no default, and the other half of the same report says `dice_pooled`.

    One global convention block would have labelled this Dice as pooled. It is a mean over cases,
    which is a different number: on a cohort with one huge case and twenty small ones the two
    disagree by more than any delta this session has measured.
    """
    _blocks_out, conventions = _blocks([_shape_row("c1", "aorta")])
    assert conventions["dice_aggregation"] == "dice_mean_per_case"
    block = _blocks_out["aorta"]
    assert block["dice_mean_per_case"]["conventions"]["dice_aggregation"] == "dice_mean_per_case"


def test_a_channel_no_case_annotates_is_absent_rather_than_present_and_empty() -> None:
    """Absent means nobody annotated it. An empty block would read as a model that found nothing.

    The two are opposite claims about the model and the report has to be able to say which.
    """
    blocks, _ = _blocks([_shape_row("c1", "aorta")], channels=("aorta", "pneumonia"))
    assert set(blocks) == {"aorta"}


def test_this_loop_does_not_inherit_the_detection_loops_control_flow() -> None:
    """A channel the DETECTION half skips still gets a shape block, and that is the point.

    The detection loop `break`s out of a channel whose `rows["sensitivity"]` is empty -- no case
    annotates it there. Shapes are a different question with a different eligibility, and a block
    written inside that loop would have inherited a decision about counts. The gate is that
    `shape_blocks` takes its channel list and its rows and nothing else: no detection result is an
    argument, so there is no path by which one could gate the other.
    """
    import inspect
    parameters = set(inspect.signature(ab.shape_blocks).parameters)
    assert parameters == {"shapes", "channels", "patient_of", "operating_point",
                          "min_candidate_volume_ml", "with_distances"}, (
        f"shape_blocks takes {sorted(parameters)}; a detection argument among them would be a "
        "path by which the count endpoint could decide whether a shape is measured"
    )


def test_the_dice_is_the_mean_over_eligible_cases_and_hd95_is_the_median() -> None:
    """Both by hand, and they are NOT the same reduction -- the registry says so per id.

    `dice_mean_per_case` aggregates as a mean and `hd95_mm` as a median, because a Hausdorff
    percentile has a tail that one bad case drags arbitrarily far. Reading both off one loop is
    how a single reduction gets applied to six metrics that need three.
    """
    rows = [
        _shape_row("c1", "aorta", dice=0.90, hd95_mm=2.0),
        _shape_row("c2", "aorta", dice=0.80, hd95_mm=4.0),
        _shape_row("c3", "aorta", dice=0.70, hd95_mm=60.0),
    ]
    block = _blocks(rows)[0]["aorta"]
    assert block["dice_mean_per_case"]["value"] == pytest.approx(0.80)
    assert block["hd95_mm"]["value"] == pytest.approx(4.0), (
        "60.0 is one case out of three; a mean would report 22.0 mm and a median 4.0, and the "
        "registry names the median for this id"
    )
    assert block["dice_mean_per_case"]["n"] == 3


def test_an_empty_reference_case_is_excluded_from_n_and_is_not_scored_zero() -> None:
    """`MOS-EVID-051`'s policy reaching the report: excluded and reported separately.

    A Dice of 0.0 for a case with nothing to segment reads as a failed segmentation. Here two
    cases are measurable at 0.9 and 0.7 and one has no reference: the honest answer is 0.8 over
    two cases, and the dishonest one is 0.533 over three.

    THE EXCLUSION ITSELF IS GUARDED TWICE and the assertion on the REASON is what makes this gate
    load-bearing. An ineligible row carries `None` in all six fields, so removing the eligibility
    branch leaves the absent-value branch to exclude the case anyway -- proved by breaking, where
    exactly that edit left this test green. What the second branch cannot supply is WHY: it would
    report `empty_prediction_surface`, blaming the model for predicting no boundary on a case that
    had no reference to predict. The reason is the half that has one guard, so it is asserted.
    """
    rows = [
        _shape_row("c1", "aorta", dice=0.9),
        _shape_row("c2", "aorta", dice=0.7),
        _shape_row("c3", "aorta", eligible=False, undefined_reason="empty_ground_truth",
                   dice=None, iou=None, volume_error_ml=None, volume_ape=None,
                   hd95_mm=None, assd_mm=None, reference_ml=0.0, predicted_ml=0.4),
    ]
    block = _blocks(rows)[0]["aorta"]
    assert block["dice_mean_per_case"]["value"] == pytest.approx(0.8)
    assert block["dice_mean_per_case"]["n"] == 2
    assert block["eligible_cases"] == 2
    assert block["empty_gt_case_count"] == 1

    carried = ab._shape_value(rows[2], "dice", SHAPE_PATIENTS, True)
    assert carried.eligible is False
    assert carried.undefined_reason == "empty_ground_truth", (
        f"the case is excluded for {carried.undefined_reason!r}. The row already decided why it "
        "was ineligible and that reason must be carried, not re-derived: re-deriving it here gives "
        "'empty_prediction_surface', which reports a model failure on a case that had no reference"
    )


def test_a_case_with_no_predicted_surface_leaves_hd95_out_but_keeps_its_dice() -> None:
    """ONE CASE, TWO ELIGIBILITIES, and they are not the same question.

    A model that predicted nothing has a Dice of 0.0 -- measured, and a real failure. It has no
    boundary at all, so it has no HD95, and a 0.0 there would be the best possible boundary score
    awarded for predicting nothing. So `n` differs between the two metrics of the same channel,
    which is the shape of answer the registry's four companions exist to make visible.
    """
    rows = [
        _shape_row("c1", "aorta", dice=0.9, hd95_mm=3.0),
        _shape_row("c2", "aorta", dice=0.0, volume_error_ml=-10.0, volume_ape=1.0,
                   predicted_ml=0.0, hd95_mm=None, assd_mm=None),
    ]
    block = _blocks(rows)[0]["aorta"]
    assert block["dice_mean_per_case"]["n"] == 2
    assert block["dice_mean_per_case"]["value"] == pytest.approx(0.45)
    assert block["hd95_mm"]["n"] == 1
    assert block["hd95_mm"]["value"] == pytest.approx(3.0)
    assert block["eligible_cases"] == 2, (
        "the case was measured; it is the distance that is undefined, not the case"
    )


def test_not_computing_the_distances_is_recorded_as_nobody_looking() -> None:
    """The same `None`, a different reason, and the difference is the model's reputation.

    With `--no-distances` every HD95 is absent because the run declined to spend two distance
    transforms per channel per case. That is not the model failing to produce a boundary, and a
    reader of `n=0` needs to be able to tell which happened. Dice and the volumes are unaffected,
    which is the whole point of the flag.
    """
    rows = [_shape_row("c1", "aorta", hd95_mm=None, assd_mm=None),
            _shape_row("c2", "aorta", hd95_mm=None, assd_mm=None)]
    block = _blocks(rows, with_distances=False)[0]["aorta"]
    assert block["hd95_mm"]["value"] is None and block["hd95_mm"]["n"] == 0
    assert block["dice_mean_per_case"]["n"] == 2

    reason = ab._shape_value(rows[0], "hd95_mm", SHAPE_PATIENTS, False).undefined_reason
    assert reason == "distances_not_computed"
    blamed = ab._shape_value(rows[0], "hd95_mm", SHAPE_PATIENTS, True).undefined_reason
    assert blamed == "empty_prediction_surface", (
        "with distances asked for, an absent HD95 IS the model's: it predicted no surface. "
        f"Reported {blamed!r}, which would blame the run for the model's miss"
    )


def test_a_measured_zero_stays_eligible_and_only_an_absent_value_does_not() -> None:
    """The distinction `MOS-EVID-051` is entirely made of, at the one place it is decided.

    `_shape_value` is the only code that turns a `CaseOverlap` field into an eligibility, so a
    falsy-instead-of-None test here -- `if not value` in place of `if value is None` -- would move
    every perfectly-agreeing volume and every total miss out of `n` at once.
    """
    zero = ab._shape_value(_shape_row("c1", "aorta", volume_error_ml=0.0),
                           "volume_error_ml", SHAPE_PATIENTS, True)
    assert zero.eligible is True and zero.value == 0.0
    dice_zero = ab._shape_value(_shape_row("c1", "aorta", dice=0.0),
                                "dice", SHAPE_PATIENTS, True)
    assert dice_zero.eligible is True and dice_zero.value == 0.0


def test_the_volume_error_keeps_its_sign_and_the_percentage_does_not() -> None:
    """Over- and under-segmentation cancelling is a FINDING, not a bug to be absoluted away.

    Two cases 5 mL over and 5 mL under give a mean signed error of 0.0 and a median APE of 0.5.
    A report that showed only the first would say the volumes are right; one that showed only an
    absolute error would hide that the model has no systematic bias. The registry carries both
    ids for that reason and this gate is that both are computed.
    """
    rows = [_shape_row("c1", "aorta", volume_error_ml=+5.0, volume_ape=0.5),
            _shape_row("c2", "aorta", volume_error_ml=-5.0, volume_ape=0.5)]
    block = _blocks(rows)[0]["aorta"]
    assert block["volume_error_ml"]["value"] == pytest.approx(0.0)
    assert block["volume_ape"]["value"] == pytest.approx(0.5)


def test_every_shape_aggregate_carries_the_companions_and_the_threshold_it_was_taken_at() -> None:
    """`MOS-EVID-056`: a bare scalar must fail validation. And the threshold, which is ours.

    None of these six needs an operating point by the registry -- they are all
    `needs_threshold: False` -- so nothing would have refused a report without one. But every one
    of them is computed from a mask that exists only at a threshold, and a Dice of 0.92 at an
    unrecorded threshold is not a reproducible measurement. So the point is passed anyway, and
    this gate is that it survives into all six documents.
    """
    block = _blocks([_shape_row("c1", "aorta"), _shape_row("c2", "aorta", dice=0.7)])[0]["aorta"]
    for metric in ab.OVERLAP_FIELDS:
        document = block[metric]
        for companion in ("n", "n_patients", "ci_low", "ci_high", "conventions"):
            assert companion in document, f"{metric} has no {companion}"
        assert document["operating_point"]["value"] == 0.5, (
            f"{metric} does not say which threshold produced the mask it was computed from"
        )
        assert document["n_patients"] == 2


def test_the_interval_is_over_patients_and_two_series_of_one_patient_do_not_count_twice() -> None:
    """`MOS-EVID-057`, at the seam: `patient_of` is consulted and the case key is not reused.

    Both rows here are one patient's two series. `n` is 2 because two cases were measured;
    `n_patients` must be 1, because the bootstrap resamples patients and a cohort read as two
    independent observations would report an interval narrower by about `sqrt(2)` with nothing
    saying so. Deriving the key from the case name by string surgery is how that happens.
    """
    blocks, _ = ab.shape_blocks(
        [_shape_row("c1", "aorta", dice=0.9), _shape_row("c2", "aorta", dice=0.7)],
        channels=["aorta"], patient_of={"c1": "same", "c2": "same"},
        operating_point=SHAPE_POINT, min_candidate_volume_ml=0.02,
    )
    document = blocks["aorta"]["dice_mean_per_case"]
    assert document["n"] == 2 and document["n_patients"] == 1
    assert document["ci_low"] == pytest.approx(document["ci_high"]), (
        "with one patient every bootstrap draw takes that patient, so both of their cases come "
        "every time and the interval is degenerate. A non-degenerate one here would mean cases "
        "were resampled independently, which MOS-EVID-057 forbids"
    )


def test_both_endpoints_read_the_same_forward_pass() -> None:
    """A PLACEMENT CHECK, AND IT SAYS SO. It cannot run `measure_clinical`; it reads it.

    Everything `shape_blocks` is given comes from a loop that needs a card, a checkpoint and a
    preprocessed cohort, so no gate in this file can execute it. What a gate CAN do is refuse the
    one arrangement that would be wrong in a way no number would reveal: shapes measured from a
    second, separate inference. Two passes of the same checkpoint over the same cases agree, so
    the report would look right -- until a non-deterministic augmenter, a different tile step, or
    a mirroring flag set on one call and not the other made the two halves describe two
    inferences while the header claimed one.

    So this asserts, over the AST and not the text: both calls sit in ONE loop, both take the SAME
    probability variable as their first argument, and that variable is still the one released by
    the `del` at the end of that iteration. A comment saying "one pass" would satisfy a text
    search; none of these three survives a second pass being introduced.

    What it does NOT check is that the numbers are right, and nothing here can. That is checked by
    running the tool, which is how three earlier defects in this file were found.
    """
    import ast

    tree = ast.parse(_TOOL.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "measure_clinical")

    def calls_named(scope, name):
        return [node for node in ast.walk(scope)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == name]

    loops = [node for node in ast.walk(function)
             if isinstance(node, ast.For) and calls_named(node, "candidates_for_case")]
    assert len(loops) == 1, (
        f"{len(loops)} loops extract candidates; with two, one of them is a second pass"
    )
    loop = loops[0]

    shape_calls = calls_named(loop, "overlap_for_case")
    assert len(shape_calls) == 1, (
        "the shape metrics are not computed inside the loop that extracts the candidates, so "
        "they are computed from a different forward pass than the counts they are reported beside"
    )
    counted = calls_named(loop, "candidates_for_case")[0]

    first = [call.args[0] for call in (counted, shape_calls[0])]
    assert all(isinstance(argument, ast.Name) for argument in first), (
        "one of the two endpoints is passed an expression rather than the pass's own array, so "
        "whether they see the same probabilities cannot be established by reading it"
    )
    assert first[0].id == first[1].id, (
        f"the counts read {first[0].id!r} and the shapes read {first[1].id!r}"
    )

    released = {target.id for node in ast.walk(loop) if isinstance(node, ast.Delete)
                for target in node.targets if isinstance(target, ast.Name)}
    assert first[0].id in released, (
        f"{first[0].id!r} is never deleted in the loop. It is a whole-volume float array per "
        "channel and the pass holds one case at a time on purpose"
    )

    second = [call.args[1] for call in (counted, shape_calls[0])]
    assert all(isinstance(argument, ast.Name) for argument in second) and \
        second[0].id == second[1].id, (
        "the two endpoints do not read the same reference. `np.asarray(segmentation)[0]` written "
        "out twice is two expressions that agree until one is edited, which is how a report comes "
        "to compare a prediction against one array and a shape against another"
    )


# =====================================================================================
# THE CANDIDATE SIDECAR: MOS-EVID-069's purpose, gated as a PROPERTY and not as a field list
# =====================================================================================
#
# The requirement's own words are the acceptance criterion: the per-candidate record "makes an
# operating threshold re-selectable without re-running inference, which is the property that stops a
# threshold change from becoming a GPU project." A gate that checked only that the named members are
# present would pass while the property failed -- which is exactly what happened here, because the
# candidate's score alone lets a reader recompute the FALSE POSITIVES and not the SENSITIVITY.
#
# Measured cost of not having it, on this cohort: two hours per question without the distance
# transforms, six with.

import json as _json  # noqa: E402  - the suite already imports json; this keeps the block portable

from medos_trainer import detection as _detection  # noqa: E402


def _sidecar_case(case: str, channel: str = "neo"):
    """One case whose component is NON-UNIFORM across two references.

    The uniform version is the fixture shape that hid a real defect: with one score everywhere, the
    component's maximum and the maximum on each reference are the same number, so a sidecar that
    persisted only the first would still reproduce every count.
    """
    import numpy as np

    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1, 1] = 1                       # weakly predicted
    truth[0, 3:6, 3:6] = 1                   # strongly predicted
    probability[0, 0, 1:6, 1:6] = 0.2
    probability[0, 0, 3:6, 3:6] = 0.93
    probability[0, 0, 1, 1] = 0.2
    probability[0, 0, 6:8, 6:8] = 0.6        # a false positive, far away
    return _detection.candidates_for_case(
        probability, truth, label_of={"neo": [1], "effusion": 2},
        channels=["neo", "effusion"], supervised=["neo"],
        spacing_mm=(2.0, 2.0, 2.0), extraction_threshold=0.1, case=case,
    )


def test_the_sidecar_carries_every_candidate_and_every_reference(tmp_path) -> None:
    """A header, then one row per candidate and one per reference, all valid JSON Lines."""
    detected = [_sidecar_case("c1"), _sidecar_case("c2")]
    patient_of = {"c1": "p1", "c2": "p2"}
    record = ab.write_candidate_records(
        tmp_path / "r.candidates.jsonl", detected, channels=["neo", "effusion"],
        patient_of=patient_of, extraction_threshold=0.1, min_candidate_volume_ml=0.02,
        spacing_mm=(2.0, 2.0, 2.0),
    )
    lines = [_json.loads(line) for line in
             (tmp_path / "r.candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    header, rows = lines[0], lines[1:]
    assert header["spec"] == "MOS-EVID-069"
    assert header["centroid_units"] == "voxel"
    assert "centroid_lps_mm" in header["centroid_departure"], (
        "the departure from the requirement's own member name is not declared in the header, so a "
        "reader would take a voxel centroid for millimetres"
    )
    candidates = [row for row in rows if row["row"] == "candidate"]
    references = [row for row in rows if row["row"] == "reference"]
    assert len(candidates) == record["candidates"] == sum(
        len(found.candidates.get(channel, ())) for found in detected
        for channel in ("neo", "effusion"))
    assert len(references) == record["references"] == 4          # two per case
    assert record["digest"].startswith("sha256:")


def test_every_candidate_row_carries_what_the_requirement_names(tmp_path) -> None:
    """The five members, plus the two this measurement adds and says why."""
    record = ab.write_candidate_records(
        tmp_path / "r.candidates.jsonl", [_sidecar_case("c1")], channels=["neo"],
        patient_of={"c1": "p1"}, extraction_threshold=0.1, min_candidate_volume_ml=0.02,
        spacing_mm=(2.0, 2.0, 2.0),
    )
    rows = [_json.loads(line) for line in
            (tmp_path / "r.candidates.jsonl").read_text(encoding="utf-8").splitlines()[1:]]
    candidate = next(row for row in rows if row["row"] == "candidate")
    for member in ("score", "volume_ml", "centroid_voxel", "matched_reference_id",
                   "match_distance_mm", "overlap_voxels", "overlap_scores",
                   "case", "patient_key", "channel"):
        assert member in candidate, f"the candidate row has no {member!r}"
    assert record["candidates"] >= 2, "the fixture no longer produces a false positive to record"


def test_a_threshold_is_re_selectable_from_the_sidecar_alone(tmp_path) -> None:
    """THE PROPERTY, AND IT IS THE ACCEPTANCE CRITERION.

    Sensitivity and PPV are recomputed at three thresholds from the FILE, with no access to the
    probability volumes, and compared against `counts_at` on the live objects. This is what makes
    the sidecar worth writing: without it the same three numbers cost three inference passes.

    It is also the gate that catches the half-measure. Persisting only each candidate's own score
    reproduces the false positives exactly and gets the sensitivity WRONG wherever a component spans
    references of different strength -- which on this cohort is the pleural effusion and the coronary
    calcifications, the two channels the whole measurement turns on.
    """
    detected = [_sidecar_case("c1"), _sidecar_case("c2")]
    patient_of = {"c1": "p1", "c2": "p2"}
    ab.write_candidate_records(
        tmp_path / "r.candidates.jsonl", detected, channels=["neo"],
        patient_of=patient_of, extraction_threshold=0.1, min_candidate_volume_ml=0.02,
        spacing_mm=(2.0, 2.0, 2.0),
    )
    rows = [_json.loads(line) for line in
            (tmp_path / "r.candidates.jsonl").read_text(encoding="utf-8").splitlines()[1:]]

    for threshold in (0.15, 0.5, 0.95):
        # --- recomputed from the FILE ---
        found_ids: set[str] = set()
        survivors = 0
        true_positives = 0
        for row in rows:
            if row["row"] != "candidate" or row["channel"] != "neo":
                continue
            if row["score"] >= threshold:
                survivors += 1
                if row["matched_reference_id"] is not None:
                    true_positives += 1
            for reference_id, on_reference in row["overlap_scores"]:
                if on_reference >= threshold:
                    found_ids.add((row["case"], reference_id))
        denominator = len([row for row in rows
                           if row["row"] == "reference" and row["channel"] == "neo"])

        # --- the live answer ---
        live = _detection.counts_at(detected, "neo", threshold=threshold,
                                    patient_of=patient_of, policy="all_components")
        live_hits = sum(row.numerator for row in live["sensitivity"])
        live_total = sum(row.denominator for row in live["sensitivity"])
        live_survivors = sum(row.denominator for row in live["ppv"])
        live_true = sum(row.numerator for row in live["ppv"])

        assert len(found_ids) == live_hits, (
            f"at {threshold}: the sidecar recomputes {len(found_ids)} found lesions and the live "
            f"counting says {live_hits}. A threshold is not re-selectable from this file"
        )
        assert denominator == live_total
        assert survivors == live_survivors, (
            f"at {threshold}: {survivors} survivors from the file against {live_survivors} live"
        )
        assert true_positives == live_true


def test_the_reference_rows_are_what_makes_the_denominator_recomputable(tmp_path) -> None:
    """A reference no candidate touched appears in NO candidate row, so it can only come from here.

    Drop the reference rows and every recomputed sensitivity gains a smaller denominator -- which
    moves the number in the flattering direction, silently.
    """
    import numpy as np

    probability = np.zeros((2, 1, 8, 8))
    truth = np.zeros((1, 8, 8), dtype=int)
    truth[0, 1, 1] = 1                       # a reference the model never predicts
    truth[0, 5, 5] = 1
    probability[0, 0, 5, 5] = 0.9            # only the second is predicted
    found = _detection.candidates_for_case(
        probability, truth, label_of={"neo": [1], "effusion": 2},
        channels=["neo", "effusion"], supervised=["neo"],
        spacing_mm=(2.0, 2.0, 2.0), extraction_threshold=0.1, case="c1",
    )
    ab.write_candidate_records(
        tmp_path / "r.candidates.jsonl", [found], channels=["neo"],
        patient_of={"c1": "p1"}, extraction_threshold=0.1, min_candidate_volume_ml=0.02,
        spacing_mm=(2.0, 2.0, 2.0),
    )
    rows = [_json.loads(line) for line in
            (tmp_path / "r.candidates.jsonl").read_text(encoding="utf-8").splitlines()[1:]]
    references = [row for row in rows if row["row"] == "reference"]
    assert len(references) == 2, "an unpredicted reference is absent, so the denominator is lost"
    untouched = [row for row in references if not row["matched_scores"]]
    assert len(untouched) == 1
    candidate_ids = {reference_id for row in rows if row["row"] == "candidate"
                     for reference_id, _score in row["overlap_scores"]}
    assert untouched[0]["reference_id"] not in candidate_ids, (
        "the fixture's unpredicted reference is reachable from a candidate row after all, so this "
        "gate proves nothing"
    )

    # AND THE DENOMINATOR `reselect_counts` USES MUST BE THESE ROWS, not what the candidates touch.
    # Proved by breaking: deriving it from the candidates' overlap ids left the agreement gate green,
    # because there every reference happens to be touched. Here one is not, and the two differ --
    # 1 instead of 2 -- which moves sensitivity in the FLATTERING direction, silently.
    counted = ab.reselect_counts(
        [row for row in rows if row["row"] == "candidate"],
        references, threshold=0.5, volume_floor_ml=0.0,
    )
    assert counted["references"] == 2, (
        f"the offline denominator is {counted['references']} where the cohort has 2 references; a "
        "denominator taken from what the candidates touch cannot see a lesion nothing predicted"
    )
    assert counted["sensitivity"] == pytest.approx(0.5), (
        "one of two references is predicted, so sensitivity is 0.5; a candidate-derived denominator "
        "reports 1.000 for a model that missed half the lesions"
    )


def test_reselect_counts_agree_with_the_live_counting_at_every_threshold(tmp_path) -> None:
    """TWO IMPLEMENTATIONS OF ONE RULE, COMPARED -- and the cheap one is the one people will use.

    `reselect` exists so that a threshold question costs a second instead of two to six hours of
    card. That is only worth having if its counting agrees with `detection.counts_at`; a second
    implementation that quietly disagrees is WORSE than none, because the cheap answer is the one
    that gets quoted.

    The comparison is at `volume_floor_ml=0.0`, which is the configuration where the two are supposed
    to be identical -- the floor is the lever `counts_at` does not have and is why this function
    exists at all.

    The fixture's component spans references of DIFFERENT strength, because that is where the two
    could differ: a candidate's own maximum and its maximum on one reference are the same number
    only when the component is uniform, and a uniform fixture already hid this exact defect once.
    """
    detected = [_sidecar_case("c1"), _sidecar_case("c2")]
    patient_of = {"c1": "p1", "c2": "p2"}
    sidecar = tmp_path / "r.candidates.jsonl"
    ab.write_candidate_records(
        sidecar, detected, channels=["neo"], patient_of=patient_of,
        extraction_threshold=0.1, min_candidate_volume_ml=0.02, spacing_mm=(2.0, 2.0, 2.0),
    )
    _header, rows = ab.read_candidate_records(sidecar)
    candidates = [row for row in rows if row["row"] == "candidate" and row["channel"] == "neo"]
    references = [row for row in rows if row["row"] == "reference" and row["channel"] == "neo"]

    for threshold in (0.1, 0.15, 0.25, 0.5, 0.6, 0.75, 0.93, 0.95):
        offline = ab.reselect_counts(candidates, references, threshold=threshold,
                                     volume_floor_ml=0.0)
        live = _detection.counts_at(detected, "neo", threshold=threshold,
                                    patient_of=patient_of, policy="all_components")
        assert offline["found"] == sum(row.numerator for row in live["sensitivity"]), (
            f"at {threshold}: sidecar says {offline['found']} found, live counting says "
            f"{sum(row.numerator for row in live['sensitivity'])}"
        )
        assert offline["references"] == sum(row.denominator for row in live["sensitivity"])
        assert offline["surviving"] == sum(row.denominator for row in live["ppv"])
        assert offline["true_positives"] == sum(row.numerator for row in live["ppv"])


def test_a_volume_floor_removes_candidates_without_touching_the_denominator(tmp_path) -> None:
    """The lever `counts_at` does not have, and the reason this function is not just a duplicate.

    A floor can only REMOVE candidates, so sensitivity is monotone non-increasing in it and the
    reference count never moves. A floor that changed the denominator would be measuring a different
    cohort at each setting, and the table would compare numbers that are not comparable.
    """
    detected = [_sidecar_case("c1")]
    sidecar = tmp_path / "r.candidates.jsonl"
    ab.write_candidate_records(
        sidecar, detected, channels=["neo"], patient_of={"c1": "p1"},
        extraction_threshold=0.1, min_candidate_volume_ml=0.02, spacing_mm=(2.0, 2.0, 2.0),
    )
    _header, rows = ab.read_candidate_records(sidecar)
    candidates = [row for row in rows if row["row"] == "candidate"]
    references = [row for row in rows if row["row"] == "reference"]

    previous = None
    for floor in (0.0, 0.02, 0.1, 1.0, 1000.0):
        counted = ab.reselect_counts(candidates, references, threshold=0.1,
                                     volume_floor_ml=floor)
        assert counted["references"] == len(references), (
            "the volume floor changed the denominator, so each row of the table describes a "
            "different cohort"
        )
        if previous is not None:
            assert counted["surviving"] <= previous["surviving"]
            assert (counted["found"] or 0) <= (previous["found"] or 0), (
                "a higher floor found MORE lesions, which a filter that only removes candidates "
                "cannot do"
            )
        previous = counted
    assert previous["surviving"] == 0, "a 1000 mL floor left something, so the floor is not applied"


def test_a_sidecar_header_from_the_wrong_file_is_refused(tmp_path) -> None:
    """A report and its sidecar are different files, and reading one as the other must say so
    rather than producing an empty table."""
    wrong = tmp_path / "report.json"
    wrong.write_text('{"measurement": "clinical", "per_channel": {}}', encoding="utf-8")
    with pytest.raises(SystemExit, match="candidate-records header"):
        ab.read_candidate_records(wrong)
