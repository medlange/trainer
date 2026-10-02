# SPDX-License-Identifier: Apache-2.0
"""ONE counting rule, applied to two trained networks.

WHY THIS EXISTS. Each arm of a masked/unmasked comparison validates under its OWN rule.
The masked arm counts only the (case, channel) pairs the cohort declares; the unmasked one
masks everything to ones, so its validation includes cases where a channel is not annotated
at all -- the reference there is empty, and any finding the network makes becomes a false
positive. The denominators differ, so subtracting the two arms' own printed numbers would
mix the effect of the mask on TRAINING with the difference in how the two COUNT. Here the
rule is one, it is declared in this file, and it is not inherited from either run's
configuration: only pairs named in `supervision.json` count.

RECALL IS THE PRIMARY METRIC, AND DICE IS SECOND. The claim under test is that unmasked
training teaches SUPPRESSION: an unannotated channel is presented as background and the
network learns not to find it. Suppression is false negatives, which is recall. Dice mixes
recall with precision, so a fall in Dice cannot distinguish "started missing things" from
"started guessing". Precision is recorded as the control on the opposite explanation: if
the masked arm's recall is higher and its precision has collapsed, it is simply predicting
more everywhere and the recall is worth nothing.

ONE ARM PER PROCESS, WHICH IS A CONSTRAINT AND NOT A STYLE. `nnunetv2/paths.py` reads its
three roots from the environment AT IMPORT and binds them to constants -- the reason
`medos_trainer/__main__.py` calls `prepare_workspace` before importing the backend. A
first draft of this file set the variables AFTER importing the trainer (so the dataset was
looked for in the image's default root) and tried to measure both arms in one process,
which is impossible for the same reason: after the first import the roots cannot change.
So `measure` takes ONE run directory and `compare` only reads finished reports.

WHY PATCHES, AND WHAT THAT OBLIGES. Whole cases through the sliding window would be
cleaner, but for a region head `perform_actual_validation` collapses the channels into one
label map via `regions_class_order`, and these channels must stay separate: a voxel can be
both pneumonia and effusion. Patches are valid on exactly one condition -- both arms see
the same sequence, which holds only if their preprocessed data is byte-identical.
`compare` CHECKS that condition through the digest each measurement records, and refuses
when it does not hold: diverged bytes mean diverged patches, and the difference would look
meaningful while measuring two different samples.

WHAT IT WRITES INTO THE RUN DIRECTORY. Constructing the trainer creates a new
`training_log_*.txt` under `work/results/.../fold_0`, so a measurement is not quite
read-only. Nothing else: checkpoints are written only by the fit, and `splits_final.json`
is read. Measuring a RUNNING fit is therefore safe but not traceless, which is why
`_epochs_done` takes the maximum over all logs rather than the newest -- the newest is its
own.

Spec: MOS-TRAIN-116 (selection reads a partition the fit did not see), MOS-TRAIN-126.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

CONFIGURATION = "3d_fullres"
DATASET = "Dataset501_MedicalOSCohort"


# =====================================================================================
# The arithmetic, on its own, so it can be checked against numbers worked out by hand
# =====================================================================================
def _macro(values: Sequence[float | None]) -> float | None:
    """The mean over channels that HAVE a value, or `None` if none does.

    WHY THE MACRO AVERAGE IS HERE AT ALL, AND WHY IT IS NOT A SECOND OPINION.
    ------------------------------------------------------------------------
    The micro average sums counts before dividing, so a channel with ten million reference
    voxels and a channel with twenty thousand contribute in that proportion. That is the right
    weighting for "how much of the reference did this network find", and it is the wrong
    weighting for "which findings can this network find at all" -- because the structures the
    partial-label problem suppresses are precisely the small ones.

    Measured on this cohort: the control arm predicts NOTHING for `coronary_calcification` at
    any threshold, and the masked arm reaches 0.55 recall there. Micro recall barely notices --
    that channel's median is 207 voxels against 10,000+ for the large structures -- while the
    masked arm's precision loss on the large channels dominates the same sum. So micro said the
    control was better everywhere on the curve, and macro says the masked arm is better at
    every threshold by 0.07 to 0.12. Both are true statements about different questions.

    The clinical weighting is the macro one: a missed nodule matters the same as a missed
    aorta, and it does not matter less because it is smaller. So both are reported, and neither
    is dropped -- a single "primary metric" is what let the effect hide for a whole evening.

    `None` is excluded rather than counted as zero, for the reason `metrics_from_counts` gives:
    a channel with nothing to measure is not a channel that scored zero.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def metrics_from_counts(
    tp: list[float], fp: list[float], fn: list[float], channels: list[str]
) -> dict[str, Any]:
    """Per-channel recall, Dice and precision from summed confusion counts, plus micro.

    SUMMED OVER THE WHOLE MEASUREMENT, not averaged over cases -- which is nnU-Net's own
    pseudo-Dice convention, so these numbers are comparable with the training logs rather
    than being a second, differently-defined quantity beside them.

    A CHANNEL WITH NOTHING TO MEASURE IS `None`, NOT ZERO. `tp + fn == 0` means no
    supervised pair in this measurement contained the finding; `tp + fp == 0` means the
    network predicted it nowhere. Those are "nothing to measure" and "found nothing", and
    collapsing either into 0.0 makes an absent measurement read as a failed one -- the same
    error as averaging a `nan` into a mean.
    """
    if not (len(tp) == len(fp) == len(fn) == len(channels)):
        raise ValueError(
            "counts and channels disagree in length: %d/%d/%d against %d channels"
            % (len(tp), len(fp), len(fn), len(channels))
        )

    def ratio(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else float(numerator / denominator)

    recall = [ratio(a, a + c) for a, c in zip(tp, fn)]
    precision = [ratio(a, a + b) for a, b in zip(tp, fp)]
    dice = [ratio(2 * a, 2 * a + b + c) for a, b, c in zip(tp, fp, fn)]

    st, sp, sn = float(sum(tp)), float(sum(fp)), float(sum(fn))
    return {
        "channels": list(channels),
        "recall_per_channel": recall,
        "dice_per_channel": dice,
        "precision_per_channel": precision,
        # THE MACRO AVERAGE AND THE COVERAGE COUNT, beside the micro ones rather than instead
        # of them. See `_macro`: micro answers "how much of the reference was found" and macro
        # answers "which findings can be found at all", and on a partially labelled cohort the
        # second is the question the mask exists for.
        "recall_macro": _macro(recall),
        "precision_macro": _macro(precision),
        "dice_macro": _macro(dice),
        #: Channels with a MEASURED recall above zero -- the coverage. A channel the network
        #: never predicts anywhere is a capability that does not exist, and no average of
        #: ratios says that as plainly as a count. The control arm sits at 8 of 10 at every
        #: threshold and the masked arm at 9.
        "channels_found": sum(1 for v in recall if v is not None and v > 0.0),
        "channels_measured": sum(1 for v in recall if v is not None),
        "recall_micro": ratio(st, st + sn),
        "dice_micro": ratio(2 * st, 2 * st + sp + sn),
        "precision_micro": ratio(st, st + sp),
        "tp": [float(x) for x in tp],
        "fp": [float(x) for x in fp],
        "fn": [float(x) for x in fn],
    }


# =====================================================================================
# What the two arms must share before their numbers may be subtracted
# =====================================================================================
def inputs_digest(preprocessed_root: Path, sample: int = 12) -> str:
    """Fingerprint of the preprocessed data: the file set, their sizes, and a few bodies.

    A full hash of ~100 GB would take longer than the measurement. The list of EVERY file
    with its size catches any difference in membership or length; the bodies are read from
    files chosen by position in the sorted list rather than at random, so two runs of this
    look at the same ones.
    """
    folder = preprocessed_root / DATASET / f"nnUNetPlans_{CONFIGURATION}"
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"{folder} holds no preprocessed files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    step = max(1, len(files) // sample)
    for path in files[::step][:sample]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


#: WHAT TWO MEASUREMENTS OF ONE CHECKPOINT ACTUALLY DIFFER BY, at 1000 iterations on the
#: 119-case validation half of this cohort. Both arms were measured twice on 2026-09-26 with
#: identical inputs, seed, checkpoint and operating point:
#:
#:   micro recall      masked 0.7190 / 0.7188      control 0.6935 / 0.6960
#:   micro precision   masked 0.7469 / 0.7443      control 0.8501 / 0.8513
#:   per channel       pleural_effusion recall, masked arm: 0.7775 / 0.7016
#:
#: So the MICRO figures are stable to about 0.003 and the PER-CHANNEL ones are not stable to
#: better than about 0.08. That is the whole argument for the micro average made measurable:
#: it sums counts before dividing, so the large channels carry it and a small channel's ratio
#: noise does not propagate.
#:
#: The consequence for a reader is printed with the table, because a per-channel delta of
#: 0.03 rendered to four decimals reads as a finding and is noise. Two per-channel deltas in
#: the first comparison this tool produced changed SIGN on the second measurement.
#:
#: The fix is not a bigger seed. It is to measure whole cases deterministically --
#: nnU-Net's own `perform_actual_validation` does sliding-window inference over every
#: validation case -- rather than 1000 sampled patches. Until that exists, this constant is
#: what keeps the table honest.
SAMPLING_NOISE: Final[dict[str, float]] = {"micro": 0.003, "per_channel": 0.08}


def comparable(reports: list[dict[str, Any]]) -> None:
    """Raise unless the reports differ ONLY in which network was measured.

    Three things must match, and each has its own reason:
      * `inputs_digest` -- diverged preprocessed bytes mean diverged patch sequences, so
        the two arms would have been shown different samples;
      * `seed` and `iterations` -- the same sampler, run the same number of times. NOT the
        same PATCHES: nnU-Net validates through `NonDetMultiThreadedAugmenter`, whose name
        says it, and batches arrive in thread-completion order. The seed fixes the
        augmentation draws and not the sample, so two measurements of ONE checkpoint at one
        seed differ. Measured, not assumed -- see `SAMPLING_NOISE` below;
      * `checkpoint` -- `checkpoint_best` is chosen by each arm's OWN metric (the masked
        arm's micro-Dice is masked, the control's is not), so comparing two "best" weights
        compares selections made by different rules. `checkpoint_final` is both arms at the
        same epoch with no selection at all;
      * `measurement` -- `patches` or `whole_case`. One samples 1000 patches through a
        non-deterministic augmenter and one sees every case; subtracting across them mixes the
        difference between two networks with the difference between two instruments;
      * `score_threshold` -- the operating point. Recall at 0.5 minus recall at 0.7 is not a
        difference between two networks, and `MOS-SVC-021` makes the threshold part of every
        reported sensitivity or PPV rather than context a reader is expected to supply. This
        was the one condition the list did not carry, so two measurements taken at different
        operating points would have been subtracted without complaint.
    """
    missing = [r.get("run", "?") for r in reports if "operating_point" not in r]
    if missing:
        raise SystemExit(
            f"{missing} carry no operating_point, so they were written before the threshold "
            "was recorded. MOS-SVC-021: a metric without a threshold MUST NOT be displayed. "
            "Re-measure -- the counts are cheap and the numbers are not interpretable "
            "without the point they were taken at."
        )
    # `iterations` is absent from a whole-case report and present in a patch-based one, so
    # `measurement` is checked FIRST: without it the two would be caught by a confusing
    # complaint about `iterations` rather than by the real objection, which is that they are
    # different quantities. Patch-based recall and whole-case recall are not two measurements
    # of one thing to a tolerance -- one samples and one does not.
    for field in ("measurement", "inputs_digest", "seed", "checkpoint", "score_threshold",
                  "iterations"):
        seen = {r["run"]: r.get(field) for r in reports}
        if len(set(seen.values())) != 1:
            raise SystemExit(
                f"the arms were not measured alike -- {field} differs: {seen}. Refusing: "
                "the difference between them would include this."
            )


def _apply_operating_point(trainer: Any, requested: float) -> tuple[float, str]:
    """Set the operating point, or establish the one this image is stuck with. Returns both.

    WHY THIS IS NOT ONE ASSIGNMENT. The evaluator is MOUNTED into the trainer image; the
    trainer is COMPILED INTO it. So `trainer._score_threshold = 0.7` against an image built
    before that attribute existed sets a field nobody reads, and the report then carries an
    operating point the code did not honour -- a wrong number wearing a correct label, which
    is the defect `MOS-SVC-021` exists to prevent rather than an instance of obeying it.

    Two honest outcomes, and the report records which:

      * the image's trainer declares `_score_threshold`: it is set, and `source` is `set`;
      * it does not: the image measures at whatever `masked_tp_fp_fn_tn` compiles in. That
        value is READ OUT OF THE INSTALLED SIGNATURE -- not assumed to be 0.5 -- and recorded
        with `source` of `image default`. Asking such an image for a different point is
        refused, because it cannot deliver one and a recorded 0.7 would be a fabrication.
    """
    if hasattr(trainer, "_score_threshold"):
        trainer._score_threshold = float(requested)
        return float(requested), "set"

    import inspect

    from medos_trainer.masked import masked_tp_fp_fn_tn

    compiled = float(
        inspect.signature(masked_tp_fp_fn_tn).parameters["threshold"].default
    )
    if float(requested) != compiled:
        raise SystemExit(
            f"--score-threshold {requested} was asked for, and this image's trainer has no "
            f"`_score_threshold`: it measures at the {compiled} compiled into "
            "`masked_tp_fp_fn_tn` and cannot be moved off it. Recording the requested value "
            "would put a number in the report that no code honoured. Rebuild the image, or "
            f"measure at {compiled}."
        )
    return compiled, "image default"


# =====================================================================================
# Whole-case counting: the arithmetic, separated from the inference so it can be checked
# =====================================================================================
def masked_counts_for_case(
    probability: Any,
    segmentation: Any,
    *,
    label_of: Mapping[str, Any],
    channels: Sequence[str],
    supervised: Iterable[str],
    threshold: float,
) -> tuple[list[float], list[float], list[float]]:
    """tp, fp, fn per channel for ONE case, counting only the channels IT supervises.

    WHY THIS IS A SEPARATE FUNCTION. Everything around it needs a card, a checkpoint and a
    preprocessed cohort; this needs three small arrays. It is also where a silent defect
    lives: the reference is an INTEGER label map and the prediction is one sigmoid channel per
    finding, so the two are joined by a mapping from channel name to label VALUE. Get that
    mapping off by one and every count is wrong in a way that looks like a bad model --
    channels shifted by one produce plausible, uniformly poor numbers.

    THE MAPPING COMES FROM `supervision.json`, NOT FROM THE CHANNEL ORDER. `label_of` is
    written by `stage_dataset` from the sealed cohort, alongside the `channels` list and the
    per-case supervision. Deriving the label value from a channel's position in the list would
    be a second spelling of the same fact, and the two would agree until a label set was ever
    reordered.

    `supervised` is what this case ANNOTATES. A channel absent from it contributes zero to
    every count -- not a negative -- which is the same rule the loss applies and the reason
    the two arms are comparable at all.
    """
    import numpy as np

    marked = set(supervised)
    unknown = sorted(marked - set(channels))
    if unknown:
        raise SystemExit(
            f"this case supervises {unknown}, which the channel list does not contain. "
            "supervision.json is inconsistent with itself and no mask can be built from it"
        )

    predicted_all = np.asarray(probability) > threshold
    truth_map = np.asarray(segmentation)
    if predicted_all.shape[0] != len(channels):
        raise SystemExit(
            f"the network emitted {predicted_all.shape[0]} channels and the cohort declares "
            f"{len(channels)}. MOS-TRAIN-223's label set and the network's head count have "
            "diverged, and a per-channel number would be attributed to the wrong finding"
        )

    tp: list[float] = []
    fp: list[float] = []
    fn: list[float] = []
    for column, channel in enumerate(channels):
        if channel not in marked:
            tp.append(0.0)
            fp.append(0.0)
            fn.append(0.0)
            continue
        value = label_of[channel]
        # A singleton region `[k]` and a bare `k` mean the same label value. Both spellings
        # appear in nnU-Net's own dataset.json, so both are accepted here rather than one
        # being assumed.
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise SystemExit(
                    f"{channel} maps to {value!r}: this counter joins ONE label value per "
                    "channel, and a multi-value region cannot be counted per channel without "
                    "deciding what overlap means"
                )
            value = value[0]
        truth = truth_map == int(value)
        predicted = predicted_all[column]
        tp.append(float(np.count_nonzero(predicted & truth)))
        fp.append(float(np.count_nonzero(predicted & ~truth)))
        fn.append(float(np.count_nonzero(~predicted & truth)))
    return tp, fp, fn


def masked_counts_sweep(
    probability: Any,
    segmentation: Any,
    *,
    label_of: Mapping[str, Any],
    channels: Sequence[str],
    supervised: Iterable[str],
    thresholds: Sequence[float],
) -> dict[float, tuple[list[float], list[float], list[float]]]:
    """Counts at EVERY threshold, from ONE prediction: the same comparison, done once per
    threshold over values that were split out of the volume once.

    WHY IT IS NOT A HISTOGRAM. A histogram of the score, split by the reference, is the
    textbook sufficient statistic for a whole curve and would answer any threshold afterwards.
    It is also approximate at the bin edge, and `p > t` against `p >= t` at an edge is exactly
    the off-by-one that makes a curve slightly wrong everywhere and obviously wrong nowhere.
    This returns the SAME numbers as the scalar counter because it performs the same
    comparison, and a differential gate asserts that against an implementation that already
    has eight tests of its own.

    IT IS ALSO NOT SLOWER THAN CALLING THE SCALAR COUNTER PER THRESHOLD -- it is faster. The
    expensive part is splitting the volume by the reference, and that happens once per channel
    here rather than once per (channel, threshold).

    WHY A SWEEP AT ALL. The whole-case measurement put the masked arm at 0.54 precision
    against the control's 0.84, at the 0.5 the trainer has always used and never recorded. A
    model that over-predicts at ONE threshold is not the same finding as a model that cannot
    be made precise, and only the curve tells the two apart. `MOS-SVC-020` already models the
    distinction: a service declares the operating points it SUPPORTS, and which one is applied
    is a deployment's choice.
    """
    import numpy as np

    marked = set(supervised)
    unknown = sorted(marked - set(channels))
    if unknown:
        raise SystemExit(
            f"this case supervises {unknown}, which the channel list does not contain. "
            "supervision.json is inconsistent with itself and no mask can be built from it"
        )
    if not list(thresholds):
        raise SystemExit("no thresholds were given, so there is no curve to compute")

    scores = np.asarray(probability)
    truth_map = np.asarray(segmentation)
    if scores.shape[0] != len(channels):
        raise SystemExit(
            f"the network emitted {scores.shape[0]} channels and the cohort declares "
            f"{len(channels)}. MOS-TRAIN-223's label set and the network's head count have "
            "diverged, and a per-channel number would be attributed to the wrong finding"
        )

    out: dict[float, tuple[list[float], list[float], list[float]]] = {
        float(t): ([], [], []) for t in thresholds
    }
    for column, channel in enumerate(channels):
        if channel not in marked:
            for triple in out.values():
                for series in triple:
                    series.append(0.0)
            continue
        value = label_of[channel]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise SystemExit(
                    f"{channel} maps to {value!r}: this counter joins ONE label value per "
                    "channel, and a multi-value region cannot be counted per channel without "
                    "deciding what overlap means"
                )
            value = value[0]
        truth = truth_map == int(value)
        # SPLIT ONCE. Every count below is over these two arrays, so another threshold costs
        # comparisons and not another pass over the reference.
        positive = scores[column][truth]
        negative = scores[column][~truth]
        for threshold, triple in out.items():
            found = float(np.count_nonzero(positive > threshold))
            triple[0].append(found)
            triple[1].append(float(np.count_nonzero(negative > threshold)))
            triple[2].append(float(positive.size) - found)
    return out


def accumulate(totals: list[float] | None, addition: Sequence[float]) -> list[float]:
    """Sum counts across cases. `None` is the first case, not zero.

    Trivial, and separated because the alternative -- initialising to a list of zeros whose
    length is guessed before the first case is read -- is how a channel count comes to be
    fixed by the wrong thing.
    """
    if totals is None:
        return [float(v) for v in addition]
    if len(totals) != len(addition):
        raise SystemExit(
            f"a case contributed {len(addition)} channels against {len(totals)} so far; the "
            "channel list changed mid-measurement"
        )
    return [a + float(b) for a, b in zip(totals, addition)]


# =====================================================================================
# The measurement
# =====================================================================================
def _prepared(run_dir: Path, seed: int, which: str, score_threshold: float) -> dict[str, Any]:
    """Everything both measurements need, in the one order that works.

    EXTRACTED RATHER THAN COPIED. The sequence below is load-bearing three times over -- the
    nnU-Net roots must be bound before the first `nnunetv2` import, the mask rule must be set
    before `initialize()`, and the checkpoint must be named by the caller -- and a second
    measurement path with its own copy of it would be a second place for that order to drift.
    The patch-based and whole-case paths differ only in what they do with the network.
    """
    work = run_dir / "work"

    # BEFORE ANY nnunetv2 IMPORT. The order is load-bearing; see the module docstring.
    sys.path.insert(0, "/opt/medos-trainer")
    from medos_trainer import backend

    backend.prepare_workspace(work)
    for name, expected in (("nnUNet_raw", work / "raw"),
                           ("nnUNet_preprocessed", work / "preprocessed"),
                           ("nnUNet_results", work / "results")):
        if os.environ.get(name) != str(expected):
            raise SystemExit(
                f"{name} is {os.environ.get(name)!r}, expected {str(expected)!r}: "
                "prepare_workspace did not set the root and nnU-Net would look elsewhere"
            )

    import numpy as np
    import torch

    from medos_trainer.masked_trainer import nnUNetTrainerMaskedChannels

    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    preprocessed = work / "preprocessed" / DATASET
    plans = json.loads(
        (preprocessed / f"{plan['plans_identifier']}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((preprocessed / "dataset.json").read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = nnUNetTrainerMaskedChannels(
        plans=plans, configuration=CONFIGURATION, fold=0,
        dataset_json=dataset_json, device=torch.device("cuda"),
    )
    # THE EVALUATOR'S RULE, SET HERE AND VISIBLE. The arm may have trained unmasked; it is
    # MEASURED masked, or its numbers are not comparable with the masked arm's.
    trained_unmasked = bool(trainer._unmasked)
    trainer._unmasked = False
    # AND THE OPERATING POINT, declared the same way and for the same reason.
    # `MOS-SVC-021`: a metric without a threshold must not be displayed or returned.
    applied, how = _apply_operating_point(trainer, score_threshold)

    trainer.initialize()
    checkpoint = checkpoint_path(work, which)
    trainer.load_checkpoint(str(checkpoint))
    trainer.network.eval()

    return {
        "work": work, "preprocessed": preprocessed, "trainer": trainer,
        "checkpoint": checkpoint, "applied": applied, "how": how,
        "trained_unmasked": trained_unmasked,
    }


def _report_head(run_dir: Path, prepared: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """The provenance every measurement carries, whatever it measured."""
    applied = prepared["applied"]
    return {
        "run": run_dir.name,
        "trained_unmasked": prepared["trained_unmasked"],
        "measured_with_mask": True,
        "checkpoint": prepared["checkpoint"].name,
        "seed": seed,
        # `MOS-SVC-021` fixes the member name as `score_threshold` everywhere it travels.
        "operating_point": {
            "id": "balanced" if applied == 0.5 else "calibrated",
            "score_threshold": applied,
            "source": prepared["how"],
        },
        "score_threshold": applied,
        "inputs_digest": inputs_digest(prepared["work"] / "preprocessed"),
        "epochs_done": epochs_done(prepared["work"]),
    }


def measure(run_dir: Path, iterations: int, seed: int, which: str, out: Path,
            score_threshold: float = 0.5) -> int:
    """The PATCH-BASED measurement: `iterations` batches through the training validation loop.

    Cheap and NOISY. Two runs of one checkpoint differ by up to `SAMPLING_NOISE["per_channel"]`
    per channel, because nnU-Net validates through a non-deterministic augmenter. Kept because
    it is the number every epoch of training already reported, so it is the one comparable
    with the training logs; `measure_whole_case` is the one to quote.
    """
    prepared = _prepared(run_dir, seed, which, score_threshold)
    trainer = prepared["trainer"]

    import numpy as np
    import torch


    _, val_loader = trainer.get_dataloaders()
    outputs = []
    with torch.no_grad():
        trainer.on_validation_epoch_start()
        for _ in range(iterations):
            outputs.append(trainer.validation_step(next(val_loader)))

    from nnunetv2.utilities.collate_outputs import collate_outputs

    collated = collate_outputs(outputs)
    report = metrics_from_counts(
        [float(x) for x in np.sum(collated["tp_hard"], 0)],
        [float(x) for x in np.sum(collated["fp_hard"], 0)],
        [float(x) for x in np.sum(collated["fn_hard"], 0)],
        list(trainer._channels),
    )
    report.update({
        **_report_head(run_dir, prepared, seed=seed),
        "measurement": "patches",
        "iterations": iterations,
        "val_loss": float(np.mean(collated["loss"])),
    })
    return _write_report(report, out)


def _write_report(report: dict[str, Any], out: Path) -> int:
    """Write the report and print what a person reads. One writer, both measurements."""
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("measurement: %s   trained unmasked: %s   measured masked: yes"
          % (report.get("measurement", "?"), report["trained_unmasked"]))
    print("checkpoint: %s   epochs: %s" % (report["checkpoint"], report["epochs_done"]))
    print("operating point: %s at score_threshold %s (%s)"
          % (report["operating_point"]["id"],
             report["operating_point"]["score_threshold"],
             report["operating_point"]["source"]))
    print("recall micro %s   dice micro %s"
          % (_cell(report["recall_micro"]).strip(), _cell(report["dice_micro"]).strip()))
    print("written:", out)
    return 0


_EPOCH_HEADER: Final[re.Pattern[str]] = re.compile(r"^.*: Epoch \d+[ \t]*$", re.MULTILINE)


def measure_whole_case(
    run_dir: Path, which: str, out: Path, *, seed: int, score_threshold: float,
    tile_step_size: float = 0.5, use_mirroring: bool = False,
) -> int:
    """The DETERMINISTIC measurement: sliding-window inference over EVERY validation case.

    WHY IT EXISTS. The patch-based path draws `iterations` random patches through nnU-Net's
    `NonDetMultiThreadedAugmenter`, so two runs of one checkpoint differ by up to 0.08 per
    channel -- and two per-channel deltas in the first comparison this tool produced changed
    SIGN on the second measurement. That is not a metric a claim can rest on. Here every case
    is seen, every voxel is counted, and nothing is sampled.

    WHAT IS ADOPTED AND WHAT IS OURS. The sliding window is nnU-Net's own predictor, wired the
    way its `perform_actual_validation` wires it (`MOS-REL-032`: adopt the adopted thing
    whole). What is ours is the counting: nnU-Net compares a written prediction against
    `gt_segmentations` using its label manager, which knows nothing about which channels a case
    annotates. `masked_counts_for_case` does, and that is the whole difference between a number
    the two arms can be compared on and a number that punishes the masked arm for the labels
    nobody drew.

    MEASURED IN PREPROCESSED SPACE, AND THE REPORT SAYS SO. nnU-Net resamples its prediction
    back to the original geometry before scoring; this counts in the space the network ran in,
    against the preprocessed reference. Two reasons: it is the same space the patch-based
    number was taken in, so the two are on one footing and the only difference is the sampling
    noise this removes; and resampling is a second transform whose own correctness would then
    be inside the measurement. A clinical figure wants the original space, and that is a
    declared later change rather than a silent one -- hence `inference.space`.

    NO TEST-TIME MIRRORING BY DEFAULT. `use_mirroring=True` is eight-fold flip augmentation:
    slower, usually better, and a different inference from the one training validated with. It
    belongs to the model as a declared choice, not to the measurement, so it is off unless
    asked for and recorded either way.
    """
    prepared = _prepared(run_dir, seed, which, score_threshold)
    trainer = prepared["trainer"]

    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # DEEP SUPERVISION OFF BEFORE ANYTHING PREDICTS, as nnU-Net's own whole-case validation
    # does. With it on the network returns a LIST of outputs, one per decoder scale, and the
    # predictor's `[0]` -- written to strip a batch dimension -- takes the first SCALE instead,
    # batch axis and all. It failed loudly here: "output with shape [10, 128, 224, 224] doesn't
    # match the broadcast shape [1, 10, 128, 224, 224]". Had the shapes happened to line up it
    # would have measured a decoder head at the wrong resolution and reported it as the model.
    #
    # NOT in `_prepared`, because the patch-based path needs deep supervision ON: it goes
    # through `validation_step`, which knows the output is a list and takes the full-resolution
    # head deliberately.
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=use_mirroring,
        # ACCUMULATE ON THE HOST. One case's logits are [C, Z, Y, X] -- about 3.7 GB at this
        # cohort's median shape with ten channels -- and keeping the assembly on the card
        # invites an OOM that would arrive per case rather than at the start.
        perform_everything_on_device=False,
        device=trainer.device, verbose=False, allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network, trainer.plans_manager, trainer.configuration_manager, None,
        trainer.dataset_json, type(trainer).__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    supervision = json.loads(
        (prepared["work"] / "raw" / DATASET / "supervision.json").read_text(encoding="utf-8")
    )
    channels = list(supervision["channels"])
    label_of = dict(supervision["label_of"])
    per_case = dict(supervision["cases"])

    _train_keys, val_keys = trainer.do_split()
    val_keys = list(val_keys)
    absent = [k for k in val_keys if k not in per_case]
    if absent:
        raise SystemExit(
            f"{len(absent)} validation cases are not in supervision.json ({absent[:3]}...). "
            "Treating them as supervising nothing would drop them from every count while the "
            "case total still said they were measured"
        )
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, val_keys)

    tp: list[float] | None = None
    fp: list[float] | None = None
    fn: list[float] | None = None
    with torch.no_grad():
        for key in val_keys:
            data, segmentation, _seg_prev, _properties = dataset.load_case(key)
            logits = predictor.predict_sliding_window_return_logits(
                torch.from_numpy(np.asarray(data))
            )
            probability = torch.sigmoid(logits.float()).cpu().numpy()
            case = masked_counts_for_case(
                probability, np.asarray(segmentation)[0],
                label_of=label_of, channels=channels,
                supervised=per_case[key], threshold=prepared["applied"],
            )
            tp, fp, fn = (accumulate(tp, case[0]), accumulate(fp, case[1]),
                          accumulate(fn, case[2]))
            del probability, logits

    if tp is None:
        raise SystemExit(
            "the validation split is empty, so nothing was measured. A report of zeros would "
            "read as a model that finds nothing"
        )
    report = metrics_from_counts(tp, fp, fn, channels)
    report.update({
        **_report_head(run_dir, prepared, seed=seed),
        "measurement": "whole_case",
        "cases": len(val_keys),
        #: PROVENANCE FOR THE INFERENCE, on the same argument as the operating point: every
        #: one of these changes the number, so a reader comparing two reports has to be able
        #: to see that they were produced the same way.
        "inference": {
            "tile_step_size": float(tile_step_size),
            "use_gaussian": True,
            "use_mirroring": bool(use_mirroring),
            "space": "preprocessed",
        },
    })
    return _write_report(report, out)


#: THE DEFAULT GRID, AND WHY IT IS SEVEN POINTS AND NOT NINETEEN.
#:
#: Each threshold costs two counts over the values split out of one channel, and the negative
#: side of one channel on this cohort is about 90 million voxels. Nineteen points would be
#: roughly 17 billion comparisons per case -- an hour added to a 26-minute pass, to resolve a
#: curve at 0.05 that seven points already show the shape of.
#:
#: A finer grid is available without that cost by quantising the score to a fixed grid and
#: counting with `bincount`, which is EXACT for thresholds on that grid. It is not here because
#: seven points answer the question in front of us -- whether 0.54 precision is a property of
#: the model or of the 0.5 nobody chose -- and a structure nobody needs yet is a structure with
#: no gate behind it.
CURVE_THRESHOLDS: Final[tuple[float, ...]] = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95)


def operating_point_id(threshold: float) -> str:
    """A name for an operating point that no other threshold can answer to.

    `MOS-SVC-020` makes the supported operating points a property of the service and the
    applied one a property of the `Deployment`; `MOS-SVC-021` carries the id beside every
    reported sensitivity. So the id is an identifier, and two thresholds sharing one is a
    metric attributed to a threshold nobody can resolve.
    """
    per_mille = round(float(threshold) * 1000)
    if not 0 <= per_mille <= 1000:
        raise SystemExit(
            f"an operating point at {threshold!r} is outside [0, 1]: a sigmoid score cannot "
            "reach it, so every count at it would be the same count"
        )
    return "t%04d" % per_mille


def measure_curve(
    run_dir: Path, which: str, out: Path, *, seed: int,
    thresholds: Sequence[float] = CURVE_THRESHOLDS,
    tile_step_size: float = 0.5, use_mirroring: bool = False,
) -> int:
    """Whole-case inference once, counted at several operating points.

    WHAT QUESTION THIS ANSWERS. The whole-case measurement put the masked arm at 0.54 precision
    against the control's 0.84 -- at 0.5, which is the default argument of a counting function
    and was never a choice anybody made. "This model over-predicts" and "this model cannot be
    made precise" are different findings with different consequences, and one number at one
    threshold cannot tell them apart.

    IT PRESENTS AND DOES NOT NOMINATE. No row here is marked best, and that is deliberate:
    `MOS-SVC-020` makes the supported operating points a property of the service and the
    applied one a property of the deployment, and changing an applied `score_threshold`
    requires a new `ServiceVersion` and re-entry into the gate. A tool that picked a row would
    be making a release decision from a table, which is also what `MOS-TRAIN-233` keeps out of
    the search machinery by refusing to let `nominate()` compute its own argument.

    Everything else is `measure_whole_case`'s: the same predictor wired the same way, the same
    per-case masked counting, the same preprocessed space, the same deep-supervision-off step.
    """
    prepared = _prepared(run_dir, seed, which, 0.5)
    trainer = prepared["trainer"]

    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # See `measure_whole_case`: with deep supervision on, the predictor's `[0]` takes the first
    # decoder SCALE rather than the batch element.
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=False, device=trainer.device, verbose=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network, trainer.plans_manager, trainer.configuration_manager, None,
        trainer.dataset_json, type(trainer).__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    supervision = json.loads(
        (prepared["work"] / "raw" / DATASET / "supervision.json").read_text(encoding="utf-8")
    )
    channels = list(supervision["channels"])
    label_of = dict(supervision["label_of"])
    per_case = dict(supervision["cases"])

    _train_keys, val_keys = trainer.do_split()
    val_keys = list(val_keys)
    absent = [k for k in val_keys if k not in per_case]
    if absent:
        raise SystemExit(
            f"{len(absent)} validation cases are not in supervision.json ({absent[:3]}...). "
            "Treating them as supervising nothing would drop them from every count while the "
            "case total still said they were measured"
        )
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, val_keys)

    grid = [float(t) for t in thresholds]
    totals: dict[float, list[list[float] | None]] = {t: [None, None, None] for t in grid}
    done = 0
    with torch.no_grad():
        for key in val_keys:
            data, segmentation, _seg_prev, _properties = dataset.load_case(key)
            logits = predictor.predict_sliding_window_return_logits(
                torch.from_numpy(np.asarray(data))
            )
            probability = torch.sigmoid(logits.float()).cpu().numpy()
            swept = masked_counts_sweep(
                probability, np.asarray(segmentation)[0],
                label_of=label_of, channels=channels,
                supervised=per_case[key], thresholds=grid,
            )
            for threshold, (tp, fp, fn) in swept.items():
                slot = totals[threshold]
                slot[0] = accumulate(slot[0], tp)
                slot[1] = accumulate(slot[1], fp)
                slot[2] = accumulate(slot[2], fn)
            del probability, logits
            done += 1
            # PROGRESS, BECAUSE A LONG RUN THAT PRINTS NOTHING IS A RUN NOBODY CAN TELL FROM A
            # HUNG ONE. The first whole-case pass printed only at the end, and the only way to
            # see it was alive was `nvidia-smi`.
            if done % 10 == 0 or done == len(val_keys):
                print("  %d/%d cases" % (done, len(val_keys)), flush=True)

    if not done:
        raise SystemExit(
            "the validation split is empty, so nothing was measured. A report of zeros would "
            "read as a model that finds nothing"
        )

    points = []
    for threshold in grid:
        tp, fp, fn = totals[threshold]
        point = metrics_from_counts(tp, fp, fn, channels)
        point["operating_point"] = {
            # PER-MILLE AND ZERO-PADDED, so two thresholds cannot share an id. The first
            # spelling was `("%g" % threshold).lstrip("0.")`, which maps 0.1 and 0.01 both to
            # `t1` -- and an operating point id is what a `ServiceVersion` declares and a
            # `Deployment` selects, so two points answering to one name is a deployment
            # applying a threshold nobody can identify.
            "id": "balanced" if threshold == 0.5 else operating_point_id(threshold),
            "score_threshold": threshold,
            "source": "swept",
        }
        points.append(point)

    report = {
        **_report_head(run_dir, prepared, seed=seed),
        "measurement": "curve",
        "cases": done,
        "channels": channels,
        "operating_points": points,
        "inference": {
            "tile_step_size": float(tile_step_size),
            "use_gaussian": True,
            "use_mirroring": bool(use_mirroring),
            "space": "preprocessed",
        },
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("measurement: curve   %d cases   checkpoint %s   epochs %s"
          % (done, report["checkpoint"], report["epochs_done"]))
    print()
    # MACRO AND COVERAGE BESIDE MICRO, because the two answer different questions and reading
    # only the first is what let a channel nobody predicts hide behind a good-looking average
    # for an evening. See `_macro`.
    print("%-10s %8s %8s %8s %8s %8s %6s"
          % ("threshold", "R_micro", "P_micro", "R_macro", "P_macro", "dice_mi", "found"))
    for point in points:
        print("%-10s %8s %8s %8s %8s %8s %4d/%d"
              % (point["operating_point"]["score_threshold"],
                 _cell(point["recall_micro"]).strip(),
                 _cell(point["precision_micro"]).strip(),
                 _cell(point["recall_macro"]).strip(),
                 _cell(point["precision_macro"]).strip(),
                 _cell(point["dice_micro"]).strip(),
                 point["channels_found"], point["channels_measured"]))
    print()
    print("no row is marked best: MOS-SVC-020 makes the applied operating point a deployment's")
    print("choice, and moving it requires a new ServiceVersion and the gate again")
    print("written:", out)
    return 0


#: The permissive threshold candidates are extracted at. Every operating point above it is then
#: free (`MOS-EVID-069`), and the component structure is fixed here -- see
#: `detection.candidates_for_case` on the one approximation that buys this.
EXTRACTION_THRESHOLD: Final[float] = 0.1

#: `MOS-EVID-056`'s convention block needs it, and it decides what counts as a false positive on
#: a case with nothing to find. 0.1 mL is about 12 voxels at this cohort's preprocessed spacing:
#: small enough that a real finding is never below it, large enough that a single stray voxel is
#: not a false positive.
FP_VOLUME_THRESHOLD_ML: Final[float] = 0.1

#: A COMPONENT SMALLER THAN THIS IS NOT A CANDIDATE. Without it, six reference lesions drew 722
#: candidates on `aorta_arch` -- precision 0.058 and every declared FP/scan point out of reach --
#: because a thresholded sigmoid over a 355x512x512 volume speckles. 0.02 mL is about 33 voxels
#: at this cohort's spacing: below any finding a reader would name and far above a speck.
MIN_CANDIDATE_VOLUME_ML: Final[float] = 0.02

#: `MOS-EVID-058` fixes B; the seed is ours and is RECORDED, because an interval nobody can
#: recompute is a decoration.
BOOTSTRAP_SEED: Final[int] = 20260926

#: The registry version these ids are resolved against (`MOS-EVID-054`, `MOS-EVID-070`).
METRIC_REGISTRY_VERSION: Final[int] = 1

#: Which `CaseOverlap` field each registry id is read from.
#:
#: WRITTEN OUT RATHER THAN DERIVED, and the reason is not that a derivation would fail today --
#: stripping `_mean_per_case` does give `dice` and `iou`, and it would work for all six. It is
#: that the id names the AGGREGATION and the field names the quantity, so such a rule makes the
#: report's field access depend on the registry's choice of reduction. `MOS-EVID-054` can rename
#: an aggregation without the measured quantity changing, and then the rule reads a field that
#: does not exist -- or, worse, one that does and means something else.
OVERLAP_FIELDS: Final[dict[str, str]] = {
    "dice_mean_per_case": "dice",
    "iou_mean_per_case": "iou",
    "volume_error_ml": "volume_error_ml",
    "volume_ape": "volume_ape",
    "hd95_mm": "hd95_mm",
    "assd_mm": "assd_mm",
}


def _shape_value(
    row: Any, field: str, patient_of: Mapping[str, str], with_distances: bool
) -> Any:
    """One `CaseOverlap` field as an `evidence.CaseValue`, carrying absence rather than zeroing.

    THREE WAYS A VALUE CAN BE ABSENT HERE AND THEY ARE NOT THE SAME THING:

      * the reference for this case and channel is empty -- `MOS-EVID-051`'s excluded case. The
        reason comes from the row, which already decided it;
      * the run asked for no distances, so HD95 and ASSD were never computed. Absent because
        nobody looked, which is not a property of the model;
      * the prediction's surface is empty -- the model predicted nothing for this case. That IS
        a property of the model and it is a measured miss at the DETECTION endpoint, but it is
        not a boundary agreement of zero millimetres. An HD95 of 0.0 here would be the best
        possible boundary score awarded for predicting nothing.

    All three are `eligible=False` with the reason recorded, so `n` counts what was measured and
    the aggregate cannot average a placeholder.
    """
    from medos_trainer import evidence

    patient_key = patient_of[row.case]
    value = getattr(row, field)
    if not row.eligible:
        return evidence.CaseValue(case=row.case, patient_key=patient_key, value=None,
                                  eligible=False, undefined_reason=row.undefined_reason)
    if value is None:
        distance_field = field in ("hd95_mm", "assd_mm")
        return evidence.CaseValue(
            case=row.case, patient_key=patient_key, value=None, eligible=False,
            undefined_reason=("distances_not_computed"
                              if distance_field and not with_distances
                              else "empty_prediction_surface"),
        )
    return evidence.CaseValue(case=row.case, patient_key=patient_key, value=float(value),
                              eligible=True)


def shape_blocks(
    shapes: Sequence[Any],
    *,
    channels: Sequence[str],
    patient_of: Mapping[str, str],
    operating_point: Mapping[str, Any],
    min_candidate_volume_ml: float,
    with_distances: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The shape half of a clinical report: per channel, six aggregates and one convention block.

    SEPARATE FROM `measure_clinical` SO THAT IT CAN BE TESTED AT ALL. Everything above it in that
    function needs a card, a trained checkpoint and a preprocessed cohort, so a gate on the
    wiring could only ever have been a text search -- and a text search cannot tell a metric that
    is computed from one that is merely mentioned in a comment. This takes rows and returns
    documents, so a test can hand it rows whose right answer is known.

    THE CONVENTION BLOCK IS THIS FUNCTION'S OWN AND SAYS `dice_mean_per_case`. `MOS-EVID-050`
    makes a criterion name one of the two Dice conventions explicitly and supplies no default;
    the detection half of the same report is written under `dice_pooled`. A single global block
    would have stated the wrong convention for half the numbers under it, which is the failure
    the requirement exists to prevent, so each half carries the convention it was computed under.

    A CHANNEL WITH NO ROWS IS ABSENT, not present and empty: no case annotated it, so there is
    nothing to say about it and an empty block would read as a measurement that found nothing.
    """
    from medos_trainer import evidence

    conventions_block = evidence.conventions(
        metric_registry_version=METRIC_REGISTRY_VERSION,
        dice_aggregation="dice_mean_per_case",
        fp_volume_threshold_ml=FP_VOLUME_THRESHOLD_ML,
        bootstrap_seed=BOOTSTRAP_SEED,
        min_candidate_volume_ml=min_candidate_volume_ml,
    )
    blocks: dict[str, dict[str, Any]] = {}
    for channel in channels:
        rows = [row for row in shapes if row.channel == channel]
        if not rows:
            continue
        block: dict[str, Any] = {
            "eligible_cases": sum(1 for row in rows if row.eligible),
            "empty_gt_case_count": sum(1 for row in rows if not row.eligible),
        }
        for metric, field in OVERLAP_FIELDS.items():
            # THE OPERATING POINT IS PASSED THOUGH THE REGISTRY DOES NOT DEMAND IT. All six carry
            # `needs_threshold: False`, so `MOS-EVID-055` would not refuse them without one --
            # but every one is computed from a mask that exists only at a threshold, and a Dice
            # whose threshold is not written beside it cannot be reproduced.
            block[metric] = evidence.aggregate(
                metric,
                [_shape_value(row, field, patient_of, with_distances) for row in rows],
                conventions_block=conventions_block, operating_point=operating_point,
            ).as_document()
        blocks[channel] = block
    return blocks, conventions_block


def _patient_of(run_dir: Path) -> dict[str, str]:
    """case_key -> patient_key, from the SEALED COHORT and from nowhere else.

    `MOS-EVID-057` resamples PATIENTS. Deriving a patient key from a case key by string surgery
    -- stripping a trailing index, say -- is how a cohort whose corpora share patients comes to
    report intervals that are too narrow with nothing anywhere saying so. The cohort declares
    `patient_key` as a required member; this reads it.

    ON THIS COHORT THE TWO ARE EQUAL for all 585 cases: 585 studies, 585 series, no study
    holding two cases. So the cluster bootstrap degenerates to a case-level one here, correctly
    and by the cohort's own record. It stops being correct the moment one patient appears in two
    source corpora, which the run cannot see -- hence `patients_equal_cases` in the report, so a
    reader is told rather than left to assume.
    """
    # TWO NAMES FOR ONE MODULE, BECAUSE THIS TOOL IS MOUNTED INTO AN IMAGE IT DID NOT BUILD.
    #
    # The run-directory contract lives at `medos.sdk.contract` in the repository
    # and at `medos_trainer.contract` in the image currently on the lab, which was built before
    # that move. The evaluator has to run against both -- that is the whole point of mounting it
    # rather than shipping it -- so it tries both and refuses naming both.
    #
    # THE SHIM HAS AN EXPIRY: when the lab image is rebuilt from the current tree, only the
    # first name exists and the second can go. It is a shim and not a fallback because either
    # import yields the SAME constant; a fallback that produced a different value would be the
    # defect this comment would otherwise be hiding.
    #
    # The paths are read from the contract rather than spelled here, because a path written
    # twice is two paths that agree until one is edited.
    RUN_DIRECTORY = None
    for module_name in ("medos.sdk.contract", "medos_trainer.contract"):
        try:
            RUN_DIRECTORY = __import__(
                module_name, fromlist=["RUN_DIRECTORY"]).RUN_DIRECTORY
            break
        except ImportError:
            continue
    if RUN_DIRECTORY is None:
        raise SystemExit(
            "neither medos.sdk.contract nor medos_trainer.contract is "
            "importable, so the cohort's paths cannot be read from the contract that declares "
            "them"
        )

    mapping: dict[str, str] = {}
    for role in ("cohort_fit", "cohort_select"):
        path = run_dir / RUN_DIRECTORY[role]
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            mapping[str(entry["case_key"])] = str(entry["patient_key"])
    if not mapping:
        raise SystemExit(
            f"no cohort rows under {run_dir}. Without patient keys the interval would have to "
            "be computed over cases, which MOS-EVID-057 forbids because it narrows the interval "
            "by a factor invisible in the result"
        )
    return mapping



def write_candidate_records(
    out: Path, detected: Sequence[Any], *, channels: Sequence[str],
    patient_of: Mapping[str, str], extraction_threshold: float,
    min_candidate_volume_ml: float, spacing_mm: Sequence[float],
) -> dict[str, Any]:
    """Persist every extracted candidate beside the report, which is what `MOS-EVID-069` is FOR.

    ITS STATED PURPOSE, QUOTED: the per-candidate record "makes an operating threshold
    re-selectable without re-running inference, which is the property that stops a threshold change
    from becoming a GPU project." Until this function existed the clinical report carried only
    COUNTS, so every question about a threshold, a minimum candidate volume or a component policy
    cost another full inference pass -- measured on this cohort at two to six hours, depending on
    whether the distance transforms run.

    That cost is not hypothetical. The first pass showed 20 to 70 surviving candidates per case for
    channels whose reference is a single structure, and the question that decides what to do about
    it -- specks below a volume floor, or real over-segmentation? -- is answerable from these
    records in seconds and not answerable otherwise.

    A SIDECAR AND NOT AN INLINE MEMBER. At 119 cases, ten channels and up to seventy candidates
    each, inlining would make the report unloadable for the questions it is normally asked. JSON
    Lines, so a reader can stream it and so a partial file is still readable up to its last line.

    THE REFERENCE ROWS ARE HERE TOO, and without them a threshold is only half re-selectable: a
    recomputed sensitivity needs its denominator, and a reference that no candidate touched cannot
    be inferred from the candidate rows at all.

    `centroid_voxel` AND NOT `centroid_lps_mm`, WHICH IS A DECLARED DEPARTURE. The requirement asks
    for LPS millimetres; LPS needs the case's original origin and direction, which this measurement
    does not carry -- it runs in nnU-Net's preprocessed geometry. Recording a voxel centroid under
    the LPS name would be the worse of the two errors, so the member is named for what it holds and
    the spacing that converts it is in the header.
    """
    newline = chr(10)
    header = {
        "record": "candidates",
        "spec": "MOS-EVID-069",
        "extraction_threshold": float(extraction_threshold),
        "min_candidate_volume_ml": float(min_candidate_volume_ml),
        "space": "preprocessed",
        "spacing_mm": [float(s) for s in spacing_mm],
        "centroid_units": "voxel",
        "centroid_departure": (
            "MOS-EVID-069 asks for centroid_lps_mm; LPS needs the case's original origin and "
            "direction, which this measurement does not carry. The centroid is in preprocessed "
            "voxels and `spacing_mm` converts it to millimetres in that space"
        ),
        "channels": list(channels),
    }
    candidates_written = 0
    references_written = 0
    with out.open("w", encoding="utf-8", newline=newline) as stream:
        stream.write(json.dumps(header, ensure_ascii=False) + newline)
        for found in detected:
            for channel in channels:
                for candidate in found.candidates.get(channel, ()):
                    stream.write(json.dumps({
                        "row": "candidate",
                        "case": found.case,
                        "patient_key": patient_of[found.case],
                        "channel": channel,
                        "score": float(candidate.score),
                        "volume_ml": float(candidate.volume_ml),
                        "centroid_voxel": [float(c) for c in candidate.centroid_voxel],
                        "matched_reference_id": candidate.matched_reference_id,
                        "match_distance_mm": candidate.match_distance_mm,
                        "overlap_voxels": int(candidate.overlap_voxels),
                        # THE PER-REFERENCE SCORES, without which a threshold is NOT re-selectable.
                        # A candidate's own score decides whether it survives; its score on each
                        # reference it touches decides whether that reference is found. Persisting
                        # only the first lets a reader recompute the false positives and not the
                        # sensitivity.
                        "overlap_scores": [
                            [reference_id, float(score)]
                            for reference_id, score in candidate.overlap_scores
                        ],
                    }, ensure_ascii=False) + newline)
                    candidates_written += 1
                for reference in found.references.get(channel, ()):
                    stream.write(json.dumps({
                        "row": "reference",
                        "case": found.case,
                        "patient_key": patient_of[found.case],
                        "channel": channel,
                        "reference_id": reference.id,
                        "volume_ml": float(reference.volume_ml),
                        "matched_scores": [float(s) for s in reference.matched_scores],
                    }, ensure_ascii=False) + newline)
                    references_written += 1
    return {
        "path": out.name,
        "candidates": candidates_written,
        "references": references_written,
        "digest": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "header": header,
    }



#: The volume floors compared side by side with the score thresholds. Chosen to bracket what a
#: speck is: 0.02 mL is the extraction floor already in use (about 2.5 voxels at this cohort's
#: spacing), and 1.0 mL is about the smallest thing a radiologist would call a finding.
VOLUME_FLOORS_ML: Final[tuple[float, ...]] = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def read_candidate_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The sidecar's header and rows. Refuses a file that is not one.

    A sidecar is JSON Lines with the header first, so a truncated file is still readable up to its
    last complete line -- which matters because it is written after a pass that takes hours, and a
    reader would rather have 90% of it than a parse error.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty")
    header = json.loads(lines[0])
    if header.get("record") != "candidates":
        raise SystemExit(
            f"{path} does not begin with a candidate-records header; its first line says "
            f"{header.get('record')!r}. A report and its sidecar are different files"
        )
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines[1:], start=2):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A TRUNCATED LAST LINE IS NOT A CORRUPT FILE. Said out loud rather than swallowed:
            # every earlier row is still usable and the reader is told how many were lost.
            print("  (line %d is incomplete and was skipped -- the pass may have been killed)"
                  % number)
            break
    return header, rows



def reselect_counts(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    volume_floor_ml: float = 0.0,
) -> dict[str, Any]:
    """The counts at one operating point, from sidecar rows alone.

    THE RULES ARE `detection.counts_at`'S, DELIBERATELY AND GATED AS SUCH. A reference is found when
    some surviving candidate's score ON THAT REFERENCE'S OWN VOXELS clears the threshold -- not when
    the candidate's own maximum does. The two differ wherever one component spans references of
    different strength, which on this cohort is the pleural effusion and the coronary calcifications,
    and using the second number there reports a lesion found at a threshold where the model predicted
    nothing on it.

    `volume_floor_ml` is the lever `counts_at` does NOT have, and that is the point of having this
    function at all: the extraction floor is fixed when the pass runs, so asking "what would a higher
    floor do" is otherwise another pass. A floor of 0.0 is therefore the configuration under which
    the two must agree exactly, and the gate uses it.
    """
    surviving = [row for row in candidates
                 if row["score"] >= threshold and row["volume_ml"] >= volume_floor_ml]
    found = {
        (row["case"], reference_id)
        for row in surviving
        for reference_id, on_reference in row["overlap_scores"]
        if on_reference >= threshold
    }
    false_positives = sum(1 for row in surviving if row["matched_reference_id"] is None)
    annotating = sorted({row["case"] for row in references})
    return {
        "found": len(found),
        "references": len(references),
        "surviving": len(surviving),
        "true_positives": len(surviving) - false_positives,
        "false_positives": false_positives,
        "annotating_cases": len(annotating),
        "sensitivity": (len(found) / len(references)) if references else None,
        "ppv": ((len(surviving) - false_positives) / len(surviving)) if surviving else None,
        "false_positives_per_case": (false_positives / len(annotating)) if annotating else None,
    }


def reselect(path: Path, *, channel: str | None = None) -> int:
    """Recompute the counts at every threshold and volume floor, from the sidecar ALONE.

    THIS IS WHAT `MOS-EVID-069` IS FOR, in its own words: the per-candidate record "makes an
    operating threshold re-selectable without re-running inference, which is the property that stops
    a threshold change from becoming a GPU project." Measured on this cohort, a pass is two hours
    without the distance transforms and six with; this is a second.

    WHY THE TWO LEVERS ARE PRINTED TOGETHER. The first pass over a trained model showed 20 to 70
    surviving candidates per case for channels whose reference is a single structure -- so precision
    at the operating point is 0.04 to 0.15 while precision under a largest-component filter is
    1.000. Two different remedies follow, and they are not equivalent:

      * a SCORE threshold removes whatever the model is least sure of, wherever it is;
      * a VOLUME floor removes small things regardless of confidence.

    If the scatter is specks, the floor removes it at no cost to sensitivity, and a floor is a
    parameter of the measurement. A component-count policy is not: `MOS-REG-048` puts a component
    filter through the same gate as a weights change, so choosing one is a release decision. Knowing
    which remedy the data supports is therefore worth a column, and until this subcommand existed it
    cost a GPU pass to find out.

    THE COUNTING RULES ARE THE ONES `detection.counts_at` USES, and that is not a coincidence: a
    gate recomputes both from the same fixtures and refuses a disagreement. A reference is found
    when some surviving candidate's score ON THAT REFERENCE'S OWN VOXELS clears the threshold -- not
    when the candidate's own maximum does, which is a different number wherever a component spans
    references of different strength.
    """
    header, rows = read_candidate_records(path)
    channels = [channel] if channel else list(header.get("channels") or [])
    candidates = [row for row in rows if row.get("row") == "candidate"]
    references = [row for row in rows if row.get("row") == "reference"]
    cases = sorted({row["case"] for row in rows})

    print("sidecar: %s" % path)
    print("  %d candidates, %d references, %d cases, extracted at %s, floor %s mL"
          % (len(candidates), len(references), len(cases),
             header.get("extraction_threshold"), header.get("min_candidate_volume_ml")))
    print("  space %s, spacing %s" % (header.get("space"), header.get("spacing_mm")))

    for name in channels:
        mine = [row for row in candidates if row["channel"] == name]
        theirs = [row for row in references if row["channel"] == name]
        if not theirs and not mine:
            continue
        annotating = sorted({row["case"] for row in theirs})
        print()
        print("== %s -- %d references over %d annotating cases, %d candidates"
              % (name, len(theirs), len(annotating), len(mine)))

        unmatched = [row for row in mine if row["matched_reference_id"] is None]
        if unmatched:
            volumes = sorted(row["volume_ml"] for row in unmatched)
            def at(share: float) -> float:
                return volumes[min(len(volumes) - 1, int(share * len(volumes)))]
            print("   false positives at extraction: %d, volume mL "
                  "min %.3f p50 %.3f p90 %.3f max %.3f"
                  % (len(unmatched), volumes[0], at(0.5), at(0.9), volumes[-1]))
            print("   share under 0.1 mL: %.0f%%   under 0.5 mL: %.0f%%"
                  % (100.0 * sum(1 for v in volumes if v < 0.1) / len(volumes),
                     100.0 * sum(1 for v in volumes if v < 0.5) / len(volumes)))

        head = "   %-9s %-9s %9s %9s %9s %7s" % (
            "threshold", "floor mL", "sens", "ppv", "fp/case", "kept")
        print(head)
        for threshold in (0.1, 0.25, 0.5, 0.75, 0.9):
            for floor in VOLUME_FLOORS_ML:
                counted = reselect_counts(mine, theirs, threshold=threshold,
                                          volume_floor_ml=floor)
                print("   %-9s %-9s %9s %9s %9s %7d" % (
                    threshold, floor,
                    "--" if counted["sensitivity"] is None else "%.3f" % counted["sensitivity"],
                    "--" if counted["ppv"] is None else "%.3f" % counted["ppv"],
                    "--" if counted["false_positives_per_case"] is None
                    else "%.2f" % counted["false_positives_per_case"],
                    counted["surviving"],
                ))
    return 0


def measure_clinical(
    run_dir: Path, which: str, out: Path, *, seed: int, operating_point: float,
    extraction_threshold: float = EXTRACTION_THRESHOLD,
    min_candidate_volume_ml: float = MIN_CANDIDATE_VOLUME_ML,
    tile_step_size: float = 0.5, use_mirroring: bool = False, limit: int = 0,
    with_distances: bool = True,
) -> int:
    """Lesion-level AND shape registry metrics over every validation case, from ONE pass.

    WHAT THIS PRODUCES THAT NOTHING BEFORE IT DID: `sensitivity` and `ppv` as the registry
    defines them -- counts over cases, not ratios of voxels -- plus `froc_sensitivity` at the
    declared FP/scan points and the five empty-reference metrics, each with the four companions
    `MOS-EVID-056` requires. Every number before this one was a voxel ratio under a name the
    metric registry does not carry.

    THE CANDIDATES ARE THE ARTEFACT. They are extracted once at `extraction_threshold` and
    persisted, so every operating point and the whole FROC curve are recomputable from the
    report without a card. That is `MOS-EVID-069`'s stated purpose and it is why this replaces
    the threshold sweep rather than joining it.

    AND THE SHAPE BLOCK, WHICH IS NOT RE-READABLE THE SAME WAY. Six of these ten channels have a
    shape endpoint and not a count -- the aorta's only cleared precedent is Dice 0.924 +/- 0.046,
    and a lesion-level "sensitivity 1.000" on a one-component structure says only that the one
    component was found. So `overlap_for_case` runs on the same probability array, in the same
    pass, before it is freed.

    THE ASYMMETRY IS REAL AND IS RECORDED. A candidate list is re-thresholdable; a Dice is not,
    because it needs the binary mask, and HD95 needs two distance transforms over that mask. The
    shape block is therefore measured AT ONE OPERATING POINT, and the report says which. Asking
    for a second one costs another pass, which is why `--no-distances` exists: it keeps Dice and
    the volumes, which are cheap, and drops the two that are not.
    """
    prepared = _prepared(run_dir, seed, which, 0.5)
    trainer = prepared["trainer"]

    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from medos_trainer import detection, evidence, overlap

    # See `measure_whole_case`: with deep supervision on, the predictor's `[0]` takes the first
    # decoder SCALE rather than the batch element.
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=False, device=trainer.device, verbose=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network, trainer.plans_manager, trainer.configuration_manager, None,
        trainer.dataset_json, type(trainer).__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    supervision = json.loads(
        (prepared["work"] / "raw" / DATASET / "supervision.json").read_text(encoding="utf-8")
    )
    channels = list(supervision["channels"])
    label_of = dict(supervision["label_of"])
    per_case = dict(supervision["cases"])
    patient_of = _patient_of(run_dir)

    spacing = list(trainer.configuration_manager.spacing)
    _train_keys, val_keys = trainer.do_split()
    val_keys = list(val_keys)
    unknown = [k for k in val_keys if k not in patient_of]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} validation cases are absent from the sealed cohort "
            f"({unknown[:3]}...), so they have no patient key and cannot be placed in a "
            "bootstrap cluster"
        )
    if limit:
        # A SMOKE PATH AND IT IS RECORDED AS ONE. The whole pass is 45 to 85 minutes and its
        # last step -- the aggregation -- runs only after every case is in, so a defect there
        # costs the entire run. `limit` exercises the whole path on a handful of cases; the
        # report carries `limited_to` so a partial measurement can never be read as a full one.
        val_keys = val_keys[:int(limit)]
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, val_keys)

    detected: list[detection.CaseDetections] = []
    shapes: list[overlap.CaseOverlap] = []
    with torch.no_grad():
        for done, key in enumerate(val_keys, start=1):
            data, segmentation, _seg_prev, _properties = dataset.load_case(key)
            logits = predictor.predict_sliding_window_return_logits(
                torch.from_numpy(np.asarray(data))
            )
            probability = torch.sigmoid(logits.float()).cpu().numpy()
            truth = np.asarray(segmentation)[0]
            detected.append(detection.candidates_for_case(
                probability, truth,
                label_of=label_of, channels=channels, supervised=per_case[key],
                spacing_mm=spacing, extraction_threshold=extraction_threshold,
                min_candidate_volume_ml=min_candidate_volume_ml, case=key,
            ))
            # BOTH ENDPOINTS OFF ONE FORWARD PASS. The pass is 45 to 85 minutes and the shape
            # metrics need the same probabilities the candidates came from, so measuring them in
            # a second run would cost a second pass and -- worse -- would let the two blocks of
            # one report describe two different inferences.
            shapes.extend(overlap.overlap_for_case(
                probability, truth,
                label_of=label_of, channels=channels, supervised=per_case[key],
                spacing_mm=spacing, threshold=float(operating_point), case=key,
                with_distances=with_distances,
            ))
            del probability, logits, truth
            if done % 10 == 0 or done == len(val_keys):
                print("  %d/%d cases" % (done, len(val_keys)), flush=True)

    conventions = evidence.conventions(
        metric_registry_version=METRIC_REGISTRY_VERSION,
        dice_aggregation="dice_pooled",
        fp_volume_threshold_ml=FP_VOLUME_THRESHOLD_ML,
        bootstrap_seed=BOOTSTRAP_SEED,
        min_candidate_volume_ml=min_candidate_volume_ml,
    )
    point = {"name": "probability", "value": float(operating_point), "selected_on": "tune"}

    per_channel: dict[str, Any] = {}
    for channel in channels:
        curve = detection.froc_curve(detected, channel)
        # BOTH POLICIES, NEITHER APPLIED. MOS-REG-048 puts a component filter through the same
        # gate as a weights change, so choosing between them is a release decision and not this
        # tool s business. What the tool can do is make that decision answerable on numbers: one
        # candidate list, read two ways, reported side by side.
        for policy in detection.COMPONENT_POLICIES:
            rows = detection.counts_at(detected, channel, threshold=float(operating_point),
                                       patient_of=patient_of, policy=policy)
            if not rows["sensitivity"]:
                break                     # no case annotates this channel
            kept = 0
            for found in detected:
                surviving = detection._surviving(
                    found.candidates.get(channel, ()), float(operating_point))
                kept += len(surviving) if policy == "all_components" else min(1, len(surviving))
            block = {
                "sensitivity": evidence.aggregate_counted(
                    "sensitivity", rows["sensitivity"], conventions_block=conventions,
                    operating_point=point).as_document(),
                "ppv": evidence.aggregate_counted(
                    "ppv", rows["ppv"], conventions_block=conventions,
                    operating_point=point).as_document(),
                "empty_gt_case_count": len(rows["empty_reference"]),
                "candidates_kept": kept,
            }
            if rows["empty_reference"]:
                volumes = sorted(r.numerator for r in rows["empty_reference"])
                exceeding = sum(1 for v in volumes if v > FP_VOLUME_THRESHOLD_ML)
                block.update({
                    "empty_gt_false_positive_rate": exceeding / len(volumes),
                    "empty_gt_mean_fp_volume_ml": sum(volumes) / len(volumes),
                    "empty_gt_p95_fp_volume_ml": volumes[min(len(volumes) - 1,
                                                             int(0.95 * len(volumes)))],
                    "empty_gt_max_fp_volume_ml": volumes[-1],
                })
            per_channel.setdefault(channel, {})[policy] = block
        if channel in per_channel:
            #: THE CURVE IS OVER ALL COMPONENTS. It describes the model own detection behaviour,
            #: and a curve computed under a filter would be a curve of the filter.
            per_channel[channel]["froc"] = {
                "%g" % fp: detection.froc_at(curve, fp) for fp in detection.FROC_POINTS
            }
            per_channel[channel]["curve_points"] = len(curve)
            per_channel[channel]["candidates"] = sum(
                len(f.candidates.get(channel, ())) for f in detected)
            per_channel[channel]["reference_lesions"] = sum(
                len(f.references.get(channel, ())) for f in detected)

    overlap_blocks, overlap_conventions = shape_blocks(
        shapes, channels=channels, patient_of=patient_of, operating_point=point,
        min_candidate_volume_ml=min_candidate_volume_ml, with_distances=with_distances,
    )
    for channel, block in overlap_blocks.items():
        per_channel.setdefault(channel, {})["overlap"] = block

    report = {
        **_report_head(run_dir, prepared, seed=seed),
        "measurement": "clinical" if not limit else "clinical_smoke",
        "cases": len(val_keys),
        #: Present ONLY on a smoke run, and the measurement name changes too, so
        #: `comparable()` refuses to subtract a partial pass from a full one.
        **({"limited_to": int(limit)} if limit else {}),
        "channels": channels,
        "per_channel": per_channel,
        "conventions": conventions,
        "operating_point": point,
        "extraction_threshold": float(extraction_threshold),
        #: TRUE on this cohort, and the reader has to know: with one patient per case the
        #: cluster bootstrap of MOS-EVID-057 gives exactly what a case-level one would, so the
        #: interval is only as trustworthy as the cohort's patient keys.
        "patients_equal_cases": len(set(patient_of.values())) == len(patient_of),
        #: The shape block's own settings and its own convention block. `space: preprocessed`
        #: is load-bearing and not decoration: `MOS-EVID-107` refuses plausibility geometry in
        #: model space, and a distance in millimetres depends on the grid it was measured on, so
        #: comparing this HD95 against a cleared device's 2.6-5.2 mm needs the resampling closed
        #: first. A Dice is unaffected by the grid; the two distances are.
        "overlap": {
            "threshold": float(operating_point),
            "with_distances": bool(with_distances),
            "space": "preprocessed",
            "spacing_mm": [float(s) for s in spacing],
            "conventions": overlap_conventions,
        },
        "inference": {
            "tile_step_size": float(tile_step_size), "use_gaussian": True,
            "use_mirroring": bool(use_mirroring), "space": "preprocessed",
        },
    }
    # THE CANDIDATE SIDECAR, WRITTEN BEFORE THE REPORT so the report can carry its digest. A
    # report naming a sidecar that does not exist is worse than one that names none.
    report["candidate_records"] = write_candidate_records(
        out.with_suffix(".candidates.jsonl"), detected, channels=channels,
        patient_of=patient_of, extraction_threshold=extraction_threshold,
        min_candidate_volume_ml=min_candidate_volume_ml, spacing_mm=spacing,
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # READ FROM THE REPORT, NOT A LITERAL. This line said "clinical" while the report said
    # "clinical_smoke", so the output claimed a full pass where a partial one had run -- the
    # printed number and the recorded one disagreeing about what was measured.
    print("measurement: %s   %d cases   checkpoint %s   epochs %s"
          % (report["measurement"], len(val_keys), report["checkpoint"],
             report["epochs_done"]))
    print("operating point %s, candidates extracted at %s, components under %s mL dropped"
          % (point["value"], extraction_threshold, min_candidate_volume_ml))
    print()
    for policy in detection.COMPONENT_POLICIES:
        print("-- %s" % policy)
        print("%-24s %18s %18s %8s %8s" % ("channel", "sensitivity [95% CI]",
                                           "ppv [95% CI]", "lesions", "kept"))
        for channel in channels:
            block = (per_channel.get(channel) or {}).get(policy)
            if block is None:
                continue
            print("%-24s %18s %18s %8d %8d" % (
                channel, _interval(block["sensitivity"]), _interval(block["ppv"]),
                per_channel[channel]["reference_lesions"], block["candidates_kept"]))
        print()
    print()
    print("froc_sensitivity at declared FP/scan points (null = the model never reaches it):")
    for channel in channels:
        block = per_channel.get(channel)
        if block is None:
            continue
        cells = " ".join("%s=%s" % (fp, "--" if v is None else "%.3f" % v)
                         for fp, v in block["froc"].items())
        print("  %-24s %s" % (channel, cells))
    print()
    print("-- shape, at probability %g%s" % (
        operating_point, "" if with_distances else " (distances not computed)"))
    head = "%-24s %18s %18s %10s %10s %6s" % (
        "channel", "dice [95% CI]", "hd95 mm [95% CI]", "vol err mL", "vol APE", "n")
    print(head)
    for channel in channels:
        block = (per_channel.get(channel) or {}).get("overlap")
        if block is None:
            continue
        print("%-24s %18s %18s %10s %10s %6d" % (
            channel, _interval(block["dice_mean_per_case"]), _interval(block["hd95_mm"]),
            _plain(block["volume_error_ml"]), _plain(block["volume_ape"]),
            block["dice_mean_per_case"]["n"]))
    print("written:", out)
    return 0


def _plain(document: Mapping[str, Any]) -> str:
    """A value with no interval, for a column too narrow to carry one honestly.

    A DASH FOR `None` AND NOT A ZERO, for the same reason `_interval` does it: this column shows
    `volume_error_ml`, where 0.0 is the best possible answer and "not measurable" is not.
    """
    return "--" if document["value"] is None else "%.3f" % document["value"]


def _interval(document: Mapping[str, Any]) -> str:
    """A value with its interval, or a dash. `MOS-EVID-056`: a bare scalar is not a metric."""
    if document["value"] is None:
        return "%18s" % "--"
    return "%.3f [%.2f-%.2f]" % (document["value"], document["ci_low"], document["ci_high"])


def epochs_done(work: Path) -> int | None:
    """Epochs the TRAINING run completed, not this process's.

    `logs[-1]` -- the newest by name -- was wrong: constructing the trainer creates its own
    timestamped log, so the newest was the evaluator's, empty, and the report said "0
    epochs" against thirty. The training log is the one with the most `Epoch` lines, which
    is true however many times the directory has been opened for measurement.

    `": Epoch " in line` WAS THE SECOND DEFECT AND IT DOUBLED THE COUNT. nnU-Net writes both
    the header `: Epoch 152 ` and, just above the next header, `: Epoch time: 126.8 s `. A
    substring test takes both, so 152 epochs read as 304 -- the same mistake this session
    made once already in a shell one-liner, where it turned 77 into 153. The header is
    matched whole instead, tolerating the trailing space nnU-Net emits; a timing line, a
    prose line mentioning an epoch, or a reformatted log cannot be counted as one.
    """
    logs = sorted(work.glob("results/*/*/*/training_log*.txt"))
    if not logs:
        return None
    return max(
        len(_EPOCH_HEADER.findall(log.read_text(encoding="utf-8", errors="replace")))
        for log in logs
    )


def checkpoint_path(work: Path, which: str) -> Path:
    """The checkpoint the CALLER named, never one this function chose.

    It used to be "best, else final". That is a silent choice and it breaks the comparison:
    `checkpoint_best` is selected by each arm's own metric, so two "best" weights are two
    selections made under different rules -- the same confusion as comparing the arms' own
    printed numbers, hidden one level deeper in which weights were loaded at all.
    """
    names = {"best": "checkpoint_best.pth", "final": "checkpoint_final.pth"}
    if which not in names:
        raise SystemExit(f"unknown checkpoint {which!r}; expected one of {sorted(names)}")
    found = sorted(work.glob(f"results/*/*/*/{names[which]}"))
    if not found:
        tail = (" The fit writes checkpoint_final last, so it is absent until training ends."
                if which == "final" else "")
        raise SystemExit(f"no {names[which]} under {work}/results.{tail}")
    if len(found) > 1:
        raise SystemExit("several %s: %s" % (names[which], [str(f) for f in found]))
    return found[0]


# =====================================================================================
# The comparison
# =====================================================================================
def _cell(value: float | None) -> str:
    """`--` for a quantity with no denominator. Not `0.0000`: nothing to measure and
    measured zero are different statements, and one column must not read as the other."""
    return "          --" if value is None else "%12.4f" % value


def compare(paths: list[Path]) -> int:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    comparable(reports)

    first = reports[0]
    print("preprocessed data agrees: %s" % first["inputs_digest"][:24])
    print("operating point: %s at score_threshold %s -- every number below is measured there"
          % (first["operating_point"]["id"], first["operating_point"]["score_threshold"]))
    # THE EXTENT IS SPELLED BY THE INSTRUMENT THAT TOOK IT. A patch-based report counts
    # iterations and a whole-case one counts cases; printing `iterations` unconditionally
    # raised KeyError on the first whole-case comparison, because the gate for this covered
    # `comparable()` and not the function a person actually calls.
    extent = ("%d cases" % first["cases"] if first.get("measurement") == "whole_case"
              else "%d iterations" % first["iterations"])
    print("%s, %s, seed %d, checkpoint %s, both measured WITH the mask"
          % (first.get("measurement", "?"), extent, first["seed"], first["checkpoint"]))
    # THE NOISE BELONGS TO THE INSTRUMENT, NOT TO THE REPORT. `SAMPLING_NOISE` was measured
    # on the patch-based path, where an augmenter chooses which patches arrive. A whole-case
    # pass sees every case and every voxel, so the same input gives the same output and
    # printing a sampling error for it would invent an uncertainty the measurement does not
    # have -- the mirror image of printing a metric with no threshold.
    if first.get("measurement") == "whole_case":
        print("resolution: deterministic -- every case, every voxel, nothing sampled")
    else:
        print("resolution: micro +-%.3f, per channel +-%.2f -- two measurements of ONE "
              "checkpoint" % (SAMPLING_NOISE["micro"], SAMPLING_NOISE["per_channel"]))
        print("            differ by that much, so a per-channel delta below it is not a "
              "finding")
    for report in reports:
        print("  %-14s trained unmasked: %-5s  epochs %s"
              % (report["run"], report["trained_unmasked"], report["epochs_done"]))

    left, right = reports[0], reports[1]
    for key, label, micro in (
        # RECALL FIRST: it is the claim under test. Dice and precision follow so that
        # "found more" can be told apart from "predicted more".
        ("recall_per_channel", "RECALL (primary)", "recall_micro"),
        ("dice_per_channel", "DICE", "dice_micro"),
        ("precision_per_channel", "PRECISION (control)", "precision_micro"),
    ):
        head = "%-24s%12s%12s%12s" % (label, left["run"][:11], right["run"][:11], "delta")
        print()
        print(head)
        print("-" * len(head))
        for index, channel in enumerate(first["channels"]):
            a, b = left[key][index], right[key][index]
            delta = None if (a is None or b is None) else a - b
            print("%-24s%s%s%s" % (channel, _cell(a), _cell(b), _cell(delta)))
        print("-" * len(head))
        a, b = left[micro], right[micro]
        delta = None if (a is None or b is None) else a - b
        print("%-24s%s%s%s" % ("micro", _cell(a), _cell(b), _cell(delta)))
        if micro == "recall_micro":
            # The macro row under the micro one, and the coverage under both: micro answers
            # how much of the reference was found, macro which findings can be found at all,
            # and the count says plainly when one cannot be found anywhere.
            ma, mb = left.get("recall_macro"), right.get("recall_macro")
            if ma is not None or mb is not None:
                md = None if (ma is None or mb is None) else ma - mb
                print("%-24s%s%s%s" % ("macro", _cell(ma), _cell(mb), _cell(md)))
                print("%-24s%10s%10s" % ("channels found",
                                          "%d/%d" % (left["channels_found"], left["channels_measured"]),
                                          "%d/%d" % (right["channels_found"], right["channels_measured"])))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B evaluator: one rule, two networks")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("measure", help="measure ONE arm (one process per arm)")
    one.add_argument("--run-dir", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--iterations", type=int, default=1000)
    one.add_argument("--seed", type=int, default=20260925)
    # NO DEFAULT, DELIBERATELY: which weights are measured is part of the claim, not a
    # convenience. See `checkpoint_path`.
    one.add_argument("--checkpoint", required=True, choices=["best", "final"])
    # DEFAULTED, unlike `--checkpoint`, because 0.5 is what the trainer's own validation
    # statistic has always used and a measurement that silently moved off it would not be
    # comparable with any number already recorded. It is RECORDED either way.
    one.add_argument("--score-threshold", type=float, default=0.5)

    cases = sub.add_parser(
        "measure-cases",
        help="measure ONE arm over every validation case, deterministically",
    )
    cases.add_argument("--run-dir", required=True)
    cases.add_argument("--out", required=True)
    cases.add_argument("--seed", type=int, default=20260925)
    cases.add_argument("--checkpoint", required=True, choices=["best", "final"])
    cases.add_argument("--score-threshold", type=float, default=0.5)
    cases.add_argument("--tile-step-size", type=float, default=0.5)
    # OFF BY DEFAULT and recorded either way: test-time mirroring is a different inference
    # from the one training validated with. See `measure_whole_case`.
    cases.add_argument("--mirroring", action="store_true")

    curve = sub.add_parser(
        "curve", help="measure ONE arm over every case at several operating points"
    )
    curve.add_argument("--run-dir", required=True)
    curve.add_argument("--out", required=True)
    curve.add_argument("--seed", type=int, default=20260925)
    curve.add_argument("--checkpoint", required=True, choices=["best", "final"])
    curve.add_argument("--tile-step-size", type=float, default=0.5)
    curve.add_argument("--mirroring", action="store_true")
    curve.add_argument(
        "--thresholds", type=float, nargs="+", default=list(CURVE_THRESHOLDS),
        help="operating points to count at; each costs comparisons, not another pass",
    )

    clinical = sub.add_parser(
        "clinical",
        help="lesion-level registry metrics over every case, from one inference pass",
    )
    clinical.add_argument("--run-dir", required=True)
    clinical.add_argument("--out", required=True)
    clinical.add_argument("--seed", type=int, default=20260925)
    clinical.add_argument("--checkpoint", required=True, choices=["best", "final"])
    clinical.add_argument("--operating-point", type=float, default=0.5)
    clinical.add_argument("--extraction-threshold", type=float,
                          default=EXTRACTION_THRESHOLD)
    clinical.add_argument("--min-candidate-volume-ml", type=float,
                          default=MIN_CANDIDATE_VOLUME_ML,
                          help="a component smaller than this is not a candidate; recorded in "
                               "the convention block because it changes what the number means")
    clinical.add_argument("--tile-step-size", type=float, default=0.5)
    clinical.add_argument("--mirroring", action="store_true")
    clinical.add_argument(
        "--no-distances", action="store_true",
        help="skip HD95 and ASSD (two distance transforms per channel per case); Dice and the "
             "volumes still come out, and the report records that nobody looked")
    clinical.add_argument("--limit", type=int, default=0,
                          help="smoke: measure only the first N cases; the report says so")

    reselection = sub.add_parser(
        "reselect",
        help="recompute counts at every threshold and volume floor from a candidate sidecar")
    reselection.add_argument("--candidates", required=True,
                             help="the .candidates.jsonl written beside a clinical report")
    reselection.add_argument("--channel", default=None,
                             help="one channel, or every channel the sidecar names")
    two = sub.add_parser("compare", help="compare two finished reports")
    two.add_argument("reports", nargs=2)

    args = parser.parse_args(argv)
    if args.command == "measure":
        return measure(Path(args.run_dir), args.iterations, args.seed,
                       args.checkpoint, Path(args.out),
                       score_threshold=args.score_threshold)
    if args.command == "reselect":
        return reselect(Path(args.candidates), channel=args.channel)
    if args.command == "clinical":
        return measure_clinical(
            Path(args.run_dir), args.checkpoint, Path(args.out), seed=args.seed,
            operating_point=args.operating_point,
            extraction_threshold=args.extraction_threshold,
            min_candidate_volume_ml=args.min_candidate_volume_ml,
            tile_step_size=args.tile_step_size, use_mirroring=args.mirroring,
            limit=args.limit, with_distances=not args.no_distances,
        )
    if args.command == "curve":
        return measure_curve(
            Path(args.run_dir), args.checkpoint, Path(args.out), seed=args.seed,
            thresholds=args.thresholds, tile_step_size=args.tile_step_size,
            use_mirroring=args.mirroring,
        )
    if args.command == "measure-cases":
        return measure_whole_case(
            Path(args.run_dir), args.checkpoint, Path(args.out), seed=args.seed,
            score_threshold=args.score_threshold, tile_step_size=args.tile_step_size,
            use_mirroring=args.mirroring,
        )
    return compare([Path(p) for p in args.reports])


if __name__ == "__main__":
    sys.exit(main())
