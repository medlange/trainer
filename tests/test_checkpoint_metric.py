# SPDX-License-Identifier: Apache-2.0
"""`checkpoint_best` is selected on a number that means the same thing every epoch.

REGISTER ENTRY 102, AND WHAT IT COST TO FIND
--------------------------------------------
nnU-Net picks the best checkpoint on `np.nanmean(dice_per_class)`. On a fully labelled
dataset every channel has validation data every epoch, so that mean is over a fixed set
and two epochs' values are comparable. Under PARTIAL supervision they are not: a channel
is `nan` whenever the epoch's validation iterations drew no patch from a case supervised
for it, and with 18 validation cases and roughly 3 supervised per channel that happens
constantly.

Measured over 70 epochs of the ten-channel fit: the mean was taken over FOURTEEN DISTINCT
CHANNEL SUBSETS, between 3 and 9 channels wide -- 20 epochs averaged 4 channels, 14
averaged 7. And it was biased, not merely noisy: `vertebral_body` and `pleural_effusion`,
the structures the model learns earliest and best, appeared in none of the four most
common subsets, while `coronary_calcification` at 4,971 voxels appeared in all of them.

A direct probe of the epoch-50 checkpoint scored `vertebral_body` at Dice 0.865 on
foreground-centred validation patches while the trainer's own log reported `nan` for it
in every epoch. The first hypothesis was that `masked_tp_fp_fn_tn` was wrong. IT IS NOT --
reading a real batch showed the target arriving as `(2, 10, 128, 128, 128)` bool,
correctly one-hot, and the metric computing correctly on correct inputs. The defect is in
what the metric is averaged OVER. The disproved hypothesis is written down because the
next person will have it too.

Needs torch and nnunetv2. A hard import, like its sibling: a missing pin is a failure
here, not a skip.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.masked_trainer import nnUNetTrainerMaskedChannels  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.entries: dict[str, list] = {}

    def log(self, key, value, epoch) -> None:  # noqa: ANN001
        self.entries.setdefault(key, []).append(value)


class _Trainer:
    """The method under test, with only what it touches.

    Constructing a real `nnUNetTrainerMaskedChannels` needs plans, a dataset.json and a
    preprocessed tree on disk; the method reads four attributes and calls two. Binding
    the unbound function keeps this a test OF that method rather than of a copy.
    """

    on_validation_epoch_end = nnUNetTrainerMaskedChannels.on_validation_epoch_end

    def __init__(self, channels: list[str]) -> None:
        self._channels = channels
        self.current_epoch = 0
        self.logger = _Logger()
        self.messages: list[str] = []

    def print_to_log_file(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.messages.append(" ".join(str(a) for a in args))


CHANNELS = ["a", "b", "c"]


def _outputs(rows):
    """rows: (tp, fp, fn) per validation iteration, each a list over channels."""
    return [
        {
            "loss": np.float32(0.5),
            "tp_hard": np.array(tp, dtype=np.float64),
            "fp_hard": np.array(fp, dtype=np.float64),
            "fn_hard": np.array(fn, dtype=np.float64),
        }
        for tp, fp, fn in rows
    ]


def _run(rows):
    trainer = _Trainer(list(CHANNELS))
    trainer.on_validation_epoch_end(_outputs(rows))
    return trainer


def _nanmean_of(rows):
    """What nnU-Net computes, reproduced so the tests can show the two disagree."""
    tp = np.sum([r[0] for r in rows], 0).astype(float)
    fp = np.sum([r[1] for r in rows], 0).astype(float)
    fn = np.sum([r[2] for r in rows], 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
    return np.nanmean(per)


def test_the_checkpoint_metric_is_defined_when_channels_are_absent() -> None:
    """Channel `c` saw no supervised pair; `a` and `b` did. The metric must still exist."""
    trainer = _run([([10, 20, 0], [1, 2, 0], [1, 2, 0])])
    micro = trainer.logger.entries["mean_fg_dice"][0]
    assert np.isfinite(micro), "the checkpoint metric must not be nan while data exists"
    assert micro == pytest.approx(60 / 66)  # 2*30 / (2*30 + 3 + 3)


#: TWO EPOCHS OF THE SAME MODEL, differing only in whether the epoch's validation
#: sampling happened to draw the one tiny channel.
#:
#: `a` is a large structure the model segments perfectly: 10,000 true positives.
#: `c` is a tiny one it gets wrong: five voxels, all of them wrong. In the real fit these
#: are `pleural_effusion` at 20.4M voxels and `coronary_calcification` at 4,971 -- a ratio
#: of four thousand to one, which a macro-average flattens to one-to-one.
_SAMPLED_THE_TINY_CHANNEL = [([10000, 0, 0], [0, 0, 5], [0, 0, 5])]
_DID_NOT = [([10000, 0, 0], [0, 0, 0], [0, 0, 0])]


def test_the_checkpoint_metric_barely_moves_when_a_tiny_channel_is_sampled() -> None:
    """THE DEFECT, stated as the comparison that actually happens. The model is identical
    in both epochs. All that differs is whether ten voxels of a tiny channel were drawn.
    A metric that swings on that is selecting checkpoints on sampling luck."""
    with_tiny = _run(_SAMPLED_THE_TINY_CHANNEL).logger.entries["mean_fg_dice"][0]
    without = _run(_DID_NOT).logger.entries["mean_fg_dice"][0]
    assert abs(with_tiny - without) < 0.01, (
        f"the checkpoint metric moved {abs(with_tiny - without):.3f} because ten voxels "
        f"of one tiny channel were sampled. It is weighted by evidence and should barely "
        f"notice."
    )


def test_nanmean_would_have_halved_on_the_same_pair() -> None:
    """Without this, the test above could pass for a metric that is merely constant. It
    shows nnU-Net's own statistic swings from 1.0 to 0.5 on the same two epochs -- the
    model unchanged, ten voxels of sampling difference, and the checkpoint decision
    inverted."""
    with_tiny = _nanmean_of(_SAMPLED_THE_TINY_CHANNEL)
    without = _nanmean_of(_DID_NOT)
    assert without == pytest.approx(1.0)
    assert with_tiny == pytest.approx(0.5)
    assert abs(with_tiny - without) > 0.4, (
        "nnU-Net's nanmean should swing hard here; if it no longer does, this fixture no "
        "longer reproduces the defect and this file proves nothing."
    )


def test_the_per_channel_numbers_are_still_reported() -> None:
    """The micro-average is for the CHECKPOINT. A human reads per-channel Dice, and
    replacing it would trade one blind spot for another."""
    trainer = _run([([10, 20, 0], [1, 2, 0], [1, 2, 0])])
    per_class = trainer.logger.entries["dice_per_class_or_region"][0]
    assert len(per_class) == len(CHANNELS)
    assert np.isfinite(per_class[0]) and np.isfinite(per_class[1])
    assert not np.isfinite(per_class[2]), "an unmeasured channel must stay nan, not become 0"


def test_the_absent_channels_are_named_in_the_log() -> None:
    """A row of `nan` tells a reader the model scored nothing. The message says the epoch
    MEASURED nothing for those channels, which is a different fact."""
    trainer = _run([([10, 20, 0], [1, 2, 0], [1, 2, 0])])
    assert trainer.messages, "nothing was logged about the absent channel"
    # Searched, not indexed: the IoU lines are printed alongside, and a test that
    # depends on which one comes first breaks on a reordering that changes nothing.
    line = next((m for m in trainer.messages if "validation saw no" in m), None)
    assert line is not None, trainer.messages
    assert "'c'" in line, line
    assert "1/3" in line, line


def test_an_epoch_that_measured_nothing_is_nan_and_not_zero() -> None:
    """`0.0` would claim the model scored zero and `_best_ema` would treat it as a real,
    very bad epoch. nan says the epoch measured nothing."""
    trainer = _run([([0, 0, 0], [0, 0, 0], [0, 0, 0])])
    assert np.isnan(trainer.logger.entries["mean_fg_dice"][0])


def test_a_perfect_epoch_scores_one_and_a_useless_one_scores_zero() -> None:
    """Bounds, so the micro-average cannot be a constant satisfying everything above."""
    perfect = _run([([10, 10, 10], [0, 0, 0], [0, 0, 0])])
    assert perfect.logger.entries["mean_fg_dice"][0] == pytest.approx(1.0)
    useless = _run([([0, 0, 0], [5, 5, 5], [5, 5, 5])])
    assert useless.logger.entries["mean_fg_dice"][0] == pytest.approx(0.0)


def test_counts_are_summed_across_validation_iterations() -> None:
    """nnU-Net accumulates over the epoch before dividing. Computing Dice per iteration
    and averaging is a different, worse statistic on small structures."""
    one_batch = _run([([20, 0, 0], [2, 0, 0], [2, 0, 0])])
    two_batches = _run([([10, 0, 0], [1, 0, 0], [1, 0, 0])] * 2)
    assert one_batch.logger.entries["mean_fg_dice"][0] == pytest.approx(
        two_batches.logger.entries["mean_fg_dice"][0]
    )


def test_the_three_logger_keys_nnunet_reads_are_all_written() -> None:
    """`on_validation_epoch_end` is an override: nnU-Net's plotting and its
    `_best_ema` selection read these three keys by name, and an override that quietly
    stops writing one degrades the run without failing it."""
    trainer = _run([([10, 20, 0], [1, 2, 0], [1, 2, 0])])
    assert set(trainer.logger.entries) == {
        "mean_fg_dice",
        "dice_per_class_or_region",
        "val_losses",
    }


# --------------------------------------------------------------------------------------
# IoU, and the arithmetic that has to be right
# --------------------------------------------------------------------------------------
#
# nnU-Net reports Dice; segmentation literature reports IoU. For the SAME confusion
# counts the two are one function of each other -- `IoU = Dice / (2 - Dice)` -- so
# logging IoU adds no information, only the number people compare against. Doing that
# conversion by hand is how a model gets recorded against the wrong baseline.
#
# The first implementation wrote `2*tp / (2*tp + fp + fn + tp)`, which returns 0.625
# where the answer is 0.8333. It was caught by checking it against the identity rather
# than by reading it, which is the only reason these tests exist as well as the code.


def _reported_iou(trainer) -> list[float]:  # noqa: ANN001
    for message in trainer.messages:
        if message.startswith("IoU per class "):
            # `nan` is a NAME to the parser, not a literal, so it cannot be
            # `literal_eval`ed directly. Swapping it for None keeps the parse total
            # and preserves "this channel was not measured" as a distinct value.
            body = message[len("IoU per class ") :].replace("nan", "None")
            return [float("nan") if v is None else v for v in ast.literal_eval(body)]
    raise AssertionError(f"no per-class IoU line in {trainer.messages}")


def _reported_micro_iou(trainer) -> float:  # noqa: ANN001
    for message in trainer.messages:
        if message.startswith("IoU (micro"):
            return float(message.rsplit(" ", 1)[1])
    raise AssertionError("no micro IoU line")


@pytest.mark.parametrize(
    ("tp", "fp", "fn"),
    [(30, 3, 3), (100, 0, 0), (0, 5, 5), (7, 2, 9), (1, 0, 0), (12345, 678, 910)],
)
def test_iou_equals_dice_over_two_minus_dice(tp, fp, fn) -> None:
    """The identity, on the reported numbers rather than on a reimplementation."""
    trainer = _run([([tp, 0, 0], [fp, 0, 0], [fn, 0, 0])])
    iou = _reported_iou(trainer)[0]
    dice = trainer.logger.entries["dice_per_class_or_region"][0][0]
    # `abs=5e-5` because the log rounds to four decimals. Comparing to full float
    # precision would be comparing against a number the log does not carry.
    assert iou == pytest.approx(tp / (tp + fp + fn), abs=5e-5)
    assert iou == pytest.approx(dice / (2 - dice), abs=5e-5), (
        f"Dice {dice} and IoU {iou} are inconsistent; for one set of counts they are "
        f"one function of each other."
    )


def test_iou_is_never_greater_than_dice() -> None:
    """A cheap invariant that the wrong formula also violated in the other direction:
    IoU <= Dice always, with equality only at 0 and 1."""
    trainer = _run([([30, 10, 1], [3, 40, 0], [3, 5, 0])])
    for iou, dice in zip(
        _reported_iou(trainer), trainer.logger.entries["dice_per_class_or_region"][0]
    ):
        if iou == iou and dice == dice:  # skip nan
            assert iou <= dice + 1e-12, f"IoU {iou} exceeds Dice {dice}"


def test_a_channel_with_no_supervised_pair_reports_nan_iou_not_zero() -> None:
    """Same rule as the Dice column: `0.0` claims the model scored nothing, `nan` says
    the epoch measured nothing, and they are different facts."""
    trainer = _run([([10, 20, 0], [1, 2, 0], [1, 2, 0])])
    iou = _reported_iou(trainer)
    assert iou[0] == pytest.approx(10 / 12, abs=5e-5)
    assert iou[1] == pytest.approx(20 / 24, abs=5e-5)
    assert iou[2] != iou[2], "an unmeasured channel must be nan"


def test_the_micro_iou_matches_the_micro_dice() -> None:
    """The checkpoint metric is a micro-averaged Dice; its IoU twin must agree with it
    through the same identity, or the two headline numbers describe different runs."""
    trainer = _run([([30, 15, 0], [3, 2, 0], [3, 2, 0])])
    micro_dice = trainer.logger.entries["mean_fg_dice"][0]
    assert _reported_micro_iou(trainer) == pytest.approx(
        micro_dice / (2 - micro_dice), abs=1e-4
    )
