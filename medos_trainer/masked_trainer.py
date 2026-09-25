# SPDX-License-Identifier: Apache-2.0
"""An nnU-Net trainer that honours a per-case channel-availability mask.

WHAT IT CHANGES, AND NOTHING ELSE
----------------------------------
Four overrides on `nnUNetTrainer`. Everything else -- planning, preprocessing, the
architecture, the optimiser, the augmentation pipeline -- is stock, deliberately, because
every line of nnU-Net this file replaces is a line whose behaviour is no longer the
behaviour nnU-Net's results were measured with.

  __init__          forces region mode. See below.
  _build_loss       MaskedDiceBCELoss instead of DC_and_BCE_loss.
  train_step        passes the mask to the loss.
  validation_step   passes the mask to the loss AND to the pseudo-Dice.

THE FORCED REGION MODE IS A DECLARED DEVIATION -- register entry 95.
`LabelManager` sets `_has_regions` from `any(isinstance(v, (tuple, list)) and len(v) > 1)`.
A label_set with one finding per channel has singleton lists, so it reports False and gets
`softmax_helper_dim0`. On a softmax head this whole approach is incoherent: the outputs sum
to one, so masking a channel out of the loss still changes what the remaining channels must
predict, and "unknown" cannot mean unknown. Forcing the flag and recomputing the regions --
in the same order `LabelManager.__init__` does -- is narrower than the alternative, which
is to write `"neo": [1, 1]` in dataset.json and lie to the library about the data in order
to obtain the behaviour we want.

WHY validation_step IS OVERRIDDEN AND NOT ONLY THE LOSS
--------------------------------------------------------
nnU-Net's online pseudo-Dice drives `checkpoint_best`. An unsupervised channel reports
I = P = G = 0, which the pseudo-Dice reads as perfect agreement -- so with a perfectly
masked loss and an unmasked metric, the shipped weights would still be selected partly by
the absence of labels. `get_tp_fp_fn_tn` already takes a `mask` argument and multiplies it
in, so the per-channel mask broadcasts into it with no change to nnU-Net.

THE ECHO-BACK COUNTER
---------------------
`applied_supervised_pairs` accumulates the (case, channel) pairs the fit ACTUALLY masked
in. It exists so the platform can compare what was applied against what the sealed record
says was intended. An all-ones mask and a correctly-varied one both have "masking enabled";
only the count tells them apart, and a mask that silently failed to load would otherwise
produce a clean, plausible, wrong run.

Spec: MOS-REL-032 (declared deviation), MOS-TRAIN-141. Register entries 95, 98.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from torch import autocast

from medos_trainer.masked import (
    MaskedDeepSupervisionWrapper,
    MaskedDiceBCELoss,
    masked_tp_fp_fn_tn,
    supervised_pair_count,
)

#: Where `medos/tools/ingest/nnunet_dataset.py` writes the mask, beside dataset.json.
SUPERVISION_FILE = "supervision.json"


def force_region_mode(manager: LabelManager) -> LabelManager:
    """Make a singleton-list label set behave as regions. Register entry 95."""
    manager._has_regions = True
    manager._regions = manager._get_regions()
    manager.inference_nonlin = torch.sigmoid
    return manager


class nnUNetTrainerMaskedChannels(nnUNetTrainer):
    """Multi-label sigmoid heads, supervised per (case, channel)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")) -> None:
        # The signature is nnUNetTrainer's, transcribed rather than absorbed into **kwargs.
        # nnUNetv2_train constructs the trainer by keyword, so a missing parameter is a
        # TypeError at construction -- which is the good case. **kwargs would accept
        # anything, including a parameter nnU-Net adds in a later version that this
        # subclass then silently drops on the floor.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # FORCE REGION MODE FOR EVERY LabelManager BUILT FROM THESE PLANS, not only the
        # trainer's own. Register entry 98, confirmed by running it.
        #
        # `perform_actual_validation` constructs an `nnUNetPredictor` and hands it
        # `self.plans_manager`; the predictor then calls `get_label_manager(dataset_json)`
        # itself and gets an UNFORCED manager, which reports C+1 segmentation heads for a
        # C-head network and dies allocating the logits buffer. Forcing only
        # `self.label_manager` fixes the fit and leaves inference broken -- which is
        # exactly what happened: the first model trained to completion and then could not
        # predict a single case.
        #
        # Patched on the INSTANCE rather than the class so it cannot leak into another
        # trainer in the same process, and wrapping the bound method rather than
        # reimplementing it so a change to how nnU-Net builds label managers is inherited
        # rather than silently diverged from.
        _build_label_manager = self.plans_manager.get_label_manager

        def _forced(dataset_json_: dict, **kwargs):
            return force_region_mode(_build_label_manager(dataset_json_, **kwargs))

        self.plans_manager.get_label_manager = _forced
        self.label_manager = force_region_mode(self.label_manager)

        #: Epoch budget. nnU-Net's 1000 is right for a real run and wrong for a first one;
        #: this is the only knob, it is read once, and it is printed into the log so a
        #: short run cannot be mistaken later for a full one.
        self.num_epochs = int(os.environ.get("MEDOS_TRAINER_EPOCHS", self.num_epochs))

        self._channels, self._supervision = self._load_supervision()
        self.applied_supervised_pairs = 0
        self.seen_cases: set[str] = set()

    # -- the mask ------------------------------------------------------------------

    def _load_supervision(self) -> tuple[list[str], dict[str, list[str]]]:
        """Read the mask. A MISSING FILE IS A HARD FAILURE, never an all-ones default.

        An all-ones fallback is the single most dangerous line this file could contain: the
        fit would run, converge, report sensible numbers and be trained on the exact
        false-negative signal the mask exists to remove -- and nothing downstream could
        tell. Refusing is the only safe behaviour.
        """
        raw = os.environ.get("nnUNet_raw")
        if not raw:
            raise RuntimeError("nnUNet_raw is unset; cannot locate " + SUPERVISION_FILE)
        # THE DATASET'S OWN FILE, OR A REFUSAL. Never "the first one found".
        #
        # The first version fell back to `sorted(glob("*/supervision.json"))[0]` when it
        # could not resolve the dataset name, and the moment a second dataset existed it
        # loaded Dataset501's mask into a Dataset599 run. The KeyError in `_mask_for`
        # caught it -- a case in the batch was absent from the table -- but only because
        # the two datasets happen to use different case keys. Two datasets sharing a key
        # would have trained happily under the wrong mask, which is the failure this whole
        # file exists to make impossible. So: the dataset's own path, and if that is not
        # there, say what was found rather than choose.
        name = getattr(self.plans_manager, "dataset_name", None)
        expected = Path(raw) / str(name) / SUPERVISION_FILE if name else None
        if expected is not None and expected.is_file():
            candidates = [expected]
        else:
            found = sorted(Path(raw).glob(f"*/{SUPERVISION_FILE}"))
            if len(found) == 1:
                candidates = found
            elif found:
                listed = "\n  ".join(str(f) for f in found)
                raise FileNotFoundError(
                    f"this run is for dataset {name!r} and there is no "
                    f"{SUPERVISION_FILE} at {expected}. {len(found)} other(s) exist:\n"
                    f"  {listed}\n"
                    f"Picking one would mean training under another dataset's channel "
                    f"mask. Build this dataset with medos/tools/ingest/nnunet_dataset.py, "
                    f"which writes the file beside dataset.json."
                )
            else:
                candidates = []
        if not candidates:
            raise FileNotFoundError(
                f"no {SUPERVISION_FILE} under {raw}. This trainer will not fall back to "
                f"supervising every channel on every case: that is precisely the "
                f"false-negative signal it exists to remove, and a run that did it would "
                f"look completely normal. Build the dataset with "
                f"medos/tools/ingest/nnunet_dataset.py, which writes the file."
            )
        document = json.loads(candidates[0].read_text(encoding="utf-8"))
        channels = list(document["channels"])
        cases = {k: list(v) for k, v in document["cases"].items()}
        self.print_to_log_file(
            f"supervision: {candidates[0]} -- {len(cases)} cases, channels {channels}, "
            f"empty_segment_is_negative={document.get('empty_segment_is_negative')}"
        )
        for channel in channels:
            n = sum(1 for v in cases.values() if channel in v)
            self.print_to_log_file(f"  {channel}: supervised on {n}/{len(cases)} cases")
        return channels, cases

    def _mask_for(self, keys: Any, batch: int, device: torch.device) -> torch.Tensor:
        """[B, C] of 1.0 where this case supervises this channel.

        A key absent from the table raises rather than defaulting, for the same reason
        `_load_supervision` refuses a missing file.
        """
        mask = torch.zeros((batch, len(self._channels)), dtype=torch.float32, device=device)
        for row, key in enumerate(list(keys)[:batch]):
            key = str(key)
            try:
                supervised = self._supervision[key]
            except KeyError:
                raise KeyError(
                    f"case {key!r} is in the batch but not in {SUPERVISION_FILE}. Treating "
                    f"it as fully supervised would silently train every channel on a case "
                    f"nobody declared."
                ) from None
            self.seen_cases.add(key)
            for column, channel in enumerate(self._channels):
                if channel in supervised:
                    mask[row, column] = 1.0
        return mask

    # -- the loss ------------------------------------------------------------------

    def _build_loss(self):  # noqa: ANN201 - matches the base signature
        loss = MaskedDiceBCELoss(
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            weight_dice=1.0,
            weight_bce=1.0,
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(scales))])
            weights[-1] = 0          # nnU-Net drops the coarsest head
            weights = weights / weights.sum()
            # OUR OWN WRAPPER, WHICH TAKES A MASK, rather than nnU-Net's, which does not.
            #
            # `MaskedDeepSupervisionWrapper` was written for this, perturbation-tested at
            # every decoder scale, and then not used: `_build_loss` reached for nnU-Net's
            # wrapper instead and paid for it in `_call_loss`, which had to synthesise
            # `[mask] * len(output)` so that a third per-scale list would survive
            # `forward(*args)`'s `zip(*args)`. That worked, and it worked by depending on
            # the argument-forwarding of a third-party class that knows nothing about
            # masks -- a dependency that needed its own test
            # (`trainer/tests/test_nnunet_internals.py`) to pin somebody else's internals,
            # and that would break silently on any upstream change to a signature nobody
            # upstream promised.
            #
            # The mask is scale-invariant: whether a reader annotated a channel is a fact
            # about the CASE, not about the resolution a decoder head runs at. Our wrapper
            # says that in its own docstring and applies one [B,C] mask at every scale.
            loss = MaskedDeepSupervisionWrapper(loss, weights)
        return loss

    def _call_loss(self, output, target, mask):
        """One call shape, wrapped or not, so train and validation cannot diverge.

        THE BRANCH IS GONE, and that is the point of the wrapper swap above. Both the bare
        `MaskedDiceBCELoss` and `MaskedDeepSupervisionWrapper` take `(prediction, target,
        mask)`; the wrapper is the one that knows a mask is scale-invariant and hands the
        same one to every scale. While nnU-Net's wrapper was in the path this function had
        to know which of the two it was calling and reshape the mask for one of them.
        """
        return self.loss(output, target, mask)

    # -- the steps -----------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([t.to(self.device, non_blocking=True) for t in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        mask = self._mask_for(batch["keys"], data.shape[0], self.device)
        self.applied_supervised_pairs += supervised_pair_count(mask)

        self.optimizer.zero_grad(set_to_none=True)
        context = (autocast(self.device.type, enabled=True)
                   if self.device.type == "cuda" else dummy_context())
        with context:
            output = self.network(data)
            loss = self._call_loss(output, target, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([t.to(self.device, non_blocking=True) for t in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        mask = self._mask_for(batch["keys"], data.shape[0], self.device)

        context = (autocast(self.device.type, enabled=True)
                   if self.device.type == "cuda" else dummy_context())
        with context:
            output = self.network(data)
            del data
            loss = self._call_loss(output, target, mask)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # THE MASKED PSEUDO-DICE, computed by `masked_tp_fp_fn_tn` and NOT by nnU-Net's
        # `get_tp_fp_fn_tn`.
        #
        # Theirs cannot take this mask. Its docstring says "mask must have shape
        # (b, 1, x, y(, z))" and it does
        #     mask_here = torch.tile(mask, (1, tp.shape[1], ...))
        # -- it assumes a SPATIAL mask with one channel and tiles it across all C. Handing
        # it a per-channel [B, C, 1, 1, 1] mask tiles 2 channels into 4 and raises. The
        # shapes happened to make that a loud failure; had C been 1 it would have been
        # silent, and the metric that selects checkpoint_best would have been wrong.
        tp, fp, fn, _ = masked_tp_fp_fn_tn(output, target, mask)

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy(),
            "fp_hard": fp.detach().cpu().numpy(),
            "fn_hard": fn.detach().cpu().numpy(),
        }

    # -- the echo back -------------------------------------------------------------

    def on_validation_epoch_end(self, val_outputs: list[dict]) -> None:
        """Select `checkpoint_best` on a number that means the same thing every epoch.

        REGISTER ENTRY 102. nnU-Net picks the best checkpoint on
        `np.nanmean(dice_per_class)`. On a fully labelled dataset every channel has
        validation data every epoch and that mean is over a fixed set. Under PARTIAL
        supervision it is not: a channel is `nan` whenever the epoch's validation
        iterations drew no patch from a case supervised for it, and with 18 validation
        cases and roughly 3 per channel that happens constantly.

        Measured over 70 epochs of the ten-channel fit: the mean was taken over
        FOURTEEN DISTINCT CHANNEL SUBSETS, between 3 and 9 channels wide. Epoch A's mean
        over four channels was compared against epoch B's over eight, and the larger
        number won. Worse than noisy, it was biased -- `vertebral_body` and
        `pleural_effusion`, the two structures the model learns earliest and best,
        appeared in none of the four most common subsets, while `coronary_calcification`
        at 4,971 voxels appeared in all of them.

        So the checkpoint metric is a MICRO-average: one Dice from the summed confusion
        counts across every channel. It is defined in every epoch that saw any supervised
        pair at all, and two epochs' values are comparable because both are computed the
        same way. The per-channel macro numbers are still logged -- they are what a human
        reads -- and the count of absent channels is logged with them, so a reader can
        see the metric thinning instead of inferring it from a row of `nan`.
        """
        from nnunetv2.utilities.collate_outputs import collate_outputs

        collated = collate_outputs(val_outputs)
        tp = np.sum(collated["tp_hard"], 0)
        fp = np.sum(collated["fp_hard"], 0)
        fn = np.sum(collated["fn_hard"], 0)
        loss_here = np.mean(collated["loss"])

        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        absent = [c for c, value in zip(self._channels, per_class) if not np.isfinite(value)]

        denominator = 2 * float(np.sum(tp)) + float(np.sum(fp)) + float(np.sum(fn))
        # Nothing supervised anywhere this epoch. `0.0` would claim the model scored zero;
        # nan says the epoch measured nothing, and `_best_ema` will not select on it.
        micro = (2 * float(np.sum(tp)) / denominator) if denominator > 0 else float("nan")

        self.logger.log("mean_fg_dice", micro, self.current_epoch)
        self.logger.log("dice_per_class_or_region", per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

        # IoU PER CLASS, printed rather than logged. For the same confusion counts the
        # two metrics are one function of each other -- `IoU = Dice / (2 - Dice)` -- so
        # this adds no information nnU-Net did not have. It adds the number people
        # actually compare against: segmentation papers report IoU, nnU-Net reports
        # Dice, and doing the conversion by hand at 3 a.m. is how a model gets recorded
        # against the wrong baseline.
        #
        # `print_to_log_file` and NOT `self.logger.log`: `nnUNetLogger` keeps a fixed
        # set of keys and `plot_progress_png` iterates them, so an unknown key risks
        # breaking the plot for a number that is already derivable. The log file is the
        # honest place for a derived quantity.
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class_iou = [
                i / (i + j + k) if (i + j + k) else float("nan")
                for i, j, k in zip(tp, fp, fn)
            ]
        micro_iou = (
            float(np.sum(tp)) / (float(np.sum(tp)) + float(np.sum(fp)) + float(np.sum(fn)))
            if (np.sum(tp) + np.sum(fp) + np.sum(fn)) > 0
            else float("nan")
        )
        shown = [
            round(float(v), 4) if np.isfinite(v) else float("nan") for v in per_class_iou
        ]
        self.print_to_log_file("IoU per class " + str(shown))
        self.print_to_log_file(f"IoU (micro, the checkpoint metric's twin) {micro_iou:.4f}")
        if absent:
            self.print_to_log_file(
                f"validation saw no supervised pair for {len(absent)}/{len(self._channels)} "
                f"channels: {absent}. checkpoint_best is selected on the micro-averaged "
                f"Dice ({micro:.4f}), which does not change shape when this list does."
            )

    def on_train_end(self) -> None:
        self.print_to_log_file(
            f"MASKING APPLIED: {self.applied_supervised_pairs} (case, channel) pairs over "
            f"{len(self.seen_cases)} distinct cases. Compare against the declared "
            f"supervision: a number equal to epochs * steps * batch * channels means the "
            f"mask was all-ones and did nothing."
        )
        stamp = Path(self.output_folder) / "masking_applied.json"
        stamp.write_text(
            json.dumps(
                {
                    "applied_supervised_pairs": int(self.applied_supervised_pairs),
                    "distinct_cases_seen": sorted(self.seen_cases),
                    "channels": self._channels,
                    "epochs": int(self.num_epochs),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        super().on_train_end()
