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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from medos_trainer.masked import (
    MaskedDeepSupervisionWrapper,
    MaskedDiceBCELoss,
    masked_tp_fp_fn_tn,
    supervised_pair_count,
)

#: Where `medos/tools/ingest/nnunet_dataset.py` writes the mask, beside dataset.json.
SUPERVISION_FILE = "supervision.json"
#: Optional per-channel loss weights, beside `SUPERVISION_FILE` in the dataset directory.
#: Absent means exact previous behaviour -- the same identity rule as the focal defaults.
CHANNEL_WEIGHTS_FILE = "class_weights.json"


def build_masked_loss(
    *, batch_dice: bool, scales: Any = None, focal_gamma: float = 0.0,
    focal_alpha: float | None = None, channel_weights: tuple[float, ...] | None = None,
) -> Any:
    """THE ONE PLACE THIS BACKEND BUILDS A LOSS, callable without a trainer.

    WHY IT IS A MODULE FUNCTION AND NOT JUST THE `_build_loss` METHOD. `_build_loss` needs a
    constructed trainer, which needs plans, a dataset and a card. So the only proof that
    this driver honours the mask was reading the method -- and reading is what a grep does.
    `test_backend_conformance.py` calls THIS, on CPU, in milliseconds, and perturbs an
    unsupervised channel to show the loss and every gradient are untouched.

    That closes the loop only if the trainer actually uses it, which reading cannot
    establish either. So the conformance suite ALSO asserts, over the syntax tree, that
    `_build_loss` calls this function and constructs no loss of its own. Two gates because
    the failure has two halves: a loss that ignores the mask, and a correct loss nobody
    calls. The second is the one that shipped once already -- `MaskedDeepSupervisionWrapper`
    was written, perturbation-tested at every scale, and then not used.

    `scales` is the deep-supervision scale list, or `None` for a single head. The mask is
    scale-invariant -- whether a reader annotated a channel is a fact about the CASE, not
    about the resolution a decoder head runs at -- so one `[B, C]` mask reaches every scale.
    """
    loss: Any = MaskedDiceBCELoss(
        batch_dice=batch_dice, smooth=1e-5, weight_dice=1.0, weight_bce=1.0,
        focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        channel_weights=channel_weights,
    )
    if scales is None:
        return loss
    weights = np.array([1 / (2**i) for i in range(len(scales))])
    weights[-1] = 0          # nnU-Net drops the coarsest head
    weights = weights / weights.sum()
    # OUR OWN WRAPPER, WHICH TAKES A MASK, rather than nnU-Net's, which does not. It was
    # written for this, perturbation-tested at every decoder scale, and then not used:
    # `_build_loss` reached for nnU-Net's wrapper instead and paid for it in `_call_loss`,
    # which had to synthesise `[mask] * len(output)` so that a third per-scale list would
    # survive `forward(*args)`'s `zip(*args)`. That worked, and it worked by depending on
    # the argument-forwarding of a third-party class that knows nothing about masks -- a
    # dependency that needed its own test to pin somebody else's internals, and that would
    # break silently on any upstream change to a signature nobody upstream promised.
    return MaskedDeepSupervisionWrapper(loss, weights)


def force_region_mode(manager: LabelManager) -> LabelManager:
    """Make a singleton-list label set behave as regions. Register entry 95."""
    manager._has_regions = True
    manager._regions = manager._get_regions()
    manager.inference_nonlin = torch.sigmoid
    return manager


class nnUNetTrainerMaskedChannels(nnUNetTrainer):
    """Multi-label sigmoid heads, supervised per (case, channel)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        # The signature is nnUNetTrainer's, transcribed rather than absorbed into **kwargs.
        # nnUNetv2_train constructs the trainer by keyword, so a missing parameter is a
        # TypeError at construction -- which is the good case. **kwargs would accept
        # anything, including a parameter nnU-Net adds in a later version that this
        # subclass then silently drops on the floor.
        #
        # AND THAT IS EXACTLY WHAT IT CAUGHT. `unpack_dataset: bool = True` sat here,
        # fifth, transcribed from 2.5.1. nnU-Net 2.6 removed it -- the `.npz` -> `.npy`
        # unpack step it switched on is gone, replaced by a compressed format read in
        # place. Against 2.6.4 the old call passed six positional arguments where five are
        # taken, binding a bool to `device`. With `**kwargs` the parameter would simply
        # have vanished into it and the trainer would have run with a silently wrong
        # device argument; transcribed, it is a TypeError at construction. Kept in that
        # form for the next time upstream moves.
        super().__init__(plans, configuration, fold, dataset_json, device)

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

        self._channels, self._supervision, self._unmasked = self._load_supervision()
        #: Per-channel loss weights from the dataset's class_weights.json, or None for the
        #: unweighted behaviour every run before this one trained under. Data-side, like the
        #: supervision map: the weights are a property of the cohort's class geometry, not of
        #: a training setting, and they are recorded in the same place the next reader looks.
        self._channel_weights: tuple[float, ...] | None = self._load_channel_weights()
        #: THE OPERATING POINT THE VALIDATION STATISTIC IS MEASURED AT (`MOS-SVC-021`).
        #:
        #: It was the default argument of `masked_tp_fp_fn_tn` and appeared nowhere else, so
        #: every pseudo-Dice this trainer logged -- and every number the A/B evaluator
        #: reported from those counts -- was measured at 0.5 and said so nowhere.
        #: `MOS-SVC-021` is explicit: "A metric without a threshold MUST NOT be displayed or
        #: returned by the API", and it fixes the member's name as `score_threshold`
        #: everywhere it travels.
        #:
        #: An attribute rather than a constructor argument for the same reason `_unmasked` is
        #: one: the trainer's signature is transcribed from nnU-Net's so that a mismatch is a
        #: TypeError at the call site, and a caller that wants a different operating point
        #: sets it in one visible line.
        self._score_threshold: float = 0.5
        #: The focal shaping of the pointwise term, set the same way and for the same reason.
        #:
        #: THESE EXIST BECAUSE THE FOCAL TERM WAS OTHERWISE UNREACHABLE. `masked.py` has carried
        #: `focal_gamma` and `focal_alpha` with six gates over them since they were written, and
        #: `_build_loss` passed neither -- so every fit this image has ever run used plain BCE and
        #: no run COULD have used anything else. A loss with tests and no caller is the same defect
        #: as a state key nobody subscribes to: every static check passes and the feature does not
        #: exist. `backend.fit` now sets these before `initialize()`, beside `num_epochs`.
        #:
        #: `0.0` is an EXACT identity with the previous behaviour, not an approximation of it --
        #: `masked._pointwise` reduces to `binary_cross_entropy_with_logits` at gamma 0 with alpha
        #: `None`, which `test_masked_loss.py` asserts against torch's own function. So a run that
        #: sets nothing trains exactly what it trained before this line existed.
        self._focal_gamma: float = 0.0
        self._focal_alpha: float | None = None
        #: The learning-rate schedule. "poly" is nnU-Net's own and the only one any run on
        #: disk trained under; "plateau" swaps in ReduceLROnPlateau on val_loss in
        #: `configure_optimizers`. Set by `backend.fit` before `initialize()`, for the same
        #: reason as the focal shaping: the scheduler is built inside `configure_optimizers`,
        #: which `initialize()` calls, so setting it afterwards would leave the run on poly
        #: while the record named plateau.
        self._lr_scheduler_kind: str = "poly"
        self.applied_supervised_pairs = 0
        self.seen_cases: set[str] = set()

    # -- the mask ------------------------------------------------------------------

    def _load_supervision(self) -> tuple[list[str], dict[str, list[str]], bool]:
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
        # READ, NOT JUST PRINTED. This value used to reach exactly one place -- the log
        # line below -- while `_mask_for` built the mask from `cases` regardless. So the
        # document declared the semantics, the trainer announced them, and the loss
        # ignored them: writing `true` produced a run whose own record contradicted what
        # it computed. It is now the switch it always claimed to be.
        unmasked = document.get("empty_segment_is_negative", False)
        if not isinstance(unmasked, bool):
            raise TypeError(
                f"{SUPERVISION_FILE} carries empty_segment_is_negative="
                f"{unmasked!r} ({type(unmasked).__name__}); it is a JSON boolean. "
                "`bool(\"false\")` is True, and this member decides whether three "
                "quarters of the (case, channel) pairs are masked out of the loss or "
                "trained as background"
            )
        self.print_to_log_file(
            f"supervision: {candidates[0]} -- {len(cases)} cases, channels {channels}, "
            f"empty_segment_is_negative={unmasked}"
        )
        if unmasked:
            # LOUD, AND AT THE TOP OF THE LOG. A control arm has to be recognisable from
            # the first screen of its own log, or the two arms of the comparison are
            # distinguishable only by whoever remembers which directory was which.
            self.print_to_log_file(
                "!! UNMASKED CONTROL ARM: every channel is supervised on every case. An "
                "unannotated finding is presented to the loss as BACKGROUND -- the "
                "false-negative signal this trainer exists to remove. This run is a "
                "measurement of that effect and MUST NOT be promoted."
            )
        for channel in channels:
            n = sum(1 for v in cases.values() if channel in v)
            declared = f"{n}/{len(cases)} cases"
            if unmasked:
                declared += f" (annotated; TRAINED on all {len(cases)})"
            self.print_to_log_file(f"  {channel}: supervised on {declared}")
        return channels, cases, unmasked

    def _load_channel_weights(self) -> tuple[float, ...] | None:
        """class_weights.json beside supervision.json, or None for unweighted.

        THE FILE IS OPTIONAL, AND ITS ABSENCE IS THE IDENTITY. Every run trained before
        this existed ran unweighted; a missing file must reproduce that exactly, or the
        comparison against those runs is against a different loss. This is the same rule
        the focal defaults follow.

        PARTIAL FILES ARE REFUSED, not padded with ones: a weights document that names
        nine of ten channels would silently leave the tenth at weight 1, and a reader
        comparing two runs would not know whether "1" was chosen or defaulted.

        The document is data, and the reason it travels with the dataset rather than the
        request is the same one as for supervision.json: these weights are a statement
        about the cohort's class geometry (measured medians live in the file's own
        record), and the next reader looks beside the labels, not in a shell history.
        """
        raw = os.environ.get("nnUNet_raw")
        if not raw:
            raise RuntimeError("nnUNet_raw is unset; cannot locate " + CHANNEL_WEIGHTS_FILE)
        # THE SAME RESOLUTION RULE AS THE SUPERVISION FILE: beside the dataset this run
        # actually trains on, never "the first one found".
        plans_name = getattr(self.plans_manager, "dataset_name", None)
        path = Path(raw) / str(plans_name) / CHANNEL_WEIGHTS_FILE if plans_name else None
        if path is None or not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        table = document.get("weights")
        if not isinstance(table, dict):
            raise TypeError(
                f"{CHANNEL_WEIGHTS_FILE} carries no `weights` mapping; it is not a "
                "weights document and guessing one would train under weights nobody wrote"
            )
        missing = [c for c in self._channels if c not in table]
        if missing:
            raise ValueError(
                f"{CHANNEL_WEIGHTS_FILE} names {sorted(table)}; the dataset's channels "
                f"are {self._channels}. Missing {missing}: padding with 1.0 would "
                "silently leave those channels unweighted and the run's record would not "
                "say so"
            )
        try:
            weights = tuple(float(table[c]) for c in self._channels)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{CHANNEL_WEIGHTS_FILE} weights must be numbers: {exc}"
            ) from exc
        self.print_to_log_file(
            f"channel weights: {path} -- {dict(zip(self._channels, weights))}"
        )
        formula = document.get("_formula")
        if formula:
            self.print_to_log_file(f"channel weights formula: {formula}")
        return weights

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
                # THE CONTROL ARM DIFFERS HERE AND NOWHERE ELSE -- one predicate, inside
                # the same walk. An early `return torch.ones(...)` above would have been
                # shorter and wrong: it skips the lookup, so the control run becomes the
                # one place a case missing from the map passes silently, and the two arms
                # then differ in more than the mask. A controlled comparison may not do
                # that. (Written that way first; the gate below caught it, which is the
                # only reason this comment is about a fixed defect and not a live one.)
                if self._unmasked or channel in supervised:
                    mask[row, column] = 1.0
        return mask

    # -- the loss ------------------------------------------------------------------

    def _build_loss(self):  # noqa: ANN201 - matches the base signature
        """Delegate, and construct nothing here.

        Everything this used to do inline now lives in `build_masked_loss`, which the
        conformance suite can call without plans, a dataset or a card. This method's whole
        remaining job is to read the two facts only a constructed trainer knows -- whether
        Dice is batched, and what the decoder's scales are -- and hand them over. A loss
        constructed here instead would be a loss no gate exercises, which is half of the
        defect this file exists to prevent.
        """
        loss = build_masked_loss(
            batch_dice=self.configuration_manager.batch_dice,
            scales=(self._get_deep_supervision_scales()
                    if self.enable_deep_supervision else None),
            focal_gamma=self._focal_gamma,
            focal_alpha=self._focal_alpha,
            channel_weights=self._channel_weights,
        )
        # PRINTED THROUGH THE TRAINER'S OWN LOGGER, so the run's log says which loss trained it.
        # `result.json` carries it too (see `backend.fit`), but the log is what a person reads when
        # two runs disagree, and "which loss was this" must not require finding a shell history.
        self.print_to_log_file(
            "masked loss: focal_gamma=%g focal_alpha=%s%s"
            % (self._focal_gamma, self._focal_alpha,
               "  (gamma 0 and alpha None is exact plain BCE)"
               if self._focal_gamma == 0.0 and self._focal_alpha is None else "")
        )
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

    # -- the learning-rate schedule --------------------------------------------------

    def configure_optimizers(self):  # noqa: ANN201 - matches the base signature
        """The base's SGD, with the scheduler swapped when the run asks for plateau.

        The optimiser is ALWAYS the base's: momentum 0.99 nesterov SGD at `initial_lr` is
        what every existing run used, and the schedule is the variable under test, so it is
        the only thing this changes. `poly` is returned untouched -- `0.0`-style identity
        with previous behaviour, like the focal default.
        """
        optimizer, scheduler = super().configure_optimizers()
        if self._lr_scheduler_kind != "plateau":
            return optimizer, scheduler
        plateau = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
        # THE MEASURED REASON FOR THE SWAP: three architectures on this cohort (SegResNetDS,
        # PlainConvUNet, ResEncUNet-M) destabilised under the poly decay from 1e-2, and the
        # one rate everything survives -- 1e-4 -- warms up too slowly to read at 200 epochs.
        # A plateau scheduler permits a hotter start because it backs off on evidence: when
        # val_loss stops improving for `patience` epochs the rate halves, which is the
        # cooling the fixed schedule never knew it needed.
        self.print_to_log_file(
            "lr scheduler: ReduceLROnPlateau(mode=min, factor=0.5, patience=8, "
            "min_lr=1e-6) stepping on val_loss at on_epoch_end; poly is NOT stepped"
        )
        return optimizer, plateau

    def on_train_epoch_start(self) -> None:
        if self._lr_scheduler_kind != "plateau":
            super().on_train_epoch_start()
            return
        self.network.train()
        # NO SCHEDULER STEP HERE, and this is the whole reason the method is overridden.
        # nnU-Net steps its poly schedule against the epoch counter at exactly this line in
        # the base implementation. ReduceLROnPlateau.step() takes a METRIC, not an epoch --
        # calling it here with the counter would drive the rate down monotonically as if
        # every epoch were a plateau epoch. The metric step lives in `on_epoch_end`, where
        # the epoch's val_loss exists. The rest of the base body is reproduced verbatim.
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: "
            f"{np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        if self._lr_scheduler_kind == "plateau":
            # AFTER super(): the epoch's val_loss was appended to the logger during it, and
            # the checkpoint decisions (latest, best-EMA) are already taken, so the rate
            # that produced this checkpoint is the rate the checkpoint's epoch recorded.
            self.lr_scheduler.step(self.logger.my_fantastic_logging["val_losses"][-1])

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
        # THE THRESHOLD IS PASSED AND NOT DEFAULTED. Leaving it to the default meant the
        # recorded operating point could disagree with the one that produced the counts,
        # which is a wrong number wearing a correct label.
        tp, fp, fn, _ = masked_tp_fp_fn_tn(
            output, target, mask, threshold=self._score_threshold
        )

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
