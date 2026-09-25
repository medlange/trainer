# SPDX-License-Identifier: Apache-2.0
"""The loss for a model trained on data where nobody annotated everything.

THE PROBLEM, STATED ONCE
------------------------
Each corpus in a partially labelled collection annotates only its own findings. A
hydrothorax case carries no nodule contour -- not because there is no nodule, but because
nobody looked. Train a multi-channel model on the naive union and every unannotated
channel is presented to the loss as background, so the model is taught to suppress exactly
the findings it exists to detect. The model then scores well on each corpus's own test
split, because each split has the same blind spot as its training data, and is worse than
the separate models it replaced on a real scan.

The fix is one sentence: a channel a case does not annotate must contribute ZERO to the
loss and ZERO to every gradient. Everything below is that sentence, made true and made
checkable.

WHY THIS IS NOT nnU-Net's EXISTING MASKED LOSS WITH A DIFFERENT MASK
--------------------------------------------------------------------
nnU-Net already carries a masked CE for ignore-label handling, and it reduces as

    (self.ce(net_output, target) * mask).sum() / torch.clip(mask.sum(), min=1e-8)

which is correct for the mask it was written for: a SPATIAL mask of shape [B,1,X,Y,Z],
where numerator and denominator count the same things -- voxels. Our mask is per (case,
channel), shape [B,C], broadcast as [B,C,1,1,1]. Reuse that reduction and the numerator
still sums over every voxel while the denominator counts (b,c) PAIRS. For a 128^3 patch
that is a factor of 2,097,152. The loss does not crash, does not warn, and does not look
like a masking bug -- it looks like the learning rate is wrong. Hence `_bce`'s denominator
is `mask.expand_as(ce).sum()`, and hence `test_masked_loss.py` asserts the scale directly.

The Dice term has a quieter version of the same defect. nnU-Net's soft Dice computes
`dc = (2I + s) / (P + G + s)` and then `dc.mean()` over all channels. For a masked channel
I = P = G = 0, so dc = s/s = 1.0 exactly: the gradient is zero, which is correct, but the
value is a constant 1.0 that is still averaged in, and the mean still divides by C. So the
effective gradient scale on the channels that ARE supervised becomes (supervised / C) and
varies from batch to batch with whichever corpora the sampler happened to draw. `_dice`
reduces over supervised (b,c) pairs only.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does not decide what is supervised. It is handed a mask and it honours it. The decision
lives in a sealed annotation manifest line, reaches the trainer as `supervision.json`, and
is echoed back after the fit so the platform can compare what was intended against what was
applied. A loss that inferred its own mask would be a loss that could not be audited.

It also does not represent overlapping ground truth. The staged exchange is one integer
label map per case (`MOS-TRAIN-141`), so two findings occupying one voxel cannot be
expressed as targets, although the sigmoid heads can predict them. That is a limitation of
the exchange, recorded rather than hidden: see the corpus manifest's `known_limitations`.

Spec: MOS-TRAIN-141 (staged layout), MOS-EVID-040 (where supervision is declared).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "MaskedDiceBCELoss",
    "MaskedDeepSupervisionWrapper",
    "masked_tp_fp_fn_tn",
    "supervised_pair_count",
]


def _broadcast(mask: Tensor, like: Tensor) -> Tensor:
    """[B,C] -> [B,C,1,...,1] with one trailing axis per spatial dimension of `like`."""
    if mask.dim() != 2:
        raise ValueError(f"mask must be [B, C]; got {tuple(mask.shape)}")
    if mask.shape[:2] != like.shape[:2]:
        raise ValueError(
            f"mask is {tuple(mask.shape)} but the prediction is {tuple(like.shape)}: the "
            f"first two axes must agree, since the mask says which (case, channel) pairs "
            f"are supervised."
        )
    return mask.view(*mask.shape, *([1] * (like.dim() - 2))).to(like.dtype)


def supervised_pair_count(mask: Tensor) -> int:
    """How many (case, channel) pairs this mask actually supervises.

    The number the fit echoes back to the platform. It is deliberately a count of pairs and
    not a boolean 'masking was on': an all-ones mask and a correctly-varied mask both have
    masking 'on', and only the count tells them apart.
    """
    return int((mask > 0).sum().item())


class MaskedDiceBCELoss(nn.Module):
    """Soft Dice + BCE over sigmoid channels, restricted to supervised (case, channel) pairs.

    Multi-LABEL, not multi-class: each channel is an independent binary problem, so a case
    may carry two findings at once and -- more importantly here -- a channel can be
    unknown for one case and known for the next without the others having to renormalise
    around it. A softmax head cannot express that: its outputs sum to one, so removing a
    channel from the loss still changes what the remaining channels must predict.

    Args:
        batch_dice: pool the Dice statistics over the batch before dividing, as nnU-Net
            does. Under masking the pooling must skip unsupervised pairs and the channel
            reduction must skip channels the batch never supervised, or an all-masked
            channel contributes a constant.
        smooth: the Dice epsilon. Applied to numerator and denominator.
        weight_dice, weight_bce: term weights.
    """

    def __init__(
        self,
        *,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        weight_dice: float = 1.0,
        weight_bce: float = 1.0,
    ) -> None:
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    # -- terms ---------------------------------------------------------------------

    def _bce(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        m = _broadcast(mask, ce)
        numerator = (ce * m).sum()
        # THE DENOMINATOR THAT MATTERS. `m.sum()` counts (b,c) pairs; `ce` was summed over
        # voxels. Expanding first makes both sides count voxels. Getting this wrong scales
        # the loss by the patch volume and presents as a learning-rate problem.
        denominator = m.expand_as(ce).sum()
        if float(denominator) == 0.0:
            return logits.sum() * 0.0  # nothing supervised; keep the graph, contribute nothing
        return numerator / denominator

    def _dice(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        probability = torch.sigmoid(logits)
        axes = tuple(range(2, logits.dim()))
        intersection = (probability * target).sum(axes)  # [B, C]
        predicted = probability.sum(axes)
        ground_truth = target.sum(axes)
        mask2d = (mask > 0).to(logits.dtype)

        if self.batch_dice:
            # Pool over the batch, counting only supervised pairs, then reduce over the
            # channels this batch supervised at all. A channel nobody in the batch
            # annotated has I = P = G = 0 and would otherwise score a constant 1.0.
            intersection = (intersection * mask2d).sum(0)
            predicted = (predicted * mask2d).sum(0)
            ground_truth = (ground_truth * mask2d).sum(0)
            coefficient = (2 * intersection + self.smooth) / (
                predicted + ground_truth + self.smooth
            )
            keep = mask2d.sum(0) > 0
        else:
            coefficient = (2 * intersection + self.smooth) / (
                predicted + ground_truth + self.smooth
            )
            keep = mask2d > 0

        if not bool(keep.any()):
            return logits.sum() * 0.0
        return 1.0 - coefficient[keep].mean()

    # -- forward -------------------------------------------------------------------

    def forward(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"prediction {tuple(logits.shape)} and target {tuple(target.shape)} differ. "
                f"This loss is multi-label: the target is a per-channel binary map of the "
                f"same shape as the prediction, not an integer label map."
            )
        target = target.to(logits.dtype)
        return self.weight_bce * self._bce(logits, target, mask) + self.weight_dice * self._dice(
            logits, target, mask
        )


class MaskedDeepSupervisionWrapper(nn.Module):
    """Apply a masked loss at every decoder scale.

    THE MASK IS SCALE-INVARIANT AND THAT IS THE POINT. Whether a reader annotated a channel
    is a fact about the case, not about the resolution a decoder head happens to run at. So
    the same [B,C] mask goes to every scale, and only the target is downsampled.

    Applying the mask at the final scale alone is the most likely way to get this wrong: the
    fit would look masked, the loss would look right, and every coarse head would still be
    learning 'unannotated means empty' -- which then propagates up through the decoder. The
    perturbation test in trainer/tests/test_masked_loss.py runs at EVERY scale for that reason.
    """

    def __init__(self, loss: nn.Module, weights: Sequence[float]) -> None:
        super().__init__()
        if not len(weights):
            raise ValueError("deep supervision needs at least one scale weight")
        self.loss = loss
        self.weights = tuple(float(w) for w in weights)

    def forward(
        self,
        logits: Sequence[Tensor],
        targets: Sequence[Tensor],
        mask: Tensor,
    ) -> Tensor:
        if not (len(logits) == len(targets) == len(self.weights)):
            raise ValueError(
                f"deep supervision has {len(self.weights)} weights but got "
                f"{len(logits)} predictions and {len(targets)} targets"
            )
        total = None
        for weight, prediction, target in zip(self.weights, logits, targets):
            if weight == 0.0:
                continue
            term = weight * self.loss(prediction, target, mask)
            total = term if total is None else total + term
        if total is None:
            raise ValueError("every deep supervision weight is zero")
        return total


def masked_tp_fp_fn_tn(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-channel confusion counts over supervised pairs only.

    WHY THIS IS NOT COSMETIC. nnU-Net's online pseudo-Dice drives `checkpoint_best`. Leave
    the validation statistic unmasked and unsupervised channels contribute a perfect score
    -- every channel a batch did not annotate reports I = P = G = 0, which the pseudo-Dice
    reads as agreement. The packaged weights would then be selected by a metric that is
    partly measuring the absence of labels. A loss can be flawlessly masked and the wrong
    checkpoint still gets shipped.

    Returns four [C] tensors, already reduced over the batch and space, counting only the
    (b, c) pairs the mask supervises.
    """
    # FLOAT32, NOT `logits.dtype`, AND THIS IS THE WHOLE BUG THIS FUNCTION ONCE HAD.
    #
    # `validation_step` runs the network under `autocast`, so `logits` arrives here as
    # float16 -- and this function is called OUTSIDE the autocast block, so nothing
    # promotes it back. These are COUNTS OF VOXELS: a 128^3 patch holds 2,097,152 of
    # them and float16 saturates at 65,504. Every count above that became `inf`, the
    # Dice became `inf/inf`, and the epoch reported `nan`.
    #
    # It presented as the opposite of a numerical bug. The ONLY channel that ever showed
    # a Dice was `coronary_calcification` at 4,971 voxels across the whole dataset --
    # the second smallest structure, and the only one small enough never to overflow.
    # Everything large read `nan`, so the metric looked like a sampling problem: the big,
    # well-learned structures appeared to be the ones the model could not do. A direct
    # probe of the epoch-50 checkpoint scored `vertebral_body` at Dice 0.865 while the
    # trainer's log had reported `nan` for it in every single epoch.
    #
    # The corroborating signal was a `RuntimeWarning: overflow encountered in reduce`
    # from numpy, thrown while nnU-Net summed these arrays across validation iterations,
    # because `.cpu().numpy()` on a float16 tensor yields a float16 array and numpy sums
    # it in float16 too.
    #
    # The loss is NOT affected: it is computed inside the autocast block, where PyTorch
    # promotes the reductions. Only this function, called after it, saw raw float16.
    counting = torch.float32
    axes = tuple(range(2, logits.dim()))
    predicted = (torch.sigmoid(logits.to(counting)) > threshold).to(counting)
    truth = (target > 0).to(counting)
    m = _broadcast((mask > 0).to(counting), predicted)

    true_positive = (predicted * truth * m).sum(axes).sum(0)
    false_positive = (predicted * (1 - truth) * m).sum(axes).sum(0)
    false_negative = ((1 - predicted) * truth * m).sum(axes).sum(0)
    true_negative = ((1 - predicted) * (1 - truth) * m).sum(axes).sum(0)
    return true_positive, false_positive, false_negative, true_negative
