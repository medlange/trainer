# SPDX-License-Identifier: Apache-2.0
"""Masked losses for the vanilla stack.

THE MASKED SEMANTICS ARE THE TRAINER'S OWN INNOVATION and survive the rewrite
verbatim (see the audit: masked region training is where this framework beats
nnU-Net and MONAI). A case carries, per channel, a binary mask of VOXELS THAT
WERE LABELLED: voxels outside it are unknown, not background. Training on
unknown voxels as if they were background teaches the model to erase findings
nobody annotated — the masked loss excludes them from every term instead.

`MaskedSegmentationLoss` combines:
  * per-voxel cross-entropy, averaged over labelled voxels only;
  * soft Dice per class, integrated over labelled voxels only;
  * deep supervision: the network returns a tuple of stage logits; the loss
    averages the same combination over the tuple (the last element is the
    full-resolution output).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _labelled(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The mask travels with the target's shape; a case with no mask is fully
    labelled. The reduction divides by the LABELLED count, never the voxel
    count — an empty mask yields zero contribution, not NaN."""
    if mask is None:
        return torch.ones_like(target, dtype=torch.float32)
    return mask.to(dtype=torch.float32)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    num_classes: int,
) -> torch.Tensor:
    """Per-voxel CE over labelled voxels only."""
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    labelled = _labelled(mask, target)
    denom = labelled.sum().clamp_min(1.0)
    return (ce * labelled).sum() / denom


def masked_soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    num_classes: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Soft Dice per class over labelled voxels, averaged over classes."""
    probs = F.softmax(logits, dim=1)
    labelled = _labelled(mask, target)
    one_hot = F.one_hot(target.long(), num_classes).movedim(-1, 1).to(probs.dtype)
    one_hot = one_hot * labelled.unsqueeze(1)
    probs = probs * labelled.unsqueeze(1)

    dims = (0, 2, 3, 4)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class MaskedSegmentationLoss(nn.Module):
    """CE + Dice over labelled voxels; nnU-Net-style deep-supervision weights.

    THE FULL-RESOLUTION HEAD CARRIES THE ANSWER, so it carries the most
    weight: w_i = 0.5^(n-1-i) over the tuple (aux heads first, the served
    output last), normalised to sum 1. The equal-weight average this
    replaced diluted the full-res head to a 1/n share — measured on the
    PulmoAI probe as part of the optimization-speed gap vs nnU-Net (their
    MultipleOutputLossTwo uses the same 1/2^i shape; see
    docs/benchmark-pulmo-2026-10-07.md, W16 addendum).
    """

    def __init__(self, num_classes: int, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, ...],
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # w grows toward the LAST element: (*aux, full_res) -> full_res heaviest.
        weights = [0.5 ** (len(outputs) - 1 - i) for i in range(len(outputs))]
        norm = sum(weights)
        total = outputs[0].new_zeros(())
        for out, w in zip(outputs, weights):
            # Deep-supervision heads sit at lower resolutions; the target and
            # mask travel NEAREST-NEIGHBOUR down to each head's grid. Labels
            # are class indices (no averaging across classes), masks are
            # binary (no fractional membership).
            t, m = self._to(out, target, mask)
            total = total + (w / norm) * (
                masked_cross_entropy(out, t, m, self.num_classes)
                + self.dice_weight * masked_soft_dice(out, t, m, self.num_classes)
            )
        return total

    @staticmethod
    def _to(out: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if out.shape[-3:] == target.shape[-3:]:
            return target, mask
        size = out.shape[-3:]
        t = F.interpolate(target.unsqueeze(1).float(), size=size,
                          mode="nearest").squeeze(1).long()
        m = None
        if mask is not None:
            # The mask already carries its channel dim (B, C, *) — unlike the
            # target it needs no unsqueeze.
            m = F.interpolate(mask.float(), size=size, mode="nearest")
        return t, m
