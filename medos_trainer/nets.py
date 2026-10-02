# SPDX-License-Identifier: Apache-2.0
"""Foreign architectures under nnU-Net's own calling convention. First-party wrappers, not nets.

WHY A WRAPPER AND NOT A RE-IMPLEMENTATION
------------------------------------------
`MOS-REL-027` makes the build-versus-adopt register normative: no subsystem is built without a
row, and a row cannot say "build" without naming an incumbent and the property that
disqualified it. The register's MONAI Core row is already **ADOPT** and lists "the 3D network
zoo" among what is adopted. Writing our own SegResNet would contradict a standing normative row,
and its success criterion would be "matches the published implementation" -- which is what
`MOS-REL-032` means by adopting the adopted thing whole.

WHY A WRAPPER AT ALL, THEN
---------------------------
Because a MONAI class cannot be named in a plan. `get_network_from_plans` builds a network with
exactly

    nw_class(input_channels=..., num_classes=..., deep_supervision=..., **arch_kwargs)

and MONAI's constructors take `in_channels`/`out_channels`, have no `num_classes`, no
`deep_supervision`, and would reject `n_stages`, `features_per_stage`, `conv_op`, `kernel_sizes`
and `strides` as unexpected keywords. So the adapter is the "genuinely added" column of the
register row, and it is the only thing here that is ours.

THE FOUR THINGS THE CONTRACT ACTUALLY DEMANDS, each established by reading the installed
nnU-Net 2.6.4 rather than its documentation:

  1. under deep supervision, return a LIST of logits FINEST-FIRST with exactly
     `len(strides) - 1` entries. The reference decoder ends with `seg_outputs[::-1]`; the trainer
     takes `output[0]` as the full-resolution prediction and the loss weights decay `1/2**i` from
     index 0. `DeepSupervisionWrapper` uses `zip`, which TRUNCATES, so returning too few outputs
     raises no clear error -- it silently pairs a head with another scale's target;
  2. with deep supervision off, return a single tensor at INPUT resolution. A length-one list
     breaks `torch.flip` during test-time mirroring, which is the crash the trainer warns about;
  3. expose a real attribute path `.decoder.deep_supervision` that SWITCHES the return type.
     `set_deep_supervision_enabled` unwraps DDP and `torch.compile` and then performs one hard
     write to it -- and this repository's own `packaging.py` walks the same path before tracing,
     raising `ContractViolation` unless the traced output is a single tensor. A wrapper hiding
     the flag elsewhere breaks packaging even when training succeeds;
  4. implement `compute_conv_feature_map_size(input_size)` if the architecture is to be PLANNED.
     The planner builds the network only to size it and reads that number against
     `UNet_reference_val_3d`, so it has to be the same quantity in the same units: the summed
     element count of every stage's activations, encoder plus decoder.

WHY (4) MATTERS, AND A CLAIM THIS FILE MADE AND THEN MEASURED
-------------------------------------------------------------
The temptation is to keep the frozen plan and swap the class name. The plan this cohort trained
on describes a `PlainConvUNet` of 31.2M parameters at six stages with features capped at 320.
`SegResNetDS` at the same six stages and the same initial 32 filters is 356.2M parameters --
eleven times larger, because it doubles per stage with no cap.

This file first said the activation footprint grew in proportion. It does not, and the two
quantities point opposite ways at the same geometry and patch 128x224x224:

    PlainConvUNet        1.44e9 activation elements     31.2M parameters
    SegResNetDS init=32  9.72e8 activation elements    356.2M parameters

The planner sizes the patch and the batch against ACTIVATIONS, so the frozen plan is not too
small for this architecture -- it is CONSERVATIVE, by about a third. What the extra parameters
cost is optimiser state: weights, gradients and SGD momentum at 356M floats is roughly 4 GB,
which a 32 GB card carries. `test_nets.py` pins both numbers, so the paragraph above is a test
rather than a recollection.

The plan is still not one anybody derived for this architecture, which is `MOS-REL-032`'s
objection and remains true. But the cost of retrofitting is a conservative patch, not an OOM,
and that is a different decision. It also keeps the difference between two experiments honest:
comparing `plain` against `resenc_l` is two fits and no preprocessing, because both are upstream
planners over one fingerprint; a MONAI architecture reuses a patch derived elsewhere until a
planner sizes it.

AND THE SAME QUESTION HAS A DIFFERENT ANSWER FOR EACH, WHICH IS WHY ALL FOUR ARE MEASURED
------------------------------------------------------------------------------------------
The paragraph above does not generalise. At the frozen patch 128x224x224, one sample -- conv
activations counted by forward hook, attention matrices counted where they are materialised:

    PlainConvUNet                 1.44e9 conv                    1.44e9     31.2M parameters
    SegResNetDS   init=32         9.72e8 conv                    9.72e8    356.2M
    SwinUNETR     feature_size=48 3.78e9 conv + 4.02e9 attention 7.80e9     62.2M
    UNETR         feature_size=16 1.20e9 conv + 3.54e8 attention 1.55e9    122.2M

"A TRANSFORMER COSTS MORE MEMORY" IS FALSE AS A GENERALISATION, and the two rows say why. Swin's
attention is WINDOWED and runs at high resolution: at 1/2 of this patch there are about 2,300
windows of 343 tokens each, and every one materialises `heads * 343**2`. UNETR's attention is
GLOBAL but runs only on the ViT's grid at 1/16 -- 8x14x14 = 1,568 tokens in total, so twelve
layers of twelve heads come to 12*12*1568**2 = 3.54e8, a fifth of its own convolution term.
UNETR is therefore about the same size as the conv U-Net at this patch and SwinUNETR is 5.4 times
it. Both figures are per sample and the plan's batch size is 2.

Neither attention term is a hypothesis: MONAI's `SABlock` materialises `att_mat` through an
`einsum` whenever `use_flash_attention` is false, which it is for all twelve of UNETR's blocks,
and Swin's `WindowAttention` materialises through an `nn.Softmax` module. Both were checked on the
built networks rather than read from documentation.

THREE CONSEQUENCES, WRITTEN INTO THE CODE RATHER THAN LEFT AS ADVICE. Gradient checkpointing
defaults to ON for SwinUNETR, because its windowed attention is the term recomputation removes --
at batch 2 the total is about 31 GB in half precision on a 32 GB card. And BOTH transformer
wrappers REFUSE `compute_conv_feature_map_size`: a conv-shaped estimate omits the attention term,
and the planner divides its VRAM target by that number, so the error runs in the direction that
sizes the patch too LARGE. For UNETR there is a second reason -- it cannot be resized after
construction at all, so a planner that shrank the patch would leave the plan's patch and the
plan's network disagreeing. `test_nets.py` pins every row above, so this section is a measurement
and not a recollection.

Spec: MOS-REL-027, MOS-REL-032, MOS-TRAIN-135, MOS-TRAIN-223.
"""

from __future__ import annotations

from typing import Any, Final, Sequence

import torch
from torch import nn

__all__ = ["ARCH_KWARGS_NOT_APPLICABLE", "MUST_BE_NONE", "MedOSSegResNetDS", "MedOSSwinUNETR",
           "MedOSUNETR", "SWIN_ARCH_KWARGS_NOT_APPLICABLE", "SWIN_STAGES",
           "UNETR_ARCH_KWARGS_NOT_APPLICABLE", "UNETR_PATCH", "UNETR_STAGES",
           "activation_elements", "uniform_halving_stages"]

#: Keys whose ABSENCE is acceptable and whose PRESENCE is not. A plan that sets one of these to
#: a real value is asking for something this architecture cannot do, and recording it as
#: "inapplicable" would be the difference between a key nobody needed and a setting that
#: disappeared.
MUST_BE_NONE: Final[tuple[str, ...]] = ("dropout_op", "dropout_op_kwargs")

#: Plan keys a residual-encoder architecture cannot honour, each with the reason.
#:
#: ACCEPTED AND RECORDED, NEVER SWALLOWED. `get_network_from_plans` hands over every key the plan
#: carries, and a wrapper with `**kwargs` would turn a key it ignores into "a setting the platform
#: believes it applied" -- the hazard the run-directory contract states in those words about its
#: own members. So every key is either used or listed here, and one that is neither raises.
ARCH_KWARGS_NOT_APPLICABLE: Final[dict[str, str]] = {
    "kernel_sizes":
        "SegResNetDS fixes its convolution kernels internally; a per-stage kernel list derived "
        "for a plain conv U-Net has nowhere to go and pretending otherwise would report a "
        "geometry the network does not have.",
    "n_conv_per_stage":
        "a SegResNet BLOCK is not a convolution, so a per-stage convolution count cannot be "
        "mapped onto `blocks_down` without inventing the ratio. The block counts come from the "
        "architecture's own recommendation and are recorded in the plan as `blocks_down`.",
    "n_conv_per_stage_decoder":
        "as above, for the decoder; SegResNetDS derives its decoder depth from `blocks_down`.",
    "conv_bias":
        "SegResNetDS sets convolution bias itself per block.",
    "dropout_op":
        "SegResNetDS has no dropout parameter. The plan sets this to null, so there is nothing "
        "to honour -- and a NON-null value is refused rather than recorded here, because a "
        "requested dropout that silently vanished would be a regularisation the platform "
        "believes it applied. See MUST_BE_NONE.",
    "dropout_op_kwargs":
        "as above: arguments for a dropout operation this architecture does not have, refused "
        "rather than dropped when non-null.",
    "nonlin_kwargs":
        "the activation is named by `nonlin`; MONAI takes the name and its own defaults.",
}


def uniform_halving_stages(strides: Sequence[Sequence[int]], *, architecture: str) -> int:
    """The stage count a plan asks for, refusing any stride pattern the wrappers cannot honour.

    BOTH WRAPPERS IN THIS FILE HALVE EVERY AXIS AT EVERY STAGE, because both foreign
    architectures do: `SegResNetDS` downsamples isotropically unless it is given a `resolution`,
    and `SwinUNETR`'s patch-merging halves all three axes by construction. nnU-Net downsamples its
    deep-supervision TARGETS by the plan's cumulative strides. So an anisotropic plan -- a final
    `[1, 2, 2]`, which nnU-Net produces for a thick-slice cohort -- gives a head whose depth is
    twice its target's.

    AND THAT WOULD NOT BE CAUGHT DOWNSTREAM. The loss would raise a shape error somewhere inside
    a Dice, with no mention of strides, at the first backward pass of a queued fit. Every 3D
    configuration in this cohort's frozen plans is `[[1,1,1], [2,2,2], ...]`, so nothing has ever
    exercised the other branch -- which is exactly why the refusal is written rather than assumed.
    """
    if not strides:
        raise ValueError(
            f"{architecture}: the plan carries no strides, which fix both the stage count and "
            "the number of deep-supervision heads the trainer pairs with targets"
        )
    axes = len(strides[0])
    for index, stride in enumerate(strides):
        if len(stride) != axes:
            raise ValueError(
                f"{architecture}: stride {index} has {len(stride)} axes and stride 0 has {axes}"
            )
        wanted = 1 if index == 0 else 2
        if any(int(step) != wanted for step in stride):
            raise ValueError(
                f"{architecture}: stride {index} is {list(stride)} and this wrapper can only "
                f"honour {[wanted] * axes}. Its deep-supervision heads sit at powers of two in "
                "every axis, and nnU-Net downsamples its targets by the plan's cumulative "
                "strides, so an anisotropic stage would pair a head with a target of another "
                "shape -- raising a shape error inside the loss that says nothing about strides"
            )
    return len(strides)


def activation_elements(
    *, init_filters: int, blocks_down: Sequence[int], input_size: Sequence[int],
    out_channels: int,
) -> int:
    """The summed activation element count, in the units the nnU-Net planner reads.

    THE SAME QUANTITY THE REFERENCE ARCHITECTURES RETURN, which is what makes the planner's VRAM
    arithmetic mean anything: `PlainConvUNet.compute_conv_feature_map_size` sums, over every
    stage, that stage's output channels times its spatial volume, halving the volume by the
    strides as it descends, and adds the decoder's. A number in different units would still be a
    number, and the planner would size the patch against it.

    SegResNetDS doubles its filters each stage from `init_filters` and halves each spatial axis
    after the first stage, so the encoder term is a geometric series; the decoder mirrors it and
    each of its levels also carries a segmentation head of `out_channels`.
    """
    if init_filters <= 0 or not len(blocks_down) or out_channels <= 0:
        raise ValueError(
            f"init_filters={init_filters!r}, blocks_down={list(blocks_down)!r}, "
            f"out_channels={out_channels!r}: a size estimate over an empty or negative geometry "
            "would be a number the planner would nonetheless size a patch against"
        )
    total = 0
    volume = 1
    for extent in input_size:
        volume *= int(extent)

    filters = int(init_filters)
    for stage, blocks in enumerate(blocks_down):
        if stage:
            # Every axis halves after the first stage, so the volume drops by 2**dims.
            volume //= 2 ** len(input_size)
            filters *= 2
        # One activation of this stage's shape per residual block, plus the stage's own output.
        total += (int(blocks) + 1) * filters * volume

    # The decoder mirrors the encoder upwards, and each level carries a head of `out_channels`.
    for _stage in range(len(blocks_down) - 1):
        volume *= 2 ** len(input_size)
        filters //= 2
        total += (filters + int(out_channels)) * volume
    return int(total)


class _DeepSupervisionFlag:
    """The object `set_deep_supervision_enabled` and `packaging.py` both write through.

    A REAL OBJECT AND NOT A PROPERTY ON THE WRAPPER, because both writers address
    `network.decoder.deep_supervision` by that exact path: the trainer's own docstring admits the
    method is specific to the default architecture, and this repository's packaging walks the same
    path before tracing. Giving the wrapper a `.decoder` that carries the flag is what makes both
    work without either of them knowing this class exists.
    """

    def __init__(self, enabled: bool) -> None:
        self.deep_supervision = bool(enabled)


class MedOSSegResNetDS(nn.Module):
    """MONAI's `SegResNetDS` with nnU-Net's constructor, output contract and size estimate."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        *,
        n_stages: int | None = None,
        features_per_stage: Sequence[int] | None = None,
        strides: Sequence[Sequence[int]] | None = None,
        blocks_down: Sequence[int] | None = None,
        conv_op: Any = None,
        norm_op: Any = None,
        norm_op_kwargs: Any = None,
        nonlin: Any = None,
        **plan_kwargs: Any,
    ) -> None:
        super().__init__()
        requested = sorted(
            key for key in MUST_BE_NONE if plan_kwargs.get(key) is not None
        )
        if requested:
            raise ValueError(
                f"the plan sets {requested}, and this architecture has no dropout to configure. "
                "Recording it as inapplicable would make a requested regularisation vanish "
                "while the plan still claimed it"
            )
        unknown = sorted(set(plan_kwargs) - set(ARCH_KWARGS_NOT_APPLICABLE))
        if unknown:
            raise ValueError(
                f"the plan carries architecture keys this wrapper neither uses nor records as "
                f"inapplicable: {unknown}. A key accepted and ignored is a setting the platform "
                "believes it applied; add it to ARCH_KWARGS_NOT_APPLICABLE with the reason, or "
                "use it"
            )
        if strides is None:
            raise ValueError(
                "the plan carries no `strides`. They fix both the number of stages and the "
                "number of deep-supervision heads the trainer will pair with targets, and "
                "guessing either makes the loss pair a head with another scale's target"
            )
        declared = uniform_halving_stages(strides, architecture="SegResNetDS")
        stages = int(n_stages) if n_stages is not None else declared
        if stages != len(strides):
            raise ValueError(
                f"n_stages={stages} and {len(strides)} strides disagree. The trainer derives its "
                "deep-supervision scales from `strides` alone, so the two must be one number"
            )
        if blocks_down is None:
            # THE ARCHITECTURE'S OWN RECOMMENDATION, EXTENDED TO THE PLAN'S DEPTH. MONAI's default
            # is (1, 2, 2, 4) for four stages; deeper plans repeat the deepest count. This is
            # recorded in the plan under `blocks_down` so a run states the geometry it trained
            # rather than leaving it to a default that could move under an upgrade.
            default = [1, 2, 2, 4]
            blocks_down = (default + [4] * stages)[:stages]
        if len(blocks_down) != stages:
            raise ValueError(
                f"{len(blocks_down)} block counts against {stages} stages"
            )

        from monai.networks.nets import SegResNetDS

        init_filters = int(features_per_stage[0]) if features_per_stage else 32
        self._heads = stages - 1
        self.decoder = _DeepSupervisionFlag(deep_supervision)
        self._geometry = {
            "init_filters": init_filters,
            "blocks_down": [int(b) for b in blocks_down],
            "out_channels": int(num_classes),
        }
        self.net = SegResNetDS(
            spatial_dims=len(strides[0]),
            in_channels=int(input_channels),
            out_channels=int(num_classes),
            init_filters=init_filters,
            blocks_down=tuple(int(b) for b in blocks_down),
            dsdepth=self._heads,
            # INSTANCE NORM, AND NOT BECAUSE IT IS NICER. The plan declares
            # `InstanceNorm3d` as its `norm_op`, so this follows the frozen plan. It also makes
            # the module's own `training` flag irrelevant to its outputs, which `forward` below
            # depends on: MONAI returns the multi-scale list only while `self.training`, and
            # nnU-Net wants that list during validation with the network in `eval()`.
            norm="instance",
            act="leakyrelu" if nonlin is None else "leakyrelu",
        )

    def forward(self, x: torch.Tensor) -> Any:
        """A finest-first list under deep supervision, one tensor without it.

        THE `training` FLAG IS FORCED, NOT READ. `SegResNetDS._forward` returns its list only
        while `self.training` is true, and nnU-Net asks for the list during validation with the
        network in `eval()`. Forcing the flag is safe here ONLY because the normalisation is
        instance norm and dropout is off, so nothing else about the module depends on it -- see
        the constructor. With batch norm this would silently corrupt validation by updating
        running statistics.
        """
        if not self.decoder.deep_supervision:
            was = self.net.training
            self.net.train(False)
            try:
                out = self.net(x)
            finally:
                self.net.train(was)
            return out[0] if isinstance(out, (list, tuple)) else out

        bottleneck = [int(extent) // (2 ** (self._heads)) for extent in x.shape[2:]]
        if any(extent < 2 for extent in bottleneck):
            # OUR MESSAGE, BECAUSE THE FORCED FLAG MAKES THEIRS MISLEADING. Instance norm in
            # training mode refuses a spatial extent of one, and `forward` forces training mode
            # to get the multi-scale list -- so a reader in `eval()` would be told "Expected more
            # than 1 spatial element when training" about a network they are not training.
            raise ValueError(
                f"a patch of {tuple(int(v) for v in x.shape[2:])} leaves {bottleneck} at the "
                f"bottleneck of {self._heads + 1} stages, and an extent of one cannot be "
                "normalised. Either the patch is too small for this depth or the depth was "
                "derived for a different patch"
            )
        was = self.net.training
        self.net.train(True)
        try:
            out = self.net(x)
        finally:
            self.net.train(was)
        if not isinstance(out, (list, tuple)):
            out = [out]
        if len(out) != self._heads:
            raise RuntimeError(
                f"the architecture returned {len(out)} deep-supervision outputs and the plan's "
                f"strides ask for {self._heads}. DeepSupervisionWrapper zips them against "
                "targets, and zip truncates: too few pairs a head with another scale's target "
                "and raises nothing"
            )
        return list(out)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        """The planner's size probe. See `activation_elements`."""
        return activation_elements(input_size=input_size, **self._geometry)


#: How many stages a plan must declare for this wrapper to serve it, and it is not negotiable.
#:
#: `SwinUNETR`'s depth is fixed by its construction: patch-merging four times after a patch embed
#: of two gives feature maps at 1/2, 1/4, 1/8, 1/16 and 1/32, and the decoder returns through all
#: of them. So there are exactly six resolutions and five of them can carry a deep-supervision
#: head. A plan of seven stages has no sixth head to give, and one of five would leave a head
#: paired with a target that does not exist -- `DeepSupervisionWrapper` zips, and zip truncates.
SWIN_STAGES: Final[int] = 6

#: Plan keys `SwinUNETR` cannot honour, each with the reason. Same rule as
#: `ARCH_KWARGS_NOT_APPLICABLE`: used, or recorded here, never silently accepted.
SWIN_ARCH_KWARGS_NOT_APPLICABLE: Final[dict[str, str]] = {
    "features_per_stage":
        "the width of every stage follows from `feature_size`, which MONAI requires to be "
        "divisible by 12 because the attention heads are (3, 6, 12, 24). This cohort's plans open "
        "at 32 channels, which is not, so the plan's widths cannot be honoured even approximately "
        "and the value actually used is recorded in the plan under `feature_size`.",
    "kernel_sizes":
        "the encoder is windowed attention and has no convolution kernels; the decoder's are "
        "fixed by `UnetrUpBlock`. A per-stage kernel list derived for a plain conv U-Net has "
        "nowhere to go.",
    "n_conv_per_stage":
        "a transformer stage is a number of attention BLOCKS, not convolutions. The block counts "
        "are `depths`, recorded in the plan; mapping one onto the other would invent the ratio.",
    "n_conv_per_stage_decoder":
        "`UnetrUpBlock` is one transposed convolution and one residual block per level, fixed by "
        "the block; there is no per-stage count to set and mapping the plan's onto it would "
        "invent a depth the decoder does not have.",
    "conv_bias":
        "every convolution here lives inside a MONAI block that sets its own bias alongside its "
        "normalisation, so there is no single flag to honour and setting one per block would "
        "change the published architecture rather than configure it.",
    "dropout_op":
        "MONAI takes three separate rates -- `drop_rate`, `attn_drop_rate`, `dropout_path_rate` "
        "-- and not an operation. The plan sets this to null, so there is nothing to honour; a "
        "NON-null value is refused rather than recorded, because a requested regularisation that "
        "vanished would be one the platform believes it applied. See MUST_BE_NONE.",
    "dropout_op_kwargs":
        "arguments for an operation this architecture does not take. The three rates it does take "
        "are named parameters with their own defaults, so a `p` intended for a dropout layer has "
        "no layer to reach and would silently become no regularisation at all.",
    "nonlin":
        "the activation inside a transformer block is GELU and inside `UnetrUpBlock` is LeakyReLU, "
        "both fixed by the blocks. A plan naming `LeakyReLU` is not contradicted, but it is not "
        "honoured either, and a wrapper that accepted the key would be claiming it was.",
    "nonlin_kwargs":
        "arguments for an activation the blocks choose themselves, so there is nothing to pass "
        "them to. A negative slope intended for LeakyReLU would reach neither the GELU in the "
        "encoder nor the block-internal LeakyReLU in the decoder.",
    "norm_op":
        "the encoder normalises with LayerNorm, which is what attention requires, and the decoder "
        "with the instance norm the plan asks for -- passed through as `norm_name`. A plan naming "
        "one operation for both cannot be honoured for both, so the key is recorded here and the "
        "value that reached the decoder is recorded in the plan under `norm_name`.",
    "norm_op_kwargs":
        "arguments for one normalisation, where this architecture uses two different ones: the "
        "encoder's LayerNorm takes an epsilon and the decoder's instance norm takes affine and "
        "momentum, and one dictionary cannot be honoured for both without guessing which.",
    "conv_op":
        "there is no single convolution operation to choose; the spatial rank comes from the "
        "plan's strides instead.",
}


class MedOSSwinUNETR(nn.Module):
    """MONAI's `SwinUNETR` with nnU-Net's constructor, output contract and refusals.

    THE DEEP-SUPERVISION HEADS ARE OURS AND THEY ARE THE ONLY ADDED PART. `SwinUNETR.forward`
    returns one tensor; nnU-Net's trainer wants five, finest-first, at the plan's cumulative
    strides. The heads are `UnetOutBlock`s -- MONAI's own 1x1 head, the same class its single
    output uses -- placed on the decoder outputs MONAI's own forward already computes. Nothing
    about the architecture is re-implemented; what is added is the tap.

    WHY `forward` RE-DISPATCHES THROUGH THE SUBMODULES. MONAI's forward computes exactly the
    tensors the heads need and then discards four of them, and there is no hook that returns a
    list. Calling it once for the finest output and again for the rest would be two forward passes
    per step. So this walks the same chain in the same order -- and `test_nets.py` pins that order
    against `inspect.getsource(SwinUNETR.forward)`, so a MONAI upgrade that reorders the decoder
    fails a test instead of quietly feeding a head the wrong stage's features.

    WHY THERE IS NO SIZE PROBE. See `compute_conv_feature_map_size`, which refuses.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        *,
        n_stages: int | None = None,
        strides: Sequence[Sequence[int]] | None = None,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        use_checkpoint: bool = True,
        use_v2: bool = False,
        norm_name: str = "instance",
        **plan_kwargs: Any,
    ) -> None:
        super().__init__()
        requested = sorted(key for key in MUST_BE_NONE if plan_kwargs.get(key) is not None)
        if requested:
            raise ValueError(
                f"the plan sets {requested}, and this architecture takes three separate dropout "
                "RATES rather than an operation. Recording it as inapplicable would make a "
                "requested regularisation vanish while the plan still claimed it"
            )
        unknown = sorted(set(plan_kwargs) - set(SWIN_ARCH_KWARGS_NOT_APPLICABLE))
        if unknown:
            raise ValueError(
                f"the plan carries architecture keys this wrapper neither uses nor records as "
                f"inapplicable: {unknown}. A key accepted and ignored is a setting the platform "
                "believes it applied; add it to SWIN_ARCH_KWARGS_NOT_APPLICABLE with the reason, "
                "or use it"
            )
        if strides is None:
            raise ValueError(
                "the plan carries no `strides`. They fix the stage count, and this architecture "
                "can serve exactly one"
            )
        declared = uniform_halving_stages(strides, architecture="SwinUNETR")
        stages = int(n_stages) if n_stages is not None else declared
        if stages != declared:
            raise ValueError(
                f"n_stages={stages} and {declared} strides disagree. The trainer derives its "
                "deep-supervision scales from `strides` alone, so the two must be one number"
            )
        if stages != SWIN_STAGES:
            raise ValueError(
                f"the plan asks for {stages} stages and SwinUNETR has exactly {SWIN_STAGES}: a "
                "patch embed of two and four patch-merges, which is its construction and not a "
                f"setting. It can therefore carry {SWIN_STAGES - 1} deep-supervision heads and "
                f"the plan asks for {stages - 1}. DeepSupervisionWrapper zips heads against "
                "targets and zip truncates, so the mismatch would not raise -- it would pair "
                "every head with the wrong scale's target. Plan this architecture at six stages "
                "or train it without deep supervision"
            )
        if int(feature_size) % 12 != 0:
            # MONAI'S OWN RULE, RAISED HERE SO IT NAMES THE PLAN. Its message is
            # "feature_size should be divisible by 12" with nothing about where 32 came from.
            raise ValueError(
                f"feature_size={feature_size} is not divisible by 12, which the attention heads "
                "(3, 6, 12, 24) require. This cohort's plans open at 32 channels, so the plan's "
                "width cannot be carried over; 24 and 48 are the published settings"
            )

        from monai.networks.blocks import UnetOutBlock
        from monai.networks.nets import SwinUNETR

        spatial_dims = len(strides[0])
        width = int(feature_size)
        self._heads = stages - 1
        #: 2**5 -- a patch embed of two and four merges. MONAI checks the same number; ours is
        #: checked in `forward` so the message can name the patch and the plan.
        self._divisor = 32
        self.decoder = _DeepSupervisionFlag(deep_supervision)
        self.net = SwinUNETR(
            in_channels=int(input_channels),
            out_channels=int(num_classes),
            feature_size=width,
            depths=tuple(int(d) for d in depths),
            num_heads=tuple(int(h) for h in num_heads),
            # GRADIENT CHECKPOINTING ON BY DEFAULT, and this is a real decision rather than a
            # cautious default: the published Swin UNETR results were obtained at 96x96x96 and
            # this cohort's frozen plan is 128x224x224, which is seven times the voxels. The
            # recomputation costs roughly a third of the step time and is the difference between
            # a fit that runs and one that does not.
            use_checkpoint=bool(use_checkpoint),
            use_v2=bool(use_v2),
            norm_name=str(norm_name),
            spatial_dims=spatial_dims,
        )
        #: The four extra heads, finest of them first, on the decoder outputs at 1/2 to 1/16.
        #: The finest scale is MONAI's own `self.net.out`, so it is not duplicated here.
        self.heads = nn.ModuleList([
            UnetOutBlock(spatial_dims=spatial_dims, in_channels=channels,
                         out_channels=int(num_classes))
            for channels in (width, width * 2, width * 4, width * 8)
        ])

    def forward(self, x: torch.Tensor) -> Any:
        """A finest-first list of five under deep supervision, one tensor without it.

        NO `training` FLAG IS FORCED HERE, unlike the SegResNetDS wrapper: the multi-scale list is
        ours, built from tensors MONAI's chain produces either way, so nothing depends on the
        module thinking it is training. That wrapper needed the force because MONAI's own
        `SegResNetDS` returns its list only while `self.training`.
        """
        wrong = [index + 2 for index, extent in enumerate(x.shape[2:])
                 if int(extent) % self._divisor]
        if wrong:
            raise ValueError(
                f"a patch of {tuple(int(v) for v in x.shape[2:])} is not divisible by "
                f"{self._divisor} on axes {wrong}. SwinUNETR embeds patches of two and merges "
                "four times, so every axis must survive five halvings -- this is the patch the "
                "plan asks for, and a plan derived for a conv U-Net is not obliged to satisfy it"
            )
        net = self.net
        hidden = net.swinViT(x, net.normalize)
        enc0 = net.encoder1(x)
        enc1 = net.encoder2(hidden[0])
        enc2 = net.encoder3(hidden[1])
        enc3 = net.encoder4(hidden[2])
        dec4 = net.encoder10(hidden[4])
        dec3 = net.decoder5(dec4, hidden[3])
        dec2 = net.decoder4(dec3, enc3)
        dec1 = net.decoder3(dec2, enc2)
        dec0 = net.decoder2(dec1, enc1)
        fused = net.decoder1(dec0, enc0)
        finest = net.out(fused)
        if not self.decoder.deep_supervision:
            return finest
        outputs = [finest] + [head(tap) for head, tap
                              in zip(self.heads, (dec0, dec1, dec2, dec3))]
        if len(outputs) != self._heads:
            raise RuntimeError(
                f"{len(outputs)} deep-supervision outputs against {self._heads} the plan's "
                "strides ask for. zip truncates, so this would pair heads with the wrong targets"
            )
        return outputs

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        """REFUSED, and the refusal is the safe answer rather than an admission.

        The planner divides its VRAM target by this number and compares the ratio against
        `UNet_reference_val_3d` -- a number measured on a plain conv U-Net, where memory really is
        dominated by convolution activations. In windowed attention it is not: the attention
        matrices are `heads * window_volume**2` per window and they do not appear in any
        activation count, so a conv-shaped estimate is an UNDERESTIMATE and the planner would size
        the patch too LARGE. That failure arrives as an out-of-memory error at the first epoch of
        a queued fit, hours after the plan was written and with nothing in the plan to point at.

        Returning a number with a comment admitting it is approximate would be worse than
        refusing, because the planner cannot read comments. This architecture is used with a plan
        derived by an upstream conv planner -- which is what `architectures.py` records, and the
        patch then being conservative for one architecture and unvalidated for another is a stated
        gap rather than a hidden one.
        """
        raise NotImplementedError(
            "SwinUNETR must not be sized by nnU-Net's planner: the planner's reference value "
            "UNet_reference_val_3d was measured on a conv U-Net, and an attention architecture's "
            f"memory is not proportional to its activation count. Asked for {tuple(input_size)}. "
            "Plan with an upstream conv planner and swap the architecture, which is what "
            "medos_trainer.architectures records"
        )


#: Stages a plan must declare for `MedOSUNETR`, for the same structural reason as `SWIN_STAGES`.
#:
#: UNETR's decoder returns through four `UnetrUpBlock`s from the ViT's grid at 1/16, so the
#: resolutions are 1, 1/2, 1/4, 1/8 and 1/16 -- five, and five heads for a six-stage plan. Measured
#: rather than read: `test_nets.py` hooks every decoder block and pins the shapes.
UNETR_STAGES: Final[int] = 6

#: The ViT's patch, and it is NOT A PARAMETER UPSTREAM. MONAI writes
#: `self.patch_size = ensure_tuple_rep(16, spatial_dims)` with the 16 inline, derives
#: `feat_size = img_size // 16` from it and freezes `proj_view_shape` from that. So 16 is part of
#: the architecture, every axis of the patch must be divisible by it, and neither fact is checked
#: by MONAI -- a patch of 100 would give 6 tokens per axis by flooring and reshape without
#: complaint, silently discarding the remainder of the image.
UNETR_PATCH: Final[int] = 16

#: Plan keys `UNETR` cannot honour, each with the reason.
UNETR_ARCH_KWARGS_NOT_APPLICABLE: Final[dict[str, str]] = {
    "features_per_stage":
        "the encoder is a plain ViT of one uniform width (`hidden_size`) and the decoder's widths "
        "double from `feature_size`, so there is no per-stage width to set. The plan's list opens "
        "at 32 and caps at 320, which describes neither; the values actually used are recorded in "
        "the plan as `feature_size` and `hidden_size`.",
    "kernel_sizes":
        "the ViT has no convolution kernels and the decoder blocks fix theirs. A per-stage kernel "
        "list derived for a plain conv U-Net has nowhere to go.",
    "n_conv_per_stage":
        "the encoder's depth is twelve transformer layers, which is `num_layers` and not a "
        "convolution count; the decoder's depth is fixed per block. Mapping one onto the other "
        "would invent the ratio.",
    "n_conv_per_stage_decoder":
        "as above for the decoder: `UnetrUpBlock` is one transposed convolution and one residual "
        "block, fixed by the block, so there is no per-stage count to honour.",
    "conv_bias":
        "every convolution sits inside a MONAI block that sets its own bias beside its "
        "normalisation; there is no single flag, and setting one per block would change the "
        "published architecture rather than configure it.",
    "dropout_op":
        "UNETR takes a single `dropout_rate` FLOAT and no operation. The plan sets this to null, "
        "so there is nothing to honour; a NON-null value is refused rather than recorded, because "
        "a requested regularisation that vanished would be one the platform believes it applied. "
        "See MUST_BE_NONE.",
    "dropout_op_kwargs":
        "arguments for an operation this architecture does not take. A `p` intended for a dropout "
        "layer has no layer to reach and would become no regularisation at all rather than an "
        "error.",
    "conv_op":
        "there is no single convolution operation to choose; the spatial rank comes from the "
        "plan's strides instead.",
    "norm_op":
        "the ViT normalises with LayerNorm, which is what attention requires, and the decoder with "
        "the instance norm the plan asks for -- passed through as `norm_name`. One operation named "
        "for both cannot be honoured for both, so the value that reached the decoder is recorded "
        "in the plan under `norm_name`.",
    "norm_op_kwargs":
        "arguments for one normalisation where two different ones are in use: LayerNorm takes an "
        "epsilon and instance norm takes affine and momentum, and one dictionary cannot be "
        "honoured for both without guessing which it was written for.",
    "nonlin":
        "the activation is GELU inside the transformer and LeakyReLU inside the decoder blocks, "
        "both fixed by the blocks. A plan naming `LeakyReLU` is not contradicted, but it is not "
        "honoured either, and accepting the key would claim it was.",
    "nonlin_kwargs":
        "arguments for an activation the blocks choose themselves, so there is nothing to pass "
        "them to. A negative slope meant for LeakyReLU would reach neither the GELU nor the "
        "block-internal LeakyReLU.",
}


class MedOSUNETR(nn.Module):
    """MONAI's `UNETR` with nnU-Net's constructor, output contract and the patch coupling made loud.

    WHAT IS DIFFERENT FROM THE OTHER TWO WRAPPERS IN THIS FILE, AND IT IS NOT COSMETIC. `UNETR`
    takes `img_size` and is BOUND to it: `proj_feat` reshapes the ViT's token sequence back into a
    grid with `view()` over a shape frozen at construction. A forward pass at any other spatial
    size raises a `view` error about numbers of elements -- or, when the token count happens to
    match, reshapes into the wrong grid and trains on scrambled features.

    nnU-Net never passes a patch size to a network, so `img_size` can only arrive through
    `arch_kwargs`, and the only honest source for it is the plan's own `patch_size`.
    `architectures.apply_overlay` therefore reads it out of the configuration it is patching --
    see `Overlay.plan_derived_kwargs` -- rather than letting a catalogue row spell a patch that
    could disagree with the plan it is dropped into.

    THE HEADS ARE OURS, as in the SwinUNETR wrapper, and for the same reason: MONAI's forward
    computes the five tensors and returns one. `forward` re-dispatches MONAI's own chain and
    `test_nets.py` pins that chain structurally against `inspect.getsource(UNETR.forward)`,
    including the ViT layer INDICES the skips are taken from -- a wrong index there would leave
    every shape correct and every feature wrong.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        *,
        n_stages: int | None = None,
        strides: Sequence[Sequence[int]] | None = None,
        img_size: Sequence[int] | None = None,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        norm_name: str = "instance",
        proj_type: str = "conv",
        res_block: bool = True,
        conv_block: bool = True,
        dropout_rate: float = 0.0,
        **plan_kwargs: Any,
    ) -> None:
        super().__init__()
        requested = sorted(key for key in MUST_BE_NONE if plan_kwargs.get(key) is not None)
        if requested:
            raise ValueError(
                f"the plan sets {requested}, and this architecture takes a dropout RATE rather "
                "than an operation. Recording it as inapplicable would make a requested "
                "regularisation vanish while the plan still claimed it"
            )
        unknown = sorted(set(plan_kwargs) - set(UNETR_ARCH_KWARGS_NOT_APPLICABLE))
        if unknown:
            raise ValueError(
                f"the plan carries architecture keys this wrapper neither uses nor records as "
                f"inapplicable: {unknown}. A key accepted and ignored is a setting the platform "
                "believes it applied; add it to UNETR_ARCH_KWARGS_NOT_APPLICABLE with the reason, "
                "or use it"
            )
        if strides is None:
            raise ValueError(
                "the plan carries no `strides`. They fix the stage count, and this architecture "
                "can serve exactly one"
            )
        declared = uniform_halving_stages(strides, architecture="UNETR")
        stages = int(n_stages) if n_stages is not None else declared
        if stages != declared:
            raise ValueError(
                f"n_stages={stages} and {declared} strides disagree. The trainer derives its "
                "deep-supervision scales from `strides` alone, so the two must be one number"
            )
        if stages != UNETR_STAGES:
            raise ValueError(
                f"the plan asks for {stages} stages and UNETR has exactly {UNETR_STAGES}: a ViT "
                "grid at 1/16 and four decoder levels back to full resolution, which is its "
                f"construction and not a setting. It can carry {UNETR_STAGES - 1} "
                f"deep-supervision heads and the plan asks for {stages - 1}. "
                "DeepSupervisionWrapper zips heads against targets and zip truncates, so this "
                "would not raise -- it would pair every head with another scale's target. Plan "
                f"this configuration at {UNETR_STAGES} stages, or choose another architecture"
            )

        # THE PATCH IS A CONSTRUCTOR ARGUMENT HERE AND THAT IS THE WHOLE DIFFICULTY.
        if img_size is None:
            raise ValueError(
                "UNETR needs `img_size` and the plan's architecture block does not carry one. It "
                "cannot be defaulted: `proj_feat` freezes the token grid from it, so a guess that "
                "disagreed with the patch the trainer actually crops would raise a `view` error "
                "deep inside the forward pass -- or reshape into the wrong grid when the token "
                "counts happen to match. It must be the plan's own `patch_size`, which "
                "`architectures.apply_overlay` copies in through `plan_derived_kwargs`"
            )
        spatial = [int(extent) for extent in img_size]
        if len(spatial) != len(strides[0]):
            raise ValueError(
                f"img_size {tuple(spatial)} has {len(spatial)} axes and the strides have "
                f"{len(strides[0])}: one of the two is not this configuration's"
            )
        indivisible = [axis + 2 for axis, extent in enumerate(spatial) if extent % UNETR_PATCH]
        if indivisible:
            # MONAI DOES NOT CHECK THIS, which is why it is here rather than left upstream. Its
            # patch embedding strides by 16 and floors, and `feat_size` floors the same way, so a
            # patch of 100 gives six tokens per axis, reshapes without complaint, and trains on
            # 96 of every 100 voxels with nothing anywhere saying so.
            raise ValueError(
                f"the plan's patch {tuple(spatial)} is not divisible by {UNETR_PATCH} on axes "
                f"{indivisible}, and UNETR's ViT patch is hard-coded to {UNETR_PATCH} upstream. "
                "MONAI floors instead of refusing, so the remainder of each such axis would be "
                "dropped silently rather than raising. Plan a patch divisible by "
                f"{UNETR_PATCH}, or choose another architecture"
            )
        grid = [extent // UNETR_PATCH for extent in spatial]
        if any(extent < 2 for extent in grid):
            raise ValueError(
                f"the plan's patch {tuple(spatial)} leaves a ViT grid of {grid}, and an extent of "
                "one cannot be upsampled through four decoder levels or normalised by the "
                "instance norm the plan asks for"
            )
        if int(hidden_size) % int(num_heads):
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}. MONAI "
                "raises for this too, with a message that does not say which plan the numbers "
                f"came from; the published UNETR setting is 768 over 12"
            )

        from monai.networks.blocks import UnetOutBlock
        from monai.networks.nets import UNETR

        width = int(feature_size)
        self._img_size: Final[tuple[int, ...]] = tuple(spatial)
        self._heads = stages - 1
        self.decoder = _DeepSupervisionFlag(deep_supervision)
        self.net = UNETR(
            in_channels=int(input_channels),
            out_channels=int(num_classes),
            img_size=tuple(spatial),
            feature_size=width,
            hidden_size=int(hidden_size),
            mlp_dim=int(mlp_dim),
            num_heads=int(num_heads),
            norm_name=str(norm_name),
            proj_type=str(proj_type),
            res_block=bool(res_block),
            conv_block=bool(conv_block),
            dropout_rate=float(dropout_rate),
            spatial_dims=len(spatial),
        )
        #: Four heads on the decoder outputs at 1/2, 1/4, 1/8 and 1/16, coarsening. The finest is
        #: MONAI's own `self.net.out` and is not duplicated. The coarsest sits on the ViT's
        #: projected output -- `hidden_size` channels, before any decoder block -- which is the
        #: tensor `decoder5` consumes and the only thing at 1/16 there is.
        self.heads = nn.ModuleList([
            UnetOutBlock(spatial_dims=len(spatial), in_channels=channels,
                         out_channels=int(num_classes))
            for channels in (width * 2, width * 4, width * 8, int(hidden_size))
        ])

    def forward(self, x_in: torch.Tensor) -> Any:
        """A finest-first list of five under deep supervision, one tensor without it."""
        shape = tuple(int(extent) for extent in x_in.shape[2:])
        if shape != self._img_size:
            # BEFORE MONAI'S `view`, SO THE MESSAGE NAMES THE CAUSE. `proj_feat` reshapes into a
            # grid frozen at construction; at another size it raises "shape [...] is invalid for
            # input of size N", which says nothing about a patch or a plan. And when the token
            # count coincides -- two patches whose products agree -- it does not raise at all.
            raise ValueError(
                f"this UNETR was built for a patch of {self._img_size} and was given {shape}. Its "
                "ViT's token grid is frozen at construction, so another size is not a smaller "
                "image: it is a different network. nnU-Net crops the plan's patch, so the two "
                "disagreeing means the network was built from a different plan than the one "
                "driving the loader"
            )
        net = self.net
        x, hidden_states_out = net.vit(x_in)
        enc1 = net.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = net.encoder2(net.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = net.encoder3(net.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = net.encoder4(net.proj_feat(x4))
        dec4 = net.proj_feat(x)
        dec3 = net.decoder5(dec4, enc4)
        dec2 = net.decoder4(dec3, enc3)
        dec1 = net.decoder3(dec2, enc2)
        out = net.decoder2(dec1, enc1)
        finest = net.out(out)
        if not self.decoder.deep_supervision:
            return finest
        outputs = [finest] + [head(tap) for head, tap
                              in zip(self.heads, (dec1, dec2, dec3, dec4))]
        if len(outputs) != self._heads:
            raise RuntimeError(
                f"{len(outputs)} deep-supervision outputs against {self._heads} the plan's "
                "strides ask for. zip truncates, so this would pair heads with the wrong targets"
            )
        return outputs

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        """REFUSED, for the reason `MedOSSwinUNETR`'s does and one more that is specific to UNETR.

        The shared reason: the planner divides its VRAM target by this number and compares the
        ratio against `UNet_reference_val_3d`, measured on a plain conv U-Net. A transformer's
        memory is not proportional to its convolution activations, so a conv-shaped estimate is an
        underestimate in the direction that sizes the patch too LARGE.

        The reason that is UNETR's alone: the planner's whole procedure is to try a patch, size
        the network for it, shrink, and try again. This network cannot be resized -- `img_size` is
        frozen into `proj_feat` at construction -- so each probe would need a new network, and the
        patch the planner settled on would then have to be written back into `arch_kwargs` for the
        fit to build the same one. Nothing upstream does that. A number returned here would
        therefore produce a plan whose patch and whose network disagree.
        """
        raise NotImplementedError(
            "UNETR must not be sized by nnU-Net's planner: its reference value "
            "UNet_reference_val_3d was measured on a conv U-Net, and this network cannot be "
            f"resized after construction in any case. Asked for {tuple(input_size)}. Plan with an "
            "upstream conv planner and swap the architecture, which is what "
            "medos_trainer.architectures records -- it copies the settled patch into `img_size`"
        )
