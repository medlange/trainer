# SPDX-License-Identifier: Apache-2.0
"""The vanilla 3D UNet — Medlange Trainer's own network, no MONAI, no nnU-Net.

THE DESIGN IS THE nnU-Net LAYOUT because that layout is the de-facto answer for
volumetric segmentation and re-implementing it from first principles is not an
act of originality worth the reader's time. What IS ours:

  * stem strides from the plan (a low-resolution corpus gets a coarser stem,
    decided at planning time, `UNetConfig.stem_stride`);
  * instance norm + leaky ReLU throughout — batch-independent, so a batch of
    one voxel is a legal forward pass;
  * deep supervision heads (1x1x1 convs) on every decoder stage except the
    last, returned as a tuple. THE SERVED ARTIFACT IS ONE TENSOR: exporting
    flips `deep_supervision` off and the served output is at input resolution —
    the same contract `packaging.torchscript_bytes` asserts for any network;
  * Kaiming init on every conv.

The module speaks plain `torch.nn`. The vertical slice (data/training/
inference) plugs into this shape contract and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UNetConfig:
    """Everything the network needs, decided by the plan (T5).

    `features` is per stage: len(features) encoder stages, the decoder mirrors
    it in reverse. `stem_stride` is the plan's answer to anisotropy/resolution
    (e.g. (1, 1, 1) for full-resolution CT, (2, 2, 1) for thick-slice data).
    `deep_supervision` is a TRAINING property: the served artifact must be
    built with it off (one output tensor, input resolution).
    """

    input_channels: int
    num_classes: int
    features: tuple[int, ...] = (32, 64, 128, 256, 320)
    stem_stride: tuple[int, int, int] = (1, 1, 1)
    deep_supervision: bool = True

    def __post_init__(self) -> None:
        if len(self.features) < 2:
            raise ValueError("a UNet needs at least two stages")
        if any(f <= 0 for f in self.features):
            raise ValueError(f"feature widths must be positive, got {self.features}")
        if any(s not in (1, 2) for s in self.stem_stride):
            raise ValueError(f"stem strides are 1 or 2 per axis, got {self.stem_stride}")


class _ConvBlock(nn.Sequential):
    """Conv(k=3) → InstanceNorm → LeakyReLU, twice. The stage workhorse."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(i, o, kernel_size=3, padding=1),
                nn.InstanceNorm3d(o),
                nn.LeakyReLU(0.01, inplace=True),
            )

        super().__init__(block(in_channels, out_channels), block(out_channels, out_channels))


class VanillaUNet(nn.Module):
    """3D UNet with deep supervision, implemented from scratch.

    FORWARD CONTRACT: with `deep_supervision` off, `forward` returns ONE tensor
    at input resolution (B, num_classes, *spatial). With it on, a tuple of
    logits — one per decoder stage, LAST element at input resolution. The
    training loss averages over the tuple; the export path takes the last.
    """

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        feats = list(config.features)

        self.stem = nn.Sequential(
            nn.Conv3d(config.input_channels, feats[0], kernel_size=3,
                      stride=config.stem_stride, padding=1),
            nn.InstanceNorm3d(feats[0]),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ENCODER: ConvBlock per stage; a stride-2 conv halves resolution
        # between stages. There is one fewer downsampling than stages, so the
        # deepest ConvBlock's output IS the bottleneck.
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = feats[0]
        for width in feats[1:]:
            self.encoders.append(_ConvBlock(prev, width))
            self.downsamples.append(nn.Conv3d(width, width, kernel_size=3, stride=2, padding=1))
            prev = width

        # DECODER mirrors the encoder: trilinear upsample + 1x1x1 channel
        # reduction, concat the matching skip, ConvBlock.
        self.upsamples = nn.ModuleList()
        self.joins = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(feats))
        for hi, lo in zip(rev, rev[1:]):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.joins.append(nn.Conv3d(hi, lo, kernel_size=1))
            # After the concat the tensor carries skip (hi) + joined (lo)
            # channels; the block fuses them down to lo.
            self.decoders.append(_ConvBlock(hi + lo, lo))

        # DEEP SUPERVISION: a 1x1x1 head on every decoder output EXCEPT the
        # final one (that goes through the main seg head). Decoder outputs are
        # the `lo` side of each pair, so the head widths are lo[0..n-2].
        decoder_out_widths = [lo for _, lo in zip(rev, rev[1:])]
        self.aux_heads = (
            nn.ModuleList(
                nn.Conv3d(width, config.num_classes, kernel_size=1)
                for width in decoder_out_widths[:-1]
            )
            if config.deep_supervision else nn.ModuleList()
        )
        self.seg_head = nn.Conv3d(feats[0], config.num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Odd spatial sizes can leave a one-voxel mismatch after upsampling --
        in EITHER direction (9 -> 18 overshoots a 17-wide skip). Bring both
        tensors to the per-axis minimum, centred: every voxel a head scores
        came from a real feature, and neither side is padded with fiction."""
        if a.shape[-3:] == b.shape[-3:]:
            return a, b
        t = [min(x, y) for x, y in zip(a.shape[-3:], b.shape[-3:])]

        def crop(x: torch.Tensor) -> torch.Tensor:
            d = [xs - tt for xs, tt in zip(x.shape[-3:], t)]
            return x[
                :, :,
                d[0] // 2: d[0] // 2 + t[0],
                d[1] // 2: d[1] // 2 + t[1],
                d[2] // 2: d[2] // 2 + t[2],
            ]

        return crop(a), crop(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders, self.downsamples):
            h = enc(h)
            skips.append(h)
            h = down(h)

        aux: list[torch.Tensor] = []
        for i, (up, join, dec) in enumerate(zip(self.upsamples, self.joins, self.decoders)):
            h = up(h)
            h = join(h)
            skip, h = self._align(skips[-(i + 1)], h)
            h = dec(torch.cat([skip, h], dim=1))
            if self.config.deep_supervision and i < len(self.aux_heads):
                aux.append(self.aux_heads[i](h))

        out = self.seg_head(h)
        if not self.config.deep_supervision:
            return out
        return (*aux, out)


def build_unet(config: UNetConfig) -> VanillaUNet:
    """Constructor seam: planning asks, this answers. Exists so the network
    factory never scatters `VanillaUNet(config)` literals across the trainer."""
    return VanillaUNet(config)
