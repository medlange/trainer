# SPDX-License-Identifier: Apache-2.0
"""W28: per-stage anisotropic downsampling strides, the nnU-Net way.

The planner now emits one stride tuple per downsampling, pooling an axis
only while its pooled patch stays >= 8 voxels (nnU-Net's own rule — their
Dataset502 plans carry [2,2,2]x4 + [2,1,2] because their 96-voxel axis
stops at 16x: 96/16 = 6 < 8). Uniform (2,2,2) everywhere is reported as
None so historical nets and bundles stay byte-identical.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.data import Case
from medos_trainer.vanilla.nets import UNetConfig, build_unet
from medos_trainer.vanilla.plan import (
    VramPreset,
    collect_fingerprint,
    plan_from_fingerprint,
)


def _cases(spacing=(1.0, 1.0, 1.0), shape=(24, 32, 40), n: int = 3,
           seed: int = 0) -> list[Case]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        image = rng.normal(-600.0, 50.0, (1, *shape)).astype(np.float32)
        label = (rng.random(shape) < 0.05).astype(np.int64)
        out.append(Case(image=image, label=label, mask=np.ones_like(image),
                        spacing_mm=spacing, case_id=f"c{i}"))
    return out


TOY = {"features": (4, 8, 16, 32, 32, 32)}


def test_config_default_strides_stay_uniform() -> None:
    """None is the historical net: stride-2 between every stage."""
    net = build_unet(UNetConfig(input_channels=1, num_classes=2, **TOY))
    for conv in net.downsamples:
        assert conv.stride == (2, 2, 2)
    for up in net.upsamples:
        assert up.scale_factor == (2, 2, 2)


def test_config_validates_the_stride_table() -> None:
    with pytest.raises(ValueError, match="downsample stride"):
        UNetConfig(input_channels=1, num_classes=2, features=(4, 8, 16),
                   downsample_strides=((2, 2, 2),))
    with pytest.raises(ValueError, match="1 or 2 per axis"):
        UNetConfig(input_channels=1, num_classes=2, features=(4, 8, 16),
                   downsample_strides=((2, 2, 2), (3, 2, 2)))


def test_anisotropic_table_restores_input_size_both_modes() -> None:
    """The real benchmark table, our axis order (our axis0 is nnU-Net's
    short axis): the axis that stops pooling early keeps resolution
    through the bottleneck, and the decoder's per-stage upsampling still
    lands the final head exactly on the input grid."""
    strides = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2))
    shape = (1, 1, 96, 160, 160)
    for ds in (False, True):
        net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                    downsample_strides=strides,
                                    deep_supervision=ds, **TOY))
        x = torch.randn(*shape)
        out = net(x)
        heads = out if isinstance(out, tuple) else (out,)
        assert heads[-1].shape == (1, 2, 96, 160, 160)
        # aux heads sit on progressively finer decoder outputs, coarsest
        # first, and none exceeds the input grid
        sizes = [h.shape[-3:] for h in heads]
        for s in sizes:
            assert s[0] <= 96 and s[1] <= 160 and s[2] <= 160


def test_stride_table_survives_the_bundle_json_roundtrip() -> None:
    """net_config.json is how a bundle rebuilds the net; tuples become
    JSON arrays there (same precedent as stem_stride), and the rebuilt
    net must behave identically."""
    strides = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2))
    config = UNetConfig(input_channels=1, num_classes=2,
                        downsample_strides=strides, deep_supervision=False,
                        **TOY)
    parsed = UNetConfig(**json.loads(json.dumps(config.__dict__)))
    net = build_unet(parsed)
    assert net(torch.randn(1, 1, 96, 160, 160)).shape == (1, 2, 96, 160, 160)


def test_planner_stops_pooling_below_eight_voxels() -> None:
    """The W28 rule, pinned on the real benchmark shape class: a ~100-voxel
    axis against ~140-180-voxel axes in a 6-stage plan stops one pool
    early — 102/16 = 6.4 < 8 — exactly where nnU-Net's own table stops
    (their [2,2,2]x4 + [2,1,2] in their axis order is this same table
    read in ours)."""
    fp = collect_fingerprint(_cases(spacing=(0.782, 0.782, 1.0)))
    plan = plan_from_fingerprint(fp, "xlarge")
    assert plan.downsample_strides is not None
    assert plan.downsample_strides == (
        (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2),
    )
    totals = (16, 32, 32)
    for voxels, total in zip(plan.patch_size, totals):
        assert voxels % total == 0, f"patch {plan.patch_size} rags total {totals}"
    assert any("min-pooled-size" in r for r in plan.reasons)
    config = plan.network_config(input_channels=1)
    assert config.downsample_strides == plan.downsample_strides


def test_planner_keeps_uniform_strides_for_large_patches() -> None:
    """Patches wide enough to pool all the way down get the historical
    uniform net: None, no reason noise. (With a 6-stage depth the rule
    needs ~256-voxel axes, so this pins the 3-stage shallow preset where
    40-voxel patches suffice.)"""
    wide = VramPreset("wide", features=(8, 16, 32), batch_size=1,
                      voxel_budget=64 * 64 * 64, target_patch_mm=(40.0, 40.0, 40.0))
    fp = collect_fingerprint(_cases(spacing=(1.0, 1.0, 1.0)))
    plan = plan_from_fingerprint(fp, wide)
    assert plan.downsample_strides is None
    assert not any("min-pooled-size" in r for r in plan.reasons)
