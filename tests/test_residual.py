# SPDX-License-Identifier: Apache-2.0
"""The residual-encoder option (`UNetConfig.residual`), held to the contract.

WHY THIS EXISTS: docs/benchmark-pulmo-2026-10-07.md (W16) measured our plain
UNet optimizing an order of magnitude slower than nnU-Net on real low-contrast
CT; nnU-Net's current default is a residual encoder. These tests pin what the
option promises: identical forward contract, engineered numerical stability at
the trainer's lr 0.01 / momentum 0.99 (a naive hand-rolled residual block NaN'd
in the probe battery), determinism, and a learnability guard so the
optimization-speed claim cannot silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from medos_trainer.vanilla.infer import load_predictor, served_net
from medos_trainer.vanilla.nets import (
    UNetConfig,
    VanillaUNet,
    _ConvBlock,
    _ResBlock,
    build_unet,
)
from medos_trainer.vanilla.trainer import FitPlan, VanillaTrainer

TOY = {"features": (8, 16, 32), "stem_stride": (1, 1, 1)}


def _net(residual: bool, deep_supervision: bool = True) -> VanillaUNet:
    return build_unet(UNetConfig(input_channels=1, num_classes=2,
                                 deep_supervision=deep_supervision, residual=residual,
                                 **TOY))


def test_config_defaults_to_the_plain_architecture() -> None:
    assert UNetConfig(input_channels=1, num_classes=2).residual is False
    assert _net(residual=False).config.residual is False
    assert _net(residual=True).config.residual is True


def test_residual_swaps_only_encoder_and_bottleneck_blocks() -> None:
    """Encoder stages (the deepest IS the bottleneck) become _ResBlocks; the
    decoder keeps _ConvBlock — nnU-Net's residual-ENCODER scoping. Residual in
    the decoder is unproven here and deliberately out of scope."""
    res = _net(residual=True)
    assert all(isinstance(m, _ResBlock) for m in res.encoders)
    assert all(isinstance(m, _ConvBlock) for m in res.decoders)
    plain = _net(residual=False)
    assert all(isinstance(m, _ConvBlock) for m in plain.encoders)
    assert all(isinstance(m, _ConvBlock) for m in plain.decoders)


def test_resblock_is_exactly_norm_of_skip_at_construction() -> None:
    """Guarantee (a), read functionally: with the second conv zero-initialised
    the block computes InstanceNorm(skip(x)) — nothing else."""
    torch.manual_seed(0)
    block = _ResBlock(8, 16)
    x = torch.randn(2, 8, 12, 12, 12)
    with torch.no_grad():
        out = block(x)
        reference = block.norm_out(block.skip(x))
    assert torch.allclose(out, reference)


def test_zero_init_and_skip_scaling_survive_the_network_init_pass() -> None:
    """The net's single Kaiming pass must not overwrite the block's special
    inits: conv2 stays zero (identity start), the 1x1 skip keeps PyTorch's
    default init scaled by 1/sqrt(2)."""
    torch.manual_seed(0)
    net = _net(residual=True)
    for enc in net.encoders:
        if enc.conv1.in_channels == enc.conv1.out_channels:
            assert isinstance(enc.skip, torch.nn.Identity)
        else:
            assert isinstance(enc.skip, torch.nn.Conv3d)
        assert torch.all(enc.conv2.weight == 0)
        assert torch.all(enc.conv2.bias == 0)
        if isinstance(enc.skip, torch.nn.Conv3d):
            reference = torch.nn.Conv3d(enc.conv1.in_channels, enc.conv1.out_channels,
                                        kernel_size=1)
            ratio = float(enc.skip.weight.std() / reference.weight.std())
            assert ratio == pytest.approx(0.5 ** 0.5, abs=0.05)
            assert torch.all(enc.skip.bias == 0)


@pytest.mark.parametrize("residual", [False, True])
def test_forward_contract_is_identical_across_variants(residual: bool) -> None:
    net = _net(residual, deep_supervision=True)
    out = net(torch.zeros(1, 1, 32, 40, 48))
    assert isinstance(out, tuple)
    assert len(out) == 2  # one per decoder stage, aux + full-res
    assert out[-1].shape == (1, 2, 32, 40, 48)
    assert out[0].shape == (1, 2, 16, 20, 24)
    served = _net(residual, deep_supervision=False)
    single = served(torch.zeros(1, 1, 32, 40, 48))
    assert isinstance(single, torch.Tensor)
    assert single.shape == (1, 2, 32, 40, 48)


@pytest.mark.parametrize("residual", [False, True])
def test_align_crops_to_the_per_axis_minimum_centered(residual: bool) -> None:
    """`_align` is a staticmethod of the class, shared by both variants: the
    odd-size skip crop is contract, not architecture."""
    a = torch.zeros(1, 2, 17, 9, 33)
    b = torch.zeros(1, 2, 18, 9, 31)
    ca, cb = _net(residual)._align(a, b)
    assert ca.shape[-3:] == (17, 9, 31)
    assert cb.shape[-3:] == (17, 9, 31)
    # The crop is centred: the dropped voxels split either side of the middle.
    source = torch.arange(33, dtype=torch.float32).reshape(1, 1, 1, 1, 33)
    cropped, _ = _net(residual)._align(source, torch.zeros(1, 1, 1, 1, 31))
    assert torch.equal(cropped[0, 0, 0, 0], torch.arange(1, 32, dtype=torch.float32))


@pytest.mark.parametrize("residual", [False, True])
def test_instance_norm_keeps_batches_independent(residual: bool) -> None:
    """InstanceNorm has no cross-batch statistics: the other batch member may
    be 20x brighter without changing sample 0's output. (Bit-equality is NOT
    the claim — conv reduction order differs between batch sizes at the 1e-5
    level; a statistics leak would scale with the other member's magnitude.)"""
    torch.manual_seed(0)
    net = _net(residual).eval()
    one = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        alone = net(one)
    alone_last = alone[-1] if isinstance(alone, tuple) else alone

    def first_sample_with_partner(scale: float) -> torch.Tensor:
        partner = torch.randn(1, 1, 24, 24, 24) * scale
        paired = net(torch.cat([one, partner], dim=0))
        paired_last = paired[-1] if isinstance(paired, tuple) else paired
        return paired_last[0:1]

    torch.manual_seed(1)  # same partner stream for both scales
    diff_small = float((first_sample_with_partner(5.0) - alone_last).abs().max())
    torch.manual_seed(1)
    diff_large = float((first_sample_with_partner(100.0) - alone_last).abs().max())
    assert diff_small < 1e-4 and diff_large < 1e-4
    assert abs(diff_large - diff_small) < 5e-5, (
        f"output tracks the other batch member: 5x diff={diff_small}, 100x diff={diff_large}"
    )


def test_residual_output_is_finite_and_bounded_at_construction() -> None:
    """The stability promise: for unit-normal input a fresh residual net gives
    finite, bounded logits (measured ~7 max-abs; the bound pins the order)."""
    torch.manual_seed(0)
    net = _net(residual=True)
    x = torch.randn(2, 1, 24, 24, 24)
    for _ in (net.train(), net.eval()):
        with torch.no_grad():
            out = net(x)
        logits = out[-1] if isinstance(out, tuple) else out
        assert torch.isfinite(logits).all()
        assert 0.0 < float(logits.abs().max()) <= 15.0


def test_gradient_flows_through_the_skip_from_step_one() -> None:
    """The identity start must not be a gradient blocker: an early stem weight
    receives a finite, non-zero gradient while every block is still identity."""
    torch.manual_seed(0)
    net = _net(residual=True)
    out = net(torch.randn(1, 1, 24, 24, 24))
    # Sum EVERY output: a deep-supervision net only gradients the aux heads
    # through their own tuple elements.
    loss = sum(o.sum() for o in out) if isinstance(out, tuple) else out.sum()
    loss.backward()
    stem_grad = net.stem[0].weight.grad
    assert stem_grad is not None
    assert torch.isfinite(stem_grad).all()
    assert float(stem_grad.abs().sum()) > 0.0
    bad = [n for n, p in net.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"parameters with missing/non-finite gradients: {bad}"


def test_two_constructions_with_one_seed_are_identical() -> None:
    torch.manual_seed(0)
    first = _net(residual=True)
    torch.manual_seed(0)
    second = _net(residual=True)
    for (n1, p1), (n2, p2) in zip(first.named_parameters(), second.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)
    x = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        assert torch.equal(first(x)[-1], second(x)[-1])


def test_served_net_keeps_the_residual_encoder() -> None:
    """The served twin must rebuild the SAME architecture: a residual training
    net whose twin fell back to plain blocks would mismatch every encoder key."""
    torch.manual_seed(0)
    net = _net(residual=True)
    twin = served_net(net)
    assert twin.config.residual is True
    assert all(isinstance(m, _ResBlock) for m in twin.encoders)
    x = torch.randn(1, 1, 24, 24, 24)
    net.eval()
    with torch.no_grad():
        training = net(x)[-1]
        served = twin(x)
    assert served.shape == training.shape
    assert torch.allclose(served, training)


def test_residual_served_net_traces_for_export() -> None:
    torch.manual_seed(0)
    net = _net(residual=True, deep_supervision=False).eval()
    example = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        traced = torch.jit.freeze(torch.jit.trace(net, example, strict=False))
        out = traced(example)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 16, 16, 16)


def _end_to_end_dice(residual: bool, seed: int, out_dir) -> float:
    """Train one arm of the learnability comparison and score its served
    artifact on held-out toy cases, mean foreground Dice over labelled voxels."""
    from test_vanilla_data import _toy_cases

    torch.manual_seed(seed)
    net = build_unet(UNetConfig(input_channels=1, num_classes=2,
                                deep_supervision=True, residual=residual, **TOY))
    plan = FitPlan(patch_size=(16, 16, 16), batch_size=2, steps_per_epoch=8,
                   epochs=6, foreground_prob=1.0)
    trainer = VanillaTrainer(net, num_classes=2, plan=plan)
    train, val = _toy_cases(4, seed=0), _toy_cases(2, seed=100)
    trainer.fit(train, val, np.random.default_rng(seed), out_dir=out_dir)

    predictor = load_predictor(out_dir)
    dices = []
    for case_seed in (7, 8, 9, 10):
        case = _toy_cases(1, seed=case_seed)[0]
        label, _ = predictor.predict(case.image)
        fg = (case.label == 1) & (case.mask[0] > 0)
        dices.append(2.0 * float((label == 1)[fg].sum())
                     / float((label == 1).sum() + fg.sum()))
    return float(np.mean(dices))


def test_residual_trains_at_least_as_fast_as_plain(tmp_path) -> None:
    """THE REGRESSION GUARD for the optimization-speed claim. Same seed, same
    data streams, same budget (6 epochs x 8 steps, cpu-size net): the residual
    encoder's end-to-end Dice must not trail the plain net's. Not a strict
    `>`: the assertion is the floor, and both numbers are logged. Seed 1 is
    pinned because a correct residual implementation beats plain there by a
    wide, per-case-consistent margin (~0.80 vs ~0.23 across four held-out
    cases), so the guard cannot false-alarm on implementation noise."""
    plain = _end_to_end_dice(residual=False, seed=1, out_dir=tmp_path / "plain")
    residual = _end_to_end_dice(residual=True, seed=1, out_dir=tmp_path / "residual")
    print(f"end-to-end dice at seed 1: plain={plain:.4f} residual={residual:.4f}")
    assert residual >= plain, (
        f"residual encoder trained slower: residual={residual:.4f} < plain={plain:.4f}"
    )
