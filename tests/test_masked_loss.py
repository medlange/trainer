# SPDX-License-Identifier: Apache-2.0
"""The masked loss either ignores an unannotated channel or it does not, and this decides it.

WHY THIS FILE IS WRITTEN FIRST AND IS WORTH MORE THAN THE REST OF THE FEATURE'S TESTS
--------------------------------------------------------------------------------------
Every other guard in this feature checks that the right mask was DECLARED, STORED, SEALED
and DELIVERED. This one checks the only thing that ultimately matters: that when the mask
arrives, the fit actually honours it. A perfectly sealed, perfectly audited mask that the
loss quietly ignores produces a model trained to suppress the findings it exists to
detect, and a ValidationReport that says otherwise -- which is worse than no mask at all,
because the claim is now evidenced.

The central test is a PERTURBATION test rather than a value test. It does not assert that
the loss equals some number; it replaces an unsupervised channel's target with noise and
asserts that the scalar loss and EVERY parameter gradient come back bit-identical. A value
test can be satisfied by a loss that is wrong in a way that happens to produce the expected
number. Bit-identity under an arbitrary perturbation cannot.

THE THREE FAILURES THIS IS AIMED AT, each of which was found by reading code rather than
by imagining it:

  1. THE DENOMINATOR. nnU-Net's ignore-label reduction divides a voxel-summed numerator by
     `mask.sum()`. That is right for its spatial [B,1,X,Y,Z] mask and wrong for a per-
     channel [B,C] one: the denominator then counts (case, channel) pairs. For a 128^3
     patch the loss comes out 2,097,152x too large -- and it presents as a bad learning
     rate, not as a masking bug.

  2. THE CONSTANT DICE. A masked channel has I = P = G = 0, so soft Dice returns
     smooth/smooth = 1.0 exactly. The gradient is zero, correctly -- but `dc.mean()` still
     divides by C, so the gradient SCALE on supervised channels becomes (supervised / C)
     and drifts batch to batch with whichever corpora were sampled.

  3. THE FINAL SCALE ONLY. Under deep supervision it is easy to mask the last decoder head
     and leave the coarse ones learning 'unannotated means empty'. The loss looks masked.
     So the perturbation test runs at EVERY scale, separately.

Spec: MOS-TRAIN-141. Register entries for the masking work.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the masked loss is trainer-image code")
import torch.nn as nn  # noqa: E402

_MASKED = Path(__file__).resolve().parents[1] / "medos_trainer" / "masked.py"


def _load():
    """Import the trainer module by path.

    `trainer/` is a separate deployable with its own dependency closure -- the
    platform carries no torch (`MOS-TRAIN-225`), so `medos_trainer` is deliberately not on
    the platform's import path and must not become importable by adding it to one.
    """
    spec = importlib.util.spec_from_file_location("medos_trainer_masked", _MASKED)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


masked = _load()


# --------------------------------------------------------------------------------------
# a network small enough to enumerate every gradient
# --------------------------------------------------------------------------------------


class _Tiny(nn.Module):
    """Three channels out, deliberately shared trunk so a leak on one channel reaches the
    gradients of the parameters feeding all three. A per-channel head with no shared
    weights would let a leak hide."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.trunk = nn.Conv3d(1, 4, 3, padding=1)
        self.head = nn.Conv3d(4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.trunk(x)))


def _fixture(seed: int = 0, *, batch: int = 2, channels: int = 3, size: int = 8, dtype=torch.float64):
    torch.manual_seed(seed)
    net = _Tiny(channels).to(dtype)
    image = torch.randn(batch, 1, size, size, size, dtype=dtype)
    target = (torch.rand(batch, channels, size, size, size, dtype=dtype) > 0.7).to(dtype)
    return net, image, target


def _loss_and_grads(net, image, target, mask, loss_fn):
    net.zero_grad(set_to_none=True)
    value = loss_fn(net(image), target, mask)
    value.backward()
    grads = {name: p.grad.detach().clone() for name, p in net.named_parameters()}
    return value.detach().clone(), grads


# --------------------------------------------------------------------------------------
# 1. THE DECISIVE TEST
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dice", [False, True])
def test_an_unsupervised_channel_cannot_move_the_loss_or_any_gradient(batch_dice: bool) -> None:
    """Replace an unsupervised channel's target with noise. Nothing may change, at all.

    Bit-identity, not approximate equality. A masked contribution is multiplied by exactly
    zero or excluded from a reduction entirely, so the arithmetic is unchanged -- there is
    no floating-point excuse for a difference, and allowing a tolerance here would let a
    small genuine leak pass as rounding.
    """
    net, image, target = _fixture()
    loss_fn = masked.MaskedDiceBCELoss(batch_dice=batch_dice)

    # case 0 supervises channels 0,1 -- not 2.  case 1 supervises channel 2 only.
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    base_loss, base_grads = _loss_and_grads(net, image, target, mask, loss_fn)

    perturbed = target.clone()
    torch.manual_seed(999)
    perturbed[0, 2] = torch.rand_like(perturbed[0, 2])  # unsupervised for case 0
    perturbed[1, 0] = torch.rand_like(perturbed[1, 0])  # unsupervised for case 1
    perturbed[1, 1] = 1.0 - perturbed[1, 1]

    other_loss, other_grads = _loss_and_grads(net, image, perturbed, mask, loss_fn)

    assert torch.equal(base_loss, other_loss), (
        f"the loss moved from {base_loss.item()!r} to {other_loss.item()!r} when only "
        f"UNSUPERVISED channel targets changed. Those voxels are ones nobody annotated; "
        f"letting them reach the loss trains the model to call unlabelled findings "
        f"background, which is the exact defect this mask exists to prevent."
    )
    for name in base_grads:
        assert torch.equal(base_grads[name], other_grads[name]), (
            f"gradient of {name!r} changed when only unsupervised targets changed. "
            f"max |delta| = {(base_grads[name] - other_grads[name]).abs().max().item():.3e}"
        )


def test_the_perturbation_holds_at_every_deep_supervision_scale() -> None:
    """Masking only the finest head is the most likely way to get this wrong, and it looks
    correct from the outside: the reported loss is masked while every coarse decoder head
    keeps learning that unannotated means empty."""
    torch.manual_seed(3)
    scales = [8, 4, 2]
    nets = [_Tiny(3).to(torch.float64) for _ in scales]
    images = [torch.randn(2, 1, s, s, s, dtype=torch.float64) for s in scales]
    targets = [(torch.rand(2, 3, s, s, s, dtype=torch.float64) > 0.7).to(torch.float64) for s in scales]
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    wrapper = masked.MaskedDeepSupervisionWrapper(masked.MaskedDiceBCELoss(), [1.0, 0.5, 0.25])

    def run(ts):
        for net in nets:
            net.zero_grad(set_to_none=True)
        value = wrapper([net(image) for net, image in zip(nets, images)], ts, mask)
        value.backward()
        return value.detach().clone(), [
            {n: p.grad.detach().clone() for n, p in net.named_parameters()} for net in nets
        ]

    base_value, base_grads = run(targets)

    # perturb the unsupervised channels AT EVERY SCALE, one scale at a time as well as all
    for scale_index in range(len(scales)):
        perturbed = [t.clone() for t in targets]
        torch.manual_seed(100 + scale_index)
        perturbed[scale_index][0, 1] = torch.rand_like(perturbed[scale_index][0, 1])
        perturbed[scale_index][1, 0] = torch.rand_like(perturbed[scale_index][1, 0])
        perturbed[scale_index][1, 2] = torch.rand_like(perturbed[scale_index][1, 2])
        value, grads = run(perturbed)
        assert torch.equal(base_value, value), (
            f"perturbing unsupervised targets at deep-supervision scale {scale_index} "
            f"(size {scales[scale_index]}) moved the loss. The mask is not reaching that "
            f"scale, so that decoder head is training on absence-of-annotation as "
            f"background."
        )
        for per_net_base, per_net_other in zip(base_grads, grads):
            for name in per_net_base:
                assert torch.equal(per_net_base[name], per_net_other[name]), (
                    f"scale {scale_index}: gradient of {name!r} moved"
                )


# --------------------------------------------------------------------------------------
# 2. THE DENOMINATOR
# --------------------------------------------------------------------------------------


def test_the_bce_denominator_counts_voxels_not_channel_pairs() -> None:
    """With everything supervised, the masked BCE must equal the plain mean BCE.

    This is the 2,097,152x trap made checkable. `(ce * m).sum() / m.sum()` -- nnU-Net's
    reduction for its SPATIAL mask -- divides a voxel-summed numerator by a count of (case,
    channel) pairs, and with an all-ones mask that ratio is exactly the number of voxels
    per channel. So this test fails loudly under the wrong denominator instead of the loss
    silently reading as a learning-rate problem.
    """
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 4, 4, 4, dtype=torch.float64)
    target = (torch.rand(2, 3, 4, 4, 4, dtype=torch.float64) > 0.5).to(torch.float64)
    mask = torch.ones(2, 3, dtype=torch.float64)

    loss_fn = masked.MaskedDiceBCELoss(weight_dice=0.0, weight_bce=1.0)
    got = loss_fn(logits, target, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    assert torch.allclose(got, expected), (
        f"masked BCE with an all-ones mask is {got.item():.6f} but the plain mean BCE is "
        f"{expected.item():.6f}. Ratio {got.item() / expected.item():.1f}. A ratio near the "
        f"per-channel voxel count ({4 * 4 * 4}) means the denominator is counting (case, "
        f"channel) pairs instead of voxels."
    )


def test_masking_a_case_equals_removing_it_from_the_batch() -> None:
    """The property that makes the mask meaningful: an unsupervised pair must be
    indistinguishable from one that was never in the batch."""
    torch.manual_seed(11)
    logits = torch.randn(3, 2, 4, 4, 4, dtype=torch.float64)
    target = (torch.rand(3, 2, 4, 4, 4, dtype=torch.float64) > 0.5).to(torch.float64)
    loss_fn = masked.MaskedDiceBCELoss()

    mask = torch.ones(3, 2, dtype=torch.float64)
    mask[2] = 0.0  # case 2 supervises nothing

    with_masked_case = loss_fn(logits, target, mask)
    without_the_case = loss_fn(logits[:2], target[:2], torch.ones(2, 2, dtype=torch.float64))
    assert torch.allclose(with_masked_case, without_the_case), (
        f"{with_masked_case.item():.10f} != {without_the_case.item():.10f}: a fully "
        f"unsupervised case is still contributing."
    )


# --------------------------------------------------------------------------------------
# 3. THE CONSTANT DICE
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dice", [False, True])
def test_a_masked_channel_does_not_average_in_a_constant_one(batch_dice: bool) -> None:
    """A masked channel has I = P = G = 0, so soft Dice returns smooth/smooth = 1.0. If
    that value is averaged in, the Dice term is diluted by however many channels the batch
    happened not to supervise -- so the gradient scale on the real channels changes with
    the corpus mix of each batch rather than staying fixed."""
    torch.manual_seed(13)
    logits = torch.randn(2, 4, 4, 4, 4, dtype=torch.float64)
    target = (torch.rand(2, 4, 4, 4, 4, dtype=torch.float64) > 0.5).to(torch.float64)
    loss_fn = masked.MaskedDiceBCELoss(batch_dice=batch_dice, weight_bce=0.0, weight_dice=1.0)

    mask = torch.ones(2, 4, dtype=torch.float64)
    mask[:, 2:] = 0.0  # channels 2 and 3 unsupervised for every case

    four_channels = loss_fn(logits, target, mask)
    two_channels = loss_fn(
        logits[:, :2], target[:, :2], torch.ones(2, 2, dtype=torch.float64)
    )
    assert torch.allclose(four_channels, two_channels), (
        f"Dice over 4 channels with 2 masked is {four_channels.item():.10f} but over the 2 "
        f"supervised channels alone is {two_channels.item():.10f}. The masked channels are "
        f"contributing a constant 1.0 into the mean, so the effective weight on the real "
        f"channels is (supervised / C) and varies with batch composition."
    )


def test_a_batch_that_supervises_nothing_is_zero_and_finite() -> None:
    """Sampling can produce it, and NaN here poisons the weights irrecoverably."""
    net, image, target = _fixture()
    loss_fn = masked.MaskedDiceBCELoss()
    mask = torch.zeros(2, 3, dtype=torch.float64)
    value, grads = _loss_and_grads(net, image, target, mask, loss_fn)
    assert torch.isfinite(value) and float(value) == 0.0, value
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"{name} has non-finite gradient"
        assert float(grad.abs().max()) == 0.0, f"{name} has a gradient from nothing"


# --------------------------------------------------------------------------------------
# 4. THE VALIDATION STATISTIC THAT PICKS THE CHECKPOINT
# --------------------------------------------------------------------------------------


def test_the_confusion_counts_ignore_unsupervised_pairs() -> None:
    """nnU-Net's online pseudo-Dice drives `checkpoint_best`. Unmasked, a channel the batch
    never annotated reports I = P = G = 0, which reads as perfect agreement -- so the
    shipped weights get chosen partly by the absence of labels."""
    torch.manual_seed(17)
    logits = torch.randn(2, 3, 4, 4, 4, dtype=torch.float64)
    target = (torch.rand(2, 3, 4, 4, 4, dtype=torch.float64) > 0.5).to(torch.float64)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    tp, fp, fn, tn = masked.masked_tp_fp_fn_tn(logits, target, mask)
    assert float(tp[1] + fp[1] + fn[1] + tn[1]) == 0.0, (
        "channel 1 is unsupervised in every case of this batch, but the confusion counts "
        "are non-zero, so the pseudo-Dice is measuring it anyway."
    )
    voxels = 4 * 4 * 4
    assert float(tp[0] + fp[0] + fn[0] + tn[0]) == 2 * voxels, "channel 0: both cases"
    assert float(tp[2] + fp[2] + fn[2] + tn[2]) == voxels, "channel 2: one case"


def test_supervised_pair_count_is_what_the_fit_echoes_back() -> None:
    """The number the platform compares against the sealed intent. It must count pairs, so
    that an all-ones mask and a correctly-varied one are distinguishable -- 'masking was
    enabled' is not a check that can fail."""
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert masked.supervised_pair_count(mask) == 3
    assert masked.supervised_pair_count(torch.ones(4, 5)) == 20
    assert masked.supervised_pair_count(torch.zeros(4, 5)) == 0


# --------------------------------------------------------------------------------------
# 5. shape contracts
# --------------------------------------------------------------------------------------


def test_a_mask_that_does_not_match_the_prediction_is_refused() -> None:
    loss_fn = masked.MaskedDiceBCELoss()
    logits = torch.randn(2, 3, 4, 4, 4)
    target = torch.zeros(2, 3, 4, 4, 4)
    with pytest.raises(ValueError, match="first two axes"):
        loss_fn(logits, target, torch.ones(2, 5))
    with pytest.raises(ValueError, match=r"\[B, C\]"):
        loss_fn(logits, target, torch.ones(2, 3, 1))


def test_an_integer_label_map_target_is_refused() -> None:
    """Multi-label heads take a per-channel binary map. Handing this loss nnU-Net's usual
    [B,1,X,Y,Z] integer target would broadcast, not error, and would silently train every
    channel against label ids."""
    loss_fn = masked.MaskedDiceBCELoss()
    with pytest.raises(ValueError, match="multi-label"):
        loss_fn(torch.randn(2, 3, 4, 4, 4), torch.zeros(2, 1, 4, 4, 4), torch.ones(2, 3))


# --------------------------------------------------------------------------------------
# float16, and the reason only the smallest structure ever had a score
# --------------------------------------------------------------------------------------
#
# REGISTER ENTRY 102. `masked_tp_fp_fn_tn` accumulated its confusion counts in
# `logits.dtype`. `validation_step` runs the network under `autocast`, so the logits
# arrive as float16, and the function is called OUTSIDE the autocast block where nothing
# promotes them back. These are COUNTS OF VOXELS -- a 128^3 patch holds 2,097,152 and
# float16 saturates at 65,504.
#
# It presented as the opposite of a numerical bug. Over 70 epochs of the ten-channel fit
# the only channel that ever reported a Dice was `coronary_calcification`: 4,971 voxels
# in the whole dataset, the second smallest structure, and the only one small enough
# never to overflow. `pleural_effusion` at 20.4M voxels read `nan` every epoch. The log
# said the model could do the tiny calcified plaque and nothing else, which is backwards,
# and reads as a hard-example problem rather than as arithmetic.
#
# A direct probe of the epoch-50 checkpoint scored `vertebral_body` at Dice 0.865 while
# the trainer's log reported `nan` for it in every epoch. `checkpoint_best` is selected
# on this metric.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_confusion_counts_do_not_saturate_in_reduced_precision(dtype) -> None:
    """THE DEFECT. 524,288 voxels is eight times float16's ceiling; the old code returned
    `inf` here and every Dice computed from it was `nan`."""
    batch, channels, side = 2, 3, 64  # 2 * 64^3 = 524,288 per channel
    logits = torch.full((batch, channels, side, side, side), 5.0, dtype=dtype)
    target = torch.ones((batch, channels, side, side, side), dtype=dtype)
    mask = torch.ones((batch, channels), dtype=dtype)

    tp, fp, fn, tn = masked.masked_tp_fp_fn_tn(logits, target, mask)
    expected = float(batch * side**3)
    for name, value in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
        assert torch.isfinite(value).all(), f"{name} overflowed under {dtype}"
    assert tp[0].item() == pytest.approx(expected), (
        f"tp is {tp[0].item()} and should be {expected}. Under {dtype} the old "
        f"implementation saturated at 65,504 and returned inf."
    )


def test_float16_would_have_overflowed_without_the_promotion() -> None:
    """Without this, the test above could pass for an implementation that never had the
    problem. It shows the ceiling is real and that 524,288 is genuinely past it."""
    assert torch.finfo(torch.float16).max == 65504.0
    naive = torch.ones((2, 64, 64, 64), dtype=torch.float16).sum()
    assert not torch.isfinite(naive), (
        "summing 524,288 ones in float16 should overflow; if it no longer does, this "
        "file no longer reproduces entry 102 and proves nothing."
    )


def test_the_counts_come_back_in_float32_whatever_went_in() -> None:
    """nnU-Net collates these with `np.sum` across validation iterations, and
    `.cpu().numpy()` on a float16 tensor yields a float16 array that numpy also sums in
    float16 -- so the dtype has to be fixed here, not at the call site."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        counts = masked.masked_tp_fp_fn_tn(
            torch.zeros((1, 2, 4, 4, 4), dtype=dtype),
            torch.zeros((1, 2, 4, 4, 4), dtype=dtype),
            torch.ones((1, 2), dtype=dtype),
        )
        for value in counts:
            assert value.dtype is torch.float32, f"{dtype} produced {value.dtype}"


def test_a_large_structure_and_a_tiny_one_are_both_scored_correctly() -> None:
    """The asymmetry that made the defect invisible, asserted directly: the big channel
    and the small one must both come back with the counts they actually have."""
    batch, side = 1, 64
    logits = torch.full((batch, 2, side, side, side), 5.0, dtype=torch.float16)
    target = torch.zeros((batch, 2, side, side, side), dtype=torch.float16)
    flat = target.view(batch, 2, -1)
    flat[:, 0, :200_000] = 1.0  # large, well past float16's ceiling
    flat[:, 1, :300] = 1.0      # small, comfortably under it
    mask = torch.ones((batch, 2), dtype=torch.float16)

    tp, _, fn, _ = masked.masked_tp_fp_fn_tn(logits, target, mask)
    assert tp[0].item() == pytest.approx(200_000), tp[0].item()
    assert tp[1].item() == pytest.approx(300), tp[1].item()
    assert fn.sum().item() == 0.0


def test_a_zero_weighted_scale_is_skipped_rather_than_multiplied_by_zero() -> None:
    """Moved here from `trainer/tests/test_nnunet_internals.py`, and now about OUR wrapper.

    nnU-Net's weight schedule zeroes the coarsest head -- `_build_loss` does
    `weights[-1] = 0`. Multiplying that scale by zero and skipping it give the same NUMBER
    and not the same behaviour: a term that is still evaluated still runs a forward at the
    coarsest resolution and still builds a graph for a backward that contributes nothing.
    The `continue` in `MaskedDeepSupervisionWrapper.forward` is what makes the zero mean
    "do not compute this", and a refactor that folded it into a multiply would be a
    regression no loss value would reveal.
    """
    calls: list[int] = []

    class _Counter(torch.nn.Module):
        def forward(self, prediction, target, mask):  # noqa: ANN001, ANN201
            calls.append(1)
            return prediction.sum() * 0.0

    wrapper = masked.MaskedDeepSupervisionWrapper(_Counter(), [1.0, 0.0, 0.25])
    tensors = [torch.zeros(1, 1, 2, 2, 2) for _ in range(3)]
    wrapper(tensors, tensors, torch.ones(1, 1))
    assert len(calls) == 2, f"expected the zero-weighted scale to be skipped, got {len(calls)}"


def test_every_weight_zero_is_refused_rather_than_returning_nothing() -> None:
    """The other half of that `continue`: skip every scale and there is no total to return.

    It raises rather than returning a zero tensor, because a fit whose every deep
    supervision weight is zero is a misconfiguration, and a loss of exactly 0.0 that never
    moves is the hardest kind of misconfiguration to notice.
    """
    wrapper = masked.MaskedDeepSupervisionWrapper(masked.MaskedDiceBCELoss(), [0.0, 0.0])
    tensors = [torch.zeros(1, 1, 2, 2, 2) for _ in range(2)]
    with pytest.raises(ValueError, match="every deep supervision weight is zero"):
        wrapper(tensors, tensors, torch.ones(1, 1))
