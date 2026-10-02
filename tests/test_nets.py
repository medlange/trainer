# SPDX-License-Identifier: Apache-2.0
"""The MONAI wrapper against nnU-Net's contract, and against the installed MONAI.

WHY EVERY CLAIM HERE IS EXECUTED AND NOT READ. The wrapper exists to satisfy a protocol nobody
upstream documents as one: it was established by reading nnU-Net 2.6.4's installed source, and
three of its four requirements fail SILENTLY when broken.

  * a deep-supervision list in the wrong ORDER trains against the wrong targets and reports a
    meaningless Dice curve, with no exception anywhere;
  * TOO FEW outputs are truncated by `DeepSupervisionWrapper`'s `zip`, pairing a head with
    another scale's target;
  * a `.decoder.deep_supervision` attribute that does not switch the return type leaves
    validation returning a list and breaks test-time mirroring -- and breaks this repository's
    own packaging, which walks that exact path before tracing.

So the order, the count, the switch and the traceability are all exercised. `test_masked_loss.py`
owns the loss; this file owns the network's side of the same interface.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.nets import (  # noqa: E402
    ARCH_KWARGS_NOT_APPLICABLE,
    MedOSSegResNetDS,
    activation_elements,
)

FIXTURE = TRAINER / "tests" / "fixtures" / "nnunet_plans_plain.json"
#: The keys `get_network_from_plans` resolves through `pydoc.locate` before the call; they arrive
#: as classes, not strings, and this wrapper does not read them.
RESOLVED_KEYS = ("conv_op", "norm_op", "dropout_op", "nonlin")


def _plan_kwargs() -> dict:
    """The architecture kwargs of the REAL plan this cohort trained on."""
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    kwargs = dict(document["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for key in RESOLVED_KEYS:
        kwargs.pop(key, None)
    return kwargs


def _net(**over):
    kwargs = _plan_kwargs()
    kwargs.update(over)
    return MedOSSegResNetDS(input_channels=1, num_classes=10, **kwargs)


# =====================================================================================
# The plan's keys: used, recorded as inapplicable, or refused
# =====================================================================================
def test_the_real_plan_constructs_the_wrapper() -> None:
    """THE WHOLE POINT OF USING THE REAL FIXTURE. A hand-written kwargs dict would be written by
    someone who already knows which keys the wrapper reads."""
    net = _net()
    assert net._heads == 5, "six stages must give five deep-supervision heads"


def test_a_plan_key_the_wrapper_neither_uses_nor_records_is_refused() -> None:
    """A key accepted and ignored is a setting the platform believes it applied -- the hazard the
    run-directory contract states in those words about its own members. An upgrade that adds a
    plan key must break here rather than be silently dropped."""
    with pytest.raises(ValueError, match="neither uses nor records as inapplicable"):
        _net(some_future_key=7)


def test_every_inapplicable_key_carries_a_reason() -> None:
    """A list of ignored keys with no reasons is a list nobody can audit, and the next reader
    cannot tell a deliberate omission from an oversight."""
    assert ARCH_KWARGS_NOT_APPLICABLE
    for key, reason in ARCH_KWARGS_NOT_APPLICABLE.items():
        assert len(reason) > 40, f"{key} has no real reason: {reason!r}"


def test_the_plan_s_own_keys_are_all_accounted_for() -> None:
    """Every key the REAL plan carries is either read by the constructor or recorded as
    inapplicable. This is the gate that fails when nnU-Net's plan format grows a field."""
    import inspect

    read = set(inspect.signature(MedOSSegResNetDS.__init__).parameters)
    for key in _plan_kwargs():
        assert key in read or key in ARCH_KWARGS_NOT_APPLICABLE, key


def test_a_plan_without_strides_is_refused() -> None:
    """`strides` fixes both the stage count and the number of heads the trainer pairs with
    targets. Guessing either makes the loss pair a head with another scale's target."""
    kwargs = _plan_kwargs()
    kwargs.pop("strides")
    with pytest.raises(ValueError, match="no `strides`"):
        MedOSSegResNetDS(input_channels=1, num_classes=10, **kwargs)


def test_a_stage_count_that_contradicts_the_strides_is_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        _net(n_stages=4)


# =====================================================================================
# The output contract
# =====================================================================================
def test_deep_supervision_returns_the_strides_count_minus_one_finest_first() -> None:
    """ORDER AND COUNT, both silent when wrong. The trainer takes `output[0]` as the
    full-resolution prediction and the loss weights decay from index 0; `zip` truncates a short
    list without complaint."""
    net = _net()
    net.eval()                       # nnU-Net asks for the list during validation, in eval()
    outputs = net(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(outputs, list)
    assert len(outputs) == 5, len(outputs)
    shapes = [tuple(o.shape[2:]) for o in outputs]
    assert shapes[0] == (64, 64, 64), f"index 0 is not full resolution: {shapes}"
    assert shapes == sorted(shapes, reverse=True), f"not finest-first: {shapes}"
    for finer, coarser in zip(shapes, shapes[1:]):
        assert all(c * 2 == f for f, c in zip(finer, coarser)), shapes


def test_with_deep_supervision_off_it_returns_one_tensor_at_input_resolution() -> None:
    """A length-one list breaks `torch.flip` during test-time mirroring, which is the crash the
    trainer warns about, and breaks the predictor's `[0]` batch index."""
    net = _net()
    net.decoder.deep_supervision = False
    out = net(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(out, torch.Tensor), type(out)
    assert tuple(out.shape) == (1, 10, 64, 64, 64)


def test_the_decoder_attribute_is_the_switch_both_writers_use() -> None:
    """`set_deep_supervision_enabled` unwraps DDP and `torch.compile` and then writes
    `mod.decoder.deep_supervision`; `packaging.py` walks the same path before tracing. A wrapper
    hiding the flag elsewhere breaks packaging even when training succeeds."""
    net = _net()
    net.eval()
    assert hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision")
    net.decoder.deep_supervision = True
    assert isinstance(net(torch.zeros(1, 1, 64, 64, 64)), list)
    net.decoder.deep_supervision = False
    assert isinstance(net(torch.zeros(1, 1, 64, 64, 64)), torch.Tensor)


def test_forward_leaves_the_module_in_the_mode_it_found_it() -> None:
    """`forward` FORCES the training flag, because MONAI returns its multi-scale list only while
    training and nnU-Net wants that list in `eval()`. Leaving the flag flipped afterwards would
    make the next caller's dropout and normalisation depend on who called first."""
    net = _net()
    net.eval()
    net(torch.zeros(1, 1, 64, 64, 64))
    assert net.net.training is False, "eval() was not restored"
    net.train()
    net(torch.zeros(1, 1, 64, 64, 64))
    assert net.net.training is True, "train() was not restored"


def test_a_patch_too_small_for_the_depth_is_refused_in_our_own_words() -> None:
    """The forced training flag makes instance norm refuse a spatial extent of one with a message
    about TRAINING -- told to a reader who is in `eval()`. Our message names the patch, the
    bottleneck and the depth instead."""
    net = _net()
    net.eval()
    with pytest.raises(ValueError, match="at the bottleneck of 6 stages"):
        net(torch.zeros(1, 1, 32, 32, 32))


def test_it_traces_with_deep_supervision_off() -> None:
    """`packaging.torchscript_bytes` traces the network and refuses unless the traced output is a
    single tensor. An architecture that trains and cannot be traced fails at the end of a run,
    after the card has been paid for."""
    net = _net()
    net.decoder.deep_supervision = False
    net.eval()
    traced = torch.jit.trace(net, torch.zeros(1, 1, 64, 64, 64), strict=False)
    out = traced(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(out, torch.Tensor), type(out)


# =====================================================================================
# The size estimate the planner reads
# =====================================================================================
def test_the_estimate_is_in_the_same_units_as_the_reference_architecture() -> None:
    """THE NUMBER THE PLANNER SIZES A PATCH AGAINST. A number in different units is still a
    number, and nnU-Net would derive a patch from it against `UNet_reference_val_3d` regardless.
    So it is compared with what the reference network returns for the same plan and patch, and
    the two must be within an order of magnitude or the calibration is meaningless.

    THE MEASURED PAIR, which also corrects a claim `nets.py` first made in prose: at the plan's
    geometry SegResNetDS has ELEVEN TIMES the parameters and about TWO THIRDS the activations.
    The planner sizes on activations, so the frozen plan is conservative for this architecture
    rather than too small -- the opposite of what proportionality would suggest.
    """
    import torch.nn as nn
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    kwargs = document["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    reference = PlainConvUNet(
        input_channels=1, n_stages=6, features_per_stage=kwargs["features_per_stage"],
        conv_op=nn.Conv3d, kernel_sizes=kwargs["kernel_sizes"], strides=kwargs["strides"],
        n_conv_per_stage=kwargs["n_conv_per_stage"], num_classes=10,
        n_conv_per_stage_decoder=kwargs["n_conv_per_stage_decoder"], conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    patch = [128, 224, 224]
    theirs = int(reference.compute_conv_feature_map_size(patch))
    ours = int(_net().compute_conv_feature_map_size(patch))

    assert theirs / 10 < ours < theirs * 10, (
        f"the estimates are not on one scale: reference {theirs:.3g} against ours {ours:.3g}. "
        "The planner derives a patch from this number and would size it against nonsense"
    )
    assert ours < theirs, (
        "the measured relation has flipped: this architecture used to need about two thirds of "
        f"the reference's activations. reference {theirs:.3g}, ours {ours:.3g}"
    )
    reference_params = sum(p.numel() for p in reference.parameters())
    our_params = sum(p.numel() for p in _net().parameters())
    assert our_params > reference_params * 5, (
        "the parameter counts no longer diverge from the activation counts, which is the whole "
        f"point of the paragraph this test pins: {our_params} against {reference_params}"
    )


def test_the_estimate_grows_with_the_patch_and_with_the_width() -> None:
    """Both directions, because the planner searches over the patch and a non-monotone estimate
    would make that search wander."""
    net = _net()
    small = net.compute_conv_feature_map_size([64, 64, 64])
    large = net.compute_conv_feature_map_size([128, 128, 128])
    assert large == pytest.approx(small * 8, rel=0.05), (
        f"doubling every axis should multiply the volume by eight: {small} -> {large}"
    )
    narrow = _net(features_per_stage=[8] + list(_plan_kwargs()["features_per_stage"])[1:])
    assert narrow.compute_conv_feature_map_size([64, 64, 64]) < small


def test_an_empty_geometry_is_refused_rather_than_estimated() -> None:
    """A size over nothing is still a number, and the planner would size a patch against it."""
    with pytest.raises(ValueError, match="would be a number the planner"):
        activation_elements(init_filters=0, blocks_down=[1, 2], input_size=[32, 32, 32],
                            out_channels=10)
    with pytest.raises(ValueError, match="would be a number the planner"):
        activation_elements(init_filters=32, blocks_down=[], input_size=[32, 32, 32],
                            out_channels=10)


def test_a_plan_asking_for_dropout_is_refused_rather_than_ignored() -> None:
    """A key that arrives `null` is a key nobody needed; a key with a real value is a setting. The
    plan this cohort trained on sets `dropout_op` to null, so the wrapper accepts it -- but a plan
    that asked for dropout would have it silently vanish, and the plan would still claim it."""
    from medos_trainer.nets import MUST_BE_NONE

    assert "dropout_op" in MUST_BE_NONE
    with pytest.raises(ValueError, match="no dropout to configure"):
        _net(dropout_op="torch.nn.Dropout3d")
    # The null the real plan carries is still fine.
    assert _net(dropout_op=None) is not None


# =====================================================================================
# SwinUNETR: the same contract, a different architecture, and one refusal the other does not need
# =====================================================================================

from medos_trainer.nets import (  # noqa: E402
    MedOSSwinUNETR,
    SWIN_ARCH_KWARGS_NOT_APPLICABLE,
    SWIN_STAGES,
    uniform_halving_stages,
)

ISOTROPIC_SIX = [[1, 1, 1]] + [[2, 2, 2]] * 5


def _swin(**over):
    kwargs = _plan_kwargs()
    kwargs.update(over)
    return MedOSSwinUNETR(input_channels=1, num_classes=10, **kwargs)


def test_the_real_plan_this_cohort_trained_on_constructs_the_swin_wrapper() -> None:
    """The 3d_fullres plan, verbatim, with nothing removed and nothing added."""
    net = _swin()
    assert net._heads == SWIN_STAGES - 1 == 5
    assert net.net.swinViT is not None


def test_the_two_dimensional_plan_is_refused_and_the_message_names_the_head_count() -> None:
    """EIGHT STAGES CANNOT BE SERVED AND THE REFUSAL IS THE POINT.

    `SwinUNETR`'s depth is its construction, not a setting. A wrapper that accepted eight stages
    and returned five heads would hand `DeepSupervisionWrapper` five outputs against seven
    targets, and `zip` truncates: no error, every head paired with the wrong scale, a loss that
    descends on the wrong comparison for the whole fit.
    """
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    two_d = dict(document["configurations"]["2d"]["architecture"]["arch_kwargs"])
    for key in RESOLVED_KEYS:
        two_d.pop(key, None)
    with pytest.raises(ValueError) as raised:
        MedOSSwinUNETR(input_channels=1, num_classes=10, **two_d)
    message = str(raised.value)
    assert "8 stages" in message and "zip truncates" in message


def test_deep_supervision_returns_five_heads_finest_first_halving_every_axis() -> None:
    """THE OUTPUT CONTRACT, against the scales nnU-Net actually builds its targets at.

    The trainer's deep-supervision scales are the cumulative products of the plan's strides with
    the coarsest dropped, so for six isotropic stages they are 1, 1/2, 1/4, 1/8, 1/16. The list
    must be finest-first, because the trainer reads `output[0]` as the full-resolution prediction
    and the loss weights decay from index 0.
    """
    net = _swin(strides=ISOTROPIC_SIX)
    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(1, 1, 32, 64, 64))
    assert isinstance(outputs, list) and len(outputs) == 5
    expected = [(32, 64, 64), (16, 32, 32), (8, 16, 16), (4, 8, 8), (2, 4, 4)]
    assert [tuple(t.shape[2:]) for t in outputs] == expected
    assert all(t.shape[1] == 10 for t in outputs)


def test_only_four_heads_are_ours_and_the_finest_scale_is_monai_own() -> None:
    """Five outputs, four added heads: the full-resolution one comes from MONAI's `self.out`.

    Duplicating it -- a fifth head of our own on the same features -- would leave the architecture
    carrying two differently-initialised full-resolution heads, of which only one is the one a
    published weight file would fill.
    """
    net = _swin()
    assert len(net.heads) == 4


def test_with_deep_supervision_off_it_returns_one_tensor_at_input_resolution() -> None:
    """A LENGTH-ONE LIST IS NOT THE SAME THING. `torch.flip` during test-time mirroring takes a
    tensor, and the predictor's `[0]` would take the first decoder scale instead of the batch --
    which is the crash that produced `[1,10,128,224,224]` against `[10,128,224,224]` in this
    session's first whole-case inference run.
    """
    net = _swin(strides=ISOTROPIC_SIX)
    net.decoder.deep_supervision = False
    net.eval()
    with torch.no_grad():
        single = net(torch.zeros(1, 1, 32, 64, 64))
    assert isinstance(single, torch.Tensor)
    assert tuple(single.shape) == (1, 10, 32, 64, 64)


def test_the_decoder_attribute_is_the_switch_both_writers_use() -> None:
    """`set_deep_supervision_enabled` and this repository's `packaging.py` both write this path.

    Neither knows the wrapper exists; both perform a hard write to
    `network.decoder.deep_supervision`. A flag kept anywhere else trains fine and fails at
    packaging, which is the worst place to find out.
    """
    net = _swin(strides=ISOTROPIC_SIX)
    net.eval()
    assert net.decoder.deep_supervision is True
    net.decoder.deep_supervision = False
    with torch.no_grad():
        assert isinstance(net(torch.zeros(1, 1, 32, 64, 64)), torch.Tensor)
    net.decoder.deep_supervision = True
    with torch.no_grad():
        assert isinstance(net(torch.zeros(1, 1, 32, 64, 64)), list)


def test_the_tap_order_is_the_chain_monai_own_forward_computes() -> None:
    """SUPERSEDED IN PLACE by `test_the_swin_tap_order_still_fails_under_the_general_extractor`.

    This gate's first version carried its own AST walker that recorded only direct calls on the
    receiver. That was enough for SwinUNETR, whose forward is a straight chain, and NOT enough for
    UNETR, which takes its skips from ViT hidden states by index in separate statements -- so the
    walker skipped exactly the lines a wrong index would appear on. Rather than keep two walkers,
    the shared `_dispatch_chain` replaced it and both architectures use the stronger one.

    Kept as a name so that a reader following the proof-by-breaking log to this test finds where it
    went, and asserting the thing that motivated the replacement.
    """
    from monai.networks.nets import SwinUNETR

    assert _dispatch_chain(MedOSSwinUNETR.forward, "net") == _dispatch_chain(
        SwinUNETR.forward, "self")


def test_an_anisotropic_plan_is_refused_by_both_wrappers() -> None:
    """THE BRANCH NOTHING IN THIS COHORT EXERCISES, which is why it is written and not assumed.

    Every 3D configuration in both frozen plans is `[[1,1,1], [2,2,2], ...]`. nnU-Net produces a
    final `[1, 2, 2]` for a thick-slice cohort, and both architectures here halve every axis, so
    the head at that stage would be twice its target's depth. The error would surface inside a
    Dice as a shape mismatch with no mention of strides.
    """
    anisotropic = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    with pytest.raises(ValueError, match="anisotropic"):
        uniform_halving_stages(anisotropic, architecture="probe")
    with pytest.raises(ValueError, match="anisotropic"):
        _swin(strides=anisotropic)
    with pytest.raises(ValueError, match="anisotropic"):
        _net(strides=anisotropic)
    assert uniform_halving_stages(ISOTROPIC_SIX, architecture="probe") == 6


def test_a_first_stride_that_downsamples_is_refused_too() -> None:
    """nnU-Net's first stage never strides, and a plan whose first one does shifts every scale.

    Checked separately from the anisotropic case because a single loop over "all twos" would pass
    a plan that begins by halving -- and then head `i` sits one scale below target `i` all the way
    down, with every shape a power of two and none of them the right one.
    """
    with pytest.raises(ValueError, match=r"stride 0 is \[2, 2, 2\]"):
        uniform_halving_stages([[2, 2, 2]] * 6, architecture="probe")


def test_a_feature_size_the_attention_heads_cannot_divide_is_refused_naming_the_plan() -> None:
    """MONAI's own rule, raised where the number came from.

    Its message is "feature_size should be divisible by 12" and says nothing about the plan. This
    cohort's plans open at 32 channels, and the useful refusal is the one that says so.
    """
    with pytest.raises(ValueError) as raised:
        _swin(feature_size=32)
    message = str(raised.value)
    # NOT "divisible by 12" -- MONAI'S OWN MESSAGE CONTAINS THAT TOO, so asserting it left this
    # gate green while the refusal was removed entirely. Proved by breaking. What only ours says
    # is where 32 came from and what to use instead.
    assert "feature_size=32" in message
    assert "this cohort's plans open at 32 channels" in message.lower()
    assert "24 and 48" in message
    assert _swin(feature_size=24)._heads == 5


def test_a_patch_that_cannot_survive_five_halvings_is_refused_in_our_own_words() -> None:
    """MONAI raises here too, and its message is about `patch_size**5` rather than the plan.

    The plan's patch is derived by a conv planner, which is under no obligation to produce one
    divisible by 32, so this is a refusal a real run can meet.
    """
    net = _swin(strides=ISOTROPIC_SIX)
    net.eval()
    with pytest.raises(ValueError) as raised:
        with torch.no_grad():
            net(torch.zeros(1, 1, 32, 64, 48))
    message = str(raised.value)
    assert "not divisible by 32" in message and "axes [4]" in message
    assert "a plan derived for a conv U-Net" in message


def test_the_size_probe_refuses_rather_than_handing_the_planner_an_underestimate() -> None:
    """AND THE MEASURED REASON IS IN THE MESSAGE'S OWN TEST BELOW.

    `compute_conv_feature_map_size` is called by one place only -- the experiment planner, which
    divides its VRAM target by the number and compares against a value measured on a conv U-Net.
    Returning a conv-shaped estimate for an attention architecture underestimates by a factor of
    two here, in the direction that sizes the patch too large, and the planner cannot read the
    comment that would have admitted it.
    """
    net = _swin()
    with pytest.raises(NotImplementedError) as raised:
        net.compute_conv_feature_map_size((128, 224, 224))
    assert "UNet_reference_val_3d" in str(raised.value)


def test_the_footprint_the_module_docstring_claims_is_the_measured_one() -> None:
    """THE THREE NUMBERS IN THE DOCSTRING, COUNTED. A paragraph, otherwise, is a recollection.

    Counted at a reduced patch and scaled: activations grow linearly in voxel count for both the
    convolution path and -- at a fixed window size -- the attention matrices, since the number of
    windows is proportional to the volume. The point the docstring rests on is the RATIO between
    the conv term and the attention term, which is what makes a conv-shaped estimate wrong by
    about half rather than by a few percent.
    """
    from torch import nn

    net = _swin(strides=ISOTROPIC_SIX, feature_size=48, use_checkpoint=False)
    net.eval()
    counted = {"conv": 0, "attention": 0}

    def observe(module, _inputs, output):
        if isinstance(output, torch.Tensor):
            counted["attention" if isinstance(module, nn.Softmax) else "conv"] += output.numel()

    handles = [m.register_forward_hook(observe) for m in net.modules()
               if isinstance(m, (nn.Conv3d, nn.Linear, nn.Softmax, nn.LayerNorm))]
    try:
        with torch.no_grad():
            net(torch.zeros(1, 1, 32, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    scale = (128 * 224 * 224) / (32 * 64 * 64)
    conv = counted["conv"] * scale
    attention = counted["attention"] * scale
    assert conv == pytest.approx(3.78e9, rel=0.02), "%.3e" % conv
    assert attention == pytest.approx(4.02e9, rel=0.02), "%.3e" % attention
    assert attention > conv, (
        "the attention term no longer dominates, so the docstring's argument for refusing the "
        f"size probe is stale: conv {conv:.3e} against attention {attention:.3e}"
    )
    assert sum(p.numel() for p in net.parameters()) == pytest.approx(62.2e6, rel=0.01)


def test_gradient_checkpointing_is_on_unless_a_caller_turns_it_off() -> None:
    """A DEFAULT THAT IS A DECISION. 31 GB of activations at the plan's batch size of 2, on a
    32 GB card; the encoder's attention is the term recomputation removes. A wrapper defaulting to
    off would OOM at the first epoch of every fit anyone queued without reading this file.
    """
    assert _swin().net.swinViT.layers1[0].blocks[0].use_checkpoint is True
    assert _swin(use_checkpoint=False).net.swinViT.layers1[0].blocks[0].use_checkpoint is False


def test_every_inapplicable_swin_key_carries_a_reason_and_the_plan_s_keys_are_covered() -> None:
    """A key accepted and ignored is a setting the platform believes it applied.

    Both halves matter: a reason nobody wrote makes the dictionary a silence list, and a plan key
    in neither the signature nor the dictionary would be swallowed by `**plan_kwargs`.
    """
    for key, reason in SWIN_ARCH_KWARGS_NOT_APPLICABLE.items():
        assert len(reason) > 40, f"{key} has no real reason: {reason!r}"
    import inspect
    signature = set(inspect.signature(MedOSSwinUNETR.__init__).parameters)
    unaccounted = sorted(set(_plan_kwargs()) - signature - set(SWIN_ARCH_KWARGS_NOT_APPLICABLE))
    assert not unaccounted, f"the plan carries {unaccounted}, which nothing uses or records"


def test_a_plan_asking_swin_for_dropout_is_refused_rather_than_recorded() -> None:
    """The one distinction `MUST_BE_NONE` exists for: a key nobody needed against a setting that
    disappeared. MONAI takes three separate rates here and no operation, so a requested dropout
    cannot be honoured -- and must not be logged as inapplicable either.
    """
    with pytest.raises(ValueError, match="dropout"):
        _swin(dropout_op=torch.nn.Dropout3d)
    assert _swin(dropout_op=None) is not None


# =====================================================================================
# UNETR: the same contract again, plus a patch coupling neither of the others has
# =====================================================================================

from medos_trainer.nets import (  # noqa: E402
    MedOSUNETR,
    UNETR_ARCH_KWARGS_NOT_APPLICABLE,
    UNETR_PATCH,
    UNETR_STAGES,
)


def _dispatch_chain(function, receiver):
    """The ordered submodule dispatch of a `forward`, normalised so local NAMES do not matter.

    WHY A NORMALISER AND NOT A TEXT COMPARISON. Two wrappers here re-dispatch through MONAI's own
    submodules, because MONAI's forward computes the tensors the deep-supervision heads need and
    returns one. The hazard is a MONAI upgrade that reorders a decoder, renames a skip, or -- for
    UNETR -- changes WHICH ViT layers the skips are taken from. Every shape would still match,
    every head would still emit logits, and the features would be wrong. Nothing in training would
    say so: the loss would simply be worse, which is indistinguishable from a worse architecture.

    Each assignment becomes a tuple naming the submodule called and, for every argument, WHICH
    EARLIER STEP produced it -- so `enc2 = self.encoder2(self.proj_feat(x2))` where
    `x2 = hidden_states_out[3]` normalises to a structure that carries the 3. The first version of
    this helper recorded only direct calls on the receiver, which skipped the `x2 = ...[3]` line
    entirely and so could not have caught a wrong layer index at all.

    A `return` whose value is a call on the receiver counts as a final step, because MONAI's two
    forwards differ in whether they assign the last head's output or return it directly.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)).body
    produced: dict[str, tuple] = {}
    steps: list[tuple] = []

    def norm(node):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == receiver):
                return ("call", node.func.attr, tuple(norm(a) for a in node.args))
            return ("other_call", tuple(norm(a) for a in node.args))
        if isinstance(node, ast.Subscript):
            return ("index", norm(node.value), ast.literal_eval(node.slice))
        if isinstance(node, ast.Name):
            return produced.get(node.id, ("input",))
        if isinstance(node, ast.Attribute):
            return ("attr", node.attr)
        return ("opaque",)

    def record(value, names):
        steps.append(value)
        index = len(steps) - 1
        if len(names) == 1:
            produced[names[0]] = ("step", index)
        else:
            for position, name in enumerate(names):
                produced[name] = ("unpack", index, position)

    for statement in body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = norm(statement.value)
            if value[0] not in ("call", "index"):
                continue
            if isinstance(target, ast.Name):
                record(value, [target.id])
            elif isinstance(target, ast.Tuple) and all(
                    isinstance(e, ast.Name) for e in target.elts):
                record(value, [e.id for e in target.elts])
        elif isinstance(statement, ast.Return) and statement.value is not None:
            value = norm(statement.value)
            if value[0] == "call":
                steps.append(value)
    return steps


UNETR_ISOTROPIC = [[1, 1, 1]] + [[2, 2, 2]] * 5
UNETR_SMALL = [64, 64, 64]


def _unetr(**over):
    kwargs = _plan_kwargs()
    kwargs.setdefault("img_size", UNETR_SMALL)
    kwargs.update(over)
    return MedOSUNETR(input_channels=1, num_classes=10, **kwargs)


def test_the_real_plan_constructs_the_unetr_wrapper_at_the_plan_s_own_patch() -> None:
    """The 3d_fullres plan verbatim, with `img_size` taken from the plan's `patch_size`."""
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    patch = document["configurations"]["3d_fullres"]["patch_size"]
    net = _unetr(img_size=patch)
    assert net._heads == UNETR_STAGES - 1 == 5
    assert net._img_size == tuple(patch)


def test_unetr_deep_supervision_returns_five_heads_finest_first() -> None:
    """Five scales, measured: 1, 1/2, 1/4, 1/8 and 1/16, the last being the ViT's own grid.

    UNETR's decoder has four `UnetrUpBlock` levels from the grid at 1/16, so the fifth and coarsest
    head sits on `proj_feat(x)` -- the projected ViT output, before any decoder block, which is the
    only tensor at that resolution and is what `decoder5` consumes.
    """
    net = _unetr(strides=UNETR_ISOTROPIC)
    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(outputs, list) and len(outputs) == 5
    assert [tuple(t.shape[2:]) for t in outputs] == [
        (64, 64, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
    assert all(t.shape[1] == 10 for t in outputs)
    assert len(net.heads) == 4, "the finest scale is MONAI's own head and must not be duplicated"


def test_unetr_with_deep_supervision_off_returns_one_tensor_at_input_resolution() -> None:
    net = _unetr(strides=UNETR_ISOTROPIC)
    net.decoder.deep_supervision = False
    net.eval()
    with torch.no_grad():
        single = net(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(single, torch.Tensor)
    assert tuple(single.shape) == (1, 10, 64, 64, 64)


def test_the_unetr_tap_order_including_the_vit_layer_indices_is_monai_own() -> None:
    """THE GATE THAT MAKES RE-DISPATCHING SAFE FOR UNETR, and it goes further than Swin's needs to.

    UNETR takes its three encoder skips from ViT hidden states 3, 6 and 9. Those indices are a
    choice MONAI made -- evenly spaced through twelve layers -- and nothing about a wrong one is
    visible downstream: hidden state 4 has exactly the shape of hidden state 3. A comparison that
    only checked which submodules were called in which order would pass while the skips came from
    the wrong depth.

    `_dispatch_chain` therefore carries each argument's producer, so `x2 = hidden_states_out[3]`
    contributes the 3 and a change to it makes the two chains differ.
    """
    from monai.networks.nets import UNETR

    theirs = _dispatch_chain(UNETR.forward, "self")
    ours = _dispatch_chain(MedOSUNETR.forward, "net")
    assert ours == theirs, (
        "the wrapper no longer walks MONAI's own chain.\n  MONAI: %r\n  ours:  %r" % (theirs, ours))

    called = [step[1] for step in theirs if step[0] == "call"]
    # THE TOP-LEVEL STEPS ONLY. The three `self.proj_feat(...)` calls that feed encoder2, encoder3
    # and encoder4 are ARGUMENTS, nested inside those steps rather than steps of their own, and the
    # comparison above covers them there -- which is where the layer indices live too. The one
    # `proj_feat` listed here is `dec4 = self.proj_feat(x)`, the only one assigned to a name.
    # A first version of this list guessed them as separate steps and was simply wrong about the
    # shape of the chain; the measured list is below.
    assert called == ["vit", "encoder1", "encoder2", "encoder3", "encoder4", "proj_feat",
                      "decoder5", "decoder4", "decoder3", "decoder2", "out"], (
        "MONAI's chain is not the one this wrapper was written against, so the comparison above is "
        f"now two wrappers agreeing with each other: {called}")
    indices = sorted(step[2] for step in theirs if step[0] == "index")
    assert indices == [3, 6, 9], (
        f"the ViT layers the skips come from are {indices} and this wrapper was written for "
        "[3, 6, 9]; hidden state 4 has the same shape as hidden state 3, so nothing downstream "
        "would object")


def test_the_swin_tap_order_still_fails_under_the_general_extractor() -> None:
    """The same comparison for SwinUNETR, through the helper UNETR needed.

    Kept as a separate gate rather than folded into the one above because the two architectures'
    chains are different lengths and a single parametrised assertion would have to know which
    submodule list belongs to which -- and a list applied to the wrong architecture is a gate that
    passes for the wrong reason.
    """
    from monai.networks.nets import SwinUNETR

    theirs = _dispatch_chain(SwinUNETR.forward, "self")
    ours = _dispatch_chain(MedOSSwinUNETR.forward, "net")
    assert ours == theirs, (
        "the Swin wrapper no longer walks MONAI's own chain.\n  MONAI: %r\n  ours:  %r"
        % (theirs, ours))
    assert [step[1] for step in theirs if step[0] == "call"] == [
        "swinViT", "encoder1", "encoder2", "encoder3", "encoder4", "encoder10",
        "decoder5", "decoder4", "decoder3", "decoder2", "decoder1", "out"]
    assert [step[2] for step in theirs if step[0] == "index"] == [], (
        "SwinUNETR now indexes its hidden states in separate statements; the index assertion from "
        "the UNETR gate should be added here too")


def test_unetr_without_an_img_size_is_refused_rather_than_defaulted() -> None:
    """A DEFAULT HERE WOULD BE A NETWORK BUILT FOR A PATCH NOBODY IS FEEDING IT.

    `proj_feat` freezes the token grid at construction. A guessed `img_size` that disagreed with
    the patch the trainer crops raises a `view` error deep in the forward pass -- or, when the two
    products coincide, reshapes into the wrong grid and trains on scrambled features without
    raising at all.
    """
    kwargs = _plan_kwargs()
    with pytest.raises(ValueError) as raised:
        MedOSUNETR(input_channels=1, num_classes=10, **kwargs)
    message = str(raised.value)
    assert "needs `img_size`" in message and "plan_derived_kwargs" in message


def test_a_patch_not_divisible_by_the_hard_coded_vit_patch_is_refused() -> None:
    """MONAI DOES NOT CHECK THIS, WHICH IS THE WHOLE REASON THE CHECK IS HERE.

    `self.patch_size = ensure_tuple_rep(16, spatial_dims)` is written with the 16 inline upstream,
    and `feat_size = img_size // 16` floors. A patch of 100 gives six tokens per axis, reshapes
    without complaint, and trains on 96 of every 100 voxels along that axis with nothing anywhere
    saying so. There is no exception to raise from below, so the refusal has to be ours.
    """
    assert UNETR_PATCH == 16
    with pytest.raises(ValueError) as raised:
        _unetr(img_size=[100, 64, 64])
    message = str(raised.value)
    assert f"not divisible by {UNETR_PATCH}" in message and "axes [2]" in message
    assert "floors instead of refusing" in message
    # AND THE NEAR MISS: 48 IS divisible by 16 and must be accepted. An assertion built on a
    # number that only looked indivisible was this gate's first version.
    assert _unetr(img_size=[64, 64, 48])._img_size == (64, 64, 48)


def test_a_patch_leaving_a_single_token_on_an_axis_is_refused() -> None:
    """Divisible by 16 is not sufficient: 16 itself leaves one token, and one cannot be upsampled
    through four decoder levels or normalised by the instance norm the plan asks for."""
    with pytest.raises(ValueError, match=r"ViT grid of \[1, 4, 4\]"):
        _unetr(img_size=[16, 64, 64])


def test_a_forward_at_a_patch_other_than_the_one_it_was_built_for_is_refused() -> None:
    """THE COUPLING MADE LOUD, and the message names the cause rather than the symptom.

    MONAI's own failure here is "shape [...] is invalid for input of size N" from inside
    `proj_feat`, which mentions neither a patch nor a plan. And when the two patches' token counts
    coincide it does not fail at all -- it reshapes into a different grid.
    """
    net = _unetr(strides=UNETR_ISOTROPIC)
    net.eval()
    with pytest.raises(ValueError) as raised:
        with torch.no_grad():
            net(torch.zeros(1, 1, 32, 64, 64))
    message = str(raised.value)
    assert "was built for a patch of (64, 64, 64)" in message
    assert "(32, 64, 64)" in message


def test_the_unetr_plan_of_the_wrong_depth_is_refused() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    two_d = dict(document["configurations"]["2d"]["architecture"]["arch_kwargs"])
    for key in RESOLVED_KEYS:
        two_d.pop(key, None)
    with pytest.raises(ValueError) as raised:
        MedOSUNETR(input_channels=1, num_classes=10, img_size=[512, 512], **two_d)
    assert "8 stages" in str(raised.value) and "zip truncates" in str(raised.value)


def test_unetr_refuses_the_size_probe_and_says_it_cannot_be_resized() -> None:
    """Two reasons, and the second is UNETR's alone.

    The planner's procedure is to try a patch, size the network, shrink, and try again. This network
    cannot be resized -- `img_size` is frozen into `proj_feat` -- so each probe would need a new
    network and the settled patch would have to be written back into `arch_kwargs`. Nothing
    upstream does that, so a number returned here would yield a plan whose patch and whose network
    disagree.
    """
    with pytest.raises(NotImplementedError) as raised:
        _unetr().compute_conv_feature_map_size((128, 224, 224))
    message = str(raised.value)
    assert "UNet_reference_val_3d" in message
    assert "cannot be resized" in message


def test_the_unetr_footprint_the_docstring_claims_is_measured_and_its_attention_is_real() -> None:
    """THE FOUR-ROW TABLE'S UNETR ROW, and the fact the row depends on.

    UNETR's attention term cannot be counted by hooking `nn.Softmax`: MONAI's `SABlock` computes
    `att_mat` through an `einsum` and there is no module to hook, which is why a first measurement
    of it read ZERO and would have supported a claim that UNETR materialises no attention at all.
    So the term is computed in closed form -- twelve layers, twelve heads, tokens squared -- and
    what is CHECKED here is the premise that makes the closed form right: that `use_flash_attention`
    is false on every block, since flash attention never materialises the matrix.
    """
    from torch import nn

    net = _unetr(strides=UNETR_ISOTROPIC)
    net.eval()
    blocks = [m for m in net.modules() if type(m).__name__ == "SABlock"]
    assert len(blocks) == 12
    assert {b.use_flash_attention for b in blocks} == {False}, (
        "UNETR's blocks now use flash attention, so they no longer materialise the attention "
        "matrix and the module docstring's 3.54e8 is an overestimate rather than a measurement"
    )
    assert {b.num_heads for b in blocks} == {12}

    counted = 0
    handles = []

    def observe(_module, _inputs, output):
        nonlocal counted
        if isinstance(output, torch.Tensor):
            counted += output.numel()

    for module in net.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear, nn.LayerNorm)):
            handles.append(module.register_forward_hook(observe))
    try:
        with torch.no_grad():
            net(torch.zeros(1, 1, 64, 64, 64))
    finally:
        for handle in handles:
            handle.remove()

    scale = (128 * 224 * 224) / (64 * 64 * 64)
    conv = counted * scale
    tokens = (128 // UNETR_PATCH) * (224 // UNETR_PATCH) * (224 // UNETR_PATCH)
    attention = 12 * 12 * tokens ** 2
    assert tokens == 1568
    assert conv == pytest.approx(1.20e9, rel=0.03), "%.3e" % conv
    assert attention == pytest.approx(3.54e8, rel=0.01), "%.3e" % attention
    assert attention < conv / 2, (
        f"UNETR's attention term {attention:.3e} is no longer small beside its convolution term "
        f"{conv:.3e}, so the docstring's comparison with SwinUNETR is stale"
    )


def test_every_inapplicable_unetr_key_carries_a_reason_and_the_plan_s_keys_are_covered() -> None:
    for key, reason in UNETR_ARCH_KWARGS_NOT_APPLICABLE.items():
        assert len(reason) > 40, f"{key} has no real reason: {reason!r}"
    import inspect
    signature = set(inspect.signature(MedOSUNETR.__init__).parameters)
    unaccounted = sorted(
        set(_plan_kwargs()) - signature - set(UNETR_ARCH_KWARGS_NOT_APPLICABLE))
    assert not unaccounted, f"the plan carries {unaccounted}, which nothing uses or records"


def test_a_width_the_attention_heads_cannot_divide_is_refused_in_our_own_words() -> None:
    """MONAI refuses this too, and its message does not say where the numbers came from.

    Asserting the substring MONAI's own message shares -- "divisible by num_heads" -- would leave
    this gate green if the check were deleted outright, which is how the SwinUNETR `feature_size`
    gate was found blind. So the assertion is on the part only ours carries.
    """
    with pytest.raises(ValueError) as raised:
        _unetr(hidden_size=770)
    message = str(raised.value)
    assert "hidden_size=770" in message
    assert "768 over 12" in message


def test_a_plan_asking_unetr_for_dropout_is_refused_rather_than_recorded() -> None:
    """UNETR takes a `dropout_rate` FLOAT, so a requested dropout OPERATION cannot be honoured --
    and must not be logged as inapplicable either, which would make a requested regularisation
    vanish while the plan still claimed it."""
    with pytest.raises(ValueError, match="dropout"):
        _unetr(dropout_op=torch.nn.Dropout3d)
    assert _unetr(dropout_op=None) is not None


def test_an_anisotropic_plan_is_refused_by_all_three_wrappers() -> None:
    """The third wrapper joins the guard, so the shared refusal covers everything in this file."""
    anisotropic = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    with pytest.raises(ValueError, match="anisotropic"):
        _unetr(strides=anisotropic)
