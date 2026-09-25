# SPDX-License-Identifier: Apache-2.0
"""What nnU-Net 2.5.1 actually does, asserted rather than read from documentation.

WHY THIS GATE EXISTS
--------------------
The partial-label design rests on four claims about a third-party library's internals.
Every one of them was, at design time, an unverified reading of source code -- and one of
the three independent designs that produced this plan got a load-bearing detail wrong (it
proposed a top-level `regions` key in dataset.json, which is not a thing). A design resting
on four unverified claims about someone else's private API is a design resting on nothing.

So this file establishes each claim by assertion against the installed pin, before any
feature code depends on it. It is the cheapest possible place to discover that an upgrade
renamed `_build_loss`.

WHY IT MUST NOT SKIP
--------------------
A skipped gate is not a gate. `tests/integration/test_trainer_boundary.py` already skips
when the stack is absent, and that is correct for a test that needs a running container --
this one needs only an installed package, so "nnunetv2 is not installed" is a real failure
of a real precondition and is reported as one. It is kept out of the default unit run by
selection (`-m trainer_internals`, its own CI job) rather than by skipping, so that a run
which does not include it says "deselected" rather than "passed".

WHAT WAS FOUND, AND THE ONE THING THAT WENT THE WRONG WAY
----------------------------------------------------------
(a) FAILED as predicted. `LabelManager` sets `_has_regions` from
    `any(isinstance(i, (tuple, list)) and len(i) > 1 ...)`, so label values that are
    SINGLETON lists -- which is what one channel per finding looks like -- report
    `has_regions == False` and the model silently gets `softmax_helper_dim0`. Multi-label
    sigmoid heads do not happen by themselves and asking for them in dataset.json is not
    enough. The decision rule was fixed in advance and is taken: force the flag in a
    `LabelManager` subclass, recompute the regions, and record it as a declared deviation
    rather than improvise.

(b), (c), (d) all came back usable. (c) is the load-bearing one: the trainer's own
    `train_step` reads only `data` and `target`, but the dataloader DOES put case
    identifiers in the batch under `keys` -- which is the hop the per-case channel mask
    travels on, and there was no second candidate if it had been absent.

Spec: MOS-REL-032 (declared deviations), MOS-TRAIN-141, MOS-TRAIN-225.
"""

from __future__ import annotations

import inspect

import pytest

# SELECTED BY PATH, not by a registered marker.
#
# Registering a marker means editing pyproject.toml, and `tests/gate/test_zero_core_change.py`
# permits the 0.3.0 release exactly one change to that file -- its own gate marker -- so a
# second registration turns that gate red. It did, and the commit that introduced it claimed
# green having run tests/unit but not tests/gate.
#
# A path selects this suite just as well: `pytest trainer/tests`. What the marker was for --
# keeping it out of the default unit run -- the directory already achieves, because
# `pytest tests/unit` does not reach here.

# Deliberately a hard import. See the module docstring: this gate reports a missing pin as
# a failure, because the pin is the thing it exists to check.
import torch  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.label_handling.label_handling import LabelManager  # noqa: E402

#: One channel per finding, which is what a multi-label label_set looks like: each value is
#: a list of exactly one label id.
SINGLETON_LABELS = {"background": 0, "neo": [1], "benign": [2]}


class MultiLabelManager(LabelManager):
    """Force region mode for singleton label values.

    THE DECLARED DEVIATION. `_has_regions` is private and this subclass sets it, which is
    reaching into another project's internals -- the thing `MOS-REL-032` wants declared
    rather than done quietly. It is done because the alternative is worse in a specific
    way: the only supported route to sigmoid heads is to spell a channel as a region of
    two or more labels, and our channels genuinely are one label each. Writing
    `"neo": [1, 1]` to satisfy the check would be lying to the library about the data in
    order to get the behaviour we want, and that lie would travel into `dataset.json`,
    into the plans file and into the packaged bundle.

    Setting the flag and recomputing is narrower and visible. `_get_regions` is called
    after `_has_regions` in `__init__`, so recomputing in that order is the same sequence
    the library itself runs -- this is not reconstructing its logic, it is re-running it.
    """

    def __init__(self, label_dict, regions_class_order=None, **kwargs):
        super().__init__(label_dict, regions_class_order, **kwargs)
        self._has_regions = True
        self._regions = self._get_regions()
        self.inference_nonlin = torch.sigmoid


# --------------------------------------------------------------------------------------
# (a) the claim that failed
# --------------------------------------------------------------------------------------


def test_singleton_label_values_do_not_get_region_mode_on_their_own() -> None:
    """The finding this whole gate existed to make. If this ever starts passing as
    `has_regions == True`, nnU-Net changed its rule and `MultiLabelManager` is no longer
    needed -- which is worth knowing loudly, because the subclass would then be reaching
    into a private attribute for no reason."""
    manager = LabelManager(SINGLETON_LABELS, regions_class_order=[1, 2])
    assert manager.has_regions is False, (
        "singleton label values now report has_regions=True. nnU-Net's rule changed; "
        "re-read MultiLabelManager, it may be unnecessary."
    )
    assert manager.inference_nonlin is not torch.sigmoid, (
        "without region mode the head is softmax, which is the whole problem: softmax "
        "outputs sum to one, so masking a channel out still changes what the remaining "
        "channels must predict, and co-occurring findings are unrepresentable."
    )
    # And the reason it matters: the background head is counted.
    assert manager.num_segmentation_heads == 3, manager.num_segmentation_heads


def test_forcing_region_mode_yields_one_sigmoid_head_per_channel() -> None:
    """The declared deviation, asserted to actually do what it claims."""
    manager = MultiLabelManager(SINGLETON_LABELS, regions_class_order=[1, 2])
    assert manager.has_regions is True
    assert manager.all_regions == [(1,), (2,)], manager.all_regions
    assert manager.foreground_regions == [(1,), (2,)], manager.foreground_regions
    assert manager.inference_nonlin is torch.sigmoid
    assert manager.num_segmentation_heads == 2, (
        "region mode must drop the background head: C channels in, C heads out."
    )


def test_the_head_count_disagrees_with_the_label_set_size_by_exactly_one() -> None:
    """THE PACKAGING TRAP, pinned here rather than discovered in a served model.

    `packaging.py` builds `channel_def` from the full `label_set` INCLUDING background,
    while `num_channels` comes from `num_segmentation_heads`. Under region mode those
    differ by one, so every declared channel name sits one index off the tensor it names.
    `bundle.py` compares `channel_def` only against the manifest's `label_map` -- both
    derived from the same `label_set` -- so the two agree with each other and disagree
    with the network. A structurally valid bundle, served at the wrong indices.
    """
    manager = MultiLabelManager(SINGLETON_LABELS, regions_class_order=[1, 2])
    assert len(SINGLETON_LABELS) == manager.num_segmentation_heads + 1, (
        "the off-by-one this test pins has changed shape; re-read packaging.py:370-378 "
        "before trusting any channel_def it writes."
    )


# --------------------------------------------------------------------------------------
# (b) the hooks the trainer subclass overrides
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("hook", ["_build_loss", "train_step", "validation_step", "initialize"])
def test_the_trainer_exposes_the_hook_the_design_overrides(hook: str) -> None:
    """Each of these is named in the build order. A rename in an upgrade would otherwise
    surface as a subclass that silently stops taking effect -- the override would simply
    never be called, the fit would run unmasked, and nothing would raise."""
    assert callable(getattr(nnUNetTrainer, hook, None)), (
        f"nnUNetTrainer has no {hook!r}. The masked-training subclass overrides it; an "
        f"override of a method that no longer exists is not an error, it is a silent "
        f"no-op, and the fit would train every channel on every case."
    )


# --------------------------------------------------------------------------------------
# (c) the hop the mask travels on
# --------------------------------------------------------------------------------------


def test_the_batch_carries_case_identifiers_under_keys() -> None:
    """THE LOAD-BEARING CLAIM. The per-case mask has to be joined to the samples in a
    batch, and the only thing that can join them is a case identifier. nnU-Net's own
    `train_step` reads `data` and `target` and ignores the rest, but the dataloader puts
    identifiers in the batch under `keys`, so an overridden `train_step` can look them up.

    There was no second candidate. If this were absent the design would have needed the
    mask baked into the staged label files, which cannot represent 'unknown' distinctly
    from 'absent'.
    """
    import nnunetv2.training.dataloading.data_loader_3d as loader_3d

    source = inspect.getsource(loader_3d)
    assert "'keys'" in source or '"keys"' in source, (
        "the 3D dataloader no longer emits 'keys'. Without a case identifier in the batch "
        "there is nothing to join the per-case channel mask onto, and the masked fit has "
        "no delivery mechanism at all."
    )
    train_step = inspect.getsource(nnUNetTrainer.train_step)
    assert "'data'" in train_step and "'target'" in train_step, train_step[:400]
# --------------------------------------------------------------------------------------
# (d) WAS HERE, AND IS GONE ON PURPOSE.
#
# Three tests pinned `DeepSupervisionWrapper.forward`: that it takes `*args`, that
# `zip(*args)` forwards a third list unmodified, and that a zero-weighted scale is
# skipped. They existed because `masked_trainer._build_loss` wrapped the masked loss in
# nnU-Net's wrapper and `_call_loss` smuggled the mask through it as `[mask] * len(output)`.
#
# It no longer does. `MaskedDeepSupervisionWrapper` -- written for exactly this, and until
# now used by nothing but its own test -- takes the mask as an argument and applies one
# [B,C] mask at every scale, because whether a reader annotated a channel is a fact about
# the CASE and not about the resolution a decoder head happens to run at.
#
# So these three pinned an upstream signature nobody upstream promised, guarding a
# dependency this trainer does not have. The one BEHAVIOUR worth keeping -- that a
# zero-weighted scale is skipped rather than multiplied by zero -- moved to
# `trainer/tests/test_masked_loss.py`, where it is asserted about OUR wrapper and where it
# can fail for a reason that is ours to fix.
# --------------------------------------------------------------------------------------
