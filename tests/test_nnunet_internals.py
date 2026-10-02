# SPDX-License-Identifier: Apache-2.0
"""What the PINNED nnU-Net actually does, asserted rather than read from documentation.

THE VERSION IS NOT NAMED IN THIS DOCSTRING ANY MORE, AND THAT IS THE POINT OF THE FIRST
TEST BELOW. It said "nnU-Net 2.5.1" while the shipped image had moved to 2.6.4 -- so every
claim in this file was being established against whichever version happened to be installed
where the suite ran, under a heading naming a different one. Locally that was 2.5.1 against
an image running 2.6.4: green, and meaningless. `test_the_installed_versions_are_the_pins`
is what makes "the pinned nnU-Net" a fact rather than a hope.

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
from pathlib import Path

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

#: The packages whose PRIVATE API the claims in this file, in `masked_trainer.py` and in the
#: architecture catalogue rest on. Not every pin in `requirements.txt`: a mismatch in
#: `blosc2` is a data-format problem the image will report itself, while a mismatch in these
#: three silently changes what a passing gate means.
PINNED_FOR_INTERNALS = ("nnunetv2", "dynamic-network-architectures", "torch", "monai")


def _pins() -> dict[str, str]:
    """The `name==version` pins from the image's own requirements file.

    Read from the file rather than restated here, because a version written in two places is
    not a pin -- it is two numbers that agree until someone edits one.
    """
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    found = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line and not line.startswith("-"):
            name, _, version = line.partition("==")
            found[name.strip().lower()] = version.strip()
    return found


def test_the_installed_versions_are_the_pins() -> None:
    """THE INSTRUMENT, CHECKED BEFORE ANY MEASUREMENT IS TAKEN WITH IT.

    Every other test in this file asserts something about a third-party private API, and is
    therefore a statement about ONE version. Nothing tied the installed version to the
    image's, and the two diverged: the local environment sat on nnunetv2 2.5.1 and
    dynamic-network-architectures 0.3.1 while `requirements.txt` had moved to 2.6.4 and
    0.4.4 for the image that actually trained the models. The suite was green on both
    machines and agreed about nothing.

    That is the same defect as a lint gate with no fixed instrument, whose verdict depends on
    the calendar. It is checked here, first, and it FAILS rather than skips: an unpinned
    instrument does not make the other gates uncertain, it makes them unrelated to the image.

    THE LOCAL BUILD TAG IS NOT PART OF THE VERSION. `torch==2.7.1+cu128` pins the CUDA
    userspace the image carries; a CPU machine running `2.7.1+cpu` has the same public API
    and the same internals, and requiring the tag would make this gate unrunnable anywhere
    a card is absent -- which is exactly where the cheap gates are supposed to run. The full
    string is still recorded, in `MEDOS_TRAINING_ENVIRONMENT`, where it is provenance rather
    than a precondition.
    """
    import importlib.metadata as metadata

    pins = _pins()
    wrong = []
    for name in PINNED_FOR_INTERNALS:
        want = pins.get(name)
        assert want, f"{name} carries no `==` pin in trainer/requirements.txt"
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            wrong.append(f"{name}: pinned {want}, NOT INSTALLED")
            continue
        if installed.split("+")[0] != want.split("+")[0]:
            wrong.append(f"{name}: pinned {want}, installed {installed}")
    assert not wrong, (
        "the installed trainer stack is not the image's: "
        + "; ".join(wrong)
        + ". Every other gate in this file asserts something about one version's private "
        "API. Against a different version they are green and unrelated to the image that "
        "trains the models. Install the pins with "
        "`pip install -r trainer/requirements.txt`"
    )


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

    FOUND BY SEARCH AND ASSERTED BY SYNTAX, FOR TWO REASONS THIS GATE LEARNED THE HARD WAY.

    It imported `nnunetv2.training.dataloading.data_loader_3d` by name and searched its
    source text for the string `'keys'`. Both halves failed:

      * 2.6.4 collapsed `data_loader_2d` and `data_loader_3d` into one `data_loader`, so the
        import raised `ModuleNotFoundError` and the gate was DEAD -- not red for its own
        reason, but uncollectable -- through the entire migration that put 2.6.4 into the
        image. The claim the whole masked design rests on went unverified while two 200-epoch
        fits ran on it. They ran correctly, which is the point: nothing was checking.
      * a source-text search cannot tell code from prose about code. This very docstring
        contains `'keys'`, so the old assertion would pass against a module whose only
        mention of it was a comment saying it had been removed.

    So the class is located by searching the `dataloading` package -- a rename is reported
    with what was found instead of raising on an import line -- and the claim is asserted
    over the RETURN STATEMENT's syntax tree.
    """
    import ast
    import importlib
    import pkgutil
    import textwrap

    import nnunetv2.training.dataloading as dataloading

    loaders = {}
    for module_info in pkgutil.iter_modules(dataloading.__path__):
        module = importlib.import_module(f"{dataloading.__name__}.{module_info.name}")
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and obj.__module__ == module.__name__
                    and name.startswith("nnUNetDataLoader")):
                loaders[f"{module_info.name}.{name}"] = obj
    assert loaders, (
        "no class named nnUNetDataLoader* under nnunetv2.training.dataloading; it holds "
        f"{[m.name for m in pkgutil.iter_modules(dataloading.__path__)]}. The dataloader is "
        "what puts a case identifier in the batch, and without one there is nothing to join "
        "the per-case channel mask onto"
    )

    for where, loader in sorted(loaders.items()):
        generate = getattr(loader, "generate_train_batch", None)
        assert generate is not None, f"{where} has no generate_train_batch"
        # `textwrap.dedent`, not `inspect.cleandoc`: cleandoc is for docstrings and leaves
        # the first line's indentation alone, so a method's source fails to parse.
        tree = ast.parse(textwrap.dedent(inspect.getsource(generate)))
        emitted = {
            ast.literal_eval(key)
            for node in ast.walk(tree)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert "keys" in emitted, (
            f"{where}.generate_train_batch returns {sorted(emitted)} and no 'keys'. Without "
            "a case identifier in the batch there is nothing to join the per-case channel "
            "mask onto, and the masked fit has no delivery mechanism at all"
        )
        assert {"data", "target"} <= emitted, (
            f"{where}.generate_train_batch returns {sorted(emitted)}; the trainer's own "
            "train_step reads 'data' and 'target'"
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
