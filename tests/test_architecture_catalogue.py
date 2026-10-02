# SPDX-License-Identifier: Apache-2.0
"""The architecture catalogue, checked against the nnU-Net that is actually installed.

WHY THIS FILE EXISTS
--------------------
`architectures.py` makes four claims about each preset that it cannot check itself, because
it deliberately imports no nnU-Net: that the planner class exists, that it is defined once,
that it instantiates the network the catalogue names, and that it writes the
`plans_identifier` the catalogue records. Every one of those is a statement about a
third-party package's internals, so every one is a statement about ONE version -- which is
why `test_nnunet_internals.py::test_the_installed_versions_are_the_pins` runs first.

A catalogue that can lie about which network a preset produces is worse than no catalogue.
`plan.json` would record `network: ResidualEncoderUNet` for a run that trained a
`PlainConvUNet`, and the record is the only place anybody would look.

AND THE COMPLETENESS GATE, WHICH IS THE ONE THAT WILL ACTUALLY FIRE ONE DAY. Every planner
upstream defines must be accounted for -- offered, excluded with a reason, or named as
ambiguous. When a future nnU-Net adds a planner, this goes red and says which. The
alternative is a catalogue that silently stops being the set of choices available.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer import architectures as cat  # noqa: E402


def _upstream_planners() -> dict[str, list[str]]:
    """Every `*Planner*` class upstream defines, by name, with the modules defining it.

    A list per name, not one module: the duplicate-name hazard `AMBIGUOUS_PLANNERS` exists
    for is exactly a name with two entries here.
    """
    import nnunetv2.experiment_planning.experiment_planners as package

    found: dict[str, list[str]] = {}
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001 - an unimportable planner is not this gate's subject
            continue
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and obj.__module__ == module.__name__
                    and "Planner" in name):
                found.setdefault(name, []).append(module.__name__)
    return found


UPSTREAM = _upstream_planners()


@pytest.mark.parametrize("preset", cat.PRESETS, ids=lambda p: p.name)
def test_the_planner_the_preset_names_exists_and_is_defined_once(preset) -> None:
    """Upstream resolves a planner BY NAME through a package walk, so a name defined twice
    means the class that runs is whichever the walk returned first -- and the plan it writes
    carries no sign of which one it was."""
    modules = UPSTREAM.get(preset.planner)
    assert modules, (
        f"{preset.name} names planner {preset.planner!r}, which the installed nnU-Net does "
        f"not define. It defines {sorted(UPSTREAM)}"
    )
    assert len(modules) == 1, (
        f"{preset.planner!r} is defined in {modules}. Upstream resolves planners by name, "
        "so which one runs depends on directory order; this preset cannot name it"
    )
    assert preset.planner not in cat.AMBIGUOUS_PLANNERS, (
        f"{preset.planner!r} is listed as ambiguous and offered as a preset at once"
    )


@pytest.mark.parametrize("preset", cat.PRESETS, ids=lambda p: p.name)
def test_the_preset_produces_the_network_and_the_plans_name_it_claims(preset) -> None:
    """THE CLAIM THAT MATTERS, because it is the one `plan.json` records.

    ESTABLISHED OVER THE SYNTAX TREE, NOT BY CONSTRUCTING THE PLANNER. `UNet_class` is
    assigned in `__init__` and is not a class attribute, so it cannot be read off the class.
    The first version of this gate constructed the planner against a dataset id that does not
    exist, on the reasoning that the assignments happen before anything touches the
    filesystem. They do not: `__init__` resolves the dataset name first and raises. So the
    assignment is read where it is written.

    This also catches the thing that was actually wrong: `reference_gb` was 9.0, taken from
    the class docstring's "Target is ~9-11 GB VRAM max", while the signature calibrates at 8.
    The rule picks the preset whose reference is closest below the budget, so a number copied
    from prose would have changed which architecture a card got.
    """
    module_name = UPSTREAM[preset.planner][0]
    planner_class = getattr(importlib.import_module(module_name), preset.planner)

    signature = inspect.signature(planner_class.__init__)
    assert signature.parameters["plans_name"].default == preset.plans_identifier, (
        f"{preset.name} records plans_identifier {preset.plans_identifier!r} but "
        f"{preset.planner} writes {signature.parameters['plans_name'].default!r}. "
        "`plan.json` carries this value, and the fit loads the plans file by it"
    )
    assert signature.parameters["gpu_memory_target_in_gb"].default == preset.reference_gb, (
        f"{preset.name} records reference_gb {preset.reference_gb!r} but {preset.planner} "
        f"calibrates at {signature.parameters['gpu_memory_target_in_gb'].default!r}. The "
        "selection rule minimises the distance between the budget and this number, so a "
        "stale copy of it silently changes which preset a card gets"
    )

    assigned = _unet_class_assigned_in(planner_class)
    assert assigned == preset.network, (
        f"{preset.name} claims {preset.network} and {preset.planner} assigns "
        f"UNet_class = {assigned}. `plan.json` would record an architecture the run did not "
        "train"
    )


def _unet_class_assigned_in(planner_class) -> str:
    """The name assigned to `self.UNet_class` in this class's `__init__`, or its base's.

    Walks the MRO because a subclass that does not set it inherits the base's -- which is
    itself a fact worth asserting: a preset whose planner never assigns one would silently
    get whatever `ExperimentPlanner` assigns.
    """
    import ast
    import textwrap

    for klass in planner_class.__mro__:
        initialiser = klass.__dict__.get("__init__")
        if initialiser is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(initialiser)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and target.attr == "UNet_class"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and isinstance(node.value, ast.Name)):
                    return node.value.id
    raise AssertionError(
        f"no `self.UNet_class = <Name>` assignment anywhere in {planner_class.__name__}'s "
        "MRO, so which network this planner builds cannot be established here at all"
    )


def test_every_planner_upstream_ships_is_accounted_for() -> None:
    """THE GATE THAT WILL FIRE ON AN UPGRADE, AND SHOULD.

    A catalogue is a claim about the set of available choices. If nnU-Net adds a planner and
    nothing here notices, the claim quietly becomes false -- and the way it becomes false is
    that a better configuration exists and nobody knows. Named, excluded with a reason, or
    ambiguous: those are the three honest states.
    """
    offered = {p.planner for p in cat.PRESETS}
    accounted = offered | set(cat.EXCLUDED_PLANNERS) | set(cat.AMBIGUOUS_PLANNERS)
    # The base classes the presets derive from are not choices a run can make.
    abstract = {"ExperimentPlanner"} & offered
    unaccounted = sorted(set(UPSTREAM) - accounted - abstract)
    assert not unaccounted, (
        f"the installed nnU-Net ships planners this catalogue does not mention: "
        f"{unaccounted}. Add each to PRESETS, or to EXCLUDED_PLANNERS with the reason it is "
        "not offered. A planner nobody decided about is a configuration nobody compared"
    )


def test_a_name_listed_as_ambiguous_really_is() -> None:
    """The exclusion has to be true, or it is a superstition carried forward. If upstream
    ever de-duplicates the name, this says so and the preset can be offered."""
    for name in cat.AMBIGUOUS_PLANNERS:
        modules = UPSTREAM.get(name, [])
        assert len(modules) > 1, (
            f"{name!r} is listed as defined more than once, but the installed nnU-Net "
            f"defines it in {modules}. If that is now unambiguous, it can become a preset"
        )


# =====================================================================================
# The rule
# =====================================================================================
@pytest.mark.parametrize("budget,expected", [
    (4.0, "plain"),          # below every reference: the floor, not a refusal
    (7.9, "plain"),
    # 8.0 IS THE TIE, and it goes to the residual encoder -- see the named gate below. This
    # table said `plain` here while it was written against a rule that had no tie-break, and
    # the two failures were the tie-break arriving.
    (8.0, "resenc_m"),
    (8.9, "resenc_m"),
    (9.0, "resenc_m"),
    (23.9, "resenc_m"),
    (24.0, "resenc_l"),
    (28.0, "resenc_l"),      # the RTX 5090 this cohort was fitted on, ~30 GB free
    (39.9, "resenc_l"),
    (40.0, "resenc_xl"),
    (80.0, "resenc_xl"),
])
def test_the_rule_is_the_largest_reference_that_fits(budget, expected) -> None:
    preset, record = cat.select(budget)
    assert preset.name == expected
    assert record["selected"] == expected
    assert record["observed_budget_gb"] == budget


def test_the_record_carries_the_alternatives_and_not_only_the_answer() -> None:
    """"Why this architecture" is the first question at a review, and a bare name cannot
    answer it. The record names every preset, whether it fitted, and the rule applied."""
    _preset, record = cat.select(28.0)
    considered = {c["name"]: c for c in record["considered"]}
    assert set(considered) == {p.name for p in cat.PRESETS}
    assert considered["resenc_xl"]["fits"] is False
    assert considered["resenc_l"]["fits"] is True
    assert "MOS-TRAIN-213" in record["rule"]


def test_the_record_says_when_the_budget_is_not_the_calibration_point() -> None:
    """Upstream warns when the target is overridden, and the warning is right: the patch is
    being scaled away from anything upstream measured. A warning nobody can see is a fact
    that was deleted, so it is recorded."""
    _p, extrapolated = cat.select(28.0)
    assert extrapolated["extrapolated_from_reference"] is True
    _p, exact = cat.select(24.0)
    assert exact["extrapolated_from_reference"] is False


def test_a_failed_vram_observation_is_refused_rather_than_taking_the_smallest() -> None:
    """`vram_observation()` floors the budget at 4 GB precisely so a busy card cannot select
    the smallest architecture. A zero means the observation failed, and planning the baseline
    on it would hide that behind a plausible plan."""
    with pytest.raises(ValueError, match="observation itself failed"):
        cat.select(0.0)


def test_an_unknown_preset_name_is_refused_and_does_not_fall_back() -> None:
    with pytest.raises(LookupError, match=r"offers \['plain', 'resenc_m'"):
        cat.named("resenc_xxl")


def test_every_preset_name_is_unique_and_every_reference_is_ordered() -> None:
    """The catalogue is read by index nowhere, but the rule takes a max over `reference_gb`
    and the docstring promises the tuple is ordered; a reader who trusts that and a rule that
    does not would disagree silently."""
    names = [p.name for p in cat.PRESETS]
    assert len(set(names)) == len(names), names
    references = [p.reference_gb for p in cat.PRESETS]
    assert references == sorted(references), references


def test_the_tie_at_eight_gigabytes_goes_to_the_residual_encoder() -> None:
    """`plain` and `resenc_m` BOTH calibrate at 8 GB, so the rule has a genuine tie and it
    has to be broken on purpose. `max` returns the first maximal element, which would have
    made the order of a tuple literal the deciding factor with nothing saying so.

    Upstream recommends the residual encoder over the plain baseline at the same budget, so
    that is the tie-break -- which means `plain` is reachable only BELOW 8 GB, and that is
    the whole of what this pipeline does differently from nnU-Net's default route: on a 30 GB
    card nnU-Net's own `nnUNetv2_plan_and_preprocess` gives you `nnUNetPlans`, and you get
    `resenc_l` only by typing a flag.
    """
    tied = [p for p in cat.PRESETS if p.reference_gb == 8.0]
    assert len(tied) > 1, (
        "there is no longer a tie at 8 GB, so this gate no longer guards anything: "
        f"{[(p.name, p.reference_gb) for p in cat.PRESETS]}"
    )
    preset, _record = cat.select(8.0)
    assert preset.network == "ResidualEncoderUNet", preset
    assert cat.select(7.9)[0].name == "plain", "below every reference the floor must apply"


# =====================================================================================
# OVERLAYS: a patched plan is worth nothing unless nnU-Net can build from it
# =====================================================================================

import json  # noqa: E402

import torch  # noqa: E402

FIXTURE = TRAINER / "tests" / "fixtures" / "nnunet_plans_plain.json"


def _plans() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [o.name for o in cat.OVERLAYS])
def test_the_overlay_produces_a_plan_nnunet_itself_can_build_and_train_from(name) -> None:
    """THE GATE THE WHOLE FILE IS FOR, and the only one that is not a statement about a catalogue.

    Everything else here reads declarations. This takes the patched plan and hands it to
    `get_network_from_plans` -- nnU-Net's own builder, with nnU-Net's own `pydoc.locate` over the
    dotted path and nnU-Net's own resolution of `_kw_requires_import` into classes -- then runs a
    forward pass and checks the deep-supervision list against the plan's OWN strides.

    A catalogue row can name a class that does not exist, a kwarg the constructor rejects, or a
    stage count that yields the wrong number of heads, and every one of those is a declaration that
    reads correctly. This is the step that cannot be satisfied by reading.

    The patch is reduced to a small one so the forward pass is cheap; the plan's own patch is
    128x224x224 and would not be a test. What is NOT reduced is the stage count, because that is
    what the head count follows from -- and 64 on every axis is the SMALLEST patch six stages
    admit: 32 would leave an extent of one at the bottleneck, which instance norm refuses and
    SwinUNETR's five halvings cannot produce. Proved by breaking, where 32x64x64 was refused.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    # THE PATCH IS REDUCED IN THE PLAN AND NOT IN THE FORWARD CALL, and that is now load-bearing
    # rather than tidy. `unetr` reads its `img_size` out of the configuration's `patch_size`
    # through `plan_derived_kwargs`, and it is BOUND to it -- so a test that shrank only the
    # tensor built a network for 128x224x224 and fed it 64x64x64, which this wrapper refuses.
    # Proved by breaking: that is exactly how this gate failed when `unetr` joined OVERLAYS.
    #
    # Reducing the plan instead makes the gate prove the derivation: whatever patch the plan
    # carries is the patch the network is built for.
    small = [64, 64, 64]
    plans = _plans()
    plans["configurations"]["3d_fullres"]["patch_size"] = list(small)
    patched = cat.apply_overlay(plans, name, configuration="3d_fullres")
    architecture = patched["configurations"]["3d_fullres"]["architecture"]
    stages = len(architecture["arch_kwargs"]["strides"])

    for keyword, source in cat.overlay(name).plan_derived_kwargs.items():
        assert architecture["arch_kwargs"][keyword] == plans["configurations"]["3d_fullres"][source], (
            f"{name} derives {keyword} from the configuration's {source}, and the plan carries "
            f"{plans['configurations']['3d_fullres'][source]!r} while the architecture block says "
            f"{architecture['arch_kwargs'][keyword]!r}. A constant written into the catalogue "
            "would pass a plan whose patch the planner later changed"
        )

    # THE DECLARED KWARGS ARE IN THE PLAN, WHICH THE BUILD BELOW CANNOT TELL YOU.
    #
    # Proved by breaking: deleting the line that merges them left this test green, because every
    # one of them has a wrapper default that happens to equal the declared value -- so the network
    # built, produced the right heads, and `plan.json` recorded none of the geometry that trained.
    # A default that moves under a MONAI or a wrapper upgrade would then change what trained with
    # nothing in the run's record changing. The point of writing them into the plan is the record,
    # and only an assertion about the plan can check a record.
    declared = cat.overlay(name).arch_kwargs
    for key, value in declared.items():  # the DECLARED ones; the derived ones are checked above
        assert architecture["arch_kwargs"].get(key) == value, (
            f"{name} declares {key}={value!r} and the plan carries "
            f"{architecture['arch_kwargs'].get(key)!r}: the fit would use a wrapper default that "
            "the run's own record does not name"
        )

    network = get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture["_kw_requires_import"],
        input_channels=1,
        output_channels=10,
        deep_supervision=True,
    )
    network.eval()
    with torch.no_grad():
        outputs = network(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(outputs, list), (
        f"{name} returned {type(outputs).__name__} under deep supervision; the trainer reads "
        "output[0] as the full-resolution prediction"
    )
    assert len(outputs) == stages - 1, (
        f"{name} returned {len(outputs)} heads against {stages - 1} the plan's strides ask for. "
        "DeepSupervisionWrapper zips and zip truncates, so this would pair heads with the wrong "
        "scales without raising"
    )
    assert tuple(outputs[0].shape[2:]) == (64, 64, 64), "the list is not finest-first"

    # AND THE OTHER HALF OF THE CONTRACT, through the path the trainer and packaging both write.
    network.decoder.deep_supervision = False
    with torch.no_grad():
        single = network(torch.zeros(1, 1, 64, 64, 64))
    assert isinstance(single, torch.Tensor)


def test_selection_by_measurement_never_returns_an_overlay() -> None:
    """THE LINE BETWEEN "по анализу" AND "по заказу", and it is load-bearing.

    An overlay has no `reference_gb`, because no planner ever calibrated one for it. A rule that
    could return one would be comparing a measured calibration point against a number somebody
    invented -- and `MOS-TRAIN-213` would then have a selection over architectures, which is a
    `ConfigurationSearch` and owes the whole of `MOS-TRAIN-218`.
    """
    overlay_names = {o.name for o in cat.OVERLAYS}
    for budget in (4.0, 8.0, 12.0, 24.0, 40.0, 80.0, 200.0):
        preset, _record = cat.select(budget)
        assert preset.name not in overlay_names, (
            f"a budget of {budget} GB selected the overlay {preset.name!r}; nothing measured can "
            "choose an architecture whose patch nobody derived"
        )
    assert not overlay_names & {p.name for p in cat.PRESETS}, (
        "an overlay and a preset share a name, so which catalogue a recorded choice came from "
        "could not be resolved from the record"
    )


def test_the_two_catalogues_do_not_resolve_each_other() -> None:
    """A planner asked for by name must not return an architecture overlay, or the reverse.

    Either fallback gives the caller a plan derived by something other than what it asked for, and
    the plan records only the result.
    """
    for entry in cat.OVERLAYS:
        with pytest.raises(LookupError, match="no architecture preset named"):
            cat.named(entry.name)
    for preset in cat.PRESETS:
        with pytest.raises(LookupError, match="no architecture overlay named"):
            cat.overlay(preset.name)
    # BOTH RAISE `LookupError` AND NEITHER RAISES `ValueError`, which is the half that matters:
    # `apply_overlay` raises ValueError for a plan an overlay cannot serve, so an unknown NAME
    # arriving as ValueError too would be indistinguishable from a plan that does not fit.
    assert not issubclass(LookupError, ValueError)


def test_every_overlay_plans_from_a_preset_that_exists_and_names_an_importable_class() -> None:
    """Both halves are references into somewhere else, and both are checked here rather than
    at the first fit: `plan_from` into this module's own preset catalogue, and
    `network_class_name` into a package, through the same `pydoc.locate` nnU-Net uses -- which
    RETURNS NONE rather than raising, so a typo would become a warning and a search of
    `dynamic_network_architectures` for a class that is not there.
    """
    import pydoc

    for entry in cat.OVERLAYS:
        assert cat.named(entry.plan_from).name == entry.plan_from
        located = pydoc.locate(entry.network_class_name)
        assert located is not None, (
            f"{entry.name} names {entry.network_class_name!r}, which pydoc.locate cannot find. "
            "nnU-Net does not raise for this -- it warns and searches "
            "dynamic_network_architectures instead"
        )


def test_every_overlay_carries_a_caveat_and_the_caveat_reaches_the_plan() -> None:
    """A CAVEAT NOBODY PERSISTED IS A CAVEAT NOBODY READ.

    The whole justification for allowing an overlay is that the run's own record says its patch
    was derived for another network. If that sentence lives only in this source file, the
    justification does not hold: `plan.json` is the only place anybody looks six months later.
    """
    for entry in cat.OVERLAYS:
        assert len(entry.caveat) > 120, f"{entry.name} has no real caveat"
        assert len(entry.why) > 60, f"{entry.name} has no real reason"
    patched = cat.apply_overlay(_plans(), "swin_unetr", configuration="3d_fullres")
    record = patched["configurations"]["3d_fullres"]["medos_architecture"]
    assert record["overlay"] == "swin_unetr"
    assert record["planned_by"] == "plain"
    assert record["planner_network"] == "PlainConvUNet"
    assert record["caveat"] == cat.overlay("swin_unetr").caveat


def test_the_record_sits_beside_the_architecture_block_and_not_inside_it() -> None:
    """AND THAT PLACEMENT IS NOT COSMETIC.

    The trainer reads `network_class_name`, `arch_kwargs` and `_kw_requires_import` from the
    architecture block and passes the second straight into the constructor as `**kwargs`. A record
    written inside it would arrive at the network as an unexpected keyword -- and both wrappers
    here refuse an unknown key by design, so the fit would fail at construction with a message
    about a plan key rather than about a provenance note.
    """
    patched = cat.apply_overlay(_plans(), "swin_unetr", configuration="3d_fullres")
    body = patched["configurations"]["3d_fullres"]
    assert "medos_architecture" in body
    assert "medos_architecture" not in body["architecture"]
    assert set(body["architecture"]) == set(
        _plans()["configurations"]["3d_fullres"]["architecture"]), (
        "the overlay added or removed a key in the architecture block; the trainer reads three "
        "of them and passes one into the constructor"
    )


def test_a_stage_count_the_network_cannot_serve_is_refused_before_the_fit_is_queued() -> None:
    """The 2d configuration is eight stages and both overlays declare six.

    Left to the wrapper, this raises at construction -- inside a container, after preprocessing,
    hours into a queue. The refusal belongs where the plan is written.
    """
    for name in ("swin_unetr", "segresnet_ds"):
        with pytest.raises(ValueError) as raised:
            cat.apply_overlay(_plans(), name, configuration="2d")
        assert "declares 8" in str(raised.value)


def test_a_per_stage_key_of_the_wrong_length_is_refused_rather_than_padded() -> None:
    """Padding would invent the missing stages' geometry and truncating would drop a declared one.

    Checked on a hand-made row rather than a real one, because both real rows also declare a stage
    count -- so `requires_stages` would catch the 2d plan first and this guard would never run. A
    future row with `requires_stages=None` and a per-stage list is the case it exists for, and
    proving it means building that row.
    """
    seven = cat.Overlay(
        name="probe", plan_from="plain",
        network_class_name="medos_trainer.nets.MedOSSegResNetDS",
        arch_kwargs={"blocks_down": [1, 2, 2, 4, 4, 4, 4]},
        requires_stages=None, caveat="x" * 200, why="y" * 80,
        per_stage_kwargs=("blocks_down",),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cat, "OVERLAYS", cat.OVERLAYS + (seven,))
        with pytest.raises(ValueError) as raised:
            cat.apply_overlay(_plans(), "probe", configuration="3d_fullres")
    assert "blocks_down" in str(raised.value) and "invent" in str(raised.value)


def test_a_key_the_plan_and_the_overlay_both_carry_is_refused_not_silently_preferred() -> None:
    """Whichever won, the plan would record a value the fit did not use.

    `feature_size` is not in a conv plan, so this is built by putting it there -- which is what a
    future planner adding a key the overlay also sets would look like.
    """
    plans = _plans()
    plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["feature_size"] = 24
    with pytest.raises(ValueError) as raised:
        cat.apply_overlay(plans, "swin_unetr", configuration="3d_fullres")
    assert "feature_size" in str(raised.value)


def test_the_overlay_does_not_mutate_the_plan_it_was_handed() -> None:
    """One plans object describing two different fits is how two runs come to share a record.

    The nesting is three deep -- `configurations`, the configuration, `architecture`, `arch_kwargs`
    -- and a shallow copy at any level leaves the original sharing the dictionary that was edited.
    """
    plans = _plans()
    before = json.dumps(plans, sort_keys=True)
    cat.apply_overlay(plans, "swin_unetr", configuration="3d_fullres")
    assert json.dumps(plans, sort_keys=True) == before, "apply_overlay edited its argument"


def test_a_plan_with_no_strides_or_no_architecture_block_is_refused() -> None:
    """Neither can be supplied by this function, and inventing either would invent a geometry."""
    plans = _plans()
    del plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["strides"]
    with pytest.raises(ValueError, match="no strides"):
        cat.apply_overlay(plans, "swin_unetr", configuration="3d_fullres")

    plans = _plans()
    plans["configurations"]["3d_fullres"]["architecture"] = {}
    with pytest.raises(ValueError, match="nothing to overlay"):
        cat.apply_overlay(plans, "swin_unetr", configuration="3d_fullres")

    # NOT `3d_cascade_fullres`: the plan HAS that configuration and it carries no architecture
    # block, so it proves the refusal above instead of this one.
    assert "3d_cascade_fullres" in _plans()["configurations"]
    with pytest.raises(ValueError, match="no configuration"):
        cat.apply_overlay(_plans(), "swin_unetr", configuration="3d_quarterres")
