# SPDX-License-Identifier: Apache-2.0
"""The typed plan, checked against the two real plans this cohort was configured with.

WHY THE ROUND TRIP IS THE FIRST GATE AND NOT A NICETY. `nnUNetTrainer.py:922` and
`predict_from_raw_data.py:244,713` write the WHOLE plans document into the results folder for
inference to read back. So a plan model that dropped a key with no training-time consumer would
produce a run whose training looked perfect and whose inference failed -- and the failure would
arrive days later, in another process, against an artifact nobody would think to suspect.

The fixtures are the real thing: `nnUNetPlans` from the `ExperimentPlanner` and
`nnUNetResEncUNetLPlans` from `nnUNetPlannerResEncL`, both over this cohort's own fingerprint,
including the `3d_cascade_fullres` stub that carries two keys and nothing else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.plan import (  # noqa: E402
    CASCADE_STUB_KEYS,
    CONFIGURATION_KEYS,
    DOTTED_PATH_VALUES,
    GLOBAL_KEYS,
    MODULE_SCAN_VALUES,
    Plan,
    PlanError,
)

FIXTURES = TRAINER / "tests" / "fixtures"
PLANS = ("nnunet_plans_plain.json", "nnunet_plans_resenc_l.json")


def _document(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# =====================================================================================
# Lossless carriage
# =====================================================================================
@pytest.mark.parametrize("name", PLANS)
def test_a_real_plan_round_trips_byte_for_byte(name) -> None:
    """DATA EQUALITY IS NOT ENOUGH, so the comparison is over the serialised text.

    `1.0 == 1` in Python, so a model that turned a float into an int would pass an `==` check on the
    parsed dictionaries. `json.dumps` renders them differently, and the spacings and median shapes
    in these plans are exactly where that could happen -- the planner writes
    `[float(i) for i in ...]` for one and `[int(round(i)) for i in ...]` for its twin.
    """
    original = _document(name)
    carried = Plan.from_document(original).to_document()
    assert carried == original
    assert json.dumps(carried, sort_keys=True) == json.dumps(original, sort_keys=True), (
        "the round trip changed a value's TYPE without changing its value"
    )


@pytest.mark.parametrize("name", PLANS)
def test_every_configuration_survives_including_the_cascade_stub(name) -> None:
    """The stub carries two keys and is still a configuration.

    A model that required the sixteen keys of a full configuration would drop
    `3d_cascade_fullres` -- and the cascade is the one nnU-Net feature this cohort's plan offers
    that we do not yet use, so dropping it would quietly remove the option.
    """
    document = _document(name)
    plan = Plan.from_document(document)
    assert set(plan.configuration_names) == set(document["configurations"])
    stub = plan.configuration("3d_cascade_fullres")
    assert stub.is_cascade_stub
    assert set(stub.body) == set(CASCADE_STUB_KEYS)
    assert stub.inherits_from == "3d_fullres"


def test_a_key_this_module_does_not_know_is_carried_and_reported() -> None:
    """A FUTURE nnU-Net KEY MUST NOT BE AN UPGRADE BLOCKER.

    It is carried verbatim, and `unknown_global_keys` says it is there. Refusing it would make this
    module the reason a version bump could not be adopted; dropping it silently is the failure the
    round-trip gate exists for.
    """
    document = _document("nnunet_plans_plain.json")
    document["something_nnunet_2_7_adds"] = {"deeply": ["nested", 1, 2.5, None]}
    plan = Plan.from_document(document)
    assert plan.unknown_global_keys == ("something_nnunet_2_7_adds",)
    assert plan.to_document()["something_nnunet_2_7_adds"] == {
        "deeply": ["nested", 1, 2.5, None]}


def test_the_held_document_cannot_be_reached_from_either_side() -> None:
    """One plans object describing two different fits is how two runs come to share a record.

    BOTH DIRECTIONS, and the first version of this gate only tested one. Mutating what
    `to_document()` hands out is the OUTPUT side; mutating the mapping that was passed to
    `from_document` is the INPUT side, and that is the more dangerous of the two here --
    `architectures.apply_overlay` holds the caller's `plans` and a caller may patch two overlays
    from one document. Proved by breaking: removing the defensive copy from `from_document` left
    this test green.
    """
    original = _document("nnunet_plans_plain.json")
    plan = Plan.from_document(original)

    # INPUT side: the caller keeps its own mapping and changes it.
    original["configurations"]["3d_fullres"]["patch_size"] = [9, 9, 9]
    original["plans_name"] = "vandalised-input"
    assert plan.configuration("3d_fullres").patch_size == [128, 224, 224], (
        "mutating the mapping passed to from_document reached into the Plan"
    )
    assert plan.plans_name == "nnUNetPlans"

    # OUTPUT side: the caller changes what it was handed.
    handed = plan.to_document()
    handed["configurations"]["3d_fullres"]["patch_size"] = [1, 1, 1]
    handed["plans_name"] = "vandalised-output"
    assert plan.configuration("3d_fullres").patch_size == [128, 224, 224]
    assert plan.plans_name == "nnUNetPlans"


# =====================================================================================
# Typed reading, and refusals that name what went wrong
# =====================================================================================
def test_the_geometry_comes_out_typed_and_the_rank_from_the_patch() -> None:
    """The RANK IS NOT DERIVED FROM THE NAME. `save_plans` merges arbitrary user-named
    configurations, so a rank taken from "3d_fullres" would be a convention and not a fact."""
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    full = plan.configuration("3d_fullres")
    assert full.patch_size == [128, 224, 224]
    assert full.batch_size == 2
    assert full.spacing == pytest.approx([1.0, 0.782, 0.782], abs=1e-3)
    assert full.spatial_rank == 3
    assert plan.configuration("2d").spatial_rank == 2
    assert full.stages == 6
    assert full.deep_supervision_heads == 5
    assert plan.configuration("2d").stages == 8


def test_a_configuration_the_plan_does_not_carry_is_refused_by_name() -> None:
    """THE REFUSAL THIS MODULE EXISTS FOR.

    The configuration was a module constant in two files while `plan.json` recorded a
    `configuration` member nothing read back, so a plan recorded for one configuration could be fit
    under another and nothing raised. Asking for one it does not carry now fails before any data is
    staged, and the message lists what it does carry.
    """
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError) as raised:
        plan.configuration("3d_quarterres")
    message = str(raised.value)
    assert "3d_quarterres" in message and "3d_fullres" in message and "nnUNetPlans" in message


def test_asking_a_cascade_stub_for_geometry_says_where_the_geometry_lives() -> None:
    """A stub has no patch of its own, and the useful refusal names what it inherits from rather
    than reporting a missing key."""
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError) as raised:
        plan.configuration("3d_cascade_fullres").patch_size
    message = str(raised.value)
    assert "cascade STUB" in message and "3d_fullres" in message


def test_a_document_that_is_not_a_plan_is_refused_rather_than_wrapped() -> None:
    with pytest.raises(PlanError, match="not a plans document"):
        Plan.from_document({"dataset_name": "Dataset501", "plans_name": "nnUNetPlans"})
    with pytest.raises(PlanError, match="is a list"):
        Plan.from_document([])                                  # type: ignore[arg-type]


# =====================================================================================
# Typed modification -- what replaces the bind mount
# =====================================================================================
def test_replacing_the_architecture_leaves_the_original_untouched() -> None:
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    patched = plan.with_architecture(
        "3d_fullres",
        network_class_name="medos_trainer.nets.MedOSSegResNetDS",
        arch_kwargs={"blocks_down": [1, 2, 2, 4, 4, 4]},
        record={"overlay": "segresnet_ds", "planned_by": "plain"},
    )
    assert patched.configuration("3d_fullres").network_class_name == (
        "medos_trainer.nets.MedOSSegResNetDS")
    assert patched.configuration("3d_fullres").arch_kwargs["blocks_down"] == [1, 2, 2, 4, 4, 4]
    assert plan.configuration("3d_fullres").network_class_name == (
        "dynamic_network_architectures.architectures.unet.PlainConvUNet"), (
        "with_architecture mutated the plan it was called on")
    assert "blocks_down" not in plan.configuration("3d_fullres").arch_kwargs


def test_the_record_sits_beside_the_architecture_block_and_not_inside_it() -> None:
    """AND THAT PLACEMENT IS NOT COSMETIC.

    The trainer reads `network_class_name`, `arch_kwargs` and `_kw_requires_import` from the block
    and passes the second straight into the constructor as `**kwargs`. Every wrapper in `nets.py`
    refuses an unknown key by design, so a note written inside would fail the fit at construction
    with a message about a plan key rather than about a note.
    """
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    patched = plan.with_architecture(
        "3d_fullres", network_class_name="medos_trainer.nets.MedOSSegResNetDS",
        arch_kwargs={"blocks_down": [1, 2, 2, 4, 4, 4]}, record={"overlay": "segresnet_ds"},
    )
    body = patched.to_document()["configurations"]["3d_fullres"]
    assert body["medos_architecture"] == {"overlay": "segresnet_ds"}
    assert "medos_architecture" not in body["architecture"]
    assert set(body["architecture"]) == {
        "network_class_name", "arch_kwargs", "_kw_requires_import"}


def test_a_bare_network_name_is_refused_because_it_resolves_to_something_else() -> None:
    """The two resolution families are the point: `pydoc.locate` needs a dotted path, and a bare
    name resolves to `None` -- after which `get_network_from_plans` warns and searches
    `dynamic_network_architectures` for the last segment, which can find a different class."""
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError) as raised:
        plan.with_architecture("3d_fullres", network_class_name="MedOSSegResNetDS")
    assert "pydoc.locate" in str(raised.value)
    assert not set(DOTTED_PATH_VALUES) & set(MODULE_SCAN_VALUES), (
        "a value is declared in both resolution families, so a modifier could move it from one to "
        "the other and the failure would be a class silently resolved by search"
    )


def test_a_key_both_the_plan_and_the_override_carry_is_refused() -> None:
    """Whichever won, the plan would record a value the fit did not use."""
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError, match="in both the plan"):
        plan.with_architecture(
            "3d_fullres", network_class_name="medos_trainer.nets.MedOSSegResNetDS",
            arch_kwargs={"strides": [[1, 1, 1]]},
        )


def test_an_import_list_naming_a_key_the_kwargs_lack_is_refused_here_not_later() -> None:
    """`get_network_from_plans` indexes every `_kw_requires_import` name unconditionally, so a
    missing one raises inside the network builder -- after the plan was accepted and the data
    staged."""
    document = _document("nnunet_plans_plain.json")
    block = document["configurations"]["3d_fullres"]["architecture"]
    block["_kw_requires_import"] = list(block["_kw_requires_import"]) + ["activation_op"]
    plan = Plan.from_document(document)
    with pytest.raises(PlanError) as raised:
        plan.with_architecture(
            "3d_fullres", network_class_name="medos_trainer.nets.MedOSSegResNetDS")
    assert "activation_op" in str(raised.value)


def test_patching_a_cascade_stub_says_what_to_patch_instead() -> None:
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError) as raised:
        plan.with_architecture(
            "3d_cascade_fullres", network_class_name="medos_trainer.nets.MedOSSegResNetDS")
    message = str(raised.value)
    assert "cascade stub" in message and "3d_fullres" in message


# =====================================================================================
# Dropping configurations without breaking the cascade
# =====================================================================================
def test_dropping_a_configuration_a_pointer_still_names_is_refused() -> None:
    """`ConfigurationManager` raises for a dangling stage only when that stage is REACHED, which
    for a cascade is after the low-resolution fit has been paid for."""
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError) as raised:
        plan.without_configurations(["3d_lowres", "3d_cascade_fullres"])
    message = str(raised.value)
    assert "3d_fullres" in message and "inherits_from" in message or "next_stage" in message


def test_keeping_a_self_contained_configuration_is_allowed() -> None:
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    reduced = plan.without_configurations(["3d_fullres"])
    assert reduced.configuration_names == ("3d_fullres",)
    assert reduced.configuration("3d_fullres").patch_size == [128, 224, 224]
    assert plan.configuration_names != ("3d_fullres",), "the original was modified"


def test_keeping_a_configuration_the_plan_lacks_is_refused() -> None:
    plan = Plan.from_document(_document("nnunet_plans_plain.json"))
    with pytest.raises(PlanError, match="does not carry"):
        plan.without_configurations(["3d_fullres", "3d_quarterres"])


# =====================================================================================
# The key catalogue is the one place the names live
# =====================================================================================
@pytest.mark.parametrize("name", PLANS)
def test_the_declared_key_names_match_what_a_real_plan_carries(name) -> None:
    """THE CATALOGUE IS LOAD-BEARING, because it is what stops the names living in seven files.

    A name that drifted from the document would make an accessor refuse a key that is present, and
    the refusal would read as a malformed plan rather than as a typo here.
    """
    document = _document(name)
    present = {key for key in document if key != "configurations"}
    assert present == set(GLOBAL_KEYS), (
        f"declared globals and the real plan differ: only in plan {sorted(present - set(GLOBAL_KEYS))}, "
        f"only declared {sorted(set(GLOBAL_KEYS) - present)}"
    )
    full = set(document["configurations"]["3d_fullres"])
    assert set(CONFIGURATION_KEYS) <= full, (
        f"declared configuration keys are not all present: {sorted(set(CONFIGURATION_KEYS) - full)}"
    )
    assert full - set(CONFIGURATION_KEYS) == set(), (
        f"the real configuration carries keys the catalogue does not name: "
        f"{sorted(full - set(CONFIGURATION_KEYS))}"
    )
