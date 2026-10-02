# SPDX-License-Identifier: Apache-2.0
"""What a run asks of its training: one object, one ladder, and refusals instead of clamps.

WHY THIS FILE IS SHORT AND STILL LOAD-BEARING. The object it gates replaced three functions that
each carried their own copy of the request-then-environment-then-default rule, and the copies had
already drifted: one CLAMPED a non-positive epoch count to 1, the other REFUSED a learning rate out
of range. Two answers to one question in one module, and nothing could have caught it because each
copy was gated against itself.

So the central gate here is parametrised over `SETTINGS` itself. A new setting added to that table
gets the precedence test for free and cannot arrive with a private ladder; a setting added OUTSIDE
the table fails the completeness check below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer.training import (  # noqa: E402
    SETTINGS,
    TrainingConfiguration,
    TrainingConfigurationError,
)


class _Request:
    """The one member `from_request` reads. Deliberately not a real `RunRequest`: this module is
    pure and must stay usable without the run-directory contract."""

    def __init__(self, budget: dict | None = None) -> None:
        self.budget = budget or {}


# =====================================================================================
# The ladder, once, for every setting
# =====================================================================================
@pytest.mark.parametrize("setting", SETTINGS, ids=[s.name for s in SETTINGS])
def test_the_request_outranks_the_environment_and_both_outrank_the_default(setting) -> None:
    """THE GATE THAT MAKES THE TABLE LOAD-BEARING.

    Parametrised over `SETTINGS`, so a new setting is covered the moment it is declared there -- and
    a setting that bypasses the table is invisible to this test, which is why the completeness gate
    below exists too.

    The ORDER is the point, not merely that each route works: a deployment-wide environment variable
    must not silently replace what one run asked for. Before this object there were three ladders and
    each was gated separately, so a fourth could have arrived with any order at all.
    """
    asked, other = _two_valid_values(setting)
    # A COMPANION WHERE A CROSS-FIELD RULE DEMANDS ONE. `focal_alpha` alone is refused, on purpose:
    # a class weighting with gamma 0 is a weighting wearing a focal loss's name. The companion is
    # supplied through the request so that it is present in all three sub-checks and cannot be
    # mistaken for the value under test, whose route is what this gate varies.
    companion = {"focal_gamma": 2.0} if setting.name == "focal_alpha" else {}

    from_default = getattr(
        TrainingConfiguration.from_request(_Request(dict(companion)), environ={}), setting.name)
    assert from_default == setting.check(setting.default, "default"), (
        f"{setting.name} does not fall back to its declared default"
    )

    from_environment = TrainingConfiguration.from_request(
        _Request(dict(companion)), environ={setting.variable: str(other)})
    assert getattr(from_environment, setting.name) == other
    assert from_environment.sources[setting.name] == setting.variable

    both = TrainingConfiguration.from_request(
        _Request({**companion, setting.name: asked}), environ={setting.variable: str(other)})
    assert getattr(both, setting.name) == asked, (
        f"the environment overrode the request for {setting.name}: a deployment-wide variable must "
        "not replace what one run asked for"
    )
    assert both.sources[setting.name] == f"request.budget.{setting.name}"


def _two_valid_values(setting):
    """Two distinct values this setting accepts, so precedence can be told apart from either route."""
    pairs = {
        "max_epochs": (260, 5),
        "iterations_per_epoch": (50, 2),
        "validation_iterations_per_epoch": (10, 3),
        "initial_lr": (0.0001, 0.001),
        "focal_gamma": (2.0, 3.0),
        "focal_alpha": (0.25, 0.75),
        "configuration": ("3d_lowres", "2d"),
        "lr_scheduler": ("plateau", "poly"),
    }
    assert setting.name in pairs, (
        f"{setting.name} was added to SETTINGS without two test values here, so its precedence is "
        "declared and unverified"
    )
    return pairs[setting.name]


def test_every_setting_is_in_the_table_and_carries_a_reason() -> None:
    """A setting outside the table has its own ladder, which is the defect this object removed.

    And a reason nobody wrote makes the table a list of names: every entry has to say why its default
    is what it is, because "why 1000 epochs" is asked at every review.
    """
    declared = {setting.name for setting in SETTINGS}
    fields = {
        name for name in TrainingConfiguration.__dataclass_fields__ if name != "sources"
    }
    assert fields == declared, (
        f"fields and SETTINGS disagree: only a field {sorted(fields - declared)}, only in the table "
        f"{sorted(declared - fields)}. A field outside the table gets no precedence and no "
        "validation"
    )
    for setting in SETTINGS:
        assert len(setting.why) > 60, f"{setting.name} has no real reason: {setting.why!r}"
        assert setting.variable.startswith("MEDOS_TRAINER_"), (
            f"{setting.name}'s variable {setting.variable!r} is not in this deployment's namespace"
        )


# =====================================================================================
# Refusals, not clamps
# =====================================================================================
def test_a_non_positive_epoch_count_is_refused_and_not_clamped_to_one() -> None:
    """THE DRIFT THIS OBJECT FIXED, asserted so it cannot come back.

    The old `_budget` did `max(1, int(...))`. A clamp trains a run nobody asked for, and the record
    then names the request rather than what happened -- so a reviewer reading "max_epochs: 0" would
    see a run that trained one epoch and a record claiming zero.
    """
    for bad in (0, -1, -1000):
        with pytest.raises(TrainingConfigurationError) as raised:
            TrainingConfiguration.from_request(_Request({"max_epochs": bad}), environ={})
        assert "Clamping" in str(raised.value), (
            "the refusal does not say why clamping was rejected, and the next person to see a "
            "zero will add the clamp back"
        )


@pytest.mark.parametrize("budget,fragment", [
    ({"initial_lr": 0}, "not a rate"),
    ({"initial_lr": 1}, "not a rate"),
    ({"initial_lr": 12}, "not a rate"),
    ({"initial_lr": "x"}, "decimal number"),
    ({"focal_gamma": -1}, "negative exponent"),
    ({"focal_gamma": "two"}, "plain BCE"),
    ({"focal_alpha": 1.5}, "share and must lie"),
    ({"iterations_per_epoch": 0}, "not a number of iterations"),
    ({"configuration": "  "}, "is empty"),
    ({"lr_scheduler": "cosine"}, "not a schedule"),
])
def test_a_value_this_platform_cannot_honour_is_refused_with_a_reason(budget, fragment) -> None:
    """Each refusal names what would otherwise have happened, because every default here is a
    WORKING value -- so a silent fall-back trains something plausible and records the request."""
    with pytest.raises(TrainingConfigurationError, match=fragment):
        TrainingConfiguration.from_request(_Request(budget), environ={})


def test_a_class_weighting_without_a_focal_exponent_is_refused() -> None:
    """A CROSS-FIELD RULE, which is why validation belongs to the object and not to each field.

    `focal_alpha` with `focal_gamma=0.0` weights the classes while the pointwise term is still plain
    BCE: a class weighting wearing a focal loss's name. Neither field alone can see it, and three
    separate functions could not have.
    """
    with pytest.raises(TrainingConfigurationError, match="wearing a focal loss"):
        TrainingConfiguration.from_request(
            _Request({"focal_alpha": 0.25}), environ={})
    assert TrainingConfiguration.from_request(
        _Request({"focal_alpha": 0.25, "focal_gamma": 2.0}), environ={}).focal_alpha == 0.25


def test_the_checks_run_on_direct_construction_too() -> None:
    """Otherwise they guard one entrance of two, and a gate that builds one by hand would escape."""
    with pytest.raises(TrainingConfigurationError):
        TrainingConfiguration(max_epochs=0)
    with pytest.raises(TrainingConfigurationError):
        TrainingConfiguration(initial_lr=5.0)


# =====================================================================================
# The record
# =====================================================================================
def test_the_document_carries_every_setting_including_the_defaults() -> None:
    """A RECORD THAT OMITTED THE DEFAULTS WOULD MAKE TWO DIFFERENT RUNS LOOK IDENTICAL.

    Two runs trained under different schedules carry the same document wherever one of them asked
    for nothing -- and "whatever the default was at the time" is not a value anybody can look up
    afterwards.
    """
    document = TrainingConfiguration.from_request(_Request(), environ={}).as_document()
    for setting in SETTINGS:
        assert setting.name in document, f"{setting.name} is not in the record"
    assert document["initial_lr"] is None, (
        "null says the backend used its own rate, which stays true if upstream changes it; a "
        "literal 1e-2 would be a claim about upstream at the moment the record was written"
    )
    assert document["save_every"] == 1000
    assert set(document["sources"]) == {s.name for s in SETTINGS}
    assert set(document["sources"].values()) == {"default"}


def test_the_record_says_where_each_setting_came_from() -> None:
    """"Who asked for this" is the first question at a review and the value alone cannot answer it."""
    configured = TrainingConfiguration.from_request(
        _Request({"max_epochs": 260}),
        environ={"MEDOS_TRAINER_INITIAL_LR": "0.0001"},
    )
    sources = configured.as_document()["sources"]
    assert sources["max_epochs"] == "request.budget.max_epochs"
    assert sources["initial_lr"] == "MEDOS_TRAINER_INITIAL_LR"
    assert sources["focal_gamma"] == "default"


def test_the_checkpoint_cadence_is_derived_and_not_asked_for() -> None:
    """It follows from the schedule. It was `max(1, int(budget["max_epochs"]))` inline in the
    backend; derived here so a reader of this object sees the whole of what a fit does."""
    assert "save_every" not in {setting.name for setting in SETTINGS}
    assert TrainingConfiguration(max_epochs=260).save_every == 260
    assert TrainingConfiguration(max_epochs=1).save_every == 1


def test_the_log_line_names_what_was_asked_for() -> None:
    """`result.json` is written only after the fit succeeds, so a run that dies at epoch 3 has this
    line and nothing else."""
    line = TrainingConfiguration.from_request(
        _Request({"max_epochs": 260}), environ={"MEDOS_TRAINER_FOCAL_GAMMA": "2"}).describe()
    assert "epochs=260x250" in line
    assert "focal_gamma=2" in line
    assert "lr=backend default" in line
    assert "asked for:" in line and "max_epochs" in line and "focal_gamma" in line
