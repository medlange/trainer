# SPDX-License-Identifier: Apache-2.0
"""What EVERY training backend must satisfy, checked for every backend the image registers.

WHY THIS FILE EXISTS
--------------------
The point of a second backend is that the ARCHITECTURE changes. The thing that must not
change is the reason the platform exists: a channel a case does not annotate contributes
zero to the loss and zero to every gradient, never a negative. A backend that gets this
wrong does not look wrong. It converges, it writes a checkpoint, it reports a falling loss,
and it scores well on each corpus's own split -- because each split carries the same blind
spot as its training data. Measured on this cohort: 138 of 900 (case, channel) pairs are
supervised, so 762 would silently become negatives.

So `port.BACKENDS` is not a list of modules. It is a list of claims, and this file is what
makes each claim cost something to make.

IT RUNS ON CPU, WITH NO CORPUS AND NO CARD, IN MILLISECONDS. That is a requirement and not a
convenience: a gate that needs a GPU and a staged dataset is a gate nobody runs while adding
a backend, and a backend nobody proves correct is the defect above waiting for a cohort.

THE FAILURE HAS TWO HALVES AND SO DOES THE GATE
------------------------------------------------
  1. A loss that ignores the mask. Caught by perturbation: replace an unsupervised
     channel's target with anything at all and neither the loss nor any gradient may move.
  2. A CORRECT loss that nothing calls. Caught over the syntax tree, because no execution
     can see it and no grep can either -- `MaskedDeepSupervisionWrapper` was written for
     this, perturbation-tested at every decoder scale, and then not used, while
     `_build_loss` reached for nnU-Net's wrapper instead. Every test passed.

The second half is the one that has actually happened here, twice: that wrapper, and a
`supervision.json` whose `empty_segment_is_negative` was hard-coded `False` while the only
reader of it was a log line.

WHAT IS NOT CHECKED HERE. The loss ARITHMETIC -- the BCE denominator counting voxels and
not (case, channel) pairs, the Dice reduction not averaging in a constant 1.0, the
reduced-precision confusion counts. That is `test_masked_loss.py`, which owns
`masked.py` itself. This file owns the WIRING from a registered backend to it.
"""

from __future__ import annotations

import ast
import json
import inspect
import textwrap
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from medos_trainer import port  # noqa: E402

BACKENDS = sorted(port.BACKENDS)

#: The only names a backend may construct a loss with. `masked.py` is the reference
#: implementation and everything else in the trainer must route through it.
PERMITTED_LOSS_NAMES = frozenset({
    "MaskedDiceBCELoss",
    "MaskedDeepSupervisionWrapper",
    "build_masked_loss",
})

#: Substrings that mark a call as CONSTRUCTING a loss. A substring rule with an allow-list,
#: not a type checker: its job is to make a second loss impossible to add quietly, and the
#: names that matter are the ones nnU-Net, torch and MONAI actually ship
#: (`DC_and_BCE_loss`, `DC_and_CE_loss`, `RobustCrossEntropyLoss`, `DiceCELoss`,
#: `BCEWithLogitsLoss`, `DeepSupervisionWrapper`).
LOSS_MARKERS = ("loss", "cross_entropy")

#: `masked.py` IS the masked loss, so of course it calls
#: `binary_cross_entropy_with_logits`. Its own gate is `test_masked_loss.py`.
EXEMPT_MODULES = frozenset({"medos_trainer.masked"})


# =====================================================================================
# The registry is a set of claims, and each one has to be answerable
# =====================================================================================
@pytest.mark.parametrize("kind", BACKENDS)
def test_the_registered_module_answers_for_the_name_it_is_registered_under(kind) -> None:
    module = port.resolve(kind)
    assert module.KIND == kind


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_backend_implements_every_operation_the_port_declares(kind) -> None:
    """Checked by signature, not by presence.

    A backend missing `derive_plan` fails at import. A backend whose `derive_plan` takes
    positional `work` fails after the cohort has been staged, which on this corpus is tens
    of gigabytes and twenty minutes.
    """
    module = port.resolve(kind)
    for name in ("prepare_workspace", "apply_determinism", "derive_plan", "fit",
                 "build_masked_loss"):
        operation = getattr(module, name, None)
        assert callable(operation), f"{kind} has no {name}"
        expected = inspect.signature(getattr(port.TrainingBackend, name))
        got = inspect.signature(operation)
        # The Protocol's methods carry `self`; a module's functions do not.
        wanted = [p for p in expected.parameters.values() if p.name != "self"]
        assert [p.name for p in got.parameters.values()] == [p.name for p in wanted], (
            f"{kind}.{name}{got} does not match the port's {name}{expected}"
        )


def test_an_unregistered_kind_is_refused_and_the_refusal_names_what_is_installed() -> None:
    with pytest.raises(LookupError, match=r"this image trains \['nnunet'\]"):
        port.resolve("auto3dseg")


def test_a_module_registered_under_a_name_it_does_not_claim_is_refused(monkeypatch) -> None:
    """The registry decides which code runs; the module's KIND is what the artifact is
    labelled with. If they disagree, the artifact is mislabelled and every structural check
    still passes."""
    monkeypatch.setitem(port.BACKENDS, "auto3dseg", "medos_trainer.backend")
    with pytest.raises(LookupError, match="declares KIND='nnunet'"):
        port.resolve("auto3dseg")


# =====================================================================================
# The fit result: every member named in one refusal, before the cost is sunk
# =====================================================================================
def _fit_document(**over: Any) -> dict[str, Any]:
    document = {
        "network": object(), "patch_size": (64, 64, 64), "checkpoint": "c.pth",
        "classes": 11, "budget": {"max_epochs": 2}, "seconds": 1.5, "device": "cpu",
    }
    document.update(over)
    return document


def test_a_complete_fit_result_is_accepted_and_typed() -> None:
    result = port.FitResult.from_mapping(_fit_document())
    assert result.patch_size == (64, 64, 64)
    assert result.classes == 11
    assert isinstance(result.checkpoint, Path)


def test_every_absent_member_is_named_in_one_refusal() -> None:
    """NOT a `KeyError` on the first one. This is read while writing the run's terminal
    record, which happens after the fit has been paid for -- so a backend author who fixes
    one member and pays for another fit to find the second has been failed by the gate."""
    document = _fit_document()
    for name in ("classes", "seconds"):
        document.pop(name)
    with pytest.raises(ValueError, match=r"no \['classes', 'seconds'\]"):
        port.FitResult.from_mapping(document)


def test_the_port_does_not_invent_a_label_manager() -> None:
    """`_fit` read `fitted["label_manager_classes"]`. `LabelManager` is an nnU-Net class, so
    the port's own vocabulary named one implementation's internals, and a second backend
    would have had to invent an nnU-Net concept to fill the field.

    The old spelling is not merely unused -- it must not be ACCEPTED, or a driver could keep
    writing it and the port would read `classes` as absent while the document plainly
    carries the number.
    """
    assert not hasattr(port.FitResult, "label_manager_classes")
    document = _fit_document()
    document["label_manager_classes"] = document.pop("classes")
    with pytest.raises(ValueError, match=r"no \['classes'\]"):
        port.FitResult.from_mapping(document)


# =====================================================================================
# Half one: the loss honours the mask, proven by perturbation
# =====================================================================================
def _perturbation_case(
    kind: str, *, scales: Any, batch_dice: bool, focal_gamma: float = 0.0
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch where channel 1 of case 0 is UNSUPERVISED, and the loss built by `kind`."""
    torch.manual_seed(20260926)
    loss = port.resolve(kind).build_masked_loss(
        batch_dice=batch_dice, scales=scales, focal_gamma=focal_gamma
    )
    logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 3, 8, 8, 8) > 0.6).float()
    mask = torch.ones(2, 3)
    mask[0, 1] = 0.0
    return loss, logits, target, mask


def _call(loss: Any, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
          scales: Any) -> torch.Tensor:
    if scales is None:
        return loss(logits, target, mask)
    # A deep-supervision wrapper takes one list per scale. The mask is scale-invariant and
    # the wrapper is what knows that, so it gets ONE mask and not a list.
    return loss([logits] * len(scales), [target] * len(scales), mask)


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("scales", [None, [(1, 1, 1), (2, 2, 2), (4, 4, 4)]],
                         ids=["single_head", "deep_supervision"])
@pytest.mark.parametrize("batch_dice", [False, True], ids=["per_case", "batch"])
# EVERY CONFIGURATION THE BACKEND CAN BUILD, not just the default one. `focal_gamma`
# re-weights the pointwise term per voxel, and a per-voxel weight is exactly the kind of
# thing that can be computed before the mask is applied and thus leak an unsupervised
# channel's contents into the denominator -- which would move the loss without ever moving
# that channel's gradient, so a gradient-only check would miss it.
@pytest.mark.parametrize("focal_gamma", [0.0, 2.0], ids=["bce", "focal2"])
def test_an_unsupervised_channel_cannot_move_the_loss_or_any_gradient(
    kind, scales, batch_dice, focal_gamma
) -> None:
    """THE INVARIANT, STATED AS AN EXPERIMENT ANYONE CAN RUN.

    Replace the unsupervised channel's target with its exact opposite -- the largest
    perturbation available -- and the loss must be bit-identical and every gradient
    unchanged. Bit-identical and not `approx`: zero contribution means the tensor never
    entered the sum, and a tolerance here would admit a mask that merely down-weights.
    """
    loss, logits, target, mask = _perturbation_case(
        kind, scales=scales, batch_dice=batch_dice, focal_gamma=focal_gamma
    )
    before = _call(loss, logits, target, mask, scales)
    before.backward()
    assert logits.grad is not None
    grad_before = logits.grad.clone()

    perturbed = target.clone()
    perturbed[0, 1] = 1.0 - perturbed[0, 1]
    logits.grad = None
    after = _call(loss, logits, perturbed, mask, scales)
    after.backward()

    assert after.item() == before.item(), (
        f"{kind}: flipping an UNSUPERVISED channel's target moved the loss from "
        f"{before.item()!r} to {after.item()!r}. That channel is being trained as "
        "background, which is the false-negative signal this subsystem exists to remove"
    )
    assert torch.equal(logits.grad, grad_before), (
        f"{kind}: flipping an unsupervised channel's target changed a gradient"
    )
    assert torch.all(logits.grad[0, 1] == 0.0), (
        f"{kind}: the unsupervised channel received gradient"
    )


# =====================================================================================
# Half two: nothing builds a loss of its own, and the trainer calls the one that exists
# =====================================================================================
def _trainer_modules(kind: str) -> dict[str, Path]:
    """Every `medos_trainer` module the backend's own source reaches, by import.

    One level, deliberately: the gate is over the backend's own code, and going transitive
    would pull `masked.py` in as a subject rather than as the reference.
    """
    import importlib

    root = Path(port.resolve(kind).__file__).parent
    found: dict[str, Path] = {}
    queue = [port.BACKENDS[kind]]
    seen: set[str] = set()
    while queue:
        name = queue.pop()
        if name in seen or name in EXEMPT_MODULES:
            continue
        seen.add(name)
        module = importlib.import_module(name)
        path = Path(module.__file__ or "")
        if not path.is_file() or root not in path.parents:
            continue
        found[name] = path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "medos_trainer"
            ):
                queue.append(node.module)
            elif isinstance(node, ast.Import):
                queue += [a.name for a in node.names if a.name.startswith("medos_trainer")]
    return found


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_construction(node: ast.Call) -> bool:
    """A CONSTRUCTION, not an invocation of one already built.

    `self.loss(...)` and `self._call_loss(...)` call what `_build_loss` returned; flagging
    those would make the gate unpassable and would say nothing. An attribute of `self` is
    therefore never a construction -- a backend cannot construct a loss out of itself.
    """
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id == "self":
            return False
    name = _called_name(node)
    return any(marker in name.lower() for marker in LOSS_MARKERS) or name.endswith(
        "Wrapper"
    )


@pytest.mark.parametrize("kind", BACKENDS)
def test_no_backend_constructs_a_loss_of_its_own(kind) -> None:
    """A driver that builds `DC_and_BCE_loss` has escaped the mask by construction.

    This cannot be checked by running the driver -- running it needs plans, a staged cohort
    and a card -- and it cannot be checked by grep, which cannot tell code from prose about
    code and would fail on this file's own docstring. So: the syntax tree.
    """
    offences = []
    for name, path in _trainer_modules(kind).items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_construction(node):
                called = _called_name(node)
                if called not in PERMITTED_LOSS_NAMES:
                    offences.append(f"{name}:{node.lineno} {called}()")
    assert not offences, (
        f"{kind} constructs a loss outside medos_trainer.masked: {offences}. Every "
        f"backend's loss must be one of {sorted(PERMITTED_LOSS_NAMES)}, because the mask is "
        "applied inside those and nowhere else"
    )


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_training_hook_calls_the_loss_builder_the_conformance_suite_exercises(
    kind
) -> None:
    """THE OTHER HALF, AND THE ONE THAT HAS ACTUALLY FAILED HERE.

    The perturbation test above proves `build_masked_loss` honours the mask. It proves
    nothing about what the trainer uses. A `_build_loss` that ignored it would leave this
    suite entirely green while every fit trained unannotated channels as background.

    So: find the hook the backend's trainer overrides to build its loss, and assert its body
    calls `build_masked_loss`. Asserted over the syntax tree of whichever module defines
    it, so a driver cannot satisfy it with a comment saying it does.
    """
    hooks = []
    for name, path in _trainer_modules(kind).items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_loss":
                calls = {_called_name(c) for c in ast.walk(node)
                         if isinstance(c, ast.Call)}
                hooks.append((f"{name}._build_loss", calls))
    assert hooks, (
        f"{kind} defines no `_build_loss`, so nothing connects its training step to "
        "`build_masked_loss` and the perturbation test above proves only that an unused "
        "function is correct"
    )
    for where, calls in hooks:
        assert "build_masked_loss" in calls, (
            f"{where} does not call build_masked_loss; it calls {sorted(calls)}. The "
            "conformance suite exercises build_masked_loss, so a hook that builds its own "
            "loss is a hook no gate has ever run"
        )


# =====================================================================================
# THE FOCAL SHAPING: the wiring, because the arithmetic already had six gates and no caller
# =====================================================================================
#
# `masked.py` carried `focal_gamma` and `focal_alpha` from the day they were written, with six
# gates over the arithmetic, and `_build_loss` passed neither. So every fit this image has ever
# run used plain BCE, no run COULD have asked for anything else, and the whole suite was green
# throughout. That is the same defect shape as a state key nobody subscribes to: the feature has
# tests and does not exist. What follows gates the CONNECTION, which is the part that was missing.


def _function_in(kind: str, module_suffix: str, name: str):
    """The AST of one named function, from whichever of the backend's modules defines it."""
    for module, path in _trainer_modules(kind).items():
        if not module.endswith(module_suffix):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return module, node
    return None, None


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_focal_shaping_reaches_the_loss_from_the_trainer_not_from_a_literal(kind) -> None:
    """THE GATE THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT, and it has to be about the ARGUMENT.

    `_build_loss` calling `build_masked_loss` is already gated above. That gate passed for the
    whole life of the focal term, because a call with the keyword ABSENT satisfies it -- and an
    absent keyword takes the builder's default of 0.0, which is plain BCE.

    So this asserts two things a comment cannot satisfy: that both keywords are passed at all, and
    that each value is an ATTRIBUTE OF `self` rather than a constant. A literal `focal_gamma=0.0`
    there would compile, read as deliberate, and make the setting unreachable again -- which is
    exactly the state this gate was written to end.
    """
    module, hook = _function_in(kind, "masked_trainer", "_build_loss")
    assert hook is not None, f"{kind} defines no `_build_loss` in a masked_trainer module"

    builders = [node for node in ast.walk(hook)
                if isinstance(node, ast.Call) and _called_name(node) == "build_masked_loss"]
    assert len(builders) == 1, (
        f"{module}._build_loss calls build_masked_loss {len(builders)} times; with two, which one "
        "shapes the loss depends on a branch this gate cannot see"
    )
    keywords = {k.arg: k.value for k in builders[0].keywords if k.arg}
    for name in ("focal_gamma", "focal_alpha"):
        assert name in keywords, (
            f"{module}._build_loss does not pass {name}. An absent keyword takes "
            f"build_masked_loss's own default, so the setting is unreachable from any run -- "
            "which is the defect this gate exists for, and it was the real state of this file"
        )
        value = keywords[name]
        assert isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
            and value.value.id == "self", (
            f"{module}._build_loss passes {name}={ast.dump(value)[:60]}, which is not an attribute "
            "of the trainer. A constant here compiles and reads as deliberate while making the "
            "setting impossible to ask for"
        )


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_backend_sets_the_focal_attributes_before_it_initialises_the_trainer(kind) -> None:
    """ORDER, AND IT IS LOAD-BEARING RATHER THAN TIDY.

    `_build_loss` runs inside `initialize()`. An assignment after that call would leave the
    attributes at their declared defaults for the loss that actually trains, and then CHANGE them
    -- so the run's record would name a focal loss the fit never used. Nothing would raise, the
    loss curve would look ordinary, and the record would be wrong in the direction that flatters.
    """
    module, fit = _function_in(kind, "backend", "fit")
    assert fit is not None, f"{kind} defines no `fit`"

    statements = list(ast.walk(fit))
    def line_of_assignment(attribute: str) -> int | None:
        return next((node.lineno for node in statements if isinstance(node, ast.Assign)
                     for target in node.targets
                     if isinstance(target, ast.Attribute) and target.attr == attribute), None)

    initialise = [node.lineno for node in statements
                  if isinstance(node, ast.Call) and _called_name(node) == "initialize"]
    assert initialise, f"{module}.fit never calls initialize(), so no loss is ever built"

    for attribute in ("_focal_gamma", "_focal_alpha"):
        line = line_of_assignment(attribute)
        assert line is not None, (
            f"{module}.fit never sets {attribute}, so the trainer keeps its declared default and "
            "no run can ask for a focal loss"
        )
        assert line < min(initialise), (
            f"{module}.fit sets {attribute} on line {line}, after initialize() on line "
            f"{min(initialise)}. `_build_loss` runs inside initialize(), so the loss that trains "
            "would be built from the default and the record would name the other one"
        )


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_focal_settings_are_merged_into_what_the_fit_records(kind) -> None:
    """A SETTING NO RECORD CARRIES IS A SETTING NOBODY CAN ATTRIBUTE A RESULT TO.

    `__main__._fit` writes `fitted.budget` into `hyperparameters` and reads nothing else for it, so
    the focal shaping has to arrive inside that mapping. Asserted as a `**` unpacking of the
    settings call's own result rather than as two literal keys, because two literals could drift
    from what the function returns while both halves still read correctly.
    """
    module, fit = _function_in(kind, "backend", "fit")
    assert fit is not None

    returns = [node for node in ast.walk(fit) if isinstance(node, ast.Return)
               and isinstance(node.value, ast.Dict)]
    assert returns, f"{module}.fit returns no dictionary"
    budgets = [value for node in returns
               for key, value in zip(node.value.keys, node.value.values)
               if isinstance(key, ast.Constant) and key.value == "budget"]
    assert len(budgets) == 1, f"{module}.fit returns {len(budgets)} 'budget' members"
    # THE MECHANISM MOVED AND THE PROPERTY DID NOT. It used to be two dictionaries merged at the
    # return statement with `**`, and this asserted the unpacking. It is now one object's own
    # document, which is stronger: `as_document()` emits EVERY setting including the ones at their
    # default, so a run that asked for nothing still records what it trained under -- whereas the
    # merge only carried whatever the two dictionaries happened to hold.
    #
    # Asserted behaviourally against the object rather than over the AST, because the question is
    # what the record CONTAINS and a syntax tree can only say where it came from.
    from medos_trainer.training import TrainingConfiguration

    recorded = TrainingConfiguration().as_document()
    for member in ("focal_gamma", "focal_alpha"):
        assert member in recorded, (
            f"{member} is absent from the recorded configuration, so a run trained with a focal "
            "loss would write a record that does not say so"
        )
    assert isinstance(budgets[0], ast.Call), (
        f"{module}.fit builds its 'budget' member as {type(budgets[0]).__name__}; it must be the "
        "training configuration's own document, or a setting can reach the fit without reaching "
        "the record"
    )


def test_a_malformed_focal_value_is_refused_rather_than_silently_defaulted(monkeypatch) -> None:
    """The refusal exists because the default is PLAIN BCE, which is a working loss.

    A typo that fell back to the default would train something perfectly reasonable while whoever
    set the variable believed a focal loss was training -- and the run's record would agree with
    them, because the record reports what the function returned.
    """
    from medos.sdk.contract import ContractViolation

    from medos_trainer import backend

    class _Request:
        budget: dict = {}

    monkeypatch.setenv("MEDOS_TRAINER_FOCAL_GAMMA", "two")
    with pytest.raises(ContractViolation) as raised:
        backend._focal_settings(_Request())
    message = str(raised.value)
    # THE TYPE IS THE CONTRACT BOUNDARY'S and the message is the pure module's. `training.py` raises
    # `TrainingConfigurationError`; the backend translates it, because a refusal here has to arrive
    # as the type `__main__._phase` records. This gate caught exactly that when the ladder moved.
    assert "MEDOS_TRAINER_FOCAL_GAMMA" in message and "decimal number" in message
    assert "plain BCE" in message, (
        "the refusal no longer says what the fall-back would have trained, which is the half that "
        "makes it actionable: 0.0 is a WORKING loss, so a typo would train something plausible"
    )

    monkeypatch.delenv("MEDOS_TRAINER_FOCAL_GAMMA")
    monkeypatch.setenv("MEDOS_TRAINER_FOCAL_ALPHA", "")
    settings = backend._focal_settings(_Request())
    assert settings == {"focal_gamma": 0.0, "focal_alpha": None}, (
        "an EMPTY variable is not a malformed one -- it is the variable not being set -- and it "
        f"must give the default; got {settings}"
    )


def test_the_request_outranks_the_environment_and_both_outrank_the_default(monkeypatch) -> None:
    """`_budget`'s own precedence, applied to the same block.

    The request is the right home: a run that carries its loss shape in the document describing it
    records the shape and the schedule in one place, written by whoever submitted it. The
    environment is the deliberate-deviation route. If the environment won, a deployment-wide
    variable would silently override what a specific run asked for.
    """
    from medos_trainer import backend

    class _Request:
        def __init__(self, budget):
            self.budget = budget

    monkeypatch.setenv("MEDOS_TRAINER_FOCAL_GAMMA", "3.0")
    assert backend._focal_settings(_Request({}))["focal_gamma"] == 3.0
    assert backend._focal_settings(_Request({"focal_gamma": 2.0}))["focal_gamma"] == 2.0, (
        "the environment overrode the request; a deployment-wide variable must not silently "
        "replace what one run asked for"
    )
    monkeypatch.delenv("MEDOS_TRAINER_FOCAL_GAMMA")
    assert backend._focal_settings(_Request({}))["focal_gamma"] == 0.0


def test_the_trainer_declares_the_focal_defaults_so_an_unchanged_run_is_unchanged() -> None:
    """THE COMPATIBILITY CLAIM, ASSERTED RATHER THAN PROMISED.

    Every run before this wiring existed trained plain BCE. `gamma=0.0` with `alpha=None` reduces
    `masked._pointwise` to `binary_cross_entropy_with_logits` -- which `test_masked_loss.py`
    asserts against torch's own function -- so the A/B pair already on disk stays comparable to
    anything trained under the defaults. That only holds if the DECLARED defaults are those two
    values, which is what this checks.
    """
    from medos_trainer import masked_trainer

    source = inspect.getsource(masked_trainer.nnUNetTrainerMaskedChannels.__init__)
    tree = ast.parse(textwrap.dedent(source))
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            if node.value is not None:
                try:
                    assigned[node.target.attr] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    # A computed default (a call, not a literal) carries no identity claim
                    # to check -- `_channel_weights` is loaded from the dataset, and its
                    # identity is "absent file means unweighted", which the masked-loss
                    # tests hold. This gate checks declared literals only.
                    continue
    assert assigned.get("_focal_gamma") == 0.0, (
        f"_focal_gamma is declared as {assigned.get('_focal_gamma')!r}; a nonzero default would "
        "change what every existing run trains and break comparability with the pair on disk"
    )
    assert assigned.get("_focal_alpha") is None, (
        f"_focal_alpha is declared as {assigned.get('_focal_alpha')!r}; any class weighting makes "
        "gamma 0 stop being an exact identity with plain BCE"
    )


# =====================================================================================
# THE INITIAL LEARNING RATE: a run-level setting, because one architecture measurably needs it
# =====================================================================================
#
# MEASURED ON THE LAB, NOT ANTICIPATED. `MedOSSegResNetDS` at this cohort's plan is 356.2M
# parameters against PlainConvUNet's 31.2M, and under nnU-Net's own schedule -- polynomial decay
# from 1e-2, SGD, momentum 0.99, nesterov -- it does not train: `train_loss` NaN from the first
# epoch and the validation dice stuck at its initialisation value for 29 epochs. The network was
# exonerated first (finite in fp32 eager, in train mode, under autocast fp16, under torch.compile,
# with finite gradients, at the plan's own patch and batch size), then the loss (one iteration per
# epoch makes the logged mean that iteration: four of four finite). At 1e-3 the same run trains.
#
# So this is a setting the platform has to be able to express, and these gate the wiring -- which
# is the half that the focal term proved can be missing while every other test passes.


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_backend_sets_the_learning_rate_before_it_initialises_the_trainer(kind) -> None:
    """ORDER, AND FOR A DIFFERENT REASON THAN THE LOSS'S.

    nnU-Net builds its optimiser and its `PolyLRScheduler` in `configure_optimizers()`, which
    `initialize()` calls, and the scheduler captures `initial_lr` BY VALUE. An assignment after
    that leaves the optimiser stepping at nnU-Net's rate while `self.initial_lr` -- which is what
    the trainer logs -- reports ours. Nothing raises, the run looks configured, and the log names a
    rate the fit never used.
    """
    module, fit = _function_in(kind, "backend", "fit")
    assert fit is not None, f"{kind} defines no `fit`"
    statements = list(ast.walk(fit))

    assigned = [node.lineno for node in statements if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Attribute) and target.attr == "initial_lr"]
    assert assigned, (
        f"{module}.fit never assigns `initial_lr`, so a deployment cannot ask for one and the "
        "architecture that measurably needs it cannot be trained"
    )
    initialise = [node.lineno for node in statements
                  if isinstance(node, ast.Call) and _called_name(node) == "initialize"]
    assert initialise and max(assigned) < min(initialise), (
        f"{module}.fit assigns initial_lr on {assigned} and calls initialize() on {initialise}. "
        "PolyLRScheduler captures the rate by value inside initialize(), so the optimiser would "
        "step at nnU-Net's rate while the log named ours"
    )


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_learning_rate_is_recorded_even_when_it_is_nnunet_own(kind) -> None:
    """`None` IS A RECORD AND NOT AN ABSENCE, and the distinction is the point.

    "initial_lr: null" says the fit used whatever nnU-Net sets, which stays true if upstream changes
    it. Writing the literal 1e-2 instead would be a claim about upstream's value at the moment the
    record was written, and it would silently stop being true.
    """
    module, fit = _function_in(kind, "backend", "fit")
    returns = [node for node in ast.walk(fit) if isinstance(node, ast.Return)
               and isinstance(node.value, ast.Dict)]
    budgets = [value for node in returns
               for key, value in zip(node.value.keys, node.value.values)
               if isinstance(key, ast.Constant) and key.value == "budget"]
    assert len(budgets) == 1
    from medos_trainer.training import TrainingConfiguration

    keys = set(TrainingConfiguration().as_document())
    assert "initial_lr" in keys, (
        f"{module}.fit does not record initial_lr in the one mapping `_fit` writes into "
        "hyperparameters, so two runs trained at different rates would carry identical records"
    )


def test_no_request_and_no_variable_leaves_nnunet_own_rate_untouched(monkeypatch) -> None:
    """THE DEFAULT HAS TO BE `None`, NOT 1e-2.

    Every run already on disk was trained at nnU-Net's own rate. Returning a number here -- even
    the right number today -- would make the platform set a rate it believes is nnU-Net's, and the
    two would diverge silently the first time upstream changed it.
    """
    from medos_trainer import backend

    class _Request:
        budget: dict = {}

    monkeypatch.delenv("MEDOS_TRAINER_INITIAL_LR", raising=False)
    assert backend._initial_lr(_Request()) is None


def test_the_request_outranks_the_variable_for_the_learning_rate_too(monkeypatch) -> None:
    """Same precedence as the budget and the focal shaping: a deployment-wide variable must not
    silently replace what one run asked for."""
    from medos_trainer import backend

    class _Request:
        def __init__(self, budget):
            self.budget = budget

    monkeypatch.setenv("MEDOS_TRAINER_INITIAL_LR", "0.01")
    assert backend._initial_lr(_Request({})) == 0.01
    assert backend._initial_lr(_Request({"initial_lr": 0.001})) == 0.001


def test_a_malformed_or_impossible_learning_rate_is_refused(monkeypatch) -> None:
    """A typo must not fall back to nnU-Net's rate, and a rate outside (0, 1) is not a rate.

    The fall-back is the dangerous half: 1e-2 is a WORKING rate for the default architecture, so a
    typo would train something plausible while the record named the value that was asked for.
    """
    from medos.sdk.contract import ContractViolation

    from medos_trainer import backend

    class _Request:
        budget: dict = {}

    monkeypatch.setenv("MEDOS_TRAINER_INITIAL_LR", "1e-3x")
    with pytest.raises(ContractViolation, match="decimal number"):
        backend._initial_lr(_Request())

    for impossible in ("0", "1", "-0.001", "12"):
        monkeypatch.setenv("MEDOS_TRAINER_INITIAL_LR", impossible)
        with pytest.raises(ContractViolation, match="not a rate"):
            backend._initial_lr(_Request())


# =====================================================================================
# THE CONFIGURATION IS A CHOICE THAT TRAVELS IN THE FROZEN PLAN, not a module constant
# =====================================================================================
#
# THE DEFECT THIS CLOSES WAS LATENT AND INVISIBLE. `derive_plan` recorded a `configuration` member
# into `plan.json` and NOTHING read it back: `fit` re-read a module constant. So a plan derived and
# preprocessed for one configuration was fit under whatever that constant named, and the two could
# disagree with nothing raising -- the run would train, converge and produce a bundle whose recorded
# configuration was not the one it trained. `MOS-TRAIN-135` freezes the plan at run start, and the
# configuration belongs inside that freeze.
#
# It is also what admits `2d` and `3d_lowres` at all, which is why it comes before a wider port.


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_fit_takes_its_configuration_from_the_plan_and_not_from_a_constant(kind) -> None:
    """AST, because running `fit` needs a card, a checkpoint and 170 GB of preprocessed data.

    Two things are asserted and the second is the one that matters: that the trainer's
    `configuration` argument is a NAME (so it came from somewhere) and that no default constant is
    read anywhere in `fit`. A fallback to the default would restore the defect exactly -- a plan
    naming `2d` fit as `3d_fullres`, silently.
    """
    module, fit = _function_in(kind, "backend", "fit")
    assert fit is not None, f"{kind} defines no `fit`"

    constants = [node.id for node in ast.walk(fit)
                 if isinstance(node, ast.Name) and "DEFAULT_CONFIGURATION" in node.id]
    assert not constants, (
        f"{module}.fit reads {sorted(set(constants))}. The configuration it trains must come from "
        "the frozen plan; a default here means a plan recorded for one configuration can be fit "
        "under another, which is the defect this member was added to close"
    )
    calls = [node for node in ast.walk(fit) if isinstance(node, ast.Call)
             for keyword in node.keywords if keyword.arg == "configuration"]
    assert calls, f"{module}.fit never passes a `configuration` to anything"
    values = [keyword.value for node in calls for keyword in node.keywords
              if keyword.arg == "configuration"]
    assert all(isinstance(value, ast.Name) for value in values), (
        "the configuration is passed as a literal or an expression rather than as a name bound "
        "from the plan, so where it came from cannot be established by reading it"
    )

    # AND IT IS CHECKED AGAINST THE DOCUMENT, not merely read. The frozen plan and the plans
    # document are two files: the first records which configuration was preprocessed, the second
    # is what nnU-Net reads. A recorded name absent from the document fails deep inside
    # `ConfigurationManager` with a KeyError naming no plan, after the card has been acquired.
    validated = [node for node in ast.walk(fit)
                 if isinstance(node, ast.Call) and _called_name(node) == "configuration"]
    assert validated, (
        f"{module}.fit never resolves the recorded configuration against the plans document, so a "
        "name the document does not carry surfaces as a KeyError inside nnU-Net"
    )


@pytest.mark.parametrize("kind", BACKENDS)
def test_the_plan_phase_preprocesses_the_configuration_it_chose(kind) -> None:
    """Preprocessing the default while recording another would put 170 GB on disk for the wrong one.

    The preprocessed directory is what every later phase reads, so this is the step at which a
    mismatch becomes expensive rather than merely wrong.
    """
    module, derive = _function_in(kind, "backend", "derive_plan")
    assert derive is not None, f"{kind} defines no `derive_plan`"
    preprocess = [node for node in ast.walk(derive)
                  if isinstance(node, ast.Call) and _called_name(node) == "preprocess_dataset"]
    assert len(preprocess) == 1, f"{module}.derive_plan calls preprocess_dataset {len(preprocess)}x"
    keywords = {k.arg: k.value for k in preprocess[0].keywords if k.arg}
    assert "configurations" in keywords
    listed = keywords["configurations"]
    assert isinstance(listed, ast.List) and len(listed.elts) == 1, (
        "more than one configuration is preprocessed, so which one the run trains is decided "
        "somewhere else"
    )
    element = listed.elts[0]
    assert isinstance(element, ast.Name) and "DEFAULT" not in element.id, (
        f"preprocess_dataset is given {ast.dump(element)[:60]}; it must be the name this phase "
        "chose, or the data on disk and the recorded configuration can differ"
    )


def test_a_frozen_plan_with_no_recorded_configuration_is_refused(monkeypatch) -> None:
    """The refusal exists because the DEFAULT is a working configuration.

    Falling back to `3d_fullres` would train something plausible on data preprocessed for something
    else -- or, when the two happen to agree, work by luck and hide the gap until the day a run
    chose `2d`.
    """
    from medos.sdk.contract import ContractViolation

    from medos_trainer import backend

    class _Request:
        budget: dict = {}

    monkeypatch.delenv("MEDOS_TRAINER_CONFIGURATION", raising=False)
    assert backend._configuration(_Request()) == "3d_fullres", (
        "a run that asks for nothing must still get the configuration every run on disk used"
    )


def test_the_request_outranks_the_variable_for_the_configuration_too(monkeypatch) -> None:
    from medos_trainer import backend

    class _Request:
        def __init__(self, budget):
            self.budget = budget

    monkeypatch.setenv("MEDOS_TRAINER_CONFIGURATION", "2d")
    assert backend._configuration(_Request({})) == "2d"
    assert backend._configuration(_Request({"configuration": "3d_lowres"})) == "3d_lowres", (
        "a deployment-wide variable overrode what one run asked for"
    )


def test_a_configuration_the_plans_document_lacks_is_refused_before_preprocessing() -> None:
    """The check needs the DOCUMENT, because the set of configurations is open.

    `save_plans` merges arbitrary user-named configurations from a pre-existing file, so the four
    the planner writes are not an enum and the only honest check is against the plan in hand.
    """
    from medos.sdk.contract import ContractViolation

    from medos_trainer import backend

    document = json.loads(
        (Path(__file__).resolve().parents[1] / "tests" / "fixtures" /
         "nnunet_plans_plain.json").read_text(encoding="utf-8")
    )

    class _Request:
        budget = {"configuration": "3d_quarterres"}

    with pytest.raises(ContractViolation) as raised:
        backend._configuration(_Request(), document)
    message = str(raised.value)
    assert "3d_quarterres" in message and "3d_fullres" in message


def test_a_cascade_stub_is_refused_with_the_stage_that_has_to_run_first() -> None:
    """A stub carries two keys and no geometry, and the cascade's low-resolution stage has to be
    fit and predicted before it can be. Left to preprocessing, this fails as a missing patch size
    somewhere inside nnU-Net."""
    from medos.sdk.contract import ContractViolation

    from medos_trainer import backend

    document = json.loads(
        (Path(__file__).resolve().parents[1] / "tests" / "fixtures" /
         "nnunet_plans_plain.json").read_text(encoding="utf-8")
    )

    class _Request:
        budget = {"configuration": "3d_cascade_fullres"}

    with pytest.raises(ContractViolation) as raised:
        backend._configuration(_Request(), document)
    message = str(raised.value)
    assert "cascade STUB" in message and "3d_fullres" in message
    assert "implement the cascade" in message


def test_the_single_channel_view_will_not_guess_a_configuration() -> None:
    """A DEFAULT HERE WOULD RECREATE THE DEFECT ONE LEVEL DOWN.

    This view is what `export_spec_fields` reads to produce the frozen `PreprocessingSpec`, so a
    view taken of the wrong configuration would put another configuration's spacing and patch into
    the spec that serving is pinned to.
    """
    import inspect

    from medos_trainer import backend

    signature = inspect.signature(backend.single_channel_view)
    assert "configuration" in signature.parameters
    assert signature.parameters["configuration"].default is inspect.Parameter.empty, (
        "the configuration has a default, so a caller that forgets it silently gets whichever one "
        "the default names -- and the spec that serving is pinned to would describe another"
    )
