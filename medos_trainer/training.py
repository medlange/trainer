# SPDX-License-Identifier: Apache-2.0
"""What a run asks of its training, as ONE declared object that can be validated and recorded.

WHY THIS EXISTS. `docs/adr/BUILD_VS_ADOPT.md` carries the row, and this is its second disqualifying
property, quoted: "Training configuration reaches the trainer ONLY by attribute assignment after
construction -- six settings do this today -- so there is no object to validate and nothing to
record; two of those six were added in this session precisely because no declared home existed."

The six, before this file:

    trainer.num_epochs                      from `_budget`
    trainer.num_iterations_per_epoch         from `_budget`
    trainer.num_val_iterations_per_epoch     from `_budget`
    trainer.save_every                       derived inline from `_budget`
    trainer._focal_gamma / _focal_alpha       from `_focal_settings`
    trainer.initial_lr                       from `_initial_lr`

Three functions, three precedence ladders written three times, and a record assembled by merging two
of the three dictionaries into a member named `budget`. A reader asking "what did this run train
under" had to know which of the three to look in, and a setting nobody had thought to record simply
was not.

WHAT THIS IS NOT. It is not a config file format, not a schema language and not a place to put
everything. It holds exactly what the platform must be able to DECIDE and RECORD about a fit, and it
refuses a value it cannot honour rather than clamping one. Anything the plan already carries -- the
patch, the batch, the strides, the architecture -- stays in the plan: two homes for one number is how
a run comes to record a value it did not use.

THE PRECEDENCE IS THE ONE `_budget` ALREADY HAD, written once. The request's own `budget` block
first, because a run that carries its settings in the document describing it records them where
whoever submitted it put them; the deployment's environment second, which is the
deliberate-deviation route; the default third. `MOS-UI-149` forbids the console displaying the epoch
budget and `TrainingRunSubmitRequest` has no member for it, so it cannot arrive from a client -- it
arrives from the deployment, and either way it is recorded.

Pure: no torch, no nnU-Net, no I/O beyond reading the environment. So a gate can exercise every
refusal for the price of a text file, which is the property `port.py`'s docstring asks of everything
on this seam.

Spec: MOS-REL-027, MOS-REL-032, MOS-TRAIN-135, MOS-TRAIN-224, MOS-UI-149.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping

__all__ = ["SETTINGS", "Setting", "TrainingConfiguration", "TrainingConfigurationError"]


class TrainingConfigurationError(ValueError):
    """A run asking for training this platform cannot honour, or cannot honour honestly.

    A distinct type so a caller can tell "this run's request is wrong" from every other
    `ValueError` a fit raises, and NOT a `ContractViolation`: this module must stay importable
    without the run-directory contract, and the caller that has one can wrap.
    """


@dataclass(frozen=True)
class Setting:
    """One thing a run can ask for: where it comes from, what it defaults to, and what it refuses.

    A TABLE AND NOT SIX IF-STATEMENTS, because the precedence was written three times before this
    file and the three copies had already drifted: `_budget` clamped a non-positive epoch count up
    to 1 with `max(1, ...)` while `_initial_lr` refused a rate outside (0, 1) -- two different
    answers to the same question in one module.
    """

    name: str
    variable: str
    default: Any
    #: Returns the value, or raises `TrainingConfigurationError` with a reason a reader can act on.
    #: Refusing rather than clamping: a clamped value trains something nobody asked for and the
    #: record then names what was asked rather than what happened.
    check: Callable[[Any, str], Any]
    why: str


def _positive_int(value: Any, where: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError(
            f"{where}={value!r} must be a whole number"
        ) from exc
    if number < 1:
        raise TrainingConfigurationError(
            f"{where}={number} is not a number of iterations. Clamping it to 1 -- which an earlier "
            "version of this did -- trains a run nobody asked for and records the request rather "
            "than what happened"
        )
    return number


def _rate_or_none(value: Any, where: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError(
            f"{where}={value!r} must be a decimal number. A malformed value must not fall back to "
            "the backend's own rate, because the fit would then train under a schedule nobody "
            "asked for while the record named another"
        ) from exc
    if not 0.0 < number < 1.0:
        raise TrainingConfigurationError(
            f"{where}={number!r}: a learning rate outside (0, 1) is not a rate. nnU-Net's own is "
            "1e-2 and this cohort's larger architectures measurably need less, not more"
        )
    return number


def _focal_gamma(value: Any, where: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError(
            f"{where}={value!r} must be a decimal number. Falling back to 0.0 would train plain "
            "BCE while whoever set it believed a focal loss was training"
        ) from exc
    if number < 0.0:
        raise TrainingConfigurationError(
            f"{where}={number!r}: a negative exponent up-weights the voxels the model already has "
            "right, which is the opposite of what focal shaping is for"
        )
    return number


def _focal_alpha(value: Any, where: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError(
            f"{where}={value!r} must be a decimal number"
        ) from exc
    if not 0.0 < number < 1.0:
        raise TrainingConfigurationError(
            f"{where}={number!r} is a share and must lie in (0, 1). On this cohort the positive "
            "voxels are a few thousandths of the volume, so a plausible-looking 0.25 changes the "
            "effective learning rate of every channel at once -- which is why it has no default"
        )
    return number


def _name(value: Any, where: str) -> str:
    text = str(value).strip()
    if not text:
        raise TrainingConfigurationError(f"{where} is empty")
    return text


def _scheduler_name(value: Any, where: str) -> str:
    text = str(value).strip()
    if text not in ("poly", "plateau"):
        raise TrainingConfigurationError(
            f"{where}={text!r} is not a schedule this image carries: 'poly' (nnU-Net's own, "
            "the default, the schedule every run on disk trained under) or 'plateau' "
            "(ReduceLROnPlateau on val_loss). Refusing rather than falling back to poly, "
            "because a misspelled name would train under poly while the record named "
            "something else"
        )
    return text


#: Every setting a run can ask for, in one table. The precedence is applied to all of them
#: identically by `TrainingConfiguration.from_request`, so a new setting cannot arrive with its own
#: private ladder -- which is how the three that existed before this file came to disagree.
SETTINGS: Final[tuple[Setting, ...]] = (
    Setting(
        "max_epochs", "MEDOS_TRAINER_EPOCHS", 1000, _positive_int,
        "nnU-Net's own default is the full run, and it is the default here too: a deployment that "
        "sets nothing gets the real thing. A truncated fit whose record does not say it was "
        "truncated reads at an approval gate as a full one.",
    ),
    Setting(
        "iterations_per_epoch", "MEDOS_TRAINER_ITERATIONS", 250, _positive_int,
        "nnU-Net's own. Reducing it to 1 is how a diagnostic probe makes a logged epoch mean equal "
        "one optimisation step, which is how the NaN in this cohort's SegResNet run was localised.",
    ),
    Setting(
        "validation_iterations_per_epoch", "MEDOS_TRAINER_VAL_ITERATIONS", 50, _positive_int,
        "nnU-Net's own. Below about ten, a channel no sampled case annotates produces a 0/0 "
        "pseudo-dice, which the log prints as NaN -- harmless, and confusing to a reader who does "
        "not know the validation was shortened.",
    ),
    Setting(
        "initial_lr", "MEDOS_TRAINER_INITIAL_LR", None, _rate_or_none,
        "`None` means DO NOT TOUCH IT, not 1e-2: a run that asks for nothing must get exactly the "
        "schedule every run already on disk was trained under, including if upstream changes it. "
        "Measured reason to ask: MedOSSegResNetDS at 356.2M parameters gives NaN from epoch 0 at "
        "1e-2, diverges at 1e-3 near epoch 40, and trains at 1e-4.",
    ),
    Setting(
        "focal_gamma", "MEDOS_TRAINER_FOCAL_GAMMA", 0.0, _focal_gamma,
        "0.0 with alpha None is an EXACT identity with plain BCE, not an approximation of it, so a "
        "run that sets nothing trains what it trained before this setting existed and the pair "
        "already on disk stays comparable.",
    ),
    Setting(
        "focal_alpha", "MEDOS_TRAINER_FOCAL_ALPHA", None, _focal_alpha,
        "No default weighting, because any value makes gamma 0 stop being an exact identity with "
        "plain BCE.",
    ),
    Setting(
        "configuration", "MEDOS_TRAINER_CONFIGURATION", "3d_fullres", _name,
        "Which of the plan's configurations this run trains. It travels in the frozen plan and the "
        "fit reads it back from there; before it did, a plan derived and preprocessed for one "
        "configuration was fit under whatever a module constant named, and nothing raised.",
    ),
    Setting(
        "lr_scheduler", "MEDOS_TRAINER_LR_SCHEDULER", "poly", _scheduler_name,
        "poly is nnU-Net's own and the schedule every run on disk trained under, so it is the "
        "default and asking for nothing changes nothing. plateau is the measured answer to this "
        "cohort destabilising under poly from 1e-2 on three architectures (SegResNetDS NaN, "
        "PlainConvUNet collapse near epoch 280, ResEncUNet-M near epoch 80): ReduceLROnPlateau "
        "halves the rate when val_loss stalls, so the schedule cools on evidence instead of on a "
        "fixed decay.",
    ),
)


@dataclass(frozen=True)
class TrainingConfiguration:
    """One run's training settings: validated at construction, recordable as a document.

    FROZEN, so a setting cannot be changed after the record of it was written. The six attribute
    assignments this replaces could each happen at any point before `initialize()`, and one of them
    -- the learning rate -- is captured BY VALUE by nnU-Net's scheduler inside that call, so setting
    it afterwards left the optimiser on one rate while the log named another.
    """

    max_epochs: int = 1000
    iterations_per_epoch: int = 250
    validation_iterations_per_epoch: int = 50
    initial_lr: float | None = None
    focal_gamma: float = 0.0
    focal_alpha: float | None = None
    configuration: str = "3d_fullres"
    lr_scheduler: str = "poly"
    #: Where each setting came from -- `request.budget.<name>`, an environment variable, or
    #: `default`. Recorded because "who asked for this" is the first question at a review and the
    #: value alone cannot answer it.
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # VALIDATED EVEN WHEN CONSTRUCTED DIRECTLY, not only through `from_request`. A gate that
        # builds one by hand, or a future caller that does, must meet the same refusals -- otherwise
        # the checks guard one entrance of two.
        for setting in SETTINGS:
            setting.check(getattr(self, setting.name), f"TrainingConfiguration.{setting.name}")
        if self.focal_alpha is not None and self.focal_gamma == 0.0:
            raise TrainingConfigurationError(
                f"focal_alpha={self.focal_alpha} with focal_gamma=0.0 weights the classes while "
                "the pointwise term is still plain BCE. That is a class weighting wearing a focal "
                "loss's name: ask for a gamma, or ask for neither"
            )

    @property
    def save_every(self) -> int:
        """How often a checkpoint is written, which follows from the schedule and is not asked for.

        One at the end. It was `max(1, int(budget["max_epochs"]))` inline in the backend; derived
        here so that a reader of this object sees the whole of what a fit does, and so that a future
        change to it is one edit rather than a search.
        """
        return max(1, int(self.max_epochs))

    @classmethod
    def from_request(
        cls, request: Any, *, environ: Mapping[str, str] | None = None
    ) -> "TrainingConfiguration":
        """The request's `budget` block, then the environment, then the default -- for every setting.

        ONE LADDER FOR ALL OF THEM. Before this, three functions each wrote their own and two had
        already drifted apart on what to do with a value they could not honour: one clamped, one
        refused. The table in `SETTINGS` makes a new setting unable to arrive with a private ladder.
        """
        variables = os.environ if environ is None else environ
        declared = dict(getattr(request, "budget", None) or {})
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for setting in SETTINGS:
            if setting.name in declared and declared[setting.name] is not None:
                where = f"request.budget.{setting.name}"
                raw: Any = declared[setting.name]
            elif variables.get(setting.variable, "").strip():
                where = setting.variable
                raw = variables[setting.variable].strip()
            else:
                where = "default"
                raw = setting.default
            values[setting.name] = setting.check(raw, where)
            sources[setting.name] = where
        return cls(**values, sources=sources)

    def as_document(self) -> dict[str, Any]:
        """The whole configuration, for `result.json`.

        EVERY SETTING, INCLUDING THE ONES AT THEIR DEFAULT, and `initial_lr: null` where the backend
        used its own. A record that omitted the defaults would make two runs trained under different
        schedules carry identical documents wherever one of them asked for nothing -- and "the
        default at the time" is not a value anybody can look up later.
        """
        document: dict[str, Any] = {
            setting.name: getattr(self, setting.name) for setting in SETTINGS
        }
        document["save_every"] = self.save_every
        document["sources"] = dict(self.sources)
        return document

    def describe(self) -> str:
        """One line for the training log, so the log says what the run trained under.

        The log is what a person reads when two runs disagree, and `result.json` is written only
        after the fit succeeds -- so a run that dies at epoch 3 has this line and nothing else.
        """
        parts = [
            f"epochs={self.max_epochs}x{self.iterations_per_epoch}",
            f"val={self.validation_iterations_per_epoch}",
            f"configuration={self.configuration}",
            f"lr={'backend default' if self.initial_lr is None else self.initial_lr:g}"
            if self.initial_lr is not None else "lr=backend default",
            f"focal_gamma={self.focal_gamma:g}",
            f"focal_alpha={'none' if self.focal_alpha is None else self.focal_alpha}",
            f"scheduler={self.lr_scheduler}",
        ]
        asked = [name for name, where in self.sources.items() if where != "default"]
        if asked:
            parts.append("asked for: " + ",".join(sorted(asked)))
        return "training: " + "  ".join(parts)
