# SPDX-License-Identifier: Apache-2.0
"""Which network architecture a run gets, and how that was decided.

WHAT THIS IS FOR
----------------
nnU-Net derives one architecture from one heuristic and never compares. It also ships three
better presets -- `ResEncM`, `ResEncL`, `ResEncXL`, its own published recommendation over
the default -- and makes you pass `-pl nnUNetPlannerResEncL` by hand to get one. A pipeline
that is meant to configure itself should not have a hand-typed flag as its only route to the
configuration upstream recommends.

So: the presets are a declared catalogue, the choice is made from a MEASURED quantity, and
both the choice and the alternatives are recorded in `plan.json`.

WHY PRESETS AND NOT A PATCHED `architecture` FIELD
---------------------------------------------------
nnU-Net 2.6 carries the architecture in the plan --
`configurations.3d_fullres.architecture.network_class_name` is a dotted path resolved at fit
time -- so writing a different class name there LOOKS like the cheapest possible way to swap
architectures. It is also wrong twice over.

`MOS-REL-032` says adopt the adopted thing whole. A planner does not merely name a class: it
derives the patch size, the batch size, the number of stages, the features per stage, the
strides and the kernel sizes TOGETHER, against a VRAM budget, from the fingerprint. A patch
that changes the class and keeps the rest produces a configuration nobody upstream
validated and nobody here derived. And `MOS-TRAIN-135` freezes what the planner produced;
a patch applied after the freeze is a re-derivation, and one applied before it is a second
planner we would then own.

Each entry below therefore names an UPSTREAM PLANNER. What we add is the selection and its
record -- not a network.

WHY THE SELECTION IS A RULE OVER A MEASUREMENT AND NOT A SEARCH
----------------------------------------------------------------
`MOS-TRAIN-213`: any procedure that produces more than one trained artifact from one cohort
and keeps a subset by a score computed on data is a `ConfigurationSearch`, and it names
"an nnU-Net comparison across 2d/3d_fullres/3d_lowres/cascade" as an example. Fingerprint
derivation is excluded, because it reads STATISTICS rather than scores.

`select()` trains nothing and scores nothing. It reads one number -- the VRAM budget
`backend.vram_observation()` already observes -- and applies a stated rule. That keeps it on
the derivation side of that line. Comparing two presets by how well they SCORE is the other
thing, it is a search, and it owes the whole of `MOS-TRAIN-218`: the space, its digest, the
selection metric and partition, the runner-up, the margin and the budget. This module is
deliberately not that, and `medos/medos/training/search.py` already is.

THE RULE, AND WHY IT IS THIS ONE
---------------------------------
Each preset carries `reference_gb`: the VRAM target at which upstream CALIBRATED its
reference activation volume. Overriding the target is supported and intended -- the planner
sets `UNet_vram_target_GB` from the argument while `UNet_reference_val_corresp_GB` stays the
calibration point, and it scales the patch from the ratio -- but it does warn, and the
further the two diverge the further the configuration is being extrapolated from anything
measured.

So: the preset with the LARGEST `reference_gb` that does not exceed the observed budget.
That minimises the extrapolation and keeps its direction upward, which is the side where
running out of memory is the failure mode rather than silently planning a patch too small
for the card that was paid for.

AND ONE THING THIS MODULE SAID IT WOULD NOT DO
-----------------------------------------------
Everything above is about upstream PLANNERS, and the argument against patching the architecture
field stands. `OVERLAYS` at the foot of this file does exactly that patch, for the architectures
MONAI ships and no nnU-Net planner can configure -- and it is narrowed rather than excused:
`select()` never returns one, so no measurement ever chooses one; the caveat is mandatory and is
written into the plan; and the stage count is checked before the fit is queued. See `Overlay`.

Spec: MOS-TRAIN-135, MOS-TRAIN-211, MOS-TRAIN-213, MOS-TRAIN-218, MOS-REL-032.
Pure: no torch, no nnU-Net import, no I/O -- so the catalogue and the rule can be read by a
test for the price of a text file, and `port.py` stays importable without a backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "AMBIGUOUS_PLANNERS",
    "OVERLAYS",
    "Overlay",
    "apply_overlay",
    "overlay",
    "EXCLUDED_PLANNERS",
    "PRESETS",
    "Preset",
    "named",
    "select",
]


@dataclass(frozen=True)
class Preset:
    """One upstream planner, named so a run can record which one produced its plan."""

    #: Our stable name. It appears in `plan.json` and must never be renamed, because a
    #: recorded choice whose name changed is a recorded choice nobody can resolve.
    name: str
    #: `plan_experiments(experiment_planner_class_name=...)`. Resolved upstream BY NAME via a
    #: package walk, which is why `AMBIGUOUS_PLANNERS` exists.
    planner: str
    #: The `plans_identifier` the planner writes. Already recorded in `plan.json`, so this is
    #: how the record can be checked against the catalogue rather than trusted.
    plans_identifier: str
    #: The VRAM target upstream calibrated this preset's reference activation volume at.
    reference_gb: float
    #: The class the planner instantiates. Asserted against the planner, not assumed.
    network: str
    why: str


#: Ordered by `reference_gb`. The default planner is first and is the floor: a budget below
#: every reference still gets a plan, and it gets the one upstream treats as the baseline.
PRESETS: Final[tuple[Preset, ...]] = (
    Preset(
        name="plain",
        planner="ExperimentPlanner",
        plans_identifier="nnUNetPlans",
        reference_gb=8.0,
        network="PlainConvUNet",
        why="the baseline nnU-Net configures by default. Chosen only below 8 GB, because at "
            "or above it `resenc_m` calibrates at the same budget and upstream "
            "recommends the residual encoder there. It is the floor, so no budget is "
            "left without a plan",
    ),
    Preset(
        name="resenc_m",
        planner="nnUNetPlannerResEncM",
        plans_identifier="nnUNetResEncUNetMPlans",
        # 8.0 AND NOT 9.0. The class docstring says "Target is ~9-11 GB VRAM max"; its
        # signature says `gpu_memory_target_in_gb: float = 8`. The signature is what the
        # planner calibrates against and the docstring is prose, so the number here is the
        # signature's -- established by `test_architecture_catalogue.py`, which is how the
        # 9.0 taken from the prose was caught before it could shift a preset choice.
        reference_gb=8.0,
        network="ResidualEncoderUNet",
        why="a residual encoder at roughly the same budget as the baseline; upstream's own "
            "recommendation over `plain` where the card allows it",
    ),
    Preset(
        name="resenc_l",
        planner="nnUNetPlannerResEncL",
        plans_identifier="nnUNetResEncUNetLPlans",
        reference_gb=24.0,
        network="ResidualEncoderUNet",
        why="the residual encoder calibrated for a 24 GB card",
    ),
    Preset(
        name="resenc_xl",
        planner="nnUNetPlannerResEncXL",
        plans_identifier="nnUNetResEncUNetXLPlans",
        reference_gb=40.0,
        network="ResidualEncoderUNet",
        why="the residual encoder calibrated for a 40 GB card",
    ),
)

#: NAMES UPSTREAM DEFINES TWICE, WHICH THIS CATALOGUE MUST NEVER USE.
#:
#: `plan_experiments` resolves a planner by NAME through `recursive_find_python_class`, a
#: walk over the planner package. `ResEncUNetPlanner` exists in BOTH
#: `experiment_planners.resencUNet_planner` and
#: `experiment_planners.residual_unets.residual_encoder_unet_planners`, so which class runs
#: depends on the order a directory walk happens to return -- and the plan it writes would
#: carry no sign of which one it was. A configuration chosen by filesystem ordering is not a
#: configuration anybody chose.
AMBIGUOUS_PLANNERS: Final[frozenset[str]] = frozenset({"ResEncUNetPlanner"})

#: Planners upstream ships that this catalogue does NOT offer, each with its reason. Listed
#: rather than omitted: `test_architecture_catalogue.py` asserts that every planner upstream
#: defines is either in `PRESETS`, here, or in `AMBIGUOUS_PLANNERS`, so an upstream addition
#: is REPORTED instead of silently ignored.
EXCLUDED_PLANNERS: Final[dict[str, str]] = {
    "nnUNetPlanner_torchres":
        "changes the RESAMPLING, not the architecture. Resampling is transcribed into the "
        "PreprocessingSpec by MOS-TRAIN-223 and pinned by the golden fixture, so it is a "
        "different axis and a different decision -- one that moves what serving must do.",
    "nnUNetPlannerResEncL_torchres":
        "as above: architecture and resampling changed together, so a comparison could not "
        "say which mattered.",
    "nnUNetPlannerResEncL_torchres_sepz":
        "as above, with separate-z resampling on top.",
    "nnUNetPlannerResEncL_noResampling":
        "its own docstring says it generates 3d_lowres as well and not to trust it. An "
        "upstream author's warning about their own planner is the cheapest possible reason "
        "to leave it out.",
}


def named(name: str) -> Preset:
    """The preset called `name`, or a refusal listing what exists.

    Used where a preset arrives as data -- a recorded plan being checked, a search space
    naming its axis -- so an unknown name must not resolve to the default. Falling back to
    `plain` would train the baseline while the record said otherwise, and the record is the
    only place anybody would look.
    """
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise LookupError(
        f"no architecture preset named {name!r}; this image offers "
        f"{[p.name for p in PRESETS]}. A preset that resolved to the default would train "
        "one architecture while the run's own record named another"
    )


def select(budget_gb: float) -> tuple[Preset, dict[str, Any]]:
    """The analysis: the preset for an observed VRAM budget, and the record of the choice.

    Returns the preset and a document for `plan.json`. The document carries the
    ALTERNATIVES and the RULE, not just the answer, because "why this architecture" is the
    first question at a review and a bare name cannot answer it.

    No scores, no trained artifacts, one measured number -- see the module docstring on
    `MOS-TRAIN-213`.
    """
    if not budget_gb > 0:
        raise ValueError(
            f"the VRAM budget is {budget_gb!r}. `backend.vram_observation()` floors it at "
            "4 GB precisely so that a card being busy cannot silently select the smallest "
            "architecture; a non-positive budget means the observation itself failed"
        )
    fitting = [p for p in PRESETS if p.reference_gb <= budget_gb]
    # THE TIE-BREAK IS EXPLICIT, because there IS a tie: `plain` and `resenc_m` both
    # calibrate at 8 GB. `max` returns the FIRST maximal element, so the winner would have
    # been decided by the order of a tuple literal with nothing saying so. Upstream's own
    # recommendation is the residual encoder over the plain baseline at the same budget, and
    # `PRESETS` is ordered so the later entry is the one to prefer -- stated here, and
    # asserted by `test_the_tie_at_eight_gigabytes_goes_to_the_residual_encoder`.
    chosen = (
        max(reversed(fitting), key=lambda p: p.reference_gb) if fitting else PRESETS[0]
    )
    return chosen, {
        "selected": chosen.name,
        "planner": chosen.planner,
        "network": chosen.network,
        "plans_identifier": chosen.plans_identifier,
        "observed_budget_gb": float(budget_gb),
        "reference_gb": chosen.reference_gb,
        "rule": (
            "MOS-TRAIN-213 derivation, not a search: the preset with the largest "
            "reference_gb not exceeding the observed VRAM budget, so the extrapolation "
            "from upstream's calibration point is the smallest available and points "
            "upward; among equal references the residual encoder, which is upstream's own "
            "recommendation over the plain baseline at the same budget"
        ),
        "considered": [
            {"name": p.name, "reference_gb": p.reference_gb, "fits": p.reference_gb <= budget_gb}
            for p in PRESETS
        ],
        #: TRUE whenever the budget is not the preset's calibration point, which is almost
        #: always. Upstream emits a warning in that case and the warning is correct: the
        #: patch size is being scaled away from anything upstream measured. Recorded rather
        #: than suppressed -- a warning nobody can see is a fact that was deleted.
        "extrapolated_from_reference": float(budget_gb) != chosen.reference_gb,
    }


# =============================================================================================
# OVERLAYS: the thing this module's own docstring says is wrong, done deliberately and narrowly
# =============================================================================================


@dataclass(frozen=True)
class Overlay:
    """A foreign architecture trained on a plan an UPSTREAM PLANNER derived for another one.

    THIS IS THE PATCH THE MODULE DOCSTRING ARGUES AGAINST, and nothing below weakens that
    argument. A planner derives the patch, the batch, the stages, the features, the strides and
    the kernels together against a VRAM budget; an overlay keeps all of that and replaces only
    `network_class_name`. So the configuration is one nobody upstream validated for this network
    and nobody here derived -- `MOS-REL-032`'s objection, restated rather than answered.

    THREE THINGS MAKE IT AN HONEST OPTION RATHER THAN A SHORTCUT:

      * `select()` NEVER RETURNS ONE. An overlay has no calibrated `reference_gb`, so there is no
        measured quantity a rule could read, and inventing one would put a number the planner
        never measured into the same rule as four it did. An overlay is reachable only by name --
        "по заказу", not "по анализу";
      * `caveat` is mandatory and is written into the plan, so the run's own record carries why
        its patch is not one derived for its network. A caveat nobody persisted is a caveat
        nobody read;
      * `requires_stages` is checked BEFORE the fit is queued. `MedOSSwinUNETR` can serve exactly
        six, and finding that out inside a container three hours into a queue is the difference
        between a refusal and an outage.
    """

    #: Our stable name, written into the plan. Never renamed, for the reason `Preset.name` gives.
    name: str
    #: The preset whose planner derives the plan this architecture is then dropped into.
    plan_from: str
    #: The dotted path `get_network_from_plans` resolves through `pydoc.locate`.
    network_class_name: str
    #: Architecture kwargs this network needs and the conv plan does not carry. Written INTO
    #: `arch_kwargs`, so the plan records the geometry that actually trained rather than leaving
    #: it to a wrapper default that could move under an upgrade.
    arch_kwargs: Mapping[str, Any]
    #: The stage count the network can serve, or `None` for any. Checked against the plan.
    requires_stages: int | None
    caveat: str
    why: str
    #: Arch kwargs whose value is READ OUT OF THE PLAN rather than spelled here: the kwarg name
    #: mapped to the configuration key it comes from, e.g. `{"img_size": "patch_size"}`.
    #:
    #: WHY THIS EXISTS RATHER THAN A LITERAL IN THE ROW ABOVE. `UNETR` freezes its ViT token grid
    #: from `img_size` at construction, and nnU-Net never passes a patch size to a network -- so
    #: the only route is `arch_kwargs`, and the only honest value is the patch the loader will
    #: actually crop. A number written into the catalogue would be a second copy of the plan's
    #: patch, correct until the planner chose a different one, and then the network would be built
    #: for a patch nobody was feeding it.
    plan_derived_kwargs: Mapping[str, str] = field(default_factory=dict)
    #: Keys in `arch_kwargs` that carry ONE ENTRY PER STAGE. Their length is checked against the
    #: plan's strides, because a catalogue row written for six stages dropped into an eight-stage
    #: plan is a geometry nobody declared -- and the wrapper would raise for it inside a container,
    #: hours into a queue. Extending the list here instead would invent the missing stages.
    per_stage_kwargs: tuple[str, ...] = ()


OVERLAYS: tuple[Overlay, ...] = (
    Overlay(
        name="segresnet_ds",
        plan_from="plain",
        network_class_name="medos_trainer.nets.MedOSSegResNetDS",
        arch_kwargs={"blocks_down": [1, 2, 2, 4, 4, 4]},
        # SIX, BECAUSE THE BLOCK COUNTS ABOVE ARE SIX. The wrapper itself serves any depth -- it
        # extends MONAI's own (1, 2, 2, 4) recommendation by repeating the deepest count -- but a
        # catalogue that DECLARES a geometry declares one depth, and leaving this `None` let the
        # eight-stage 2d plan through to fail at construction instead of here.
        requires_stages=6,
        per_stage_kwargs=("blocks_down",),
        caveat="the patch was derived for a PlainConvUNet. Measured at this cohort's patch, "
               "SegResNetDS's activation footprint is 9.72e8 elements against the conv U-Net's "
               "1.44e9, so the patch is CONSERVATIVE by about a third rather than too large -- "
               "but it is still not a patch anybody derived for this network. Its 356M parameters "
               "cost roughly 4 GB of optimiser state that the plan's budget does not account for.",
        why="a residual encoder-decoder with native deep supervision, adopted whole from MONAI "
            "Core, whose register row is already ADOPT. The wrapper serves any depth; this row "
            "declares six, which is what both of this cohort's 3D configurations carry",
    ),
    Overlay(
        name="swin_unetr",
        plan_from="plain",
        network_class_name="medos_trainer.nets.MedOSSwinUNETR",
        arch_kwargs={"feature_size": 48, "depths": [2, 2, 2, 2],
                     "num_heads": [3, 6, 12, 24], "use_checkpoint": True,
                     "norm_name": "instance"},
        requires_stages=6,
        caveat="the patch was derived for a PlainConvUNet and this network's footprint at it is "
               "5.4 times larger -- 3.78e9 conv plus 4.02e9 attention elements against 1.44e9 -- "
               "which at the plan's batch size of 2 is about 31 GB in half precision. "
               "`use_checkpoint` is therefore ON and is not optional. The attention term is the "
               "reason this network refuses to be sized by nnU-Net's planner at all: the "
               "planner's reference value was measured on a conv U-Net, so a conv-shaped estimate "
               "underestimates by about half, in the direction that sizes the patch too large.",
        why="windowed attention in the encoder, which is the one architecture family in MONAI's "
            "zoo that is not another convolutional U-Net, so it is the comparison that could "
            "answer something the residual presets cannot. Serves six stages only",
    ),
)


#: UNETR, appended separately from the tuple above only because its row needs a comment of its own
#: about where its patch comes from. It is part of `OVERLAYS`; see the assignment below.
_UNETR = Overlay(
    name="unetr",
    plan_from="plain",
    network_class_name="medos_trainer.nets.MedOSUNETR",
    arch_kwargs={"feature_size": 16, "hidden_size": 768, "mlp_dim": 3072,
                 "num_heads": 12, "norm_name": "instance", "proj_type": "conv",
                 "res_block": True, "conv_block": True},
    requires_stages=6,
    caveat="the patch was derived for a PlainConvUNet, and this network is BOUND to whatever patch "
           "it is built with: `proj_feat` freezes the ViT token grid at construction, so the "
           "network cannot be resized and `img_size` is copied out of the plan rather than "
           "declared here. Every axis of that patch must be divisible by 16, which is hard-coded "
           "upstream and NOT checked by MONAI -- an indivisible axis would be floored and its "
           "remainder trained on nothing. Measured at this cohort's patch its footprint is 1.20e9 "
           "conv plus 3.54e8 attention elements against the conv U-Net's 1.44e9, so the patch is "
           "roughly the right size for it -- which is a coincidence and not a derivation.",
    why="a plain ViT encoder with a convolutional decoder: global attention, but only on the grid "
        "at 1/16, which is why it costs a fifth of SwinUNETR. It is the cheapest way to ask "
        "whether attention helps this cohort at all. Serves six stages only",
    plan_derived_kwargs={"img_size": "patch_size"},
)

OVERLAYS = OVERLAYS + (_UNETR,)


def overlay(name: str) -> Overlay:
    """The overlay by name, or a refusal naming what this image carries.

    SEPARATE FROM `named()` AND NOT A FALLBACK IN IT. A caller asking for a planner and getting
    an architecture overlay -- or the reverse -- would get a plan derived by something other than
    what it asked for, and the plan records only the result.
    """
    for candidate in OVERLAYS:
        if candidate.name == name:
            return candidate
    # LookupError, MATCHING `named()`. The two catalogues refuse in the same way on purpose: a
    # caller that handles one kind of "no such choice" handles both, and a ValueError here would
    # be caught by the same `except ValueError` that handles a plan this overlay cannot serve --
    # two different failures arriving as one.
    raise LookupError(
        f"no architecture overlay named {name!r}; this image offers "
        f"{[o.name for o in OVERLAYS]}. Planner presets are a different catalogue and are "
        f"resolved by `named()`: {[p.name for p in PRESETS]}"
    )


def apply_overlay(
    plans: Mapping[str, Any], name: str, *, configuration: str
) -> dict[str, Any]:
    """A COPY of `plans` whose one configuration trains this architecture, with the record.

    THE CHECKS HAPPEN HERE, BEFORE THE FIT IS QUEUED, and that is the function's main job. Each
    of them would otherwise surface inside a container: a stage count the network cannot serve
    raises at construction, a patch the network cannot halve raises at the first forward, and both
    arrive hours after the plan was written with nothing in the plan to point at.

    WHAT IT WRITES INTO THE PLAN, beside the class name: the overlay's own `arch_kwargs`, its
    name, the preset the plan came from, and the caveat. `MOS-TRAIN-135` freezes what the planner
    produced -- so this is a recorded, named departure from the frozen plan rather than an edit
    that leaves the plan claiming to be the planner's own.

    The copy is deep over what it touches and shares the rest. A mutation in place would leave the
    caller holding a plan whose architecture had changed under it, which is how the same plans
    object comes to describe two different fits.
    """
    chosen = overlay(name)
    configurations = plans.get("configurations") or {}
    if configuration not in configurations:
        raise ValueError(
            f"the plan has no configuration {configuration!r}; it carries "
            f"{sorted(configurations)}"
        )
    body = configurations[configuration]
    architecture = body.get("architecture")
    if not architecture:
        raise ValueError(
            f"configuration {configuration!r} carries no `architecture` block, so there is "
            "nothing to overlay and writing one would invent a geometry"
        )

    plan_kwargs = dict(architecture.get("arch_kwargs") or {})
    strides = plan_kwargs.get("strides")
    if not strides:
        raise ValueError(
            f"configuration {configuration!r} declares no strides, which fix both the stage "
            "count and the number of deep-supervision heads"
        )
    if chosen.requires_stages is not None and len(strides) != chosen.requires_stages:
        raise ValueError(
            f"{chosen.name} serves exactly {chosen.requires_stages} stages and "
            f"{configuration!r} declares {len(strides)}. DeepSupervisionWrapper zips heads "
            "against targets and zip truncates, so this would not raise at training time -- it "
            "would pair every head with another scale's target for the whole fit. Plan this "
            f"configuration at {chosen.requires_stages} stages, or choose another architecture"
        )
    mismatched = {
        key: len(chosen.arch_kwargs[key]) for key in chosen.per_stage_kwargs
        if len(chosen.arch_kwargs[key]) != len(strides)
    }
    if mismatched:
        raise ValueError(
            f"{chosen.name} declares {mismatched} entries for per-stage keys and "
            f"{configuration!r} has {len(strides)} stages. Padding the list here would invent the "
            "missing stages' geometry, and truncating it would drop a declared one"
        )
    # READ OUT OF THE PLAN, and a missing source is a refusal rather than a default. `img_size`
    # absent would let UNETR's own constructor refuse -- which it does, clearly -- but only inside
    # the container; the plan is where the patch lives and this is where it can be copied.
    derived: dict[str, Any] = {}
    for keyword, source in chosen.plan_derived_kwargs.items():
        if source not in body:
            raise ValueError(
                f"{chosen.name} reads its {keyword!r} from the configuration's {source!r}, and "
                f"{configuration!r} does not carry it. It cannot be defaulted: the value the "
                "network is built with has to be the one the loader will feed it"
            )
        derived[keyword] = body[source]

    collision = sorted((set(chosen.arch_kwargs) | set(derived)) & set(plan_kwargs))
    if collision:
        raise ValueError(
            f"{collision} are in both the plan and {chosen.name}'s own kwargs. Silently "
            "preferring one would make the plan record a value the fit did not use; if the "
            "overlay is meant to override a planned key, say so by removing it from the plan"
        )

    overlapping = sorted(set(chosen.arch_kwargs) & set(derived))
    if overlapping:
        raise ValueError(
            f"{chosen.name} both declares and derives {overlapping}; one of the two would win "
            "silently and the plan would record a value the fit did not use"
        )
    # THE PLAN IS EDITED THROUGH `plan.Plan`, NOT BY COPYING DICTIONARIES HERE.
    #
    # Everything below the catalogue's own checks -- the deep copy, the arch_kwargs merge, the
    # collision refusal, the `_kw_requires_import` completeness check, the placement of the record
    # BESIDE the architecture block rather than inside it, and the refusal of a bare dotted path --
    # lives in `plan.py` and is gated there. This function was doing all of it by hand, which is
    # exactly the seam `docs/adr/BUILD_VS_ADOPT.md`'s training-plan row names: a plan modified by
    # string indexing, in one of seven places that did it.
    #
    # What stays here is what belongs to the CATALOGUE and not to the plan: which overlays exist,
    # how many stages each serves, which of its kwargs come from the plan, and the caveat.
    from medos_trainer.plan import Plan, PlanError

    try:
        edited = Plan.from_document(plans).with_architecture(
            configuration,
            network_class_name=chosen.network_class_name,
            arch_kwargs={**dict(chosen.arch_kwargs), **derived},
            record={
                "overlay": chosen.name,
                "planned_by": chosen.plan_from,
                "planner_network": named(chosen.plan_from).network,
                "caveat": chosen.caveat,
                "why": chosen.why,
            },
        )
    except PlanError as exc:
        # RE-RAISED AS A `ValueError` WITH THE OVERLAY NAMED. `PlanError` is a ValueError subclass,
        # so a caller catching ValueError already works; what it would not have is which overlay
        # asked for the edit, and this function's callers are choosing between overlays.
        raise ValueError(f"{chosen.name}: {exc}") from exc
    return edited.to_document()
