# SPDX-License-Identifier: Apache-2.0
"""The training plan as a first-party object: typed access, typed modification, lossless carriage.

WHY THIS EXISTS. `docs/adr/BUILD_VS_ADOPT.md` carries the row, and its disqualifying property is
measured rather than asserted: nnU-Net's `PlansManager` is a READ-ONLY accessor with no write path
at all. It exposes twelve read-only properties over `self.plans`, which is a public mutable plain
`dict` (`plans_handler.py:226`), and offers no constructor from typed fields, no validation and no
modification API. So everything this platform must CHANGE -- the architecture, the configuration,
a recorded departure -- could until now only be changed by indexing that dict, which this tree did
in seven files. Two consequences were already realised and neither was visible in any test:

  * the configuration name was a module constant in two files while `plan.json` recorded a
    `configuration` member that nothing read back, so a plan recorded for one configuration could
    be fit under another;
  * swapping an architecture was done by bind-mounting a patched copy of the document over the
    read-only original.

WHY THE DOCUMENT IS HELD VERBATIM RATHER THAN TRANSCRIBED FIELD BY FIELD, which is the one design
decision in this file worth arguing. `nnUNetTrainer.py:922` and `predict_from_raw_data.py:244,713`
write the WHOLE document into the results folder for inference to read back. A dataclass enumerating
today's ten global and sixteen per-configuration keys would therefore silently drop any key a future
nnU-Net adds -- and the run whose training looked perfect would fail at inference. A field list is
the fragile design here, not the safe one. So:

  * the document is the source of truth and is never mutated. The round trip is lossless BY
    CONSTRUCTION, not by a transcription table somebody has to maintain;
  * typing lives in the ACCESSORS and the MODIFIERS, which is where it prevents mistakes: an
    accessor refuses a missing key with a message naming the plan, and a modifier returns a new
    `Plan` and records what it changed;
  * the key names live HERE, once, instead of in seven files.

WHAT THIS DOES NOT DO. It does not derive a plan -- nnU-Net's planner is ADOPTED and re-deriving its
heuristics would mean re-deriving a calibration fitted across many datasets. It does not validate
the plan against the planner's rules; `PlansManager` remains the reader on nnU-Net's own side of the
boundary, and this module's job is to hand it a document it can read.

TWO FAMILIES OF RUNTIME-RESOLVED STRING, AND CONFUSING THEM IS A REAL FAILURE. Some values are
DOTTED PATHS resolved by `pydoc.locate`; others are BARE CLASS OR FUNCTION NAMES resolved by a
module scan (`recursive_find_python_class`, `recursive_find_reader_writer_by_name`). A dotted path
where a bare name is expected raises at run time, and a bare name where a dotted path is expected
resolves to `None` and is then "recovered" by a search that may find something else entirely
(`get_network_from_plans` warns and searches `dynamic_network_architectures`). The two sets are
declared below so a modifier cannot move a value from one to the other by accident.

Pure: no torch, no nnU-Net import, no I/O. So the whole module is testable for the price of a JSON
file, and `port.py` stays importable without a backend.

Spec: MOS-REL-027, MOS-REL-032, MOS-TRAIN-135, MOS-TRAIN-223, MOS-TRAIN-225.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

__all__ = [
    "CASCADE_STUB_KEYS",
    "CONFIGURATION_KEYS",
    "Configuration",
    "DOTTED_PATH_VALUES",
    "GLOBAL_KEYS",
    "MODULE_SCAN_VALUES",
    "Plan",
    "PlanError",
]


class PlanError(ValueError):
    """A plan that cannot be read or cannot be modified as asked.

    A distinct type, so a caller can tell "this plan is not what I thought" from every other
    `ValueError` a pipeline raises. It is not a `ContractViolation`: this module is pure and must
    stay importable without the run-directory contract, and the caller that has one can wrap.
    """


#: The global keys an nnU-Net 2.6.4 plan carries, read off both of this cohort's real plans rather
#: than from the planner's source -- the planner writes these and a merge can add more.
#: `configurations` is named separately because it is the only one with structure.
GLOBAL_KEYS: Final[tuple[str, ...]] = (
    "dataset_name",
    "plans_name",
    "original_median_spacing_after_transp",
    "original_median_shape_after_transp",
    "image_reader_writer",
    "transpose_forward",
    "transpose_backward",
    "experiment_planner_used",
    "label_manager",
    "foreground_intensity_properties_per_channel",
)

#: The keys a FULL configuration carries. A cascade stub carries `CASCADE_STUB_KEYS` instead, and
#: `next_stage` / `previous_stage` appear only on the configurations that have one.
CONFIGURATION_KEYS: Final[tuple[str, ...]] = (
    "data_identifier",
    "preprocessor_name",
    "batch_size",
    "patch_size",
    "median_image_size_in_voxels",
    "spacing",
    "normalization_schemes",
    "use_mask_for_norm",
    "resampling_fn_data",
    "resampling_fn_data_kwargs",
    "resampling_fn_seg",
    "resampling_fn_seg_kwargs",
    "resampling_fn_probabilities",
    "resampling_fn_probabilities_kwargs",
    "architecture",
    "batch_dice",
)

#: A cascade configuration is a STUB and carries only these two: it inherits everything else from
#: the configuration it names. `3d_cascade_fullres` in both of this cohort's plans has exactly
#: these and nothing more, which is why `Configuration` must not require the sixteen above.
CASCADE_STUB_KEYS: Final[tuple[str, ...]] = ("inherits_from", "previous_stage")

#: Values resolved by `pydoc.locate`, so they MUST be dotted paths. `network_class_name` plus
#: whatever `_kw_requires_import` names inside `arch_kwargs` -- which on this cohort's plans is
#: `conv_op`, `norm_op`, `dropout_op`, `nonlin`. A BARE name here resolves to `None`, and
#: `get_network_from_plans` then warns and searches `dynamic_network_architectures` for the last
#: dotted segment, which can find a different class than the one intended.
DOTTED_PATH_VALUES: Final[tuple[str, ...]] = ("network_class_name",)

#: Values resolved by a MODULE SCAN over a package, so they MUST be bare class or function names.
#: A dotted path here raises: `recursive_find_reader_writer_by_name` does `getattr(module, name)`
#: over `nnunetv2/imageio` and `recursive_find_python_class` the same over its own package.
MODULE_SCAN_VALUES: Final[tuple[str, ...]] = (
    "image_reader_writer",
    "label_manager",
    "experiment_planner_used",
    "preprocessor_name",
    "resampling_fn_data",
    "resampling_fn_seg",
    "resampling_fn_probabilities",
)


@dataclass(frozen=True)
class Configuration:
    """One configuration of a plan: a typed view, not a copy.

    `body` is the configuration's own sub-mapping, held verbatim for the reason the module
    docstring gives. A cascade STUB is a legitimate configuration with two keys, so nothing here
    requires the full sixteen; an accessor refuses when the key it needs is absent, which is the
    point at which the absence actually matters.
    """

    name: str
    body: Mapping[str, Any]
    #: The plan this came from, for error messages. A refusal that cannot name the plan sends the
    #: reader looking through every plan on the machine.
    plans_name: str = ""

    def _require(self, key: str) -> Any:
        if key not in self.body:
            if set(self.body) <= set(CASCADE_STUB_KEYS):
                raise PlanError(
                    f"configuration {self.name!r} of plan {self.plans_name!r} is a cascade STUB "
                    f"carrying only {sorted(self.body)}; it inherits {key!r} from "
                    f"{self.body.get('inherits_from')!r} and does not carry it itself"
                )
            raise PlanError(
                f"configuration {self.name!r} of plan {self.plans_name!r} has no {key!r}; it "
                f"carries {sorted(self.body)}"
            )
        return self.body[key]

    # -- geometry ------------------------------------------------------------------------
    @property
    def patch_size(self) -> list[int]:
        return [int(n) for n in self._require("patch_size")]

    @property
    def batch_size(self) -> int:
        return int(self._require("batch_size"))

    @property
    def spacing(self) -> list[float]:
        return [float(s) for s in self._require("spacing")]

    @property
    def spatial_rank(self) -> int:
        """2 or 3, from the patch rather than from the configuration's NAME.

        The name is a convention -- `save_plans` merges arbitrary user-named configurations -- so a
        rank derived from it would be a guess. The patch is the thing the loader crops.
        """
        return len(self.patch_size)

    # -- the architecture ----------------------------------------------------------------
    @property
    def architecture(self) -> Mapping[str, Any]:
        return self._require("architecture")

    @property
    def network_class_name(self) -> str:
        block = self.architecture
        if "network_class_name" not in block:
            raise PlanError(
                f"configuration {self.name!r} of plan {self.plans_name!r} has an architecture "
                f"block without `network_class_name`: {sorted(block)}. A plan written before "
                "nnU-Net 2.4 carries the legacy keys instead, and `ConfigurationManager` "
                "reconstructs the block from them -- this module does not, because writing a "
                "reconstruction nobody can check against the original would be a guess"
            )
        return str(block["network_class_name"])

    @property
    def arch_kwargs(self) -> Mapping[str, Any]:
        return dict(self.architecture.get("arch_kwargs") or {})

    @property
    def requires_import(self) -> tuple[str, ...]:
        return tuple(self.architecture.get("_kw_requires_import") or ())

    @property
    def strides(self) -> list[list[int]]:
        kwargs = self.arch_kwargs
        if "strides" not in kwargs:
            raise PlanError(
                f"configuration {self.name!r} of plan {self.plans_name!r} declares no `strides`. "
                "They fix both the stage count and the number of deep-supervision heads the "
                "trainer pairs with targets, and guessing either makes the loss pair a head with "
                "another scale's target"
            )
        return [[int(step) for step in stride] for stride in kwargs["strides"]]

    @property
    def stages(self) -> int:
        return len(self.strides)

    @property
    def deep_supervision_heads(self) -> int:
        """`stages - 1`, which is what nnU-Net's own scale list length is.

        Named rather than left to each caller to subtract, because `DeepSupervisionWrapper` zips
        heads against targets and `zip` TRUNCATES: an off-by-one here pairs every head with
        another scale's target and raises nothing.
        """
        return self.stages - 1

    @property
    def is_cascade_stub(self) -> bool:
        return set(self.body) <= set(CASCADE_STUB_KEYS)

    @property
    def inherits_from(self) -> str | None:
        value = self.body.get("inherits_from")
        return None if value is None else str(value)

    @property
    def next_stage(self) -> str | None:
        value = self.body.get("next_stage")
        return None if value is None else str(value)


@dataclass(frozen=True)
class Plan:
    """An nnU-Net plans document, held verbatim, read and modified through typed operations.

    `document` is never mutated -- every modifier returns a new `Plan` over a deep copy -- so a
    caller holding a `Plan` cannot have it changed underneath, which is how one plans object comes
    to describe two different fits.
    """

    document: Mapping[str, Any]

    # -- construction --------------------------------------------------------------------
    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Plan":
        """Wrap a parsed plans document, refusing one that is not a plan at all.

        The checks are deliberately minimal: `configurations` present and a mapping, and nothing
        else. A stricter gate here would refuse plans nnU-Net itself accepts -- a merged plan with
        a user-named configuration, a plan from an older version with legacy architecture keys --
        and this module's job is to carry what nnU-Net can read, not to hold an opinion about it.
        """
        if not isinstance(document, Mapping):
            raise PlanError(
                f"a plan is a mapping and this is a {type(document).__name__}"
            )
        configurations = document.get("configurations")
        if not isinstance(configurations, Mapping):
            raise PlanError(
                "the document carries no `configurations` mapping, so it is not a plans "
                f"document; it has {sorted(document)}"
            )
        return cls(document=copy.deepcopy(dict(document)))

    def to_document(self) -> dict[str, Any]:
        """A deep copy of the document, which is lossless by construction.

        A copy and not the held mapping, so a caller that mutates what it is given -- which is what
        `save_json` does not do but a patch script might -- cannot reach back into this `Plan`.
        """
        return copy.deepcopy(dict(self.document))

    # -- reading -------------------------------------------------------------------------
    def _require(self, key: str) -> Any:
        if key not in self.document:
            raise PlanError(
                f"plan {self.document.get('plans_name', '?')!r} has no {key!r}; it carries "
                f"{sorted(k for k in self.document if k != 'configurations')}"
            )
        return self.document[key]

    @property
    def plans_name(self) -> str:
        return str(self._require("plans_name"))

    @property
    def dataset_name(self) -> str:
        return str(self._require("dataset_name"))

    @property
    def configuration_names(self) -> tuple[str, ...]:
        return tuple(self.document["configurations"])

    def configuration(self, name: str) -> Configuration:
        """One configuration by name, or a refusal naming the ones this plan has.

        THE REFUSAL IS THE POINT. The configuration was a module constant in two files and a
        `plan.json` member nothing read back, so a plan recorded for one configuration could be fit
        under another with nothing raising. Asking a plan for a configuration it does not carry now
        fails here, before any data is staged.
        """
        configurations = self.document["configurations"]
        if name not in configurations:
            raise PlanError(
                f"plan {self.document.get('plans_name', '?')!r} has no configuration {name!r}; "
                f"it carries {sorted(configurations)}"
            )
        return Configuration(
            name=name,
            body=configurations[name],
            plans_name=str(self.document.get("plans_name", "")),
        )

    @property
    def unknown_global_keys(self) -> tuple[str, ...]:
        """Global keys this module does not name, which is information and not an error.

        A future nnU-Net key lands here, is carried verbatim, and a reader can see that it exists.
        Refusing it would make this module the reason an upgrade could not be adopted.
        """
        known = set(GLOBAL_KEYS) | {"configurations"}
        return tuple(sorted(key for key in self.document if key not in known))

    # -- modification --------------------------------------------------------------------
    def with_architecture(
        self,
        configuration: str,
        *,
        network_class_name: str,
        arch_kwargs: Mapping[str, Any] | None = None,
        record: Mapping[str, Any] | None = None,
        record_key: str = "medos_architecture",
    ) -> "Plan":
        """A NEW plan whose one configuration names a different network, with the departure recorded.

        THIS REPLACES A BIND MOUNT. The architecture used to be swapped by writing a patched copy of
        the document to another path and mounting that file over the read-only original inside the
        container. That worked and left no trace in the plan of what had been done to it.

        `record` goes BESIDE the architecture block and never inside it. The trainer reads exactly
        `network_class_name`, `arch_kwargs` and `_kw_requires_import` from that block and passes the
        second straight into the network's constructor as `**kwargs`, so a provenance note written
        inside would arrive as an unexpected keyword -- and every wrapper in `nets.py` refuses an
        unknown key by design, so the fit would fail at construction with a message about a plan key
        rather than about a note.
        """
        target = self.configuration(configuration)          # refuses an absent configuration
        if target.is_cascade_stub:
            raise PlanError(
                f"configuration {configuration!r} is a cascade stub carrying only "
                f"{sorted(target.body)}; it has no architecture of its own to replace. Patch "
                f"{target.inherits_from!r}, which is where its geometry comes from"
            )
        if "." not in network_class_name:
            raise PlanError(
                f"network_class_name={network_class_name!r} is a bare name, and this value is "
                "resolved by `pydoc.locate`, which needs a dotted path. A bare name resolves to "
                "None and `get_network_from_plans` then searches "
                "`dynamic_network_architectures.architectures` for the last segment, which can "
                "find a different class than the one intended"
            )

        merged = dict(target.arch_kwargs)
        added = dict(arch_kwargs or {})
        collision = sorted(set(added) & set(merged))
        if collision:
            raise PlanError(
                f"{collision} are in both the plan and the requested arch_kwargs. Preferring one "
                "silently would make the plan record a value the fit did not use; remove the key "
                "from the plan if the override is intended"
            )
        merged.update(added)

        # EVERY NAME IN `_kw_requires_import` MUST EXIST AS A KEY, even when its value is null.
        # `get_network_from_plans` does `architecture_kwargs[ri]` unconditionally, so a missing one
        # is a KeyError AFTER the plan was accepted and the data staged.
        missing = [name for name in target.requires_import if name not in merged]
        if missing:
            raise PlanError(
                f"`_kw_requires_import` names {missing}, which the merged arch_kwargs do not "
                "carry. nnU-Net indexes each of those unconditionally, so this would raise inside "
                "the network builder rather than here"
            )

        patched = self.to_document()
        body = dict(patched["configurations"][configuration])
        body["architecture"] = {
            **dict(target.architecture),
            "network_class_name": str(network_class_name),
            "arch_kwargs": merged,
        }
        if record is not None:
            body[record_key] = dict(record)
        patched["configurations"][configuration] = body
        return Plan(document=patched)

    def without_configurations(self, keep: Sequence[str]) -> "Plan":
        """A NEW plan carrying only the named configurations.

        Why this is a first-party operation rather than a caller's dict comprehension: dropping a
        configuration a `next_stage` or an `inherits_from` still points at leaves a plan whose
        cascade cannot resolve, and `ConfigurationManager` raises for that only when the missing
        stage is reached -- which for a cascade is after the low-resolution fit has been paid for.
        """
        keeping = list(keep)
        unknown = sorted(set(keeping) - set(self.configuration_names))
        if unknown:
            raise PlanError(
                f"asked to keep {unknown}, which this plan does not carry: "
                f"{sorted(self.configuration_names)}"
            )
        patched = self.to_document()
        patched["configurations"] = {
            name: body for name, body in patched["configurations"].items() if name in keeping
        }
        dangling: list[str] = []
        for name, body in patched["configurations"].items():
            for pointer in ("inherits_from", "next_stage", "previous_stage"):
                target = body.get(pointer)
                if target is not None and target not in patched["configurations"]:
                    dangling.append(f"{name}.{pointer} -> {target!r}")
        if dangling:
            raise PlanError(
                f"keeping {keeping} would leave {dangling} pointing at a configuration that is no "
                "longer in the plan. ConfigurationManager raises for that only when the missing "
                "stage is reached, which for a cascade is after the first fit has been paid for"
            )
        return Plan(document=patched)
