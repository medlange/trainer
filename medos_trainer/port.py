# SPDX-License-Identifier: Apache-2.0
"""The training backend PORT: what a backend must be, and what it must return.

WHY THIS FILE EXISTS
--------------------
The port was already here, spelled implicitly. `__main__._phase` imported one module, bound
it to a local named `nnunet`, and passed it into `_fit` as a positional argument; `fit`
returned an undeclared dict that `_fit` indexed by seven keys. That is an interface with one
implementation, no name and no substitute -- which `MOS-REL-048` says is not a seam -- and
its failure mode is expensive in a specific way: a missing key raises a `KeyError` AFTER the
whole fit has been paid for, on a machine that has been busy for hours.

WHAT VARIES AND WHAT MUST NOT
------------------------------
The point of a second backend is that the ARCHITECTURE changes. Everything the platform
exists for must not:

  * the per-(case, channel) supervision mask -- an unannotated channel contributes zero to
    the loss and zero to every gradient, never a negative (`masked.py`, which is pure torch
    and already knows nothing about nnU-Net);
  * the plan frozen at run start and never re-derived (`MOS-TRAIN-135`, `MOS-TRAIN-225`);
  * the provenance: which image, which backend, which version, which commit;
  * the bundle layout and the `PreprocessingSpec` transcription, which REFUSES rather than
    defaults (`MOS-TRAIN-223`).

So this file names the four operations and the one result shape, and `conformance.py` beside
it states the invariants every driver must satisfy -- on CPU, against synthetic tensors, with
no corpus and no card. A backend that cannot be added without a GPU is a backend nobody will
prove correct.

WHY `classes` AND NOT `label_manager_classes`
----------------------------------------------
`_fit` read `fitted["label_manager_classes"]`. `LabelManager` is an nnU-Net class, so the
port's own vocabulary named one implementation's internals and a second backend would have
had to invent an nnU-Net concept to fill the field. The quantity is the number of output
classes. It is called that here.

WHAT THIS PORT DELIBERATELY DOES NOT DECIDE
--------------------------------------------
It does not decide WHICH backend a run uses. That arrives in `request.json` as
`training_backend.kind`, sealed by the platform, and `resolve()` refuses a kind this image
did not install rather than substituting the one it has. `MOS-REL-037` forbids a
compatibility range on a recorded version, and the same argument applies to a backend: an
image that answered for a second backend would record a version for a planner that did not
run.

It also does not decide whether a second backend belongs in THIS image or a sibling one.
`__init__.py` records the decision for `auto3dseg` -- sibling, because two fingerprint
documents behind one `pip freeze` would pin a planner that did not run -- and that argument
is about PLANNERS. A hand-configured backend has none, so it is not what that decision
bars; it still needs its own row in `BACKENDS` and its own declared version before it can
answer for anything.

Spec: MOS-TRAIN-135, MOS-TRAIN-211, MOS-TRAIN-223, MOS-TRAIN-225, MOS-REL-037, MOS-REL-048.
Pure: imports no torch, no nnU-Net and no backend, so a contract test can read it for the
price of a text file.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, runtime_checkable

__all__ = [
    "BACKENDS",
    "FitResult",
    "TrainingBackend",
    "implemented_kinds",
    "resolve",
]

#: `training_backend.kind` -> the module that implements it, for the backends THIS image
#: installed. It is a map and not a constant because the refusal below has to be able to
#: say what the image does implement; a one-entry map and a bare string differ in exactly
#: that, and the difference is the whole error message.
#:
#: A row here is a claim that the named module implements every operation in
#: `TrainingBackend` AND passes `trainer/tests/test_backend_conformance.py`. Adding a row
#: without the second half is how a backend that trains an unannotated channel as
#: background gets shipped: it converges, it writes a checkpoint, and it scores well on
#: each corpus's own split.
BACKENDS: Final[dict[str, str]] = {
    "nnunet": "medos_trainer.backend",
}


@dataclass(frozen=True)
class FitResult:
    """What a fit did, in the port's own words rather than one backend's.

    `network` is a live `torch.nn.Module`, typed `Any` so this module stays importable
    without torch. `checkpoint` is a path to the weights the backend wrote, which the
    bundle carries verbatim: the port does not re-serialise it, because a checkpoint the
    platform rewrote is a checkpoint whose digest no longer matches what the trainer
    produced.
    """

    network: Any
    patch_size: tuple[int, ...]
    checkpoint: Path
    #: Output classes. NOT `label_manager_classes`: see the module docstring.
    classes: int
    #: `max_epochs`, `iterations_per_epoch`, `validation_iterations_per_epoch`. A
    #: deployment property (`MOS-UI-149` forbids a client sending one), recorded because a
    #: run that trained 5 epochs and one that trained 1000 are not comparable.
    budget: Mapping[str, Any]
    seconds: float
    device: str

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> FitResult:
        """Build from a backend that still returns a dict, naming every absent member.

        One refusal listing everything missing, not a `KeyError` on the first one: a
        backend author reading "absent: classes, seconds" fixes both, and one reading
        `KeyError: 'classes'` pays for another fit to discover the second.
        """
        absent = [
            name for name in
            ("network", "patch_size", "checkpoint", "classes", "budget", "seconds", "device")
            if name not in document
        ]
        if absent:
            raise ValueError(
                f"the backend's fit returned no {absent}. Every member of FitResult is "
                "read while writing the run's terminal record, which happens after the "
                "fit has been paid for -- so an absent member costs a whole run"
            )
        return cls(
            network=document["network"],
            patch_size=tuple(int(v) for v in document["patch_size"]),
            checkpoint=Path(document["checkpoint"]),
            classes=int(document["classes"]),
            budget=dict(document["budget"]),
            seconds=float(document["seconds"]),
            device=str(document["device"]),
        )


@runtime_checkable
class TrainingBackend(Protocol):
    """The four operations, in the order a run performs them.

    `prepare_workspace` and `apply_determinism` run in BOTH phases, before anything the
    backend imports binds a path or a seed. `derive_plan` runs in the plan phase and
    nowhere else; `fit` runs in the fit phase and MUST NOT be able to derive a plan
    (`MOS-TRAIN-225`: the only defence against an invisible re-derivation is having no code
    path that can produce one).
    """

    #: The value of `training_backend.kind` this module answers for. Asserted against its
    #: key in `BACKENDS` by the conformance suite, so a module cannot be registered under a
    #: name it does not claim.
    KIND: str

    def prepare_workspace(self, work: Path) -> dict[str, str]:
        """Bind the backend's roots under this run's `work/`. Returns what it bound."""

    def apply_determinism(self, request: Any) -> dict[str, Any]:
        """Apply the request's seeds and flags. Returns what was applied, for the record."""

    def derive_plan(self, run: Any, request: Any, *, work: Path) -> dict[str, Any]:
        """Derive and FREEZE the plan from the fit partition alone.

        The returned document is written to `plan.json` verbatim and must carry
        `backend` (`{"kind": ..., "version": ...}`), `hyperparameters`, `fit_cases`,
        `select_cases` and `spec_fields`. An auto-configuring backend must also carry
        `fingerprint_digest`; a hand-configured one must not pretend to.
        """

    def fit(self, run: Any, request: Any, plan: Mapping[str, Any]) -> Mapping[str, Any]:
        """Fit against the frozen plan. Returns a mapping `FitResult.from_mapping` reads."""

    def build_masked_loss(
        self, *, batch_dice: bool, scales: Any = None, focal_gamma: float = 0.0,
        focal_alpha: float | None = None,
    ) -> Any:
        """The ONE place this backend builds a loss, callable without plans or a card.

        IT IS PART OF THE PORT AND NOT AN IMPLEMENTATION DETAIL, because it is the only
        operation whose correctness can be established cheaply and whose incorrectness is
        invisible. `trainer/tests/test_backend_conformance.py` calls it on CPU with synthetic
        tensors and flips an unsupervised channel's target: the loss must be bit-identical
        and every gradient unchanged.

        Returning a loss the trainer does not actually use satisfies that and means nothing,
        so the conformance suite also asserts over the syntax tree that the backend's
        `_build_loss` hook calls this and constructs no loss of its own. Both halves are
        needed: the defect that shipped here once was a correct masked wrapper that nothing
        called.

        `scales` is the deep-supervision scale list, or `None` for a single head. The
        returned object takes `(logits, target, mask)` with `mask` of shape `[B, C]` --
        ONE mask whatever the number of scales, because whether a reader annotated a channel
        is a fact about the case and not about a decoder's resolution.
        """


def implemented_kinds() -> tuple[str, ...]:
    """The kinds this image can train, sorted, for a refusal or a declaration to name."""
    return tuple(sorted(BACKENDS))


def resolve(kind: str) -> Any:
    """The module for `kind`, or a refusal naming what this image does implement.

    THE REFUSAL IS NOT A FALLBACK. An image that answered for a backend it did not install
    would record a version for a planner that did not run, which `MOS-REL-037` forbids for
    a recorded version and which is worse here: the artifact would carry a backend label
    that names software absent from the image that produced it.
    """
    try:
        module_name = BACKENDS[kind]
    except KeyError:
        raise LookupError(
            f"this image trains {list(implemented_kinds())} and the run binds {kind!r}. "
            "MOS-REL-037 forbids a compatibility range on a recorded version and the same "
            "argument applies to a backend: an image that pretended to be a second "
            "backend would record a version for a planner that did not run"
        ) from None
    module = import_module(module_name)
    declared = getattr(module, "KIND", None)
    if declared != kind:
        raise LookupError(
            f"{module_name} is registered for {kind!r} and declares KIND={declared!r}. "
            "The registry and the module must agree, because the registry decides which "
            "code runs and the module's own KIND is what the artifact is labelled with"
        )
    return module
