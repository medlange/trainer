# SPDX-License-Identifier: Apache-2.0
"""nnU-Net v2, in two phases: derive the plan, then fit against it and never re-derive.

WHY nnU-NET AND NOT Auto3DSeg
------------------------------
`MOS-UI-148` requires a dataset-fingerprint auto-configuring backend for the no-code
console -- `nnunet` or `auto3dseg` -- and forbids offering `monai_supervised`.
`MOS-TRAIN-211` makes the same choice the default for a `label` capability. Between the
two permitted ones, four things decide it and the last is the one that decides it:

  1. `MOS-TRAIN-223` FIXES A MAPPING TABLE PER BACKEND, AND ONE OF THE TWO ROWS IS
     ALREADY IMPLEMENTED. `medos/training/autoconfig.py::DERIVED_QUANTITIES` names
     `configurations.3d_fullres.spacing`, `transpose_forward` and
     `foreground_intensity_properties_per_channel.0.percentile_00_5` -- nnU-Net v2
     `plans.json` keys, exactly as this version writes them. The `auto3dseg` column of the
     same table points at `hyper_parameters.canonical_axis_order` and
     `hyper_parameters.crop_mode`, which MONAI's `AutoRunner` does not emit under those
     names; choosing Auto3DSeg would mean the exporter refuses on arrival, and the
     exporter is deliberately the component that "MUST refuse rather than substitute".
  2. ONE FINGERPRINT DOCUMENT VERSUS N. `MOS-TRAIN-223` digests "the fingerprint document
     itself -- `plans.json` for nnU-Net, `datastats.yaml` plus the per-algorithm
     `hyper_parameters.yaml` for `Auto3DSeg`". One file has one digest. Auto3DSeg's is a
     set whose membership depends on which algorithms `BundleGen` generated, and
     `fingerprint_digest` is a single `sha256_digest` column.
  3. ONE ARTIFACT VERSUS AN ENSEMBLE. `AutoRunner`'s natural output is N models and a
     combination rule; `MOS-TRAIN-227` requires that to be registered as ONE
     `ModelVersion` with the rule declared, and `MOS-TRAIN-137` refuses anything less.
     nnU-Net run at one fold produces one network, so the shortest honest path to a
     registrable artifact does not pass through the ensemble machinery at all.
  4. `MOS-TRAIN-135` NAMES nnU-NET AND ITS FREEZE EXPLICITLY, and `MOS-TRAIN-136` names
     the transcription obligation against it. The requirement set is simply more specific
     about this backend, and a backend the spec is specific about is a backend a reviewer
     can check.

An Auto3DSeg image is a SIBLING of this one, not a mode of it: two backends in one image
means two fingerprint documents behind one `pip freeze`, and `backend_versions` would have
to name a version for a planner that did not run.

THE TWO PHASES ARE TWO PROCESSES, AND THAT IS `MOS-TRAIN-135`
---------------------------------------------------------------
"Its self-configured plan MUST be frozen at run start and recorded as
`training_backend.plan_digest`." `medos.training.runs.start` refuses an auto-configured
run that reaches `RUNNING` without a `fingerprint_digest`, and `0013_training.up.sql`'s
`training_runs_guard()` then refuses any later change to it. So the fingerprint must exist
BEFORE the row leaves `PENDING` -- which means the deriving process has exited before the
fitting one starts. A single process writing the fingerprint partway through would make
the freeze a race with the poller.

`fit` therefore READS `plan.json` and refuses to run without it. It never calls the
planner. `MOS-TRAIN-225`'s argument is that a re-derivation is invisible -- "a materially
different transform, executed by a byte-identical container, against a byte-identical
weights digest, with every structural check passing" -- and the only defence that survives
is that the fitting phase has no code path that can derive one.

THE PLAN IS DERIVED FROM THE FIT PARTITION AND NOTHING ELSE
-------------------------------------------------------------
`MOS-TRAIN-135`: "A run whose plan was derived from any cohort other than the
`fit_partition` of the pinned split MUST be refused: plan derivation reads spacing,
intensity and foreground statistics, and deriving them over `train U tune U test` is a
test-set read performed by a configuration step." `stage_dataset()` below is handed ONE
cohort file and writes ONE nnU-Net dataset from it; the select partition is staged into a
separate dataset directory that the planner is never pointed at.

Spec: MOS-TRAIN-032, MOS-TRAIN-116, MOS-TRAIN-126, MOS-TRAIN-135, MOS-TRAIN-136,
MOS-TRAIN-137, MOS-TRAIN-141, MOS-TRAIN-211, MOS-TRAIN-223 to MOS-TRAIN-225, MOS-UI-148.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from medos_trainer.contract import CohortEntry, ContractViolation, RunDirectory, RunRequest

__all__ = [
    "DATASET_ID",
    "DATASET_NAME",
    "apply_determinism",
    "derive_plan",
    "fit",
    "prepare_workspace",
    "single_channel_view",
]

#: nnU-Net addresses a dataset by a three-digit id and a `DatasetNNN_Name` directory. The
#: id is constant because each run gets its own `nnUNet_raw`/`nnUNet_preprocessed` root
#: (the run directory's `work/`), so two concurrent runs cannot collide on it -- and a
#: per-run id would leak the run's ordering into a path.
DATASET_ID: Final[int] = 501
DATASET_NAME: Final[str] = f"Dataset{DATASET_ID:03d}_MedicalOSCohort"

_CONFIGURATION: Final[str] = "3d_fullres"

#: `MOS-TRAIN-032`: "nnU-Net's built-in cross-validation split MUST NOT be used as a
#: `DatasetSplit` ... The pipeline MUST hand nnU-Net a fold assignment derived from the
#: frozen `DatasetSplit`." `splits_final.json` below is that hand-off, and `"all"` is
#: never used: a fold of `all` is nnU-Net choosing its own validation cases out of the fit
#: partition, which is a split the platform did not freeze.
_SPLITS_FILE: Final[str] = "splits_final.json"


def prepare_workspace(work: Path) -> dict[str, str]:
    """Point nnU-Net's three roots INSIDE this run's directory. Must run before import.

    MEASURED, NOT ANTICIPATED. `nnunetv2/paths.py` reads `nnUNet_raw`,
    `nnUNet_preprocessed` and `nnUNet_results` from the environment AT MODULE IMPORT and
    binds them to module-level constants. Setting them after the first `import nnunetv2`
    has no effect, and the first run of this backend wrote its dataset into the image's
    default `/var/lib/medos-trainer/raw` instead of the run directory -- where a second
    concurrent run would have overwritten it under the same `DATASET_ID`, and where a
    `reap()` of the run directory would not have removed it.

    So this is called from `__main__._phase` before `medos_trainer.backend` is even
    imported, and the three roots are per-run. A run's whole footprint is then one
    directory: `MOS-TRAIN-121` C1's "its own working directory", meant literally.
    """
    roots = {
        "nnUNet_raw": str(work / "raw"),
        "nnUNet_preprocessed": str(work / "preprocessed"),
        "nnUNet_results": str(work / "results"),
    }
    for variable, path in roots.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        os.environ[variable] = path
    return roots


def apply_determinism(request: RunRequest) -> dict[str, Any]:
    """Seed and set what `request.json` says, and report what was actually applied.

    `MOS-TRAIN-126`: "the determinism settings actually used are recorded rather than
    asserted". The values come from the REQUEST, which came from
    `MEDOS_TRAINING_ENVIRONMENT`, which came from `medos_trainer.environment.DETERMINISM`.
    One chain, no second copy, and the returned dict is what a caller writes into
    `result.json` so the record is of the application rather than of the intention.
    """
    import numpy as np
    import torch

    seeds = dict(request.seeds)
    determinism = dict(request.determinism)

    random.seed(int(seeds["python"]))
    np.random.seed(int(seeds["numpy"]))
    torch.manual_seed(int(seeds["torch"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seeds["torch"]))

    workspace = str(determinism.get("cublas_workspace_config") or "")
    if workspace:
        # Read by cuBLAS at first handle creation, so it has to be set before any CUDA
        # work. Recorded either way.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace

    deterministic = bool(determinism.get("torch_use_deterministic_algorithms"))
    # `warn_only=True`: MOS-TRAIN-126 says bit-exactness "MUST NOT be required and MUST
    # NOT be claimed", and several 3D convolution backward kernels have no deterministic
    # CUDA implementation. Raising here would turn a recorded limitation into a failed
    # run; warning records it in the run log where a reader can see which kernel it was.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = bool(determinism.get("cudnn_benchmark"))
    torch.backends.cudnn.deterministic = deterministic
    tf32 = bool(determinism.get("tf32_allowed"))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # `dataloader_worker_base`. WHAT IT CONTROLS AND WHAT IT DOES NOT, because
    # overstating this would be exactly the assertion MOS-TRAIN-126 forbids.
    #
    # nnU-Net's augmenters are constructed inside `trainer.initialize()` and draw their
    # per-worker seeds from the PARENT process's numpy RNG at that moment. So `fit()`
    # re-seeds numpy with this value immediately before that call, which fixes which
    # seeds the workers receive. It does NOT make the augmentation bit-identical:
    # `MOS-TRAIN-126` names "dataloader worker interleaving" as one of the four reasons
    # bit-exactness is unavailable, and the ORDER in which seeded workers return batches
    # is still a scheduling property. What the seed buys is that a re-run draws from the
    # same sequence rather than from a different one, which is the "comparable run"
    # the requirement does promise.
    base = int(seeds["dataloader_worker_base"])
    os.environ["MEDOS_DATALOADER_WORKER_BASE"] = str(base)

    return {
        "seeds": {k: int(v) for k, v in seeds.items()},
        "applied": {
            "torch_use_deterministic_algorithms": deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
            "dataloader_worker_base": base,
        },
    }


# =====================================================================================
# Staging: the cohort, as an nnU-Net dataset
# =====================================================================================
def _dataset_json(
    cases: Sequence[CohortEntry], *, labels: Mapping[str, Any]
) -> dict[str, Any]:
    """nnU-Net v2's `dataset.json`.

    `channel_names: {"0": "CT"}` is what makes the planner choose `CTNormalization`, which
    is the ONLY normalisation scheme `medos.training.autoconfig._NORMALISATION_MAP` maps
    exactly (`MOS-TRAIN-223`: "fixes CTNormalization -> zscore_dataset and nothing else").
    A non-CT channel name here produces `ZScoreNormalization` and the export then REFUSES,
    which is the correct behaviour and a confusing one to debug -- hence this comment.
    """
    # AND `regions_class_order`, WITHOUT WHICH THE MASKED TRAINER CANNOT START.
    #
    # `nnUNetTrainerMaskedChannels.__init__` calls `force_region_mode`, which sets
    # `_has_regions` and then `_get_regions()`; that method asserts `regions_class_order
    # is not None`. This function built a dataset.json with neither regions nor an order,
    # so the masked trainer died at construction with an assertion from inside nnU-Net.
    # It had never been seen, because `fit` constructed the STOCK trainer, which does not
    # force region mode and does not need either.
    #
    # The order is the label values ascending, which is the order `io.label_set` is
    # already sorted into and therefore the order the output channels carry. It is not a
    # free choice: `regions_class_order` decides which channel wins a voxel when several
    # sigmoid heads fire, and a model whose channel order came from anywhere but the
    # registered spec is a model whose output meanings live outside the signed artifact
    # (`MOS-TRAIN-136`).
    order = sorted(
        value[0] if isinstance(value, list) else int(value)
        for name, value in labels.items()
        if name != "background"
    )
    return {
        "channel_names": {"0": "CT"},
        "labels": dict(labels),
        "regions_class_order": order,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def _labels_from_spec(spec_document: Mapping[str, Any]) -> dict[str, Any]:
    """nnU-Net's `{name: value}` from `MOS-IMG-049`'s `io.label_set`.

    The label set is the spec's, never nnU-Net's own numbering: `MOS-TRAIN-136` requires
    the derived plan to be transcribed into the registered `PreprocessingSpec`, and a
    model whose class indices came from a dataset.json nobody registered is a model whose
    output channel meanings live outside the signed artifact.
    """
    label_set = list((spec_document.get("io") or {}).get("label_set") or [])
    if not label_set:
        raise ContractViolation(
            "the bound PreprocessingSpec declares no io.label_set, so there is no "
            "mapping from an output channel to a structure. MOS-IMG-049 requires one for "
            "an output_kind of 'label' and nnU-Net cannot be handed a dataset without it"
        )
    # SINGLETON LISTS, NOT BARE INTEGERS, for everything except background.
    #
    # `{"neo": 1}` is a label; `{"neo": [1]}` is a REGION of one label, and the difference
    # is the difference between one softmax over mutually exclusive classes and one
    # sigmoid head per finding. This dataset is partially labelled: a case that annotates
    # a nodule says nothing about whether it also has an effusion, so the classes are NOT
    # mutually exclusive and a softmax over them would be a claim the data does not make.
    # `force_region_mode` in `masked_trainer.py` exists to hold nnU-Net to that reading,
    # and it can only do so if the label set arrives shaped as regions.
    #
    # Background stays a bare 0: it is not a finding and it is not a head.
    out: dict[str, Any] = {}
    for entry in label_set:
        name = str(entry["name"])
        value = int(entry["value"])
        out[name] = value if name == "background" else [value]
    if "background" not in out:
        raise ContractViolation(
            "io.label_set declares no `background`; nnU-Net requires label 0 to be named"
        )
    return out


def stage_dataset(
    run: RunDirectory,
    cases: Sequence[CohortEntry],
    *,
    raw_root: Path,
    spec_document: Mapping[str, Any],
) -> Path:
    """Write ONE nnU-Net raw dataset from ONE partition's cases. Links, never copies.

    Hard links where the filesystem allows them and a copy otherwise: a 3D CT corpus is
    tens of gigabytes and duplicating it per run is a deployment problem, but correctness
    does not depend on which happened.

    THE CASE IDENTIFIER IS THE `case_key` AND NOTHING ELSE. `MOS-EVID-010` makes
    `patient_key` an HMAC and `case_key` the case's own opaque identifier; neither is a
    patient identifier, and no accession number, name, MRN or study description is written
    into a filename here. `MOS-SEC-033` -- never log a PHI value -- applies to paths as
    much as to log lines, and nnU-Net prints every case name it loads.
    """
    dataset = raw_root / DATASET_NAME
    images, labels = dataset / "imagesTr", dataset / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for case in cases:
        source = run.staged_path(case.image, where=f"cohort case {case.case_key}")
        _link(source, images / f"{case.case_key}_0000.nii.gz")
        if case.label is None:
            raise ContractViolation(
                f"case {case.case_key} carries no `label`. The annotation set is the "
                "reference standard MOS-TRAIN-124 binds into the run; a supervised fit "
                "over cases with no reference standard is not the run that was submitted"
            )
        annotation = run.staged_path(case.label, where=f"cohort label {case.case_key}")
        _link(annotation, labels / f"{case.case_key}.nii.gz")

    labels = _labels_from_spec(spec_document)
    (dataset / "dataset.json").write_text(
        json.dumps(_dataset_json(cases, labels=labels), indent=2),
        encoding="utf-8",
    )
    _write_supervision(dataset, cases, labels=labels)
    return dataset


def _write_supervision(
    dataset: Path, cases: Sequence[CohortEntry], *, labels: Mapping[str, Any]
) -> None:
    """`supervision.json`, from the cohort, beside the dataset the cohort became.

    THIS IS WHY THE MASKED PATH HAD NEVER RUN. `nnUNetTrainerMaskedChannels` reads this
    file from `$nnUNet_raw/<dataset>/` and refuses without it. The only writer of it was
    `tools/ingest/nnunet_dataset.py`, a side tool that builds a dataset OUT OF BAND and
    drops the map beside THAT one -- a different directory from the one `stage_dataset`
    builds here, out of the cohort the platform sealed. The two never met, so a fit driven
    by this backend could not be masked, and the first one that tried died at construction
    with a FileNotFoundError naming the side tool.

    ALL OR NOTHING, AND THE MIDDLE IS AN ERROR. A cohort where some cases declare their
    channels and others do not cannot produce a usable map: the undeclared case reaches a
    batch and is either rejected mid-epoch by the trainer, or -- if somebody "fixes" that
    with a default -- supervised on every channel, which is the false-negative signal the
    whole subsystem exists to remove. So a partial cohort is refused HERE, before a GPU is
    reserved, with both counts in the message.

    NO DECLARATION AT ALL writes nothing, deliberately. The trainer's own refusal is then
    the one the reader sees, and it is the better message: it names the file, the
    directory it looked in, and what falling back would have cost.
    """
    declared = [c for c in cases if c.supervises is not None]
    if not declared:
        return
    if len(declared) != len(cases):
        raise ContractViolation(
            f"{len(declared)} of {len(cases)} cohort cases declare `supervises`. A "
            "supervision map covering part of a cohort is worse than none: the cases it "
            "omits reach the loss with no mask, and a channel nobody annotated then "
            "trains as background. Either every case declares its channels or none does"
        )

    channels = [name for name in labels if name != "background"]
    unknown = sorted(
        {c for case in declared for c in (case.supervises or ()) if c not in channels}
    )
    if unknown:
        raise ContractViolation(
            f"the cohort supervises channels the label set does not declare: {unknown}. "
            f"The label set is {sorted(channels)}, and it comes from the registered "
            "PreprocessingSpec (MOS-TRAIN-136); a channel named only in the cohort would "
            "be supervision applied to an output head that does not exist"
        )

    document = {
        "_note": (
            "Which channels each case SUPERVISES. A channel absent from a case's list is "
            "UNKNOWN for that case and must contribute zero to the loss and zero to every "
            "gradient -- not a negative."
        ),
        "_written_by": "medos_trainer.backend.stage_dataset, from the sealed cohort",
        "empty_segment_is_negative": False,
        "channels": channels,
        "label_of": {name: labels[name] for name in channels},
        "cases": {c.case_key: list(c.supervises or ()) for c in declared},
    }
    (dataset / "supervision.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )


def _link(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        # Cross-device, or a filesystem with no hard links (a bind mount from a Windows
        # host is the common one). Copying is slower and identical in effect.
        shutil.copyfile(source, target)


def write_fold_assignment(
    preprocessed: Path, fit: Sequence[CohortEntry], select: Sequence[CohortEntry]
) -> Path:
    """`splits_final.json`: the platform's frozen split, handed to nnU-Net. `MOS-TRAIN-032`.

    "nnU-Net's built-in cross-validation split MUST NOT be used as a `DatasetSplit`. It is
    generated by a hash of the case identifiers ... The pipeline MUST hand nnU-Net a fold
    assignment derived from the frozen `DatasetSplit`."

    ONE fold, and its validation half is the SELECT partition. `MOS-TRAIN-116` makes
    selection a read of a partition the fit did not see, and nnU-Net's own five-fold
    generator would instead carve the validation cases out of the fit partition by hashing
    their names -- a split nobody froze and nobody can reproduce from the manifest.
    """
    preprocessed.mkdir(parents=True, exist_ok=True)
    path = preprocessed / _SPLITS_FILE
    path.write_text(
        json.dumps([
            {
                "train": [c.case_key for c in fit],
                "val": [c.case_key for c in select],
            }
        ], indent=2),
        encoding="utf-8",
    )
    return path


# =====================================================================================
# Phase 1 -- derive
# =====================================================================================
def single_channel_view(plans: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nnU-Net's per-channel lists for the one key the exporter reads as a scalar.

    A DEFECT IN THE PLATFORM'S EXPORTER, REPORTED HERE RATHER THAN WORKED AROUND SILENTLY.
    `medos/training/autoconfig.py` maps `foreground_crop` from
    `configurations.3d_fullres.use_mask_for_norm` and `_coerce` handles `bool` -- but
    nnU-Net v2 writes that field as a LIST OF BOOLS, one per input channel, and a list
    falls through to the string branch and is refused as `crop_mode_not_mapped`. So the
    shipped exporter cannot read a real nnU-Net `plans.json` at all. The mapping table in
    `MOS-TRAIN-223` says "crop-to-nonzero; `configurations.3d_fullres.use_mask_for_norm`"
    without saying it is per channel, so the requirement is arguably the thing that is
    imprecise; either way the fix belongs in `medos/training/autoconfig.py`, which this
    change does not own.

    What this function does instead is NARROW and REFUSES rather than picks: with exactly
    one channel the list has exactly one element and flattening it loses nothing; with
    more than one it raises, because "the first channel's crop rule" is precisely the kind
    of nearest-available substitution `MOS-TRAIN-223` forbids the exporter to make.

    The returned view is used ONLY for the export. `fingerprint_digest` is taken over the
    verbatim `plans.json`, because that is the document `MOS-TRAIN-223` retains for audit.
    """
    view = json.loads(json.dumps(dict(plans)))  # a deep copy, by value
    configuration = dict(view.get("configurations", {}).get(_CONFIGURATION, {}))
    mask = configuration.get("use_mask_for_norm")
    if isinstance(mask, list):
        if len(mask) != 1:
            raise ContractViolation(
                "plans.json declares use_mask_for_norm per channel with "
                f"{len(mask)} channels. MOS-TRAIN-223's mapping table gives "
                "foreground_crop ONE source key, so a multi-channel crop rule has no "
                "exact PreprocessingSpec field and the exporter MUST refuse rather than "
                "take the first channel's value"
            )
        configuration["use_mask_for_norm"] = bool(mask[0])
        view["configurations"][_CONFIGURATION] = configuration
    return view


def derive_plan(
    run: RunDirectory, request: RunRequest, *, work: Path
) -> dict[str, Any]:
    """Phase 1. Fingerprint the FIT partition, plan, and transcribe. Returns `plan.json`.

    nnU-Net's own entry points are called rather than reimplemented (`MOS-REL-032`: adopt
    the adopted thing whole). `extract_fingerprint` and `plan_experiments` are the two
    halves of `nnUNetv2_plan_and_preprocess`; `preprocess_dataset` is run here too because
    the fitting phase must not be able to touch the raw images -- after this phase the
    only thing `fit` reads is the preprocessed tensor cache and the frozen plan.
    """
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess_dataset,
    )

    from medos.training.autoconfig import export_spec_fields, fingerprint_digest

    # `prepare_workspace` already set the three roots and `__main__` called it before
    # this module was imported -- see its docstring for why the order is load-bearing.
    raw = Path(os.environ["nnUNet_raw"])
    preprocessed = Path(os.environ["nnUNet_preprocessed"])

    spec_document = run.spec_document()
    fit_cases = run.cohort("cohort_fit")
    select_cases = run.cohort("cohort_select")

    # ONE dataset, from the FIT partition only (MOS-TRAIN-135). The select cases are
    # staged into the same nnU-Net dataset because nnU-Net's fold assignment addresses
    # cases by name within one dataset -- and the fingerprint is extracted BEFORE they are
    # staged, so no statistic is read from them. The ordering is the control; see below.
    stage_dataset(run, fit_cases, raw_root=raw, spec_document=spec_document)

    # `verify_dataset_integrity=True`: a geometry mismatch between an image and its label
    # is a silent loss of supervision, and nnU-Net's checker is the one already written.
    # `check_dataset_integrity` is nnU-Net's own image/label geometry check. A mismatch
    # is a silent loss of supervision -- the label lands on a grid the image is not on --
    # and the checker is already written, so it is turned on rather than reimplemented.
    extract_fingerprints(
        [DATASET_ID], check_dataset_integrity=True, clean=True,
        num_processes=_processes(), verbose=False,
    )
    plan_identifier = plan_experiments(
        [DATASET_ID], gpu_memory_target_in_gb=_vram_target()
    )

    # Only now are the select cases visible to nnU-Net, and only as cases to validate
    # against. Nothing between this line and the end of the function reads a statistic.
    stage_dataset(run, select_cases, raw_root=raw, spec_document=spec_document)
    preprocess_dataset(
        DATASET_ID, plans_identifier=plan_identifier,
        configurations=[_CONFIGURATION], num_processes=[_processes()],
    )
    write_fold_assignment(preprocessed / DATASET_NAME, fit_cases, select_cases)

    plans_path = preprocessed / DATASET_NAME / f"{plan_identifier}.json"
    if not plans_path.is_file():  # pragma: no cover - nnU-Net always writes it
        raise ContractViolation(f"the planner wrote no {plans_path}")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))

    # The digest is over the VERBATIM document (MOS-TRAIN-223 retains this one for audit);
    # the export reads the single-channel view. See `single_channel_view`.
    digest = fingerprint_digest(plans)
    exported = export_spec_fields(single_channel_view(plans), backend="nnunet")

    run.write("fingerprint", plans)
    configuration = plans["configurations"][_CONFIGURATION]
    return {
        "backend": {"kind": "nnunet", "version": request.backend_version},
        "plans_identifier": plan_identifier,
        "configuration": _CONFIGURATION,
        "fingerprint_digest": digest,
        "fingerprint_document": run.path("fingerprint").name,
        "spec_fields": exported["spec_fields"],
        # MOS-TRAIN-224: the derived TRAINING batch size, which is not a
        # PreprocessingSpec field. See `packaging.derive_spec_document` for where it does
        # NOT go, and trainer/README.md for the column that cannot carry it.
        "hyperparameters": exported["hyperparameters"],
        "fit_cases": len(fit_cases),
        "select_cases": len(select_cases),
        "derived_from_partition": request.fit_partition,
        "nnunet_patch_size": list(configuration["patch_size"]),
        "work": str(work),
    }


def _vram_target() -> float:
    """The planner's VRAM budget, in GB. Observed, not assumed.

    nnU-Net's default target is 8 GB. Passing the device's actual size lets the planner
    choose a patch size that fits the card the run will execute on, which is the whole
    point of auto-configuration -- and passing MORE than the card has produces a plan that
    OOMs in epoch 1 after the fingerprint has already been frozen.
    """
    import torch

    if not torch.cuda.is_available():
        return 8.0
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    override = os.environ.get("MEDOS_TRAINER_VRAM_GB", "").strip()
    if override:
        return float(override)
    # Leave a margin for the allocator, the CUDA context and cuDNN's workspace.
    return max(4.0, round(total - 2.0, 1))


def _processes() -> int:
    raw = os.environ.get("MEDOS_TRAINER_PROCESSES", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, min(8, (os.cpu_count() or 2) // 2))


# =====================================================================================
# Phase 2 -- fit
# =====================================================================================
def fit(run: RunDirectory, request: RunRequest, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Phase 2. Train against the FROZEN plan. Returns what the fit did.

    There is no planner import in this function and there must not be: `MOS-TRAIN-225`'s
    whole argument is that a re-derivation is invisible, and the only defence that holds
    is that the fitting phase has no code path that can produce one. It reads
    `plan.json`'s `plans_identifier` and nnU-Net loads that file from the preprocessed
    folder phase 1 wrote.

    THE EPOCH BUDGET IS A DEPLOYMENT PROPERTY AND IS RECORDED. `MOS-UI-149` forbids the
    console displaying it and `TrainingRunSubmitRequest` has no member for it, so it
    cannot arrive from a client. It arrives from the trainer deployable's own environment
    and is written into the result, because a run that trained for 5 epochs and a run that
    trained for 1000 are not comparable and the record has to say which happened.
    """
    import torch
    # THE MASKED TRAINER, NOT THE STOCK ONE, AND THIS LINE IS THE WHOLE SUBSYSTEM.
    #
    # This imported `nnUNetTrainer` and constructed it. `nnUNetTrainerMaskedChannels`
    # -- 345 lines, holding the per-(case, channel) mask, the forced region mode and the
    # supervision map -- was referenced by exactly one test and by nothing on this path.
    # So was `masked.py` under it, because only the subclass calls it.
    #
    # A fit built this way RUNS. It converges, writes a checkpoint and reports a falling
    # loss. What it learns is the defect `masked.py`'s own docstring opens with: every
    # channel a case does not annotate is presented to the loss as background, so the
    # model is taught to suppress exactly the findings it exists to detect -- and it then
    # scores well on each corpus's own test split, because each split carries the same
    # blind spot as its training data.
    #
    # Measured on the cohort this was caught with: 138 of 900 (case, channel) pairs are
    # supervised. The other 762 would have been negatives.
    #
    # The signature is transcribed from nnU-Net's rather than absorbed into **kwargs (see
    # the subclass), so this is a drop-in and a mismatch would be a TypeError here.
    from medos_trainer.masked_trainer import nnUNetTrainerMaskedChannels

    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    plans = json.loads(
        (preprocessed / f"{plan['plans_identifier']}.json").read_text(encoding="utf-8")
    )
    dataset_json = json.loads((preprocessed / "dataset.json").read_text(encoding="utf-8"))

    budget = _budget(request)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = nnUNetTrainerMaskedChannels(
        plans=plans,
        configuration=_CONFIGURATION,
        # Fold 0 is the ONE fold `write_fold_assignment` wrote, and its validation half is
        # the select partition. nnU-Net's `"all"` is never used: it would make nnU-Net
        # choose validation cases out of the fit partition by hashing their names.
        fold=0,
        dataset_json=dataset_json,
        device=device,
    )
    trainer.num_epochs = int(budget["max_epochs"])
    trainer.num_iterations_per_epoch = int(budget["iterations_per_epoch"])
    trainer.num_val_iterations_per_epoch = int(budget["validation_iterations_per_epoch"])
    trainer.save_every = max(1, int(budget["max_epochs"]))

    started = time.monotonic()
    # See `apply_determinism`: this is the one point at which
    # `seeds.dataloader_worker_base` can be applied, because `initialize()` is where
    # nnU-Net constructs the augmenters and draws their per-worker seeds.
    import numpy as np

    np.random.seed(int(request.seeds["dataloader_worker_base"]))
    trainer.initialize()
    trainer.run_training()
    seconds = time.monotonic() - started

    checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
    if not checkpoint.is_file():  # pragma: no cover - run_training writes it last
        raise ContractViolation(
            f"the fit finished and wrote no {checkpoint}; there is nothing to export"
        )
    return {
        "budget": budget,
        "seconds": round(seconds, 1),
        "device": str(device),
        "output_folder": str(trainer.output_folder),
        "checkpoint": str(checkpoint),
        "patch_size": [int(n) for n in plans["configurations"][_CONFIGURATION]["patch_size"]],
        # Deep supervision OFF before the network leaves this function. nnU-Net's own
        # method is used rather than reaching into the decoder, because the trainer is
        # what knows whether the module is compiled and whether the heads are wrapped.
        # `packaging.torchscript_bytes` checks the result rather than trusting it: a
        # traced module that returns a TUPLE is a model the serving runner indexes by
        # position, which is how a network comes to be served at the wrong scale with
        # every structural check passing.
        "network": _without_deep_supervision(trainer),
        "label_manager_classes": int(trainer.label_manager.num_segmentation_heads),
    }


def _budget(request: RunRequest) -> dict[str, int]:
    """How much training this deployment is asking for, from its own configuration.

    nnU-Net's own defaults are 1000 epochs of 250 iterations, which is the full run. They
    are the defaults HERE too -- a deployment that sets nothing gets the real thing -- and
    `MEDOS_TRAINER_EPOCHS` is how a short run is asked for deliberately. Recorded either
    way: a truncated fit whose record does not say it was truncated is a candidate that
    reads at the approval gate as a full one.
    """
    declared = dict(request.budget)

    def _read(name: str, variable: str, default: int) -> int:
        if name in declared:
            return max(1, int(declared[name]))
        raw = os.environ.get(variable, "").strip()
        return max(1, int(raw)) if raw else default

    return {
        "max_epochs": _read("max_epochs", "MEDOS_TRAINER_EPOCHS", 1000),
        "iterations_per_epoch": _read(
            "iterations_per_epoch", "MEDOS_TRAINER_ITERATIONS", 250
        ),
        "validation_iterations_per_epoch": _read(
            "validation_iterations_per_epoch", "MEDOS_TRAINER_VAL_ITERATIONS", 50
        ),
    }


def _without_deep_supervision(trainer: Any) -> Any:
    """`trainer.network` with deep supervision turned off, by nnU-Net's own method.

    `nnUNetTrainer.set_deep_supervision_enabled` exists for exactly this and knows about
    the `torch.compile` wrapper and the DDP wrapper; reaching into `network.decoder`
    from here would be a second implementation that goes wrong on either.
    """
    setter = getattr(trainer, "set_deep_supervision_enabled", None)
    if callable(setter):
        setter(False)
    return trainer.network
