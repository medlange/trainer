# SPDX-License-Identifier: Apache-2.0
"""The autonomous entry: the trainer without the platform.

THE AUDIT'S VERDICT THIS ANSWERS. A researcher with a folder of volumes
could not reach a trained model: the only door in was a platform-written
run directory. This module is the second door — loaders and CLI-backed
commands that need nothing but data:

    vanilla-plan             --data cases/ --preset cpu --out plan.json
    vanilla-fit              --data cases/ --preset cpu --out bundle/
    vanilla-crossval         --data cases/ --out crossval/ --folds 5
    vanilla-evaluate         --checkpoint-dir bundle/ --data cases/ --out eval.json
    vanilla-import-nnunet    --images imagesTr/ --labels labelsTr/ --out cases/
    vanilla-import-dicom     --series dicom/ --out case.npz

The module's TOP LEVEL imports only numpy and `vanilla.data` — the plan,
network and trainer layers import torch, and the importers (nibabel for
NIfTI, pydicom for DICOM) lazy-import their dependency inside the one
function that needs it, so an environment without them pays nothing. That
keeps the importers usable on a conversion box that has no torch at all.

`--data` names a directory of `.npz` cases — the format `Case` already
speaks (`image`, `label`, optional `mask`). Spacing is read from an
optional `spacing_mm` array member, defaulting to isotropic 1 mm; for
real data pass it, because the plan's patch size is a PHYSICAL decision
and wrong spacing is how a plan quietly becomes wrong.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from medos_trainer.vanilla.data import Case, load_case_npz

if TYPE_CHECKING:
    from medos_trainer.vanilla.infer import SlidingWindowPredictor


def load_cases_dir(data_dir: str | Path) -> list[Case]:
    """Every `.npz` case in a directory, sorted so the census is deterministic."""
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"{data_dir} holds no .npz cases")
    cases = []
    for path in paths:
        with np.load(path) as z:
            spacing = (
                tuple(float(v) for v in z["spacing_mm"])
                if "spacing_mm" in z.files
                else (1.0, 1.0, 1.0)
            )
        cases.append(load_case_npz(path, spacing_mm=spacing))
    return cases


def import_nnunet_dataset(
    images_dir: str | Path, labels_dir: str | Path, out_dir: str | Path
) -> int:
    """nnU-Net's NIfTI layout (imagesTr/ + labelsTr/) -> `.npz` cases.

    THE MIGRATION BRIDGE the audit asked for: a corpus prepared for
    nnU-Net is one conversion away from the vanilla stack. nibabel is
    imported HERE, lazily — an environment that never imports has no
    idea it exists, and the vanilla purity gate stays meaningful.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on the ambient env
        raise RuntimeError(
            "importing nnU-Net layouts needs nibabel; pip install nibabel"
        ) from exc

    images = sorted(Path(images_dir).glob("*.nii.gz"))
    if not images:
        raise ValueError(f"{images_dir} holds no .nii.gz images")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path in images:
        # nnU-Net names labels identically under labelsTr/.
        label_path = Path(labels_dir) / image_path.name
        if not label_path.is_file():
            raise ValueError(f"no matching label for {image_path.name} in {labels_dir}")
        image = nib.load(str(image_path))
        label = nib.load(str(label_path))
        data = np.asarray(image.dataobj, dtype=np.float32)[None]  # (1,K,J,I)
        seg = np.asarray(label.dataobj)
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        name = image_path.name[: -len(".nii.gz")]
        np.savez(out / f"{name}.npz", image=data, label=seg, spacing_mm=np.asarray(spacing))
        written += 1
    return written


def import_dicom_series(series_dir: str | Path, out_dir: str | Path) -> int:
    """ONE DICOM series -> one `.npz` case (`image` (1,K,J,I), `spacing_mm`).

    DESPITE THE PARAMETER NAME, `out_dir` IS THE TARGET FILE PATH — the CLI
    spells it `--out case.npz`. There is no label: this is the door IN for
    inference, the mirror of the NIfTI importer.

    THE READING RULES, each earned from a real failure mode:

      * EVERY FILE in the directory is tried with `pydicom.dcmread`, and
        files that are not readable DICOM are skipped — real series
        directories carry OS droppings and secondary captures.
      * MULTIPLE SERIES ARE REFUSED, by UID, rather than silently merged:
        a directory holding two series is a directory the caller has not
        described.
      * SLICES ARE SORTED by ImagePositionPatient projected on the slice
        normal from ImageOrientationPatient — filenames lie about order.
      * SPACING follows DICOM convention: `(slice, PixelSpacing[0],
        PixelSpacing[1])` matching the `(K, J, I)` axis order. The slice
        spacing is the median delta of the sorted projections; positions
        that deviate from uniform by more than 1 mm get a printed warning
        rather than a refusal, because real scanners wobble more than the
        standard admits and the median is still the honest central answer.
      * RescaleSlope/RescaleIntercept are applied per slice (default 1/0) —
        storage pixels are not image pixels until they are.

    pydicom is imported lazily here, same discipline as nibabel above.
    Returns 1 — one series, one file.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on the ambient env
        raise RuntimeError(
            "importing DICOM series needs pydicom; pip install pydicom"
        ) from exc

    slices = []
    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path))
        except Exception:  # noqa: BLE001 - unreadable files are expected noise
            continue
        if "PixelData" not in ds or not hasattr(ds, "ImagePositionPatient"):
            continue  # non-image DICOM (reports, secondary captures)
        slices.append(ds)
    if not slices:
        raise ValueError(f"{series_dir} holds no readable DICOM image slices")

    series_uids = sorted({str(getattr(ds, "SeriesInstanceUID", "unknown")) for ds in slices})
    if len(series_uids) > 1:
        raise ValueError(
            f"{series_dir} holds {len(series_uids)} series; --series expects one: "
            + ", ".join(series_uids)
        )

    iop = [float(v) for v in slices[0].ImageOrientationPatient]
    row, col = np.asarray(iop[0:3]), np.asarray(iop[3:6])
    normal = np.cross(row, col)
    normal = normal / np.linalg.norm(normal)

    def position(ds: object) -> float:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient])  # type: ignore[attr-defined]
        return float(np.dot(ipp, normal))

    slices.sort(key=position)
    positions = np.asarray([position(ds) for ds in slices])
    deltas = np.diff(positions)
    slice_spacing = abs(float(np.median(deltas))) if len(deltas) else 1.0
    if len(deltas):
        wobble = float(np.max(np.abs(deltas - np.median(deltas))))
        if wobble > 1.0:
            print(
                f"warning: slice positions deviate from uniform by up to {wobble:.2f} mm; "
                f"using the median spacing {slice_spacing:.3f} mm"
            )

    # PixelSpacing is [row spacing, column spacing] = (J, I); K comes from the
    # sorted slice positions. The tuple matches the (K, J, I) axis order.
    pixel_spacing = [float(v) for v in slices[0].PixelSpacing]
    spacing_mm = (slice_spacing, pixel_spacing[0], pixel_spacing[1])
    volume = np.stack([
        ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0))
        + float(getattr(ds, "RescaleIntercept", 0.0))
        for ds in slices
    ])
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, image=volume[None], spacing_mm=np.asarray(spacing_mm, dtype=np.float64))
    return 1


def _planned_run(
    cases: list[Case],
    preset: str,
    epochs: int | None,
    steps_per_epoch: int | None,
    foreground_prob: float | None = None,
):
    """The plan over `cases`, with smoke-run overrides applied the way
    `fit_command` always applied them: by rebuilding the frozen record."""
    from medos_trainer.vanilla.plan import (
        PlannedRun,
        collect_fingerprint,
        plan_from_fingerprint,
    )

    plan = plan_from_fingerprint(collect_fingerprint(cases), preset)
    if epochs is not None or steps_per_epoch is not None or foreground_prob is not None:
        plan = PlannedRun(
            fingerprint=plan.fingerprint,
            preset=plan.preset,
            patch_size=plan.patch_size,
            stem_stride=plan.stem_stride,
            steps_per_epoch=steps_per_epoch or plan.steps_per_epoch,
            epochs=epochs or plan.epochs,
            foreground_prob=(
                plan.foreground_prob if foreground_prob is None else foreground_prob
            ),
            reasons=plan.reasons,
        )
    return plan


def plan_command(data_dir: str | Path, preset: str, out: str | Path) -> dict:
    """Fingerprint + plan, written as JSON with the reasons attached."""
    cases = load_cases_dir(data_dir)
    plan = _planned_run(cases, preset, epochs=None, steps_per_epoch=None)
    document = {
        "median_spacing": [float(v) for v in plan.fingerprint.median_spacing],
        "median_shape": [int(v) for v in plan.fingerprint.median_shape],
        "class_counts": [[int(c), int(n)] for c, n in plan.fingerprint.class_counts],
        "labelled_fraction": plan.fingerprint.labelled_fraction,
        "patch_size": [int(v) for v in plan.patch_size],
        "stem_stride": [int(v) for v in plan.stem_stride],
        "preset": plan.preset.name,
        "features": [int(v) for v in plan.preset.features],
        "batch_size": plan.preset.batch_size,
        "reasons": list(plan.reasons),
    }
    Path(out).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def _dice_rows(
    predictor: SlidingWindowPredictor, cases: list[Case], max_cases: int | None = None
) -> dict:
    """Per-case per-class Dice against ground truth, restricted to supervised
    voxels; the aggregate the report is judged by. Shared by
    `evaluate_command`, `fit_command`'s post-fit evaluation and
    `crossval_command`'s per-fold evaluation, so the number means ONE thing
    everywhere it appears."""
    selected = cases if max_cases is None else cases[:max_cases]
    num_classes = int(predictor.net.config.num_classes)
    rows = []
    for case in selected:
        label, _ = predictor.predict(case.image)
        supervised = (
            np.ones(case.label.shape, dtype=bool)
            if case.mask is None
            else case.mask.max(axis=0) > 0
        )
        per_class = []
        for cls in range(num_classes):
            predicted = (label == cls) & supervised
            truth = (case.label == cls) & supervised
            denom = int(predicted.sum()) + int(truth.sum())
            per_class.append(
                2.0 * float((predicted & truth).sum()) / denom if denom else 0.0
            )
        foreground = float(np.mean(per_class[1:])) if num_classes > 1 else 0.0
        rows.append({
            "case_id": case.case_id,
            "per_class_dice": per_class,
            "foreground_mean": foreground,
        })
    aggregate = {
        "per_class_dice_mean": [
            float(np.mean([row["per_class_dice"][cls] for row in rows]))
            for cls in range(num_classes)
        ],
        "foreground_mean_mean": (
            float(np.mean([row["foreground_mean"] for row in rows])) if rows else 0.0
        ),
        "cases_used": len(rows),
    }
    return {"cases": rows, "aggregate": aggregate}


def evaluate_command(
    checkpoint_dir: str | Path,
    data_dir: str | Path,
    out: str | Path,
    max_cases: int | None = None,
) -> dict:
    """A bundle's per-case and aggregate Dice over a cases directory, as JSON.

    Restricted to supervised voxels where a case carries a mask: an
    unannotated voxel is unknown, and scoring against unknowns is how a
    model learns to hide findings the metric then credits it for not finding.
    """
    from medos_trainer.vanilla.infer import load_predictor

    cases = load_cases_dir(data_dir)
    report = _dice_rows(load_predictor(checkpoint_dir), cases, max_cases=max_cases)
    Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def fit_command(
    data_dir: str | Path,
    preset: str,
    out_dir: str | Path,
    *,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    resume_from: str | Path | None = None,
    max_val_cases: int = 8,
    use_amp: bool = False,
    augment_resample: bool | None = None,
    foreground_prob: float | None = None,
) -> dict:
    """The whole autonomous pipeline: data -> fingerprint -> plan -> fit -> bundle.

    `resume_from` names a bundle directory with a `training_state.pt` (every
    best checkpoint writes one): net, optimizer and scheduler are restored and
    training starts at the recorded epoch + 1, with the checkpoint still
    selected against the run-wide best — a resumed run can never regress the
    artifact. THE GENERATOR IS NOT RESUMED (see `VanillaTrainer.fit`): only a
    fresh-from-seed run replays exactly.

    `max_val_cases` caps the post-fit evaluation (the val split it trains
    against; 8 by default, None for all of it). `augment_resample` overrides
    the plan's default (on for real plans); `use_amp` enables autocast+scaler
    on CUDA devices. `foreground_prob` overrides the sampler's foreground
    bias — on a tiny corpus where background patches would let the net
    converge to "all background", pushing it toward 1.0 is the honest knob
    (every patch then carries the structure being taught).
    """
    import torch
    from medos_trainer.vanilla.infer import load_predictor
    from medos_trainer.vanilla.nets import build_unet
    from medos_trainer.vanilla.trainer import VanillaTrainer

    cases = load_cases_dir(data_dir)
    plan = _planned_run(cases, preset, epochs, steps_per_epoch, foreground_prob)
    fit_plan = plan.fit_plan()
    if augment_resample is not None or use_amp:
        fit_plan = replace(
            fit_plan,
            augment_resample=fit_plan.augment_resample
            if augment_resample is None
            else augment_resample,
            use_amp=use_amp,
        )
    torch.manual_seed(seed)
    net = build_unet(plan.network_config(input_channels=cases[0].image.shape[0]))
    trainer = VanillaTrainer(net, num_classes=plan.fingerprint.num_classes,
                             plan=fit_plan, device=device)
    resume = None
    if resume_from is not None:
        state = trainer.load_state(resume_from)
        record = json.loads(
            (Path(resume_from) / "checkpoint.json").read_text(encoding="utf-8")
        )
        resume = {**state,
                  "best_val_masked_dice_loss": float(record["val_masked_dice_loss"])}
    split = max(1, len(cases) // 5)
    train, val = cases[split:], cases[:split]
    result = trainer.fit(train, val, np.random.default_rng(seed),
                         out_dir=out_dir, resume=resume)
    evaluation = _dice_rows(load_predictor(out_dir), val, max_cases=max_val_cases)
    return {"best_val_masked_dice_loss": result["best_val_masked_dice_loss"],
            "patch_size": list(plan.patch_size),
            "preset": plan.preset.name,
            "reasons": list(plan.reasons),
            "history": result["history"],
            "evaluation": evaluation}


def crossval_command(
    data_dir: str | Path,
    preset: str,
    out_dir: str | Path,
    *,
    folds: int = 5,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    max_val_cases: int | None = None,
    foreground_prob: float | None = None,
) -> dict:
    """K-fold cross-validation over a cases directory, ONE report.

    THE FOLD SCHEME IS PART OF THE REPORT, because an aggregate nobody can
    reproduce is not a result: cases are taken in `case_id` order (the
    directory loader's order), the rank list is
    `np.random.default_rng(seed).permutation(len(cases))`, and a case's fold
    is its RANK MODULO THE FOLD COUNT. Same cases, same seed, same folds.

    ONE PLAN — fingerprinted over ALL cases, standard practice; the plan is a
    statement about the corpus, not about a split — is refit per fold FROM
    SCRATCH: `torch.manual_seed(seed + fold)` reseeds the net init and the
    fit's numpy generator comes from `default_rng(seed + fold)`. Each fold
    trains on the complement of its fold and validates on the fold itself,
    writes a full bundle at `out/fold-{k}/`, and evaluates that bundle on its
    own fold. `report.json` aggregates the per-fold best validation scores
    with the statistics module (mean and sample stdev).
    """
    import torch
    from medos_trainer.vanilla.infer import load_predictor
    from medos_trainer.vanilla.nets import build_unet
    from medos_trainer.vanilla.trainer import VanillaTrainer

    cases = load_cases_dir(data_dir)
    if folds < 2:
        raise ValueError(f"cross-validation needs at least 2 folds, got {folds}")
    if len(cases) < folds:
        raise ValueError(
            f"{folds} folds over {len(cases)} cases leaves an empty fold; "
            "choose fewer folds or more cases"
        )
    plan = _planned_run(cases, preset, epochs, steps_per_epoch, foreground_prob)
    fit_plan = plan.fit_plan()

    permutation = np.random.default_rng(seed).permutation(len(cases))
    fold_assignment = {
        cases[idx].case_id: int(rank % folds) for rank, idx in enumerate(permutation)
    }

    results = []
    for fold in range(folds):
        val = sorted(
            (cases[idx] for rank, idx in enumerate(permutation) if rank % folds == fold),
            key=lambda c: c.case_id,
        )
        val_ids = {c.case_id for c in val}
        train = [c for c in cases if c.case_id not in val_ids]
        # REFIT FROM SCRATCH, reseeded per fold: the folds must differ only
        # in their data, never in inherited weights.
        torch.manual_seed(seed + fold)
        net = build_unet(plan.network_config(input_channels=cases[0].image.shape[0]))
        trainer = VanillaTrainer(net, num_classes=plan.fingerprint.num_classes,
                                 plan=fit_plan, device=device)
        fold_dir = Path(out_dir) / f"fold-{fold}"
        result = trainer.fit(train, val, np.random.default_rng(seed + fold),
                             out_dir=fold_dir)
        evaluation = _dice_rows(load_predictor(fold_dir), val,
                                max_cases=max_val_cases)
        results.append({
            "fold": fold,
            "val_cases": [c.case_id for c in val],
            "best_val_masked_dice_loss": result["best_val_masked_dice_loss"],
            "evaluation": evaluation,
        })

    scores = [row["best_val_masked_dice_loss"] for row in results]
    report = {
        "folds": folds,
        "fold_assignment": fold_assignment,
        "results": results,
        "aggregate": {
            "mean": float(statistics.mean(scores)),
            "std": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
