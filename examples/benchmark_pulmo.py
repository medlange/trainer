# SPDX-License-Identifier: Apache-2.0
"""Head-to-head benchmark data prep: PulmoAI hydrothorax -> both frameworks.

THE BENCHMARK THIS FEEDS. Medlange Trainer (vanilla, CPU) vs nnU-Net v2 (CPU)
on 20 real CT cases from ``F:\\WorkSpace\\PulmoAI`` (hydrothorax_seg_100:
per-case ``N/N.nrrd`` + ``N/N.seg.nrrd``). Same split, same iteration budget,
same Dice evaluator. The full protocol and the numbers live in
``trainer/docs/benchmark-pulmo-2026-10-06.md``.

WHAT THIS SCRIPT IS, AND WHAT IT IS NOT. It is a CONVERTER plus a thin
orchestrator. It imports NOTHING from ``medos_trainer`` — the Medlange phase
runs the CLI as a subprocess, and nnU-Net's phase is printed as commands for
its own interpreter (``/tmp/nnunet-venv``). The one piece that must import
the trainer — evaluating nnU-Net's predictions with the trainer's own
``_dice_rows`` code path — is a separate driver the report reproduces in
full; an example must not be coupled to the package it demonstrates.

    convert (default)                 NRRD -> Medlange .npz train/val
                                      + nnU-Net raw Dataset501 + val NIfTIs,
                                      split.json, verification, RUNS.md;
                                      then runs the Medlange phase via
                                      subprocess (fit + evaluate, timed)
    --convert-only                    conversion, split.json, RUNS.md — no
                                      training subprocesses (the CI/dry-run
                                      shape; safe on any machine)
    --import-nnunet-preds PREDS_DIR   nnU-Net's NIfTI predictions + the val
                                      NIfTIs -> <out>/nnunet_eval_npz cases
                                      (image/label/spacing_mm/pred), ready
                                      for the evaluation driver

THE SPLIT is ``assign_split(case_ids, seed)`` — a pure function, unit-tested
in ``trainer/tests/test_benchmark_script.py`` (determinism, disjointness,
coverage) so the real-data run stays out of CI.

Run it (monorepo venv, from anywhere):

    python trainer/examples/benchmark_pulmo.py --src <cases-root> --out trainer/bench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: nnU-Net dataset id this benchmark converts into.
DATASET_ID = 501
DATASET_NAME = "Dataset501_PulmoHydro"

#: The benchmark's training budget. nnU-Net's short trainer
#: (``nnUNetTrainer_5epochs`` in nnunetv2 2.8.1, ``nnUNetTrainer_5epoch`` in
#: 2.6.x) runs 5 epochs x 250 iterations = 1250; Medlange is driven with the
#: same 5 x 250 so both sides train the SAME iteration count. The benchmark
#: report states this where the original brief said 100 steps/epoch.
EPOCHS = 5
STEPS_PER_EPOCH = 250
SEED = 0


def assign_split(
    case_ids: list[str] | tuple[str, ...],
    seed: int,
    val_fraction: float = 0.2,
) -> dict[str, str]:
    """Deterministic 80/20 train/val assignment over ``case_ids``.

    Pure function of (ids, seed, fraction): ids are sorted, permuted by
    ``np.random.default_rng(seed)``, and the first ``round(n * val_fraction)``
    of the permutation (at least one case, at most n - 1) become "val";
    the rest become "train". Same inputs, same dict — that is the property
    the unit test pins, because the real-data run is too heavy for CI.
    """
    ids = sorted(case_ids)
    if not ids:
        raise ValueError("assign_split needs at least one case id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"case ids must be unique, got {ids}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction in (0, 1), got {val_fraction}")
    n_val = int(round(len(ids) * val_fraction))
    n_val = max(1, min(n_val, len(ids) - 1))
    permutation = np.random.default_rng(seed).permutation(len(ids))
    val_ranks = set(permutation[:n_val].tolist())
    return {
        case_id: ("val" if rank in val_ranks else "train")
        for rank, case_id in enumerate(ids)
    }


def _nrrd():
    """pynrrd, lazy and named — the conversion box may lack it."""
    try:
        import nrrd
    except ImportError as exc:  # pragma: no cover - depends on the ambient env
        raise RuntimeError("reading NRRD cases needs pynrrd: pip install pynrrd") from exc
    return nrrd


def _nibabel():
    """nibabel, lazy and named — same discipline as the trainer's importers."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on the ambient env
        raise RuntimeError("writing NIfTI needs nibabel: pip install nibabel") from exc
    return nib


def discover_cases(src: Path) -> tuple[list[str], dict[str, str]]:
    """Numeric-order VALID case ids under ``src`` — ``N/N.nrrd`` + ``N/N.seg.nrrd``.

    Non-numeric sibling directories (``__MACOSX``, exports) are skipped. A
    numeric directory missing either file, or whose image/seg shapes disagree
    (the PulmoAI collection ships a few broken pairs — 3, 73, 75 at survey
    time), is SKIPPED, loudly, and recorded in the returned dict; a benchmark
    converter curates by naming, it does not crash half a corpus for the
    dataset's sins.
    """
    nrrd = _nrrd()
    cases = []
    skipped: dict[str, str] = {}
    for child in src.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        image_path = child / f"{child.name}.nrrd"
        label_path = child / f"{child.name}.seg.nrrd"
        if not image_path.is_file():
            skipped[child.name] = "missing N.nrrd"
            continue
        if not label_path.is_file():
            skipped[child.name] = "missing N.seg.nrrd"
            continue
        try:
            image_header = nrrd.read_header(str(image_path))
            label_header = nrrd.read_header(str(label_path))
        except Exception as exc:  # noqa: BLE001 - unreadable files are data noise
            skipped[child.name] = f"unreadable NRRD: {exc}"
            continue
        if tuple(image_header["sizes"]) != tuple(label_header["sizes"]):
            skipped[child.name] = (
                f"image {tuple(image_header['sizes'])} != seg {tuple(label_header['sizes'])}"
            )
            continue
        cases.append(child.name)
    if not cases:
        raise ValueError(f"{src} holds no usable per-case N/<id>.nrrd + N/<id>.seg.nrrd dirs")
    return sorted(cases, key=int), skipped


def _spacing_mm(header: dict, case_id: str) -> tuple[float, float, float]:
    """Per-axis voxel spacing from the NRRD header, in array axis order."""
    directions = header.get("space directions")
    if directions is not None:
        spacing = [
            float(np.linalg.norm(row)) if row is not None else None for row in directions
        ]
        if all(s is not None and s > 0 for s in spacing) and len(spacing) == 3:
            return (spacing[0], spacing[1], spacing[2])
    spacings = header.get("spacings")
    if spacings is not None and all(s is not None and float(s) > 0 for s in spacings):
        return tuple(float(s) for s in spacings[:3])
    raise ValueError(f"case {case_id}: no usable spacing in the NRRD header")


def _renumber_contiguous(label: np.ndarray) -> np.ndarray:
    """Every unique value mapped to its rank: {0,1} stays {0,1}; any sparse
    label set becomes contiguous from 0."""
    out = np.zeros(label.shape, dtype=np.int64)
    for new, old in enumerate(int(v) for v in np.unique(label)):
        out[label == old] = new
    return out


def read_case(
    src: Path, case_id: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """One case: image (1,K,J,I) float32 (RAW stored values), label int64,
    spacing_mm (K, J, I) from the NRRD ``space directions``."""
    nrrd = _nrrd()
    image_path = src / case_id / f"{case_id}.nrrd"
    label_path = src / case_id / f"{case_id}.seg.nrrd"
    image, header = nrrd.read(str(image_path))
    label, _ = nrrd.read(str(label_path))
    if label.shape != image.shape:
        raise ValueError(
            f"case {case_id}: label {label.shape} != image {image.shape}"
        )
    spacing = _spacing_mm(header, case_id)
    return (
        np.asarray(image, dtype=np.float32)[None],
        _renumber_contiguous(np.asarray(label)),
        spacing,
    )


def write_npz_case(out: Path, case_name: str, image: np.ndarray,
                   label: np.ndarray, spacing: tuple[float, float, float]) -> Path:
    path = out / f"{case_name}.npz"
    np.savez(
        path,
        image=image,
        label=label,
        spacing_mm=np.asarray(spacing, dtype=np.float64),
    )
    return path


def write_nifti_pair(image_dir: Path, label_dir: Path | None, case_name: str,
                     image: np.ndarray, label: np.ndarray,
                     spacing: tuple[float, float, float]) -> None:
    """NIfTI in ARRAY axis order: affine diagonal = (sK, sJ, sI). Both the
    trainer's importer and this script's prediction reader treat ``dataobj``
    as file order, so the round trip is self-consistent end to end."""
    nib = _nibabel()
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    image_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(image[0]), affine),
             str(image_dir / f"{case_name}_0000.nii.gz"))
    if label_dir is not None:
        label_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(label.astype(np.uint8)), affine),
                 str(label_dir / f"{case_name}.nii.gz"))


def convert(src: Path, out: Path, limit: int, seed: int) -> dict:
    """The whole conversion; returns the split document it wrote."""
    usable, skipped = discover_cases(src)
    case_ids = usable[:limit]
    assignment = assign_split(case_ids, seed)
    split = {
        "source": str(src),
        "seed": seed,
        "val_fraction": 0.2,
        "limit": limit,
        "selection": f"first {limit} valid cases in numeric order",
        "train": sorted((c for c in case_ids if assignment[c] == "train"), key=int),
        "val": sorted((c for c in case_ids if assignment[c] == "val"), key=int),
        "skipped": skipped,
    }
    print(f"split (seed={seed}): {len(split['train'])} train / {len(split['val'])} val")
    print(f"  val: {split['val']}")
    if skipped:
        print(f"  skipped {len(skipped)} broken cases: {skipped}")

    npz_train = out / "npz_train"
    npz_val = out / "npz_val"
    nnunet_raw = out / "nnunet_raw" / DATASET_NAME
    val_images = out / "nnunet_val_images"
    val_labels = out / "nnunet_val_labels"
    for d in (npz_train, npz_val, nnunet_raw / "imagesTr", nnunet_raw / "labelsTr",
              val_images, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        name = f"case_{int(case_id):04d}"
        image, label, spacing = read_case(src, case_id)
        if assignment[case_id] == "train":
            write_npz_case(npz_train, name, image, label, spacing)
            write_nifti_pair(nnunet_raw / "imagesTr", nnunet_raw / "labelsTr",
                             name, image, label, spacing)
        else:
            write_npz_case(npz_val, name, image, label, spacing)
            write_nifti_pair(val_images, val_labels, name, image, label, spacing)
        print(f"  converted case {case_id} -> {name} "
              f"shape={tuple(image.shape)} spacing={spacing} [{assignment[case_id]}]")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "hydrothorax": 1},
        "numTraining": len(split["train"]),
        "file_ending": ".nii.gz",
    }
    (nnunet_raw / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    split_path = out / "split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    verify(out, split)
    print(f"wrote {split_path}")
    return split


def verify(out: Path, split: dict) -> None:
    """Re-open one artifact per format so a corrupt write never reaches the
    benchmark silently. Refuses, named, on the first inconsistency."""
    npz = np.load(out / "npz_train" / f"case_{int(split['train'][0]):04d}.npz")
    image, label = npz["image"], npz["label"]
    if image.ndim != 4 or image.shape[1:] != label.shape:
        raise ValueError(f"verification failed: npz image {image.shape} vs label {label.shape}")
    if "spacing_mm" not in npz.files or len(npz["spacing_mm"]) != 3:
        raise ValueError("verification failed: npz carries no spacing_mm")
    nib = _nibabel()
    nifti = nib.load(str(out / "nnunet_raw" / DATASET_NAME / "imagesTr"
                         / f"case_{int(split['train'][0]):04d}_0000.nii.gz"))
    if tuple(nifti.dataobj.shape) != tuple(image.shape[1:]):
        raise ValueError(
            f"verification failed: nifti {nifti.dataobj.shape} vs npz {image.shape[1:]}"
        )
    print("verification: npz and NIfTI round-trip shapes consistent")


def write_runs_md(out: Path, split: dict) -> None:
    """The exact benchmark commands, written beside the data they use."""
    env = (f"NNUNET_RAW={out}/nnunet_raw "
           f"NNUNET_PREPROCESSED={out}/preprocessed "
           f"NNUNET_RESULTS={out}/results")
    lines = [
        "# Benchmark runbook (generated by benchmark_pulmo.py)",
        "",
        "Medlange (monorepo venv, run from the trainer root):",
        "",
        "```",
        f"{sys.executable} -m medos_trainer vanilla-fit --data {out}/npz_train "
        f"--preset cpu --out {out}/ml-bundle --epochs {EPOCHS} "
        f"--steps-per-epoch {STEPS_PER_EPOCH} --seed {SEED}",
        f"{sys.executable} -m medos_trainer vanilla-evaluate "
        f"--checkpoint-dir {out}/ml-bundle --data {out}/npz_val "
        f"--out {out}/ml-eval.json",
        "```",
        "",
        "nnU-Net (its own venv; the short trainer class is nnUNetTrainer_5epochs",
        "in nnunetv2 2.8.1, nnUNetTrainer_5epoch in 2.6.x):",
        "",
        "```",
        f"{env} nnUNetv2_plan_and_preprocess -d {DATASET_ID} -npfp 4 -np 4 "
        f"--verify_dataset_integrity",
        f"OMP_NUM_THREADS=4 {env} nnUNetv2_train {DATASET_ID} 3d_fullres 0 "
        f"-tr nnUNetTrainer_5epochs --npz -device cpu",
        f"{env} nnUNetv2_predict -d {DATASET_ID} -c 3d_fullres -f 0 "
        f"-tr nnUNetTrainer_5epochs -i {out}/nnunet_val_images -o {out}/nnpred "
        f"-device cpu",
        "```",
        "",
        "Evaluation of the nnU-Net predictions with the trainer's own",
        "`_dice_rows` code path — the driver source is in the report",
        "(`trainer/docs/benchmark-pulmo-2026-10-06.md`), run from the trainer",
        "root:",
        "",
        "```",
        f"{sys.executable} {out}/eval_nnunet_preds.py --bench {out}",
        "```",
        "",
        f"Split: seed {split['seed']}, val = {split['val']}.",
        "",
    ]
    (out / "RUNS.md").write_text("\n".join(lines), encoding="utf-8")


def run_medlange_phase(out: Path) -> dict:
    """The Medlange half of the benchmark, driven as CLI subprocesses so this
    example never imports the package. Times both subprocesses."""
    trainer_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(trainer_root) + os.pathsep + env.get("PYTHONPATH", "")
    timing: dict[str, float] = {}

    fit_cmd = [sys.executable, "-m", "medos_trainer", "vanilla-fit",
               "--data", str(out / "npz_train"), "--preset", "cpu",
               "--out", str(out / "ml-bundle"),
               "--epochs", str(EPOCHS), "--steps-per-epoch", str(STEPS_PER_EPOCH),
               "--seed", str(SEED)]
    print("+", " ".join(fit_cmd))
    start = time.monotonic()
    fit = subprocess.run(fit_cmd, cwd=trainer_root, env=env)
    timing["fit_wall_seconds"] = round(time.monotonic() - start, 1)
    if fit.returncode != 0:
        raise SystemExit(f"vanilla-fit failed with exit code {fit.returncode}")

    eval_cmd = [sys.executable, "-m", "medos_trainer", "vanilla-evaluate",
                "--checkpoint-dir", str(out / "ml-bundle"),
                "--data", str(out / "npz_val"), "--out", str(out / "ml-eval.json")]
    print("+", " ".join(eval_cmd))
    start = time.monotonic()
    ev = subprocess.run(eval_cmd, cwd=trainer_root, env=env)
    timing["evaluate_wall_seconds"] = round(time.monotonic() - start, 1)
    if ev.returncode != 0:
        raise SystemExit(f"vanilla-evaluate failed with exit code {ev.returncode}")

    (out / "ml-timing.json").write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Medlange phase done: {json.dumps(timing)}")
    return timing


def import_nnunet_predictions(preds: Path, out: Path) -> list[Path]:
    """nnU-Net's NIfTI predictions -> .npz cases the trainer evaluator reads.

    One npz per val case: image/label/spacing_mm (from the val NIfTIs this
    script exported — the same files nnU-Net predicted from) plus the extra
    ``pred`` member the evaluation driver serves back through the trainer's
    own `_dice_rows` code path.
    """
    written = []
    eval_dir = out / "nnunet_eval_npz"
    eval_dir.mkdir(parents=True, exist_ok=True)
    nib = _nibabel()
    val_images = sorted((out / "nnunet_val_images").glob("case_*_0000.nii.gz"))
    if not val_images:
        raise ValueError(
            f"{out / 'nnunet_val_images'} holds no val NIfTIs; run the conversion first"
        )
    for image_path in val_images:
        name = image_path.name[: -len("_0000.nii.gz")]
        pred_path = preds / f"{name}.nii.gz"
        if not pred_path.is_file():
            raise ValueError(f"nnU-Net prediction for {name} missing from {preds}")
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(out / "nnunet_val_labels" / f"{name}.nii.gz"))
        pred_nii = nib.load(str(pred_path))
        image = np.asarray(image_nii.dataobj, dtype=np.float32)[None]
        label = np.asarray(label_nii.dataobj, dtype=np.int64)
        pred = np.asarray(pred_nii.dataobj, dtype=np.int64)
        if pred.shape != label.shape or image.shape[1:] != label.shape:
            raise ValueError(
                f"{name}: pred {pred.shape} / image {image.shape} / "
                f"label {label.shape} disagree"
            )
        spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])
        path = eval_dir / f"{name}.npz"
        np.savez(path, image=image, label=label,
                 spacing_mm=np.asarray(spacing, dtype=np.float64), pred=pred)
        written.append(path)
        print(f"  {name}: pred foreground {int((pred == 1).sum())} voxels "
              f"(truth {int((label == 1).sum())})")
    print(f"wrote {len(written)} eval cases into {eval_dir}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_pulmo",
        description="Convert the PulmoAI hydrothorax NRRD corpus into Medlange "
                    ".npz cases and an nnU-Net raw dataset, drive the Medlange "
                    "half of the CPU benchmark, and (later) package nnU-Net's "
                    "predictions for evaluation with the trainer's own "
                    "evaluator. Imports nothing from medos_trainer.",
    )
    parser.add_argument("--src", type=Path,
                        default=Path(r"F:\WorkSpace\PulmoAI\NRRD_DATASET_HYDROTHORAX"
                                     r"\hydrothorax_seg_100\hydrothorax_seg_100"),
                        help="per-case NRRD root (N/N.nrrd + N/N.seg.nrrd)")
    parser.add_argument("--out", type=Path, required=True,
                        help="benchmark output directory (bench/)")
    parser.add_argument("--limit", type=int, default=20,
                        help="how many cases (numeric order) to take (default 20)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="split seed, written to split.json (default 0)")
    parser.add_argument("--convert-only", action="store_true",
                        help="convert and write split.json/RUNS.md, but do NOT "
                             "launch the Medlange fit/evaluate subprocesses "
                             "(the dry-run mode for CI and data prep)")
    parser.add_argument("--import-nnunet-preds", type=Path, default=None,
                        metavar="PREDS_DIR",
                        help="convert nnUNetv2_predict's NIfTI output into "
                             "<out>/nnunet_eval_npz for the evaluation driver")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_nnunet_preds is not None:
        import_nnunet_predictions(args.import_nnunet_preds, args.out)
        return 0
    split = convert(args.src, args.out, args.limit, args.seed)
    write_runs_md(args.out, split)
    print(f"wrote {args.out / 'RUNS.md'}")
    if args.convert_only:
        print("--convert-only: conversion done; training commands are in RUNS.md")
        return 0
    run_medlange_phase(args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - a script entry point
    raise SystemExit(main())
