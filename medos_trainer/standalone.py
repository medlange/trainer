# SPDX-License-Identifier: Apache-2.0
"""The autonomous entry: the trainer without the platform.

THE AUDIT'S VERDICT THIS ANSWERS. A researcher with a folder of volumes
could not reach a trained model: the only door in was a platform-written
run directory. This module is the second door — three loaders and two
CLI-backed commands that need nothing but data:

    vanilla-plan  --data cases/ --preset cpu --out plan.json
    vanilla-fit   --data cases/ --preset cpu --out bundle/
    vanilla-import-nnunet --images imagesTr/ --labels labelsTr/ --out cases/

The loaders live OUTSIDE `vanilla/` because the purity gate over that
package forbids every optional dependency, and importers are nothing but
optional dependencies: nibabel for NIfTI is imported lazily inside the
one function that needs it, so an environment without it pays nothing.

`--data` names a directory of `.npz` cases — the format `Case` already
speaks (`image`, `label`, optional `mask`). Spacing is read from an
optional `spacing_mm` array member, defaulting to isotropic 1 mm; for
real data pass it, because the plan's patch size is a PHYSICAL decision
and wrong spacing is how a plan quietly becomes wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medos_trainer.vanilla.data import Case, load_case_npz
from medos_trainer.vanilla.plan import PlannedRun, collect_fingerprint, plan_from_fingerprint


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


def plan_command(data_dir: str | Path, preset: str, out: str | Path) -> dict:
    """Fingerprint + plan, written as JSON with the reasons attached."""
    fingerprint = collect_fingerprint(load_cases_dir(data_dir))
    plan = plan_from_fingerprint(fingerprint, preset)
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


def fit_command(
    data_dir: str | Path,
    preset: str,
    out_dir: str | Path,
    *,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """The whole autonomous pipeline: data -> fingerprint -> plan -> fit -> bundle."""
    import torch

    from medos_trainer.vanilla.nets import build_unet
    from medos_trainer.vanilla.trainer import VanillaTrainer

    cases = load_cases_dir(data_dir)
    plan = plan_from_fingerprint(collect_fingerprint(cases), preset)
    if epochs is not None or steps_per_epoch is not None:
        plan = PlannedRun(
            fingerprint=plan.fingerprint,
            preset=plan.preset,
            patch_size=plan.patch_size,
            stem_stride=plan.stem_stride,
            steps_per_epoch=steps_per_epoch or plan.steps_per_epoch,
            epochs=epochs or plan.epochs,
            reasons=plan.reasons,
        )
    torch.manual_seed(seed)
    net = build_unet(plan.network_config(input_channels=cases[0].image.shape[0]))
    trainer = VanillaTrainer(net, num_classes=plan.fingerprint.num_classes,
                             plan=plan.fit_plan(), device=device)
    split = max(1, len(cases) // 5)
    train, val = cases[split:], cases[:split]
    result = trainer.fit(train, val, np.random.default_rng(seed), out_dir=out_dir)
    return {"best_val_masked_dice_loss": result["best_val_masked_dice_loss"],
            "patch_size": list(plan.patch_size),
            "preset": plan.preset.name,
            "reasons": list(plan.reasons)}
