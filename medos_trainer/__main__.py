# SPDX-License-Identifier: Apache-2.0
"""`python -m medos_trainer` — the Medlange Trainer command line.

THE SUBCOMMANDS, AND NOTHING ELSE

    vanilla-plan             fingerprint a directory of `.npz` cases and write the
                             derived plan (patch size, batch size, schedule) with the
                             reasons attached
    vanilla-fit              plan + train + write an inference bundle, no platform:
                             cases in, `model.pt` + `net_config.json` + `fit_plan.json`
                             + `preprocess.json` out
    vanilla-crossval         k-fold cross-validation: one plan, per-fold refits,
                             `fold-{k}/` bundles and a `report.json` with the fold
                             assignment and the aggregate
    vanilla-evaluate         per-case and aggregate Dice of a bundle over a cases dir
    vanilla-import-nnunet    convert an nnU-Net NIfTI layout (imagesTr/ + labelsTr/)
                             into `.npz` cases the vanilla stack reads
    vanilla-import-dicom     convert ONE DICOM series into one `.npz` case
                             (image + spacing, no label — the inference door in)
    export                   export a bundle's served net to TorchScript and/or ONNX
    predict                  run a saved bundle over one case (sliding window) —
                             or every fold bundle of a cross-validation output
                             as one probability-averaging ensemble
    declare-environment      emit `MEDOS_TRAINING_ENVIRONMENT`'s nine keys, observed
    doctor                   what this installation can and cannot do right now, as a
                             document — the answer to "is the GPU attached" that does
                             not require starting a training run

THE MODULE IMPORTS NO `medos`. The trainer is a standalone vanilla-PyTorch
framework; every input is a path on the local filesystem and every output is a
file. There is no run directory, no platform handshake and no socket — a
researcher with a folder of volumes reaches a trained model through
`vanilla-fit` alone.

THE TORCH IMPORTS ARE DELIBERATELY INSIDE THE FUNCTIONS that need them, so
`doctor` and `declare-environment` answer on a machine that can import this
package without paying for a torch import they may not need... and so the
module imports cleanly in environments where only the plan/data layers are
wanted. torch IS THE ONE DEPENDENCY THE INSTALLER CHOOSES (the CUDA build is
a property of the machine, so `medos-trainer`'s package metadata cannot pin
it — see README.md's Install section): the training subcommands therefore
REFUSE WITH A NAMED MESSAGE when torch is absent, instead of surfacing a
ModuleNotFoundError from three frames deep. `doctor` reports the same fact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

#: Subcommands that cannot do anything without torch. Guarded up front in
#: `main` so a torch-less installation gets a sentence it can act on, not a
#: traceback. `vanilla-plan` needs it too (the plan layer builds FitPlans and
#: network configs); only `doctor`, `declare-environment` and `--help` truly
#: run without it.
_TORCH_COMMANDS = {
    "vanilla-fit",
    "vanilla-crossval",
    "vanilla-evaluate",
    "vanilla-plan",
    "vanilla-import-nnunet",
    "vanilla-import-dicom",
    "predict",
    "export",
}


def _logging() -> None:
    logging.basicConfig(
        level=os.environ.get("MEDOS_TRAINER_LOG_LEVEL", "INFO").upper(),
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )


def _torch_or_refuse(command: str) -> str | None:
    """The named refusal a torch-less installation gets for training commands.

    torch is deliberately NOT a package dependency (the right build depends on
    the machine's CUDA), so the missing-dependency failure lands HERE, in the
    user's language, with the fix spelled out. Returns the refusal text, or
    None when torch imports.
    """
    if importlib.util.find_spec("torch") is not None:
        return None
    return (
        f"medos-trainer: '{command}' needs PyTorch, which is not installed in this "
        "environment. Medlange Trainer does not pin torch — install the build "
        "matching your machine's CUDA (or the CPU build):\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "See the Install section of the README."
    )


def _predict(args: argparse.Namespace) -> int:
    """The vanilla stack's standalone inference: bundle in, prediction out.

    EXISTS BECAUSE THE AUDIT NAMED ITS ABSENCE: a trainer a stranger cannot
    run a trained model with is a component, not a framework. No platform,
    no run directory, no database — a checkpoint directory from `fit`, one
    case `.npz`, one output `.npz`.

    THE PREDICTOR IS EITHER one bundle (`--checkpoint-dir`) or the ensemble
    of every fold bundle a cross-validation wrote (`--ensemble-dir` — the
    mean of the folds' probability maps). A crossval output holding no
    `fold-*/model.pt` bundles is refused, named, with exit code 2.
    """
    import numpy as np
    from medos_trainer.vanilla.data import load_case_npz
    from medos_trainer.vanilla.infer import load_ensemble, load_predictor

    # The bundle knows its TARGET spacing; only the caller knows the grid the
    # incoming volume is defined on. Our importers write a `spacing_mm`
    # member into the case npz — read it here (load_cases_dir does exactly
    # this) so a bundle with preprocessing can resample up and back, and a
    # bundle without one can ignore it.
    with np.load(args.input) as z:
        spacing = (
            tuple(float(v) for v in z["spacing_mm"])
            if "spacing_mm" in z.files
            else (1.0, 1.0, 1.0)
        )
    case = load_case_npz(args.input, spacing_mm=spacing)
    if args.ensemble_dir is not None:
        fold_dirs = sorted(
            d for d in Path(args.ensemble_dir).glob("fold-*")
            if d.is_dir() and (d / "model.pt").is_file()
        )
        if not fold_dirs:
            print(
                f"predict: --ensemble-dir {args.ensemble_dir} holds no fold-* "
                "bundles (looked for fold-*/model.pt); run vanilla-crossval "
                "there first, or pass --checkpoint-dir for a single bundle",
                file=sys.stderr,
            )
            return 2
        predictor = load_ensemble(
            [str(d) for d in fold_dirs], overlap=args.overlap,
            batch_size=args.batch_size, device=args.device,
        )
        report_path = Path(args.ensemble_dir) / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("cross-validation aggregate:")
            print(json.dumps(report.get("aggregate", {}), indent=2))
    else:
        predictor = load_predictor(
            args.checkpoint_dir, overlap=args.overlap,
            batch_size=args.batch_size, device=args.device,
        )
    label, probabilities = predictor.predict(case.image, spacing_mm=case.spacing_mm)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, label=label, probabilities=probabilities)
    print(f"wrote {out} label={label.shape} probabilities={probabilities.shape}")
    return 0


def _export(args: argparse.Namespace) -> int:
    """Bundle in, serving artefacts out — TorchScript and/or ONNX plus the
    export.json manifest that names the file, the shapes and the source
    bundle's sha256. A requested format whose optional dependency is absent
    is REFUSED UP FRONT, named, before anything is written — half an export
    directory is worse than none."""
    from medos_trainer.vanilla.export import export_bundle

    formats = ("torchscript", "onnx") if args.format == "both" else (args.format,)
    if "onnx" in formats and importlib.util.find_spec("onnx") is None:
        print(
            "medos-trainer: ONNX export needs the onnx package: pip install onnx",
            file=sys.stderr,
        )
        return 3
    written = export_bundle(args.checkpoint_dir, args.out, formats=formats,
                            device=args.device)
    for fmt, path in written.items():
        print(f"wrote {path} ({fmt})")
    print(f"wrote {Path(args.out) / 'export.json'}")
    return 0


def _declare_environment(out: str | None, allow_cpu: bool) -> int:
    from medos_trainer.environment import HardwareUnavailable, declare

    try:
        document = declare(allow_cpu=allow_cpu or None)
    except (FileNotFoundError, ValueError, HardwareUnavailable) as exc:
        # REFUSED, NAMED: an unstamped tree or a GPU-less host is a fact about this
        # installation, not a crash. The message names which of the nine keys could
        # not be observed and why.
        print(f"declare-environment: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and then moved, so a reader that opens the file while this is
        # running never sees half a declaration and reports eight of nine keys missing.
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        target.chmod(0o644)
        print(f"wrote {target}")
    print(text)
    return 0


def _doctor() -> int:
    """What this installation can do, as a document. No training run required."""
    from medos_trainer.environment import HardwareUnavailable, observe_hardware
    from medos_trainer.stamp import read_stamp

    report: dict[str, Any] = {}
    try:
        report["stamp"] = {
            k: v for k, v in read_stamp().items() if k != "image_inventory"
        }
    except (FileNotFoundError, ValueError) as exc:
        report["stamp"] = {"error": str(exc)}
    if importlib.util.find_spec("torch") is None:
        # TORCH IS THE INSTALLER'S CHOICE, and a doctor that crashed on its
        # absence would be useless on exactly the machine that needs it. The
        # report says so, plainly, and every training subcommand refuses with
        # the same fact in its message.
        report["torch"] = "not installed"
        report["hardware"] = {"error": "torch is not installed; no hardware probe ran"}
        report["can_fit"] = False
    else:
        report["torch"] = "installed"
        try:
            report["hardware"] = observe_hardware()
            report["can_fit"] = True
        except HardwareUnavailable as exc:
            report["hardware"] = {"error": str(exc)}
            report["can_fit"] = False
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("can_fit") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medos-trainer",
        description="Medlange Trainer — a vanilla-PyTorch framework for volumetric "
                    "medical image segmentation. No platform, no run directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    vanilla_plan = sub.add_parser(
        "vanilla-plan",
        help="fingerprint a cases directory and write the derived plan",
    )
    vanilla_plan.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_plan.add_argument("--preset", default="cpu")
    vanilla_plan.add_argument("--out", required=True, help="plan JSON path")

    vanilla_fit = sub.add_parser(
        "vanilla-fit",
        help="plan and train over a cases directory, no platform",
    )
    vanilla_fit.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_fit.add_argument("--preset", default="cpu")
    vanilla_fit.add_argument("--out", required=True, help="bundle directory")
    vanilla_fit.add_argument("--epochs", type=int, default=None,
                           help="override the plan's epochs (smoke runs)")
    vanilla_fit.add_argument("--steps-per-epoch", type=int, default=None)
    vanilla_fit.add_argument("--batch-size", type=int, default=None,
                             help="override the preset's batch — the lever for "
                                  "shared GPUs where the planned batch OOMs "
                                  "beside another tenant's processes")
    vanilla_fit.add_argument("--lr-schedule", choices=("plateau", "poly"),
                             default=None,
                             help="override the LR schedule. THE BENCHMARK "
                                  "FOUND THE PLATEAU ARM FIRES TOO EARLY: "
                                  "patience=2 on a noisy val anneals lr to "
                                  "~0 by epoch ~15 of 80 and the run freezes "
                                  "(0.427 at 40 and 80 epochs alike); poly "
                                  "anneals by design to the run's end")
    vanilla_fit.add_argument("--seed", type=int, default=0)
    vanilla_fit.add_argument("--device", default="cpu")
    vanilla_fit.add_argument(
        "--resume-from", default=None, metavar="BUNDLE_DIR",
        help="bundle with training_state.pt: restore net/optimizer/scheduler "
             "and continue at the recorded epoch + 1",
    )
    vanilla_fit.add_argument(
        "--max-val-cases", type=int, default=8,
        help="cap the post-fit evaluation's val cases (None = all; default 8)",
    )
    vanilla_fit.add_argument(
        "--no-augment-resample", action="store_true",
        help="disable the plan's scale/elastic augmentation (mirror/rotate stays on)",
    )
    vanilla_fit.add_argument(
        "--augment-resample", action="store_true",
        help="force scale/elastic augmentation ON. It is already a real plan's "
             "default since the W19 torch-native rewrite (~0.4 s/patch, see the "
             "benchmark report); this flag only re-affirms the plan.",
    )
    vanilla_fit.add_argument(
        "--no-augment-texture", action="store_true",
        help="disable the plan's texture augmentation (gaussian noise / blur / "
             "low-resolution simulation). The W21 benchmark rerun keeps the "
             "tier off: a prefetch-thread gaussian-blur conv3d deadlocked "
             "against AMP's CUDA sync on the benchmark box (GPU idle, two "
             "identical faulthandler dumps 10 min apart). Off until that "
             "race is root-caused.",
    )
    vanilla_fit.add_argument(
        "--augment-texture", action="store_true",
        help="force texture augmentation ON (a real plan's default). See "
             "--no-augment-texture for why the opt-out exists.",
    )
    vanilla_fit.add_argument(
        "--amp", action="store_true",
        help="autocast + GradScaler on CUDA devices (ignored on CPU)",
    )
    vanilla_fit.add_argument(
        "--foreground-prob", type=float, default=None,
        help="override the sampler's foreground bias in [0, 1] (plan default 1/3)",
    )
    vanilla_fit.add_argument(
        "--cascade-from", default=None, metavar="COARSE_BUNDLE_DIR",
        help="cascade mode: this coarse bundle predicts every case, its "
             "foreground probability becomes an extra image channel, and the "
             "fit trains the fine model on C+1 channels — writes OUT/fine/ "
             "plus OUT/cascade.json",
    )

    vanilla_crossval = sub.add_parser(
        "vanilla-crossval",
        help="k-fold cross-validation: one plan, per-fold bundles, one report",
    )
    vanilla_crossval.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_crossval.add_argument("--preset", default="cpu")
    vanilla_crossval.add_argument("--out", required=True, help="output directory")
    vanilla_crossval.add_argument("--folds", type=int, default=5)
    vanilla_crossval.add_argument("--epochs", type=int, default=None)
    vanilla_crossval.add_argument("--steps-per-epoch", type=int, default=None)
    vanilla_crossval.add_argument("--seed", type=int, default=0)
    vanilla_crossval.add_argument("--device", default="cpu")
    vanilla_crossval.add_argument(
        "--max-val-cases", type=int, default=None,
        help="cap each fold's evaluation (folds are small; None = all)",
    )
    vanilla_crossval.add_argument(
        "--foreground-prob", type=float, default=None,
        help="override the sampler's foreground bias in [0, 1] (plan default 1/3)",
    )
    vanilla_crossval.add_argument(
        "--no-augment-resample", action="store_true",
        help="disable the plan's scale/elastic augmentation in every fold",
    )
    vanilla_crossval.add_argument(
        "--augment-resample", action="store_true",
        help="force scale/elastic augmentation ON in every fold (already a "
             "real plan's default since W19)",
    )
    vanilla_crossval.add_argument(
        "--no-augment-texture", action="store_true",
        help="disable the plan's texture augmentation in every fold (see "
             "vanilla-fit's --no-augment-texture: a W21 deadlock)",
    )
    vanilla_crossval.add_argument(
        "--augment-texture", action="store_true",
        help="force texture augmentation ON in every fold (a real plan's default)",
    )

    vanilla_evaluate = sub.add_parser(
        "vanilla-evaluate",
        help="per-case and aggregate Dice of a bundle over a cases directory",
    )
    vanilla_evaluate.add_argument("--checkpoint-dir", required=True, help="bundle directory")
    vanilla_evaluate.add_argument("--data", required=True, help="directory of .npz cases")
    vanilla_evaluate.add_argument("--out", required=True, help="report JSON path")
    vanilla_evaluate.add_argument("--max-cases", type=int, default=None)
    vanilla_evaluate.add_argument("--device", default="cpu")

    vanilla_import = sub.add_parser(
        "vanilla-import-nnunet",
        help="convert an nnU-Net NIfTI layout to .npz cases",
    )
    vanilla_import.add_argument("--images", required=True, help="imagesTr/ directory")
    vanilla_import.add_argument("--labels", required=True, help="labelsTr/ directory")
    vanilla_import.add_argument("--out", required=True, help="output .npz directory")

    dicom_import = sub.add_parser(
        "vanilla-import-dicom",
        help="convert ONE DICOM series into one .npz case (image + spacing)",
    )
    dicom_import.add_argument("--series", required=True,
                              help="directory holding the series' slices")
    dicom_import.add_argument("--out", required=True, help="output .npz file path")

    export_parser = sub.add_parser(
        "export",
        help="export a bundle to TorchScript and/or ONNX serving artefacts",
    )
    export_parser.add_argument("--checkpoint-dir", required=True, help="bundle directory")
    export_parser.add_argument("--out", required=True, help="output directory")
    export_parser.add_argument("--format", choices=("torchscript", "onnx", "both"),
                               default="both")
    export_parser.add_argument("--device", default="cpu")

    declare_parser = sub.add_parser(
        "declare-environment", help="emit MEDOS_TRAINING_ENVIRONMENT's nine keys"
    )
    declare_parser.add_argument("--out", default=None)
    declare_parser.add_argument(
        "--allow-cpu", action="store_true",
        help="record a deployment with no GPU AS one, instead of refusing",
    )

    sub.add_parser("doctor", help="what this installation can and cannot do right now")

    predict_parser = sub.add_parser(
        "predict",
        help="run a saved bundle — or a cross-validation's fold ensemble — "
             "over one case (sliding window)",
    )
    predict_source = predict_parser.add_mutually_exclusive_group(required=True)
    predict_source.add_argument(
        "--checkpoint-dir", default=None,
        help="directory fit wrote (model.pt, net_config.json, fit_plan.json)",
    )
    predict_source.add_argument(
        "--ensemble-dir", default=None, metavar="CROSSVAL_DIR",
        help="a vanilla-crossval output: every fold-*/ bundle votes, the "
             "prediction is the mean of their probability maps",
    )
    predict_parser.add_argument("--input", required=True, help="case .npz (image)")
    predict_parser.add_argument("--output", required=True, help="output .npz")
    predict_parser.add_argument("--overlap", type=float, default=0.5)
    predict_parser.add_argument("--batch-size", type=int, default=2)
    predict_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    _logging()
    args = build_parser().parse_args(argv)
    if args.command != "doctor":
        refusal = _torch_or_refuse(args.command)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 3
    if args.command == "declare-environment":
        return _declare_environment(args.out, args.allow_cpu)
    if args.command == "predict":
        return _predict(args)
    if args.command == "export":
        return _export(args)
    if args.command == "vanilla-plan":
        from medos_trainer.standalone import plan_command

        plan_command(args.data, args.preset, args.out)
        print(f"wrote {args.out}")
        return 0
    if args.command == "vanilla-fit":
        from medos_trainer.standalone import fit_command

        summary = fit_command(
            args.data, args.preset, args.out,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            seed=args.seed, device=args.device,
            resume_from=args.resume_from, max_val_cases=args.max_val_cases,
            use_amp=args.amp,
            augment_resample=(True if args.augment_resample
                              else False if args.no_augment_resample else None),
            augment_texture=(True if args.augment_texture
                             else False if args.no_augment_texture else None),
            foreground_prob=args.foreground_prob, batch_size=args.batch_size,
            lr_schedule=args.lr_schedule,
            cascade_from=args.cascade_from,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "vanilla-crossval":
        from medos_trainer.standalone import crossval_command

        report = crossval_command(
            args.data, args.preset, args.out,
            folds=args.folds, epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch, seed=args.seed, device=args.device,
            max_val_cases=args.max_val_cases, foreground_prob=args.foreground_prob,
            augment_resample=(True if args.augment_resample
                              else False if args.no_augment_resample else None),
            augment_texture=(True if args.augment_texture
                             else False if args.no_augment_texture else None),
        )
        print(json.dumps(report["aggregate"], indent=2))
        print(f"wrote {Path(args.out) / 'report.json'}")
        return 0
    if args.command == "vanilla-evaluate":
        from medos_trainer.standalone import evaluate_command

        report = evaluate_command(
            args.checkpoint_dir, args.data, args.out, max_cases=args.max_cases,
            device=args.device,
        )
        print(json.dumps(report["aggregate"], indent=2))
        print(f"wrote {args.out}")
        return 0
    if args.command == "vanilla-import-nnunet":
        from medos_trainer.standalone import import_nnunet_dataset

        n = import_nnunet_dataset(args.images, args.labels, args.out)
        print(f"converted {n} cases into {args.out}")
        return 0
    if args.command == "vanilla-import-dicom":
        from medos_trainer.standalone import import_dicom_series

        import_dicom_series(args.series, args.out)
        print(f"wrote {args.out}")
        return 0
    return _doctor()


if __name__ == "__main__":  # pragma: no cover - a process entry point
    raise SystemExit(main())
