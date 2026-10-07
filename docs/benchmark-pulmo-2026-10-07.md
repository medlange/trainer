# Benchmark: Medlange Trainer vs nnU-Net on PulmoAI (hydrothorax, 20 cases)

Date: 2026-10-07. Host: remote lab box, 2× NVIDIA RTX 5090 32 GB (shared with
other tenants), 48 cores, 250 GB free disk. Both frameworks ran in ONE conda
env (`python 3.11`, `torch 2.14.1+cu130` — installed by nnunetv2's resolver;
the cu128 pin is the reproducible floor, cu130 was accepted for this run),
`nnunetv2 2.8.1`, `medlange-trainer` from this repository.

## TL;DR

| | nnU-Net v2 | Medlange Trainer |
|---|---|---|
| Foreground Dice on held-out val (4 cases, one evaluator) | **0.764** | **0.000** |
| Wall clock (5 epochs) | ~11 min (63 s/epoch) | ~5.6 h incl. one OOM retry |

nnU-Net wins this budget decisively. The loss is diagnostic, not mysterious:
the vanilla pipeline lacks the two preprocessing steps nnU-Net's planner
exists to produce — intensity normalisation and resampling to a common
spacing — and at 5 epochs the difference is the whole game. What the
benchmark already forced us to fix is listed below; what it says we must
build next is in the roadmap (W16).

## Protocol (same split, same nominal budget)

- Data: `F:\WorkSpace\PulmoAI\NRRD_DATASET_HYDROTHORAX\hydrothorax_seg_100`
  (paired `N/N.nrrd` + `N/N.seg.nrrd`, CT, labels {0,1}), first 20 cases in
  numeric order, converted ONCE by `examples/benchmark_pulmo.py` into both
  frameworks' formats from the same arrays.
- Split: `assign_split` seed 0 — train 16, val = cases 9, 11, 13, 15
  (`bench/split.json`). nnU-Net trained on its own 12/4 fold split inside the
  16 (its standard cross-validation layout) and was PREDICTED on our val 4
  via `nnUNetv2_predict` — so both sides are scored on the identical 4 cases
  by the identical code path (`medos_trainer.standalone.evaluate_predictor`,
  per-class Dice over supervised voxels).
- Budget: nnU-Net `nnUNetTrainer_5epochs` (5 epochs × 250 iterations,
  planned batch 2, its own planned patch); Medlange `vanilla-fit --preset
  large --epochs 5 --steps-per-epoch 250 --seed 0`, planned patch
  (96,176,144) voxels = (75,140,140) mm physical, batch 2 → OOM'd beside a
  tenant burst → automatic fallback to batch 1 (the `--batch-size` lever
  this benchmark added).
- Metric: foreground (class-1) Dice, mean over the 4 val cases.

## Results

| case | nnU-Net fg Dice | Medlange fg Dice |
|---|---|---|
| case_0009 | 0.723 | 0.000 |
| case_0011 | 0.696 | 0.000 |
| case_0013 | 0.737 | 0.000 |
| case_0015 | 0.900 | 0.000 |
| **mean** | **0.764** | **0.000** |

Background Dice ~0.99 both sides (an all-background predictor scores that
for free). Medlange's training-time masked val metric sat at ~0.50 for all 5
epochs — the model never left the all-background attractor.

## Why (the honest analysis)

1. **No intensity normalisation.** nnU-Net's fingerprint records foreground
   intensity statistics and its preprocessor z-scores every case with them.
   Our pipeline feeds raw HU into InstanceNorm — scale-invariant in theory,
   saturating in practice — and 1,250 SGD steps never escaped the
   all-background minimum on a 0.2%-foreground task.
2. **No resampling to median spacing.** nnU-Net resamples the whole corpus
   to the planned spacing; we train at native spacing and only read spacing
   for patch geometry. A known gap, now measured.
3. **Budget realism.** Five epochs is far below either framework's real
   envelope (nnU-Net defaults to 1,000), but nnU-Net's preprocessing lets it
   learn in minutes at this budget. "Same nominal budget" is what makes the
   comparison fair; the size of the gap is what makes the missing
   preprocessing urgent.
4. **Wall clock.** Medlange's per-step cost was data-pipeline-bound
   (single-threaded patch sampling + elastic augmentation of 176³ patches);
   nnU-Net's dataloader workers kept its 5090 at 93%.

## What this benchmark already bought the framework

- Planner: per-VRAM-preset physical patch target (`VramPreset.target_patch_mm`)
  — a global 12/20/20 mm target handed a 32 GB card an 8×-smaller patch than
  nnU-Net plans (found during prep, fixed before the run).
- `augment_mirror_rotate`: rot90 is conditional on J == I (plan-shaped
  patches broke batch stacking).
- `vanilla-fit --batch-size` override (shared-GPU OOM fallback).
- `vanilla-evaluate --device` (was CPU-only).
- `pyproject [tool.pytest.ini_options] pythonpath` (CI shape pinned).
- nnU-Net env var lesson, recorded for the next operator: nnU-Net reads
  `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` (mixed case);
  `NNUNET_*` silently resolves to None.

## Reproduction

Data prep (any machine with the PulmoAI NRRD tree):

```
python examples/benchmark_pulmo.py --out bench --limit 20
```

Medlange (trainer venv):

```
python -m medos_trainer vanilla-fit --data bench/npz_train --preset large \
    --out bench/ml-bundle --epochs 5 --steps-per-epoch 250 --seed 0 --batch-size 2
python -m medos_trainer vanilla-evaluate --checkpoint-dir bench/ml-bundle \
    --data bench/npz_val --out bench/ml-eval.json --device cuda:0
```

nnU-Net (own venv; mixed-case env vars):

```
export nnUNet_raw=.../bench/nnunet_raw nnUNet_preprocessed=.../bench/preprocessed nnUNet_results=.../bench/results
nnUNetv2_plan_and_preprocess -d 501 -npfp 4 -np 4 --verify_dataset_integrity
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_5epochs
nnUNetv2_predict -d 501 -i bench/nnunet_val_images -o bench/nnpred -f 0 -tr nnUNetTrainer_5epochs -c 3d_fullres
python examples/benchmark_pulmo.py --out bench --import-nnunet-preds bench/nnpred
python bench/eval_nnunet_preds.py --bench bench
```

## Next (roadmap W16)

Fingerprint foreground-intensity statistics + dataset z-score normalisation;
resampling to median spacing in the pipeline; dataloader workers for the
patch pipeline; a 100-epoch rerun on the same split to separate
"preprocessing gap" from "convergence gap".
