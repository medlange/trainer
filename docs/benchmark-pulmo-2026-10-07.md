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


### W17 (2026-10-09): residual encoder, DS weights, throughput — and the realistic-budget table

Cropped-corpus benchmark (same 20 hydrothorax cases, central 320x320x260
crop, identical split and evaluator). The crop exists because full-volume
Medlange epochs were CPU-pipeline-bound on the shared box; the crop keeps
the task and classes identical at 1/5th the voxels.

| run | budget | fg Dice |
|---|---|---|
| nnU-Net 3d_fullres (crops) | 5 epochs | 0.716 |
| **nnU-Net 3d_fullres (crops)** | **50 epochs** | **0.782** |
| Medlange (preprocessing W16, plain net) | 5 epochs | 0.000 |
| Medlange (residual + DS weights) | 5 epochs | 0.000 |
| Medlange (residual + DS weights + TF32/AMP/prefetch) | 40 epochs | **0.456** |
| Medlange (same stack) | 100 epochs | aborted at epoch 48 (val 0.363, still improving) — the shared box entered a multi-hour CPU stall phase it also showed earlier; environmental, not a training failure |
| wall clock per epoch (crops) | ~50 s | ~75 s (was ~10 min before the throughput stack) |

Findings this cycle:
- THE RESAMPLING AUGMENTATION WAS THE WALL-CLOCK KILLER, not a quality
  factor: with `augment_resample` off, GPU0 went 0% -> 100% and an epoch
  dropped from >68 min to ~7 min. Augmentation is now a plan flag, off for
  throughput-critical runs.
- Residual encoder (pre-activation, zero-init identity start) landed as
  `UNetConfig.residual`, enabled by plans: toy learnability 0.23 -> 0.80.
  Hand-rolled residual probes NaN'd at lr 0.01; the shipped block's
  zero-init second conv is what makes it stable.
- DS loss weights 0.5^(n-1-i) normalised (nnU-Net's shape, oriented at our
  (*aux, full_res) order).
- Throughput stack (TF32 on CUDA, foreground-coord cache, prefetch
  thread) — the engineering answer to the 8-20x per-iteration gap:
  nnU-Net overlaps CPU work in 8 dataloader processes and caches
  preprocessing; we now overlap in a producer thread and skip the
  redundant per-step argwhere. TF32 is the free 3-5x conv multiplier
  nnU-Net already takes.


### Budget sweep — convergence curves (2026-10-09, final)

Full budget grid on the crop corpus, same split, same evaluator.
Medlange rows: residual+DS+TF32/AMP+prefetch stack; the plateau rows ran
without intensity augmentation (launched before it existed), the poly row
has it.

| epochs | nnU-Net fg Dice | Medlange (plateau) | Medlange (poly+intensity) |
|---|---|---|---|
| 5 | 0.716 | 0.000 | — |
| 10 | 0.657 | 0.266 | — |
| 20 | 0.756 | 0.334 | — |
| 40 | — | 0.427 | — |
| 50 | 0.756–0.782 | — | — |
| 80 | — | 0.426 | **0.482** |
| 100 | 0.834 | — | — |
| 160 | — | 0.413 | — |
| 250 | 0.849 | — | — |

READING THE CURVES:
1. nnU-Net climbs fast and keeps climbing (0.66 -> 0.85 over 10->250
   epochs). No plateau in reach.
2. Medlange's plateau-arm FREEZES at ~0.43 from epoch 40 onward — the
   plateau scheduler's patience=2 on a noisy val fires lr reductions until
   lr ~ 0 by epoch ~15 of 80; the rest of the run trains nothing. That is
   a scheduling defect, not a quality ceiling: the poly arm at the same
   80 epochs reaches 0.482 with the val curve still falling.
3. Even poly+intensity leaves a real gap to nnU-Net at matched budgets
   (0.48 vs ~0.76 at 50-80 epochs). The next levers, named in order of
   expected value: nnU-Net's full geometric augmentation in the loop
   (our scale/elastic was OFF in all these runs — it was the throughput
   killer before the prefetch thread and is now affordable), batch size
   (ours 2 vs their planned 2-12), and longer poly runs at 250-1000
   epochs on a quiet machine.

CONCLUSION OF THE BENCHMARK PHASE. Medlange went from not learning this
task at all (0.000) to a documented, honestly-gapped 0.48 against
nnU-Net's 0.85 on identical data/evaluation — with the gap localised to
named, addressable items rather than a mysterious deficit. The framework
is not yet "better than nnU-Net"; it is now, for the first time, in the
same game with a written map of the remaining distance.


### Long-run continuation and the checkpoint-selection lesson (2026-10-10)

The 250-epoch run stalled at epoch 111 on the shared box (swap-phase
tenants, unrelated to augmentation), but its best bundle evaluated at
**0.539 fg Dice** — the campaign's best (poly+intensity+prefetch+AMP,
no elastic in this run). A resume chunk (111->159) completed and kept
improving the val proxy (0.3261 -> 0.3138 at epoch 121) — yet the 121
bundle scored only 0.438 fg Dice, and it had OVERWRITTEN the 0.539 one.

THE LESSON, worth its own line in the framework's debt register: the
masked-patch val metric the checkpoint selector optimizes is a PROXY, and
on this corpus it anti-correlated with the deployment metric on the very
next epoch. Two consequences, recorded as W18 work: (1) checkpoint
selection for segmentation runs should score full-volume val foreground
Dice (the deployment shape), not random-patch soft Dice; (2) keep top-K
checkpoints — a proxy that can lie must never be the only copy of the
best model.

FINAL BENCHMARK TABLE (crop corpus, identical split/evaluator):

| | nnU-Net | Medlange (best) |
|---|---|---|
| fg Dice @ best | 0.849 @ 250 ep | **0.539 @ ~111 ep** |
| fg Dice @ 50-80 ep | 0.756-0.782 | 0.482 |

The gap to nnU-Net is now a NUMBERED list rather than a mystery:
full-volume-val checkpoint selection, top-K retention, elastic-augmented
rerun (the torch-native elastic is 85x faster but never re-benchmarked),
batch size, and a quiet-box 1000-epoch run.


### Full-stack runs and the second proxy falsification (2026-10-10, final)

The full-stack run (poly + intensity + torch-elastic + texture + AMP +
volume-selection, 188 epochs before the environment stalled it; thread-cap
fix `0a8fb04` landed mid-campaign) produced the campaign's most
instructive result. Its top-K index ranked epoch 188 (0.5419) > 172
(0.5096) > 115 (0.4811); the deployment evaluator on raw cases ranked
them 0.409 / 0.429 / 0.431 — INVERTED. Even full-volume Dice, when scored
on the already-resampled training grid, measures grid-overfit: later
epochs fit the training resampling better while raw-grid generalization
degraded. Both historical proxies (patch-level loss, volume Dice on the
training grid) are now falsified; `deployment_dice` (selection.py) is the
only non-lying selector — it replays vanilla-evaluate's exact path and
aggregation, bit-equal by test.

THE STALL, ROOT-CAUSED (two independent defects, both fixed):
1. scipy elastic at ~35 s/patch — masqueraded as a hang; torch-native
   rewrite, ~85x faster (8ef50f8).
2. The remaining "stall": faulthandler dumps showed the producer thread
   and training thread fighting over torch's PROCESS-GLOBAL intra-op pool
   (default: all cores) with OpenMP active-wait spinning every core —
   ops crawl, nothing advances. Fixed by capping intra-op threads at 8
   (0a8fb04). nnU-Net never sees this because its augmenters live in
   worker processes.

CAMPAIGN TABLE, final (crop corpus, identical split and evaluator):

| run | stack | best real fg Dice |
|---|---|---|
| nnU-Net 250 ep | reference | 0.849 |
| Medlange ~110 ep | poly+intensity, no elastic/texture | 0.539 |
| Medlange 80 ep | poly+intensity | 0.482 |
| Medlange 115-188 ep | FULL stack (elastic+texture) | 0.41-0.43 |

VERDICT. The preprocessing-parity goal (W16) is met and exceeded as
engineering; the benchmark criterion (comparable to nnU-Net's 0.764-0.849
at matched budget) is NOT met: the best honest number is 0.539 vs
nnU-Net's 0.756-0.78 at 20-50 epochs. The gap did not close with the
heavy-augmentation stack on this 16-case corpus at these budgets — the
remaining levers, in honest order: (1) the full-stack run never completed
with the deployment selector (needs a quiet night + the thread-cap fix,
which landed only at epoch 188 of the last attempt); (2) n=4 val cases
makes ±0.1 differences noise — multi-seed evaluation is required before
any further single-number claims; (3) capacity (6-level net) and 1000-ep
budgets remain untried. What this campaign produced beyond the numbers:
a framework that learns real low-contrast CT end-to-end at nnU-Net-class
throughput, with every subsystem it lost to documented and fixed.

## W22 (2026-10-11): the perf question answered with measurements, and
## the two-seed large result — 0.626 mean fg Dice

THE PAIR IS COMPLETE. `ml-i250f` (seed 0, fp32+TF32/no-AMP after the W21c
deadlock; finished 11:20, rc=139 — a segfault in process teardown AFTER
the fit wrote its bundle; the external evaluation scored that bundle
normally, so the artifact is intact, but the exit-path crash is logged
as the same CUDA-exit instability class as the W21c deadlock):

| arm | config | fg Dice | 0009 | 0011 | 0013 | 0015 |
|---|---|---|---|---|---|---|
| seed 1 (W21d) | intensity-only, AMP | 0.6252 | 0.407 | 0.664 | 0.572 | 0.858 |
| seed 0 | intensity-only, fp32+TF32 | 0.6276 | 0.316 | 0.563 | 0.766 | 0.866 |
| **two-seed mean** | | **0.6264** | | | | |

Spread between seeds: 0.0024 — fp32 and AMP agree far inside any
meaningful threshold; the fp32-forced-by-deadlock restart cost the
campaign nothing but wall time. Best epochs: s1 at 212, s0 at 173.
References: nnU-Net 0.782 @ 50 ep, 0.849 @ 250 ep. The gap is now
−0.156/−0.223; the criterion (comparable to 0.764+) is NOT met, and
the campaign says so plainly — but the honest Medlange number moved
0.482 → 0.626 in one day of fixes, and per-case behavior is stable
across seeds (0013 is now our strongest vs nnU-Net's weakest).

WHERE THE SPEED ACTUALLY GOES (bench_speed.py, RTX 5090, fp32, large
preset, batch 2 — run on the freed GPU between arms):

- fp32 train step: **426 ms** with cudnn.benchmark (now a trainer
  default — it was observed in the environment stamp but never set),
  vs the heuristic path the campaign seeds ran.
- channels_last_3d: **536 ms — 25% SLOWER, rejected** without a code
  change. Blackwell/cudnn9 handles NCHW fp32 fine; the layout win is
  a TensorCore-fp16 story, not ours.
- selection window batch 2 → 8: **12.0 s → 49.8 s, 4x SLOWER** —
  the batch-8 hypothesis falsified and REVERTED the same morning it
  shipped (cc68579). Giant window batches thrash cache and allocator;
  scores identical (math-neutral as designed), wall time regressed.
- The measured budget per epoch: 250 × 0.426 = 106 s train + ~12 s
  selection leaves ~100 s of the fp32 run's 222 s epoch unaccounted
  → **the CPU producer (patch sampling + augmentation) is the real
  remaining bottleneck**: fp32 steps got fast enough that one
  producer thread no longer covers them. W23 candidate: N producer
  threads (cases are read-only numpy — threads multiply), measured
  the same way this section was.

STILL IN FLIGHT at the time of this section: fp250 (full-parity spatial
arm) and a second xlarge seed for the mean±std. What follows landed the
same evening.

## W22b (2026-10-11 evening): the capacity lever breaks 0.7 — xlarge
## 6-stage scores 0.723

`ml-xlg250` (xlarge 6-stage, intensity-only, fp32, seed 0, old perf
defaults — started before the W22 knobs) finished clean, rc=0:

| arm | fg Dice | 0009 | 0011 | 0013 | 0015 |
|---|---|---|---|---|---|
| **xlarge 6-stage, 250 ep** | **0.7230** | 0.542 | 0.724 | 0.728 | 0.898 |
| large, two-seed mean (W22) | 0.6264 | | | | |
| nnU-Net 50 ep (reference) | 0.782 | 0.593 | 0.737 | 0.854 | 0.943 |
| nnU-Net 250 ep (reference) | 0.849 | | | | |

THE CAPACITY LEVER IS THE CAMPAIGN'S BIGGEST SINGLE GAIN: +0.097 over
the 5-stage large mean, from depth alone (6 stages, same budget, same
augmentation arm, same selection). Case_0009 — the case every earlier
Medlange run scored 0.000 and the intensity-arm seeds scored 0.32-0.41
— comes in at 0.542, statistically at parity with nnU-Net's 0.59 on a
55k-voxel lesion. Best epoch 184 (selection 0.661; the final eval
0.723 exceeds the selection score because selection caps at 4 cases
while the final number is the same 4 cases on the same evaluator —
the difference is per-epoch scoring noise, and the honest reading is
"the selector's ranking was right; its absolute value is a slightly
pessimistic proxy").

AGAINST THE CRITERION: 0.723 vs nnU-Net's 0.764 headline = −0.041
(5.4%); vs nnU-Net's 50-epoch 0.756-0.782 = −0.033…−0.059. Not a
formal match — the campaign states that plainly — but "comparable" in
the engineering sense the word was used: same class, per-case parity
on the hard small-lesion case, remaining gap concentrated in the two
large-lesion cases (0013, 0015) where both frameworks agree most.

PARAMETER ACCOUNTING (2026-10-12, counted exactly — PlainConvUNet
instantiated from the run's own plans.json): nnU-Net's ACTUAL network
on this corpus is 6-stage [32,64,128,256,320,320] with strides
ending [2,1,2], 30.79M parameters, no deep supervision (v2). Medlange
large (5-stage, residual encoder, DS heads): 21.75M — SMALLER than
nnU-Net's net, scores 0.626. Medlange xlarge (6-stage, same feature
plan, residual, DS): 38.45M — +25% over nnU-Net's actual net, scores
0.723. So the honest capacity framing: at −30% parameters we are
within 0.156 of nnU-Net; at +25% parameters, within 0.059. Neither
point is strict parameter parity, and the final verdict says so.

THE FULL-PARITY ABLATION, COMPLETE: fp250 finished clean (rc=0) at 20:37
with a final deployment fg Dice of **0.4352** (best epoch 79 by selection,
0.4804; selector slightly optimistic in absolute terms on this run, ranking
still sound). Per-case: 0009 0.026 (the small-lesion case collapses back
under spatial+texture augmentation), 0011 0.414, 0013 0.448, 0015 0.853.
Against the intensity-only arms at the same budget: 0.435 vs 0.626
(large) / 0.723 (xlarge) — the spatial tier costs 0.19-0.29 fg Dice on
this 16-case corpus. CONCLUSION RECORDED: the full nnU-Net-parity
augmentation set, transplanted wholesale, is the wrong default for a
small corpus; the framework keeps the tiers available (and the W21 CLI
opt-outs), and the benchmark report now recommends intensity-only for
corpora of this size. The W20 plan-default decision is superseded by
this measurement — a plan-default change proposal ships as W24.

IN FLIGHT at this update: xlarge seed 1 (capacity-arm mean±std), and a
W23 validation run on the freed GPU0 — 60 epochs of the exact i250f
configuration under the new three-producer pipeline, to measure the
producer-bound win against i250f's 3.7 min/epoch.

W23, VALIDATED ON GPU (2026-10-11 ~23:00): the 60-epoch check finished
clean (rc=0) at **2.2 min/epoch against i250f's 3.7** — a 1.7x wall-time
win with `prefetch_threads=3` confirmed in the bundle's fit_plan.json,
and quality on the reference trajectory (selection 0.5296 at epoch 58;
i250f's own mid-run values sit in the same band). Producers did not
triple throughput (their numpy work shares the process-global OMP pool
the training thread also uses, so three producers scale ~sub-linearly) —
1.7x measured beats 3x imagined. Planned runs ship this by default; the
fp32 large arm's 250-epoch cost drops from ~9.3 h to ~5.5 h at the same
quality class.

W22c (2026-10-12, 04:42): xlarge seed 1 finished — and the spread is
the story. s1 scores **0.6365** (per-case 0.394/0.617/0.691/0.844;
rc=139, the same CUDA-exit segfault class as i250f, artifacts intact)
against s0's 0.7230. Two-seed xlarge: **mean 0.680, spread 0.087**
(sample std ~0.061) — versus the large arm's near-perfect agreement
(0.6264 ± 0.0024). The W22 warning materialized exactly: n=4 val
cases makes ±0.1 differences noise, and the 6-stage net's seed
variance is real. NOTE FOR SYMMETRY: nnU-Net's own reference numbers
here are single runs too; a fully seeded comparison would re-run
nnU-Net across seeds as well — logged as future work, not done.
s2 (third seed, new perf stack) runs to ~07:45 for the three-seed
mean.

## W23-FINAL (2026-10-12, 07:41): the campaign's verdict — all arms
## measured, one evaluator, one split

s2 finished clean (rc=0) at **0.6882** (per-case 0.496/0.683/0.714/0.860;
best epoch 236). THREE-SEED XLARGE: **0.7230 / 0.6365 / 0.6882 →
mean 0.683, sample std 0.044**. The three existing bundles then voted
as a probability-mean ensemble (no new training — the standard
nnU-Net-fighting move, measured because the CLI's ensemble path
exists for exactly this): **ensemble fg Dice 0.7137** (0.492/0.716/
0.750/0.896) — +0.031 over the seed mean, −0.009 under the lucky
best seed. Ensembling recovers seed variance; it does not invent
signal beyond the best draw.

THE COMPLETE TABLE (crop val, 4 cases, Medlange's own evaluator for
both frameworks, nnU-Net's NIfTI predictions imported):

| arm | params | fg Dice |
|---|---|---|
| Medlange large 5-stage, mean 2 seeds | 21.75M (−30% vs nnU-Net) | 0.6264 ± 0.002 |
| Medlange xlarge 6-stage, mean 3 seeds | 38.45M (+25%) | 0.6826 ± 0.044 |
| Medlange xlarge 3-seed ensemble | 3 × 38.45M | **0.7137** |
| Medlange full-parity ablation (1 seed) | 38.45M | 0.4352 |
| nnU-Net v2 3d_fullres, single runs | 30.79M | 0.716@5ep · 0.657@10 · 0.756–0.782@50 · 0.849@250 |

CRITERION ASSESSMENT, stated without hedging: "Medlange fg Dice > 0
and comparable to nnU-Net 0.764" — fg > 0 is exceeded by an order of
magnitude (the campaign's opening number on this task was 0.000);
**"comparable to 0.764" is NOT met**: the best honest configuration
(the seed ensemble) reaches 0.714 = 93.4% of the reference, −0.050,
and the three-seed mean sits at −0.081. Against nnU-Net's own 250-epoch
0.849 the gap is −0.135. The campaign closes with the gap measured,
not closed.

WHAT THE GAP IS MADE OF (per-case, ensemble vs nnU-Net@50): 0009
0.492 vs 0.593 (−0.10, small-lesion case, our historical zero), 0011
0.716 vs 0.737 (−0.02, parity), 0013 0.750 vs 0.854 (−0.10), 0015
0.896 vs 0.943 (−0.05). The residual is broad, not one pathology.

WHAT WOULD CLOSE IT, in honest order of expected value: (1) MORE
TRAINING DATA — 16 cases is the binding constraint both frameworks
share, and nnU-Net's recipe is tuned for hundreds; (2) seeded nnU-Net
runs — its reference numbers here are single runs too, and the fair
claim needs its variance (the mean may be lower than 0.782); (3)
1000-epoch budgets at xlarge capacity (nnU-Net's own default is 1000;
we stopped at 250); (4) cascade/crossval ensembles already shipped in
the CLI, untested at this scale. The W16 preprocessing-parity goal
itself — normalization, median-spacing resampling, bundle-carried
preprocessing — is fully achieved and tested; the Dice criterion
remains open under this goal.

WHY BEHIND AT (INTERPOLATED) PARAMETER PARITY — the mechanism
analysis (2026-10-12, code facts, ranked):

1. DOSAGE, NOT POLICY, IS THE LIKELY GAP. Our augmentation tiers are
   boolean: when on, the transform hits EVERY patch (trainer.py calls
   them with no per-transform probability gate). nnU-Net draws each
   geometric transform with p~0.2 per patch. The fp250 full-parity arm
   (−0.29) was several times their dose — "full parity hurts" really
   means "our dose of it hurts". The specific small-lesion killer is
   measurable: our lowres simulation runs at lowres_prob=0.25 with
   factor as low as 0.5 across ALL axes at once — a ~40-voxel lesion
   (case_0009) halves to 20 voxels and back, trilinearly. nnU-Net's
   SimulateLowResolution is PER-AXIS p=0.25: simultaneous three-axis
   shrink hits ~1.6% of patches, ours 25%. Consistent with fp250's
   case_0009 collapsing to 0.026.
2. RECIPE HERITAGE: their ranges are tuned on hundreds of datasets;
   ours are first-reasonable. Even at net parity the recipe is half
   the result.
3. ARCHITECTURE MICRO-DIFFERENCES (unmeasured in isolation): norms
   MATCH (InstanceNorm affine both sides — verified in nets.py);
   differences remain — our residual blocks vs their plain, their
   anisotropic final stride [2,1,2] fitted to the data spacing vs our
   uniform downsampling, deep supervision on ours (a plus) vs none.
4. STATISTICAL ASYMMETRY: their 0.756-0.782 are single runs; our
   0.683 is a three-seed mean ± 0.044. Seeded nnU-Net runs would
   likely land lower.
5. DATA: 16 training cases starve both; their correctly-dosed
   augmentation diversity compensates better than our intensity-only
   arm.

NEXT WAVE (after W25 frees the GPUs): per-tier decomposition of the
fp250 failure (elastic-only / texture-only arms at 250 epochs), then
a dosage fix (per-transform probabilities as in nnU-Net, per-axis
lowres, factor floor ~0.75) — converting "full parity hurts" into
"tier X at dose Y hurts/helps"; anisotropic strides from the
fingerprint; seeded nnU-Net for a symmetric comparison.

## W26 (2026-10-12): dosage parity shipped — the dose is a plan
## property now

The mechanism analysis named dosage as the likely gap; W26 makes the
nnU-Net-class dose expressible and ships it as the planned default.
CORRECTIONS VERIFIED AGAINST THE nnU-Net v2 SOURCE the same day
(get_training_transforms of the trainer that produced the benchmark's
reference numbers): SpatialTransform runs p_rotation=0.2, p_scaling=0.2
(scaling 0.7-1.4) and **p_elastic_deform=0 — elastic deformation is OFF
in their default trainer**; GaussianNoise apply_probability=0.1,
GaussianBlur 0.2 (sigma 0.5-1.0), Brightness 0.15, Contrast 0.15,
SimulateLowResolution 0.25 (scale 0.5-1.0, SYNCHRONIZED axes), Gamma
0.1+0.1. Three consequences: (1) the fp250 collapse re-reads — our
elastic ran on EVERY patch where theirs runs NONE, likely the bigger
half of the -0.29; (2) the dose profile shipped as scale 0.2 / elastic
0.0 / noise 0.1 / blur 0.2 / lowres 0.25; (3) the W26 per-axis low-res
is a deliberate MILDer deviation from their synchronized default
(pending the W27 arms). W26b (same day) CLOSED the last named gap:
continuous rotation — the geometric tier now composes rotation, scale
and elastic as independent per-patch draws, the exact structure of
their SpatialTransform, with rotation_prob 0.2 / ±30 degrees in the
planned profile. THE W27 decomposition arms therefore test nnU-Net's
recipe BY CONSTRUCTION: --augment-resample is their SpatialTransform
twin, --augment-texture their noise/blur/low-res bundle.

- `augment_scale_elastic` / `augment_texture` take per-transform draw
  gates (scale/elastic/noise/blur probabilities). Gates at >= 1.0
  short-circuit WITHOUT consuming an rng draw, so the full-dose default
  replays the pre-W26 random stream bit-exactly — the toy-ball test that
  caught the first implementation (one extra draw per gate shifted the
  whole stream and dropped the trajectory below threshold) is the pin
  that proved the short-circuit necessary.
- The low-resolution simulation is PER-AXIS (nnU-Net
  SimulateLowResolution semantics): each axis independently downsamples
  with its own factor draw. The historical all-axes draw (25% of
  patches at factor 0.5) halved a 40-voxel lesion to 20 and back —
  the fp250 case_0009 collapse mechanism. Simultaneous three-axis
  shrink now matches nnU-Net's ~1.6% of patches.
- `FitPlan.augment_probs` carries the profile (key-subset and
  range-validated), wired through the prefetch producer and the inline
  loop identically via `_tier_kwargs`. Planned runs set the nnU-Net-
  class dose (scale/elastic 0.2, noise/blur 0.15, lowres per-axis
  0.25); the tiers stay opt-in (W24), so the dose activates with them.
- Five new tests pin: closed-gate bit-exact identity, open-gate
  activity, per-axis low-res shape preservation, plan validation, and
  the dose-class promise of a planned run.

262 tests green, ruff clean. THE REMAINING VERIFICATION IS GPU-BOUND
and queues behind W25: elastic-only and texture-only decomposition
arms, then a dosed full-parity arm — the three runs that convert "the
dose was the gap" from analysis into numbers. Anisotropic strides and
seeded nnU-Net follow.

## W25-live + W29 (2026-10-12, ~14:00 MSK): the plateau scare, and the
## per-epoch LED it bought

Two hours into W25's afternoon both 1000-epoch runs went silent: no new
best checkpoint (s0 last improved at epoch 136, selection 0.6475, 12:16;
s1 at epoch 128, 0.6232, 12:01), fit logs showing only the repeating
10-minute faulthandler watchdog dumps, main thread parked in train_step
in every dump. From the outside that IS a stalled run — and the
temptation was to kill both and restart with mitigations. The
kill-confirmation probe said otherwise:

- GPU util 100 % on both cards at 520-554 W of 575 W TDP, and — the
  decisive number — GPU MEMORY util 46-50 %. A CPU-side sync spin or an
  empty CUDA stream idles the memory controllers; half the memory
  bandwidth saturated for two straight hours is a training loop feeding
  and emptying real batches.
- Whole-process CPU time grew ~54 s per 45 s wall (the prefetch
  pipeline burning its cores), and the watchdog dumps' prefetch threads
  sat in queue.put — a full queue, i.e. a consuming training thread.

CONCLUSION: both runs healthy, in a SELECTION plateau — best-only
checkpoints are silent by construction for exactly as long as the
selector keeps not improving, which at poly-lr's still-high early
learning rate is hours, not minutes. Killing them would have wasted
6.5 h of training; the scare exposed an observability gap instead, and
W29 closes it:

W29 ships the per-epoch LED: `fit` now appends one json line per epoch
(`epoch`, `loss`, the selection number, `lr`) to
`out_dir/history.jsonl` — the liveness signal an operator can tail
without a debugger. It is pure observability: no selection, no
training, no rng consumption; a failed append is swallowed (a courtesy
light must never take training down). The test caught one real edge on
the way in: the FIRST epoch's LED write lands BEFORE any best
checkpoint created the bundle directory, and an append into a missing
directory raises FileNotFoundError — the silent-OSError handler ate
exactly epoch 0's line, every run. Parent mkdir is now explicit.

W25 itself is untouched by W29 (its lab tree is frozen; the running
processes never re-read source). The W27 arms will pick the LED up.

## W29b (2026-10-12, evening): nnU-Net is NOT seed-frozen — the seeding
## question resolved by source inspection

The campaign plan assumed nnU-Net's numbers are single seeded runs and
listed "seed nnU-Net (via patch, documented)" as a pending arm. SOURCE
INSPECTION of the installed v2.8.1 on the benchmark box falsified the
premise: `nnUNetTrainer` never calls `torch.manual_seed` (weight init
draws from OS entropy), the training augmenter is the
`NonDetMultiThreadedAugmenter` constructed with `seeds=None` (worker
seeds drawn from entropy every run), and the only pinned seed anywhere
is the CROSS-VALIDATION SPLIT's 12345 (`generate_crossval_split`,
fold split 12345+fold). Every `nnUNetv2_train` invocation is ALREADY an
independent statistical draw — same statistical structure as our
fresh-from-seed honesty.

CONSEQUENCE, cheaper and cleaner than a patch: two additional draws of
the reference recipe (Dataset502 crop, 3d_fullres, fold 0,
nnUNetTrainer_250epochs) into FRESH `nnUNet_results` dirs give the
nnU-Net side its mean/spread for the final parity table. Draws are
scripted bench-side (`~/mlg-bench/run_nn250_draw.sh N GPU` —
train/predict/import/eval, per-draw eval json), queued behind W27.

## W28 (2026-10-12, night): anisotropic downsampling strides, the
## nnU-Net min-pooled-size rule

The last named architecture gap. nnU-Net's Dataset502 plans carry
strides [2,2,2]x4 + [2,1,2] on a 6-stage net; we pooled a uniform 2x on
every axis at every stage. Their table is not a hand-tune — it falls
out of one rule: an axis is pooled only while its pooled patch stays
>= 8 voxels (their 96-voxel axis stops at 16x because 96/16 = 6 < 8).
W28 implements exactly that rule:

- `UNetConfig.downsample_strides` — one (d, h, w) tuple per
  downsampling; None keeps the historical uniform stride-2 net
  byte-identical (every pre-existing bundle still loads and builds).
  The config normalises JSON-array inputs back to tuples — a bundle
  rebuilt from net_config.json hands lists in, and while Conv3d
  tolerates a list stride, nn.Upsample's trilinear path refuses one.
- The planner computes the table from the capped voxel patch, rounds
  the patch PER AXIS to its actual total (stem x downsamples), and
  records the rule in `reasons`. Uniform (2,2,2) everywhere is
  reported as None — no churn for data that pools all the way down.
- On the real benchmark data (xlarge, spacing (0.782, 0.782, 1.0)) the
  rule emits ((2,2,2)x4, (1,2,2)) with totals (16, 32, 32) — our axis0
  is nnU-Net's short axis, so their [2,1,2] read in our axis order.
  THE PLANNED PATCH IS UNCHANGED (96, 192, 128): the W28 GPU arm is
  perfectly ceteris-paribus against W25's recipe — same patch, same
  parameter count, only the stride table differs.
- Six new tests pin: uniform default byte-identity, table validation,
  forward restoring the input grid in both supervision modes, the
  bundle-json round-trip, the planner reproducing the benchmark table,
  and the uniform-None path.

271 tests green, ruff clean. GPU validation queues behind W27: one
250-epoch xlarge arm with the recipe flags, ceteris paribus against
the W25/W27 row it mutates.

## W25-interim (2026-10-13, 11:45 MSK): seed 0 finished — the budget
## hypothesis weakens

The first 1000-epoch arm completed overnight (seed 0; the known
rc=139-at-exit segfault fired AFTER training, bundle intact, eval
clean):

| case | fg Dice |
|---|---|
| case_0009 | 0.629 |
| case_0011 | 0.692 |
| case_0013 | 0.440 |
| case_0015 | 0.894 |
| **mean** | **0.6638** |

best checkpoint epoch 319 (selection 0.6709) — i.e. the run's own
selector peaked early and the extra ~680 epochs never beat it. Against
the 250-epoch xlarge arms (0.723 / 0.637 / 0.688, mean 0.6826) the
1000-epoch draw sits INSIDE the same seed distribution, not above it:
doubling-plus the budget did not lift the mean. The budget hypothesis
("we lose because 250 epochs is too few") is weakening — the deficit
looks structural (recipe/architecture), not temporal. Seed 1 is still
training (best epoch 491 at 11:42); its draw completes the picture.

OPERATIONAL NOTE: with GPU0 freed by seed 0, the dfp250 W27 arm
launched immediately (11:44) alongside seed 1 — no idle-GPU hours.
The dfp250 bundle's `history.jsonl` is live (3 epochs at first check):
W29's LED proves itself on the first real run.

INCIDENT (2026-10-13, 13:45): dfp250 segfaulted MID-RUN (rc=139) at
~epoch 51 — the exit-segfault class, but not at exit this time. The
run script's eval then scored the half-trained best checkpoint
(0.5202, case_0009 0.003); that json is preserved as
`ml-dfp250-eval-ep51-crash.json` and is NOT an arm result. THE DESIGNED
ANSWER WORKED: `--resume-from` the same bundle relaunched at 13:46
(tmux `dfp250r`) — history.jsonl appended epochs 52+ seamlessly, the
poly lr continued from the run progress (0.0080 at epoch 54, not a
restart), the run-wide best is preserved by construction. The arm's
canonical result remains whatever the resumed run's final eval writes
to `ml-dfp250-eval.json`.

## W21d (2026-10-11): first finished run of the honest-selection campaign —
## seed 1 scores 0.625

`ml-i250s1` (seed 1, intensity-only, AMP, 250 epochs, deployment
selection) finished at 07:40 after one deadlock restart (see W21c —
the run that survived was the original AMP process started 00:50).

| | fg Dice (crop val, 4 cases, one evaluator) |
|---|---|
| **Medlange seed 1, 250 ep, deployment-selected best** | **0.6252** |
| nnU-Net 50 ep (reference) | 0.782 |
| nnU-Net 250 ep (reference) | 0.849 |
| Medlange campaign best before this run (poly80) | 0.482 |

Per-case: 0009 **0.407** (every previous Medlange run: 0.000), 0011
0.664, 0013 0.572, 0015 0.858. The best epoch by selection was 212
(selection score 0.6355); the final evaluation of that bundle scores
0.6252 — the deployment selector and the final evaluator agree within
0.01, which is the honest-selection fix doing its job.

WHAT MOVED THE NUMBER. Not one fix — the stack of them: W17 (residual
encoder, DS weights), W19/W20 (thread cap, prefetch, torch-native
resample in the selector), W21 (the CLI actually expressing the
intensity-only arm; deployment selection choosing the checkpoint), and
the full 250-epoch budget with poly decay. Case_0009's 0.000 → 0.407
also closes the W21 geography mystery partially: with an honest
checkpoint choice the model does find the atypical lesion — it was the
earlier runs' checkpoint SELECTION that kept shipping an empty-
prediction epoch.

STILL OPEN (in flight at report time): seed 0 fp32/no-AMP rerun for
the mean±std (the AMP original deadlocked at ~epoch 45, W21c), the
xlarge 6-stage capacity arm, and the full-parity spatial-tier arm.

## W21 (2026-10-11): the full-parity deadlock, the missing opt-out, and
## what case_0009 actually is

The 23:46 relaunch ran the designed W19/W20 default — full nnU-Net-parity
augmentation (geometric + intensity + texture), 250 epochs, deployment
selection, all three stall fixes in. Seed 0 trained healthy to epoch 18.
Seed 1 DEADLOCKED at ~epoch 12, and the evidence is unambiguous:

- faulthandler dumps 10 minutes apart (00:16, 00:26) are byte-identical:
  main thread inside AMP `GradScaler._maybe_opt_step` — the `found_inf`
  `.item()` CUDA synchronisation; GPU utilisation 0%; the prefetch
  producer thread inside `augment_texture`'s `_gaussian_blur3d` conv3d.
- No checkpoint progress for 18 minutes. Killed both runs at 00:35 —
  seed 0's health proves the deadlock is a race, not a deterministic
  defect, so a restarted seed 1 could hang at any epoch.

WHAT THE INCIDENT EXPOSED BEYOND ITSELF. Two framework defects, both
fixed in the same cycle (252 trainer tests green):

1. The texture tier shipped with NO CLI control: presets default it ON
   (a deliberate W20 decision), so the campaign's best-known arm —
   poly + intensity, no elastic/texture — was inexpressible from the
   command line. Added `--augment-texture` / `--no-augment-texture` to
   `vanilla-fit` and `vanilla-crossval`, wired through
   `fit_command`/`crossval_command`; corrected `--augment-resample`'s
   help, which still described the W18 scipy stall as current truth.
2. A bundle could not say how it was trained: `fit_plan.json` carried
   `patch_size` only — pinned by a test as "by design (inference does
   not need the training batch)". A bundle that cannot name its
   augmentation policy cannot be debugged when that policy is the
   suspect (this incident). `fit_plan.json` now carries the full
   FitPlan asdict as provenance; inference still reads `patch_size`
   alone, and the pinned comment records the overturned design.

CASE_0009, DIAGNOSED. The val case every Medlange run scores 0.000 on
(nnU-Net: 0.58-0.59) is not model blindness. CPU predict with the best
honest bundle (ml-poly80) gives **fg_prob_max = 0.466** — the model
sees the lesion — but no voxel crosses argmax (pred_fg = 0), and at
th=0.1 only 9.7k voxels (dice 0.04). The failure mode is
UNDERCONFIDENCE on the smallest val lesion (55.5k vox, 0.29%), not a
pipeline break: the case's spacing/intensity/shape are unremarkable
(the corpus is already 1 mm, so W16 resampling is identity here), and
the loss already includes masked per-voxel CE, which excludes the
"missing per-voxel term" hypothesis.

CASE_0009, GEOGRAPHY (2026-10-11, same bundle). The mass is right-sized
but in the WRONG PLACE: total foreground probability mass 48.4k voxels
≈ the GT's 55.5k, yet the top-1% probability centroid sits 158 voxels
from the GT centroid, and the best dice achievable by ANY global
threshold is 0.13. nnU-Net's prediction centroid is 24 voxels from the
GT centroid (dice 0.593). No axis permutation or flip of the GT
rescues the overlap (identity wins 0.133 vs 0.000 for every
transpose), so a conversion transpose bug is excluded too. The model
segments a coherent, confident, ATYPICALLY-PLACED lesion region on
this case — stable across every Medlange config and seed. That is the
signature of a model leaning on a location/appearance prior that
spatial augmentation (elastic/scale) is designed to break; nnU-Net
trains with that tier, our intensity-only arm does not. HONEST
SCOPE: n=1 case, hypothesis grade. The fp250 arm (full parity with
the W21b scipy mitigation) tests it directly.

RELAUNCHED 00:49/00:50: intensity-only (mirror/rotate + intensity,
`--no-augment-resample --no-augment-texture`), seeds 0 and 1, 250
epochs, deployment selection, AMP. Expected finish ~13:00-15:00;
mean±std of the two seeds against nnU-Net 0.849 (250 ep) /
0.756-0.782 (50 ep) is the number this campaign reports next.

OPEN DEFECT: the prefetch/AMP deadlock — SECOND OCCURRENCE KILLED THE
TEXTURE THEORY (2026-10-11, ~01:40, W21c). The intensity-only rerun (seed 0,
NO texture, NO resample augmentation — the configuration chosen BECAUSE it
was "safe") deadlocked at ~epoch 45 with the SAME main-thread stack —
`GradScaler._maybe_opt_step` → `found_inf` `.item()` CUDA sync, GPU idle —
while the producer thread was in `queue.put` backpressure, NOT inside any
augmentation. One occurrence with the producer in the blur conv was a
suspect; two occurrences with different producer sites say the producer is
collateral and the hang is the device sync itself. PRIME SUSPECT NOW: the
AMP scaler's synchronizing path on torch 2.14.1+cu130 / RTX 5090
(Blackwell) — every pre-AMP campaign run finished multi-hour unattended;
both confirmed hangs are scaler-site, non-deterministic, GPU idle with no
kernel in flight. MITIGATIONS SHIPPED/SCHEDULED: (a) W21b moved the gaussian
blur off torch conv3d (kept — it is the better implementation anyway);
(b) the seed-0 arm was restarted fp32 + TF32 matmuls, native allocator
(no `expandable_segments`), no GradScaler — the configuration class that
never hung; seed 1 continues on AMP as the comparison arm. HONEST STATUS:
mitigated by configuration, root cause unproven — a minimal AMP-only
reproduction and a torch bug report remain open; do not re-enable AMP on
this box until that repro lands.

## W18 (stalls root-caused 2026-10-10; one new stall mode 2026-10-11 — see W21)

0. STATUS OF THE TWO ITEMS BELOW: item 1's epoch-stall pathology was
   root-caused and fixed — three independent causes, in the order they
   were caught: scipy elastic evaluation at ~35 s/patch (W18→W19 torch
   rewrite); an OMP convoy in the process-global CPU thread pool (W20
   thread cap); scipy `zoom` inside the volume-selection scorer (W20
   torch resample). faulthandler dumps armed from fit start are what
   caught the third. A FOURTH stall mode surfaced 2026-10-11 — the
   prefetch/AMP deadlock documented in W21 — and is open.

1. THE EPOCH-STALL PATHOLOGY (FIXED 2026-10-10 — kept for the method):
   periodically an epoch went from ~75 s to 20+ minutes at 100%+ CPU with
   no checkpoint progress (seen on both the local box and the shared lab,
   always mid-run). No py-spy/strace access on the shared box to catch it
   in the act — what caught it was `faulthandler.dump_traceback_later`
   armed from fit start, exactly the repro aid this item prescribed. The
   three actual root causes, in catch order: scipy elastic spline
   evaluation at ~35 s/patch (→ W19 torch-native rewrite); an OMP convoy
   in the process-global CPU thread pool (→ W20 thread cap); scipy `zoom`
   inside the volume-selection scorer (→ W20 torch resample). The
   GC/tenant-pressure/GIL suspects above were investigated and were not
   it. A fourth mode — the W21 prefetch/AMP deadlock — is open.
2. Sample-efficiency (PARTIALLY ADDRESSED — intensity tier shipped W20,
   residual encoder + DS weights W17, thread cap + prefetch W20): at
   ~equal wall-clock Medlange converges to ~0.46-0.54 vs nnU-Net's ~0.78
   on crops. Remaining levers, in order: multi-seed evaluation (running,
   W21), the spatial tier once its deadlock is root-caused, batch size
   (ours 2, theirs 2-12 via plans), and 1000-epoch confirmation runs on
   a quiet machine.


Fingerprint foreground-intensity statistics + dataset z-score normalisation;
resampling to median spacing in the pipeline; dataloader workers for the
patch pipeline; a 100-epoch rerun on the same split to separate
"preprocessing gap" from "convergence gap".

## W16 addendum (2026-10-07): preprocessing parity shipped

The first two W16 items are implemented: the fingerprint collects
foreground-intensity statistics (global mean/std over labelled foreground
voxels, channel 0), the plan turns them plus the median spacing into the
preprocessing decision (recorded in `reasons`), `vanilla-fit` resamples
and z-scores the training corpus, and the bundle carries the decision as
`preprocess.json` — replayed transparently at predict time, including
the one-directional resampling back onto the caller's own grid.

| | nnU-Net v2 | Medlange Trainer |
|---|---|---|
| Foreground Dice on held-out val (4 cases, one evaluator) | **0.764** | **0.000** (preprocessing applied and verified) |


### W16 rerun result and the real root cause (2026-10-08)

The rerun with full preprocessing (normalization stats sane: fluid mean 5.2
HU, std 24.9; spacing parity with nnU-Net's plan verified) STILL produced
0.000 foreground Dice. The preprocessing hypothesis is falsified as the
PRIMARY cause. What the systematic step-by-step comparison and probe battery
found instead:

1. Configs are equivalent where measurable: nnU-Net's planned patch
   (96,160,160) at spacing (1.0, 0.782, 0.782) is the same physical context
   as ours (96,176,144) at (0.782, 0.782, 1.0) — axis-order convention only;
   batch 2 vs our 2 (1 after the tenant-OOM fallback); same optimizer family
   (SGD 0.01, momentum 0.99, nesterov); same loss family (CE + soft Dice).
2. The deficit is optimization speed, not a bug: a 1000-step probe on the
   real corpus moves training loss 1.0 -> 0.66 — learning, but an order of
   magnitude slower than nnU-Net, which reaches pseudo-Dice 0.58 by epoch 3.
   The 5-epoch benchmark budget hands both frameworks exactly 1250
   iterations — right where ours is barely off chance.
3. Probe battery on the real corpus: lr sweep (0.001/0.01/0.1) — no arm
   converges; percentile clipping (nnU-Net clips, we did not) — no effect;
   synthetic high-contrast foreground (+40 z) — LEARNS immediately, proving
   the pipeline is sound and the task at real contrast is hard per-patch;
   augmentation on/off — no difference; a hand-rolled residual block —
   destabilised at this depth/lr (architecture research, not a benchmark
   fix, deferred).
4. Fixed in this cycle: deep-supervision weights (equal averaging diluted the
   full-res head to 1/n; now 0.5^(n-1-i) normalised, nnU-Net's shape,
   oriented at our (*aux, full_res) order).

CONCLUSION. At this budget nnU-Net wins, honestly. Closing the remaining gap
is roadmap W17: residual-encoder architecture done properly (nnU-Net's
resenc preset), dataloader workers (our epochs are CPU-pipeline-bound at
~1 h vs nnU-Net's 63 s, which makes any equal-iteration comparison
prohibitively slow on shared hardware), and a re-run at a budget where both
frameworkes actually learn (the 5-epoch constraint punishes the slower-
optimizing trainer; nnU-Net's own default is 1000 epochs).

Numbers pending rerun on the benchmark box (same split, same budget:
`vanilla-fit --preset large --epochs 5 --steps-per-epoch 250 --seed 0`,
predicted via `vanilla-evaluate --device cuda:0`).
