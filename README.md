# Medlange Trainer

**A vanilla-PyTorch framework for volumetric medical image segmentation.**

Medlange Trainer is an nnU-Net-class trainer written from scratch: its own
3D UNet with deep supervision, its own fingerprint-based planning, its own
masked-loss training loop and its own sliding-window inference. It replaces
nnU-Net and MONAI — neither is installed, imported or required anywhere in
this tree — and it has no platform dependency: every input is a path on the
local filesystem and every output is a file.

A researcher with a folder of volumes reaches a trained model through one
command (`vanilla-fit`), evaluates it through another (`vanilla-evaluate`),
and ships it through a third (`export`). There is no web surface, no socket
and no database.

## Install

From this directory:

```sh
pip install -e .                    # the framework: numpy + scipy floor
pip install -e ".[nifti,dicom]"     # + the optional importers' dependencies
```

Then install torch YOURSELF, the build matching your machine — this is
deliberate, and it is why torch is not a package dependency:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cpu       # CPU
```

A CUDA build installed on a laptop with no NVIDIA anything is a broken
environment, and only the installer knows which machine this is. The
training subcommands refuse with a named message when torch is absent
(telling you exactly what to install), and `doctor` reports the same fact
in its document.

The optional extras mirror the lazy-import discipline: nibabel exists only
inside `vanilla-import-nnunet`, pydicom only inside
`vanilla-import-dicom`, so an environment without them pays nothing.

For reproducible environments (the training image, CI) `requirements.txt`
pins the exact development set — torch 2.7.1+cu128 included.

## Multi-GPU (DDP)

A fit runs on one device by default; with more than one GPU (or more than
one training process), launch the SAME command through `torchrun` — no CLI
flag changes:

```sh
torchrun --nproc_per_node=2 -m medos_trainer vanilla-fit \
    --data cases/ --preset cpu --out bundle/ --device cuda:0
```

`torchrun` sets `WORLD_SIZE`/`RANK`/`MASTER_ADDR`/`MASTER_PORT`; the trainer
sees a real world, initializes the process group (NCCL on CUDA, Gloo on
CPU) and wraps the net in `DistributedDataParallel`. `--device cuda:0` is
unchanged on purpose: `torchrun` isolates each process's
`CUDA_VISIBLE_DEVICES`, so every rank's `cuda:0` is its own GPU.

THE CONTRACT UNDER DDP: rank 0 validates, selects checkpoints, owns the
printed history and writes the bundle (checkpoints carry no `module.`
prefix and load exactly like a single-process run's); every rank trains.
Each rank's patch stream is seeded `seed+rank`, so no two ranks draw the
same patches — the sampler draws WITH REPLACEMENT, so overlapping draws are
expected and harmless, and the ranks' gradient all-reduce averages the
differences exactly as a larger single-process batch would. The lr is
identical on every rank at every epoch boundary: the poly law is a pure
function of progress, and the plateau law's scheduled value is broadcast
from rank 0 with the gradients. The CPU/Gloo world is what the test suite's
real two-process run exercises (`MEDOS_RUN_DDP=1 pytest tests/
test_distributed.py`).

## The case format

Everything the framework reads is a directory of `.npz` cases. One case, one
file:

| member | shape | meaning |
|---|---|---|
| `image` | (C, K, J, I) float32 | the volume, already resampled to the spacing `spacing_mm` names; C = 1 for plain CT, more for multi-modal data |
| `label` | (K, J, I) int | the segmentation; 0 is background, positive values are mutually exclusive classes |
| `mask` | (C, K, J, I) float, optional | 1 where a channel's loss counts, 0 where the case does not annotate it |
| `spacing_mm` | (3,) float, optional | voxel spacing in mm, (K, J, I) order; defaults to isotropic 1 mm |

`vanilla-import-nnunet` converts an nnU-Net NIfTI layout (an imagesTr
directory and a labelsTr directory) into this format, preserving spacing
from the NIfTI zooms. `vanilla-import-dicom` converts one DICOM series into
a single unlabelled case — the door in for inference.

## The CLI

`python -m medos_trainer <subcommand>` (or `medos-trainer <subcommand>`
after a pip install) — ten of them, and nothing else:

```sh
# fingerprint a cases directory and write the derived plan, reasons attached
python -m medos_trainer vanilla-plan --data cases/ --preset cpu --out plan.json

# plan + train + write an inference bundle (model.pt, net_config.json,
# fit_plan.json, checkpoint.json, training_state.pt, preprocess.json) + evaluate it
python -m medos_trainer vanilla-fit --data cases/ --preset cpu --out bundle/ \
    --epochs 50 --steps-per-epoch 100 --seed 0 --device cpu

# k-fold cross-validation: one plan, per-fold bundles, one report.json
python -m medos_trainer vanilla-crossval --data cases/ --preset cpu \
    --out crossval/ --folds 5

# per-case and aggregate Dice of a trained bundle over a cases directory
python -m medos_trainer vanilla-evaluate --checkpoint-dir bundle/ \
    --data cases/ --out evaluation.json

# bring a corpus prepared for nnU-Net into the case format
python -m medos_trainer vanilla-import-nnunet \
    --images imagesTr/ --labels labelsTr/ --out cases/

# bring ONE DICOM series in as an unlabelled case (image + spacing)
python -m medos_trainer vanilla-import-dicom --series dicom/ --out case.npz

# export the bundle's served net to TorchScript and/or ONNX
python -m medos_trainer export --checkpoint-dir bundle/ --out serving/ --format both

# run a trained bundle over one case, sliding window — or the fold
# ensemble of a cross-validation output (mean of the folds' probability maps)
python -m medos_trainer predict --checkpoint-dir bundle/ \
    --input cases/case_000.npz --output pred.npz --overlap 0.5 --batch-size 2
python -m medos_trainer predict --ensemble-dir crossval/ \
    --input cases/case_000.npz --output pred.npz

# observe the nine MEDOS_TRAINING_ENVIRONMENT keys (refuses without a GPU
# unless --allow-cpu)
python -m medos_trainer declare-environment --out environment.json

# what this installation can and cannot do, as a document
python -m medos_trainer doctor
```

### vanilla-fit's switches

`vanilla-fit` carries the plan from `vanilla.plan` (median spacing and
shape in, patch size, stem stride, batch size and schedule out, every
choice with the reason attached) and never re-derives it. On top of the
plan:

- `--resume-from BUNDLE_DIR` continues a run from its best checkpoint's
  `training_state.pt`: net, optimizer and scheduler come back, training
  starts at the recorded epoch + 1, and the artifact can only improve on
  the run-wide best. The random stream is NOT resumed — only a
  fresh-from-seed run replays exactly; a resumed run continues the
  schedule with a new patch stream.
- `--amp` runs the forward under autocast with a GradScaler on CUDA
  devices. On CPU it is ignored: there is nothing to accelerate and the
  CPU arithmetic stays exactly what it always was.
- `--no-augment-resample` turns off the scale/elastic augmentation tier
  (on by default for real plans). Mirror/rotate augmentation is always on;
  it cannot invent anatomy, and neither can the resampling tier — zoom-out
  pads by reflecting the patch's own border, and the elastic warp clamps
  out-of-range lookups to the nearest real voxel.
- `--max-val-cases N` caps the post-fit evaluation over the validation
  split (default 8; the evaluation also appears in the command's JSON
  summary under `evaluation`).
- `--foreground-prob P` overrides the sampler's foreground bias (plan
  default 1/3). On a tiny corpus with no background variety, background
  patches let the net converge to "all background" before it learns the
  structure — pushing toward 1.0 (every patch carries the foreground) is
  the honest answer there; on real corpora the plan default is right.
- `--cascade-from COARSE_BUNDLE_DIR` switches the fit to cascade mode: the
  coarse bundle predicts every case, its foreground probability becomes an
  extra image channel (mask gains a matching all-ones channel), and the
  fine model is planned and fit with `input_channels=C+1` — the fine bundle
  and a cascade.json recording the coarse bundle, the channel count and the
  fit summary land in the output directory.

### Preprocessing: what the plan decides about the pixels

The fingerprint collects two more census numbers than the geometry, and
the plan turns both into preprocessing — the gap the PulmoAI benchmark
measured (nnU-Net 0.764 vs 0.000 foreground Dice with neither in place;
see `docs/benchmark-pulmo-2026-10-07.md`):

- the corpus's **median spacing is the resampling target**. `vanilla-fit`
  resamples every training case whose spacing differs onto the target
  grid (image cubic, label and mask nearest — a fractional class or a
  fractional "labelled" flag would both be lies);
- the **global mean/std of channel 0 over labelled foreground voxels**
  is the z-score applied to every image after resampling. Single-channel
  CT is the supported modality. A corpus whose foreground has no
  intensity spread is refused at plan time, named — there is no z-score
  to train with.

The decision lands in the bundle as **`preprocess.json`**
(`{"target_spacing": [...], "normalization": {"mean", "std"}}`) and in
the fit summary under `preprocessing`. Inference replays it
transparently: `predict` z-scores the incoming image, resamples it up to
the target grid when the caller passes a `spacing_mm` that differs, runs
the windows, and resamples the probability map BACK to the caller's grid
— the returned label lives on the input volume's own grid, whatever the
training spacing was. `vanilla-evaluate` and the `predict` subcommand
pass each case's own spacing automatically; library callers spell it
`predictor.predict(image, spacing_mm=case.spacing_mm)`. Bundles written
before preprocessing existed carry no `preprocess.json` and load as
identity: old bundles keep predicting exactly as they always did.

AUGMENTATION, the family a fit applies on top of those pixels: mirror along
any axis and 90-degree in-plane rotation (always on — they cannot invent
anatomy); a random zoom plus a smooth elastic warp (`augment_resample`, on
for planned runs, `--no-augment-resample` opts out); and the intensity trio
— random brightness shift, contrast scaling and gamma, applied to the image
alone after the geometric tiers (`augment_intensity`, the nnU-Net-parity
tier the PulmoAI benchmark named as a gap: nnU-Net augments intensity, a
geometry-only stack memorizes one intensity regime and loses foreground Dice
on low-contrast data). Hand-written `FitPlan`s default both flags off and
keep byte-exact dynamics; planned runs turn both on.

### Cross-validation, and what a fold is

`vanilla-crossval` fingerprints ALL cases once and refits the network from
scratch per fold (reseeded with `seed + fold`), because the plan is a
statement about the corpus, not about a split. The fold assignment is
deterministic and part of `report.json`: cases in `case_id` order, one
`np.random.default_rng(seed).permutation`, rank modulo the fold count.
Same cases, same seed, same folds — an aggregate nobody can reproduce is
not a result. Each fold leaves a full bundle at `crossval/fold-{k}/` and is
evaluated on its own fold; the aggregate is the plain mean and sample
standard deviation of the per-fold best validation scores. The report also
carries an `ensemble` row: every fold bundle votes on the union of all
val cases (each case once, predicted by the full ensemble, the same Dice
code as everywhere else), with the aggregate and `cases_used` recorded.

### Evaluation, and what the number means

`vanilla-evaluate` — and the automatic post-fit evaluation in both
`vanilla-fit` and `vanilla-crossval` — reports per-class Dice per case,
restricted to SUPERVISED voxels where a case carries a mask: an unannotated
voxel is unknown, not background, and scoring against unknowns is how a
model learns to hide findings the metric then credits it for not finding.
The aggregate is the per-class mean across cases plus a foreground mean;
`cases_used` names how many cases the aggregate is over (the cap, when one
was passed).

### Export: from bundle to serving runtime

`export` rebuilds the SERVED net — the same load path inference uses:
deep supervision off, one tensor at input resolution — and freezes it:

- **TorchScript**: `torch.jit.trace` on a dummy batch shaped
  (1, C, *patch_size) read from `fit_plan.json`, the patch the weights
  were trained on.
- **ONNX**: the classic exporter (dynamo=False) at opset 17, named
  input/output, static batch — the serving cost model here is one volume
  at a time. The `onnx` package is imported lazily inside the export and
  the failure names the fix (`pip install onnx`).

Both land in the output directory with `export.json`: per format the file,
input/output shapes and opset, plus `bundle_digest` — the sha256 of the
source bundle's `model.pt`, so a deployment can assert the artefact and
the checkpoint it claims to be are one file.

## As a library

The CLI is a thin shell over `medos_trainer/vanilla/`, which is the whole
framework and is meant to be imported:

```python
from medos_trainer.vanilla.data import Case, load_case_npz
from medos_trainer.vanilla.plan import collect_fingerprint, plan_from_fingerprint
from medos_trainer.vanilla.nets import build_unet
from medos_trainer.vanilla.trainer import VanillaTrainer
from medos_trainer.vanilla.infer import load_predictor

cases = [load_case_npz(p) for p in sorted(Path("cases").glob("*.npz"))]
plan = plan_from_fingerprint(collect_fingerprint(cases), "cpu")
net = build_unet(plan.network_config(input_channels=cases[0].image.shape[0]))
trainer = VanillaTrainer(net, num_classes=plan.fingerprint.num_classes,
                         plan=plan.fit_plan(), device="cpu")
trainer.fit(cases[2:], cases[:2], np.random.default_rng(0), out_dir="bundle")
label, probabilities = load_predictor("bundle").predict(
    cases[0].image, spacing_mm=cases[0].spacing_mm)
```

The library path above is the hand-rolled one: it fits the cases it is
given. `medos_trainer.standalone.fit_command` is the door that also
applies the plan's preprocessing — resample to `plan.target_spacing`,
z-score with `plan.normalization`, both replayed from the bundle at
predict time (see "Preprocessing" above).

`trainer/examples/toy_pipeline.py` is the same pipeline end to end —
synthetic corpus, plan, fit, held-out prediction — in one page, runnable
with:

```sh
python trainer/examples/toy_pipeline.py
```

## The module layout

| module | what it owns |
|---|---|
| `medos_trainer/vanilla/data.py` | the `Case` format, patch sampling, batching, mirror/rotate, scale/elastic and intensity augmentation |
| `medos_trainer/vanilla/plan.py` | the fingerprint, the plan derivation, the reasons, the preprocessing decision (target spacing, foreground z-score) |
| `medos_trainer/vanilla/preprocess.py` | resampling to the plan's target spacing, the foreground z-score, the `preprocess.json` record |
| `medos_trainer/vanilla/nets.py` | the 3D UNet with deep supervision and its configuration |
| `medos_trainer/vanilla/losses.py` | the masked segmentation loss |
| `medos_trainer/vanilla/trainer.py` | the fit loop, AMP, checkpointing, the resume record, the bundle, the plateau/poly lr laws |
| `medos_trainer/vanilla/distributed.py` | the DDP seam: `maybe_init_distributed`, rank helpers, the world-of-one-is-not-distributed rule |
| `medos_trainer/vanilla/infer.py` | sliding-window inference, the bundle predictor, the fold ensemble (`EnsemblePredictor`) |
| `medos_trainer/vanilla/cascade.py` | the coarse→fine hand-off: the coarse foreground probability as an extra image channel |
| `medos_trainer/vanilla/export.py` | TorchScript/ONNX export of the served net |
| `medos_trainer/standalone.py` | the autonomous entry: cases in, plan/fit/crossval/evaluate/bundle out, DICOM and NIfTI importers |
| `medos_trainer/environment.py` | the nine-key environment declaration, observed |
| `medos_trainer/stamp.py` | the build stamp, recorded at image build time |
| `medos_trainer/detection.py`, `evidence.py`, `overlap.py` | metric and evidence utilities |
| `medos_trainer/__main__.py` | the CLI above |

No module imports `medos` — not the platform, not the SDK, nothing under
that name. The package imports torch, numpy, scipy and — lazily, inside the
importer alone — nibabel or pydicom.

## What is not here (yet)

The honest list, so nobody plans against a feature that does not exist:

- **PyPI publication.** Install from this tree (`pip install -e .`). The
  name is reserved, the packaging metadata is real, the upload is a
  decision about support surface that has not been made.
- **Region-based labels.** Classes are mutually exclusive (softmax + one
  hot); overlapping or hierarchical label sets would need a different
  head and a different loss.
- **Test-time augmentation, model soups.** The ensemble is a plain mean of
  fold probability maps; there is no TTA loop and no weight averaging.

## The environment declaration, and the GPU

`declare-environment` emits the nine `MEDOS_TRAINING_ENVIRONMENT` keys:
the build stamp (`code_commit`, `code_dirty`, `image_digest`), the
observed hardware (`torch.cuda` + `nvidia-smi`), the framework versions,
the seeds and determinism settings the fit will apply, and — recorded
honestly as empty — the backend and preprocessing bindings: the vanilla
stack IS the backend, and a standalone run binds no deployment spec.

Without a CUDA device the declaration REFUSES rather than degrading
(`--allow-cpu` records a CPU deployment as one), because a four-day CPU fit
and a four-hour GPU fit are not the same experiment and must not wear the
same record. `doctor` answers "is the GPU attached" without starting a run.

## The image

```sh
trainer/build.sh            # medlange/trainer:0.3.0.dev0
```

`docker build` by hand is refused: `code_commit` and `code_dirty` are build
arguments the script reads out of git, and the Dockerfile fails the build
without them. The image installs exactly `requirements.txt`, copies the
package, stamps itself, and runs as a non-root user.

## Tests

In `tests/`, beside the code they read — run them with:

```sh
cd trainer && pytest tests
```

The suite covers the vanilla stack end to end (`test_vanilla_plan.py`,
`test_vanilla_nets.py`, `test_vanilla_data.py`, `test_vanilla_infer.py`,
`test_multichannel.py`, `test_augment_resample.py`), the autonomous entry
(`test_standalone.py`, `test_crossval.py`, `test_evaluate.py`,
`test_resume.py`), the newer capabilities (`test_ensemble.py`,
`test_cascade.py`, `test_lr_schedule.py`, `test_distributed.py` — the real
two-process DDP run is opt-in via `MEDOS_RUN_DDP=1`), serving
(`test_export.py`), the importers (`test_dicom_import.py`, plus the NIfTI
round-trip in `test_standalone.py`) and the metric utilities
(`test_detection.py`, `test_evidence.py`, `test_overlap.py`). It needs
torch, numpy and scipy; the nibabel and pydicom importer tests skip when
their package is absent, and the ONNX equivalence test skips when
onnxruntime is absent.

**The rule for this directory: a test here may read `trainer/` and nothing
else.** What the rest of the monorepo asserts about this tree — that it
imports no `medos` module at all — lives in the platform's suite, because
those are facts about two things rather than about this one.
