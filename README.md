# Medlange Trainer

**A vanilla-PyTorch framework for volumetric medical image segmentation.**

Medlange Trainer is an nnU-Net-class trainer written from scratch: its own
3D UNet with deep supervision, its own fingerprint-based planning, its own
masked-loss training loop and its own sliding-window inference. It replaces
nnU-Net and MONAI — neither is installed, imported or required anywhere in
this tree — and it has no platform dependency: every input is a path on the
local filesystem and every output is a file.

A researcher with a folder of volumes reaches a trained model through one
command (`vanilla-fit`). There is no web surface, no socket and no database.

## Install

```sh
pip install -r requirements.txt
```

That is the whole installation: torch (2.7.1, CUDA wheels via the
extra-index line at the top of the file), numpy, scipy, and nibabel for the
NIfTI importer. Nothing else.

## The case format

Everything the framework reads is a directory of `.npz` cases. One case, one
file:

| member | shape | meaning |
|---|---|---|
| `image` | (1, K, J, I) float32 | the volume, already resampled to the spacing `spacing_mm` names |
| `label` | (K, J, I) int | the segmentation; 0 is background, positive values are classes |
| `mask` | (1, K, J, I) float, optional | 1 where a channel's loss counts, 0 where the case does not annotate it |
| `spacing_mm` | (3,) float, optional | voxel spacing in mm; defaults to isotropic 1 mm |

`vanilla-import-nnunet` converts an nnU-Net NIfTI layout (an imagesTr
directory and a labelsTr directory) into this format, preserving spacing
from the NIfTI zooms.

## The CLI

`python -m medos_trainer <subcommand>` — six of them, and nothing else:

```sh
# fingerprint a cases directory and write the derived plan, reasons attached
python -m medos_trainer vanilla-plan --data cases/ --preset cpu --out plan.json

# plan + train + write an inference bundle (model.pt, net_config.json, fit_plan.json)
python -m medos_trainer vanilla-fit --data cases/ --preset cpu --out bundle/ \
    --epochs 50 --steps-per-epoch 100 --seed 0 --device cpu

# bring a corpus prepared for nnU-Net into the case format
python -m medos_trainer vanilla-import-nnunet \
    --images imagesTr/ --labels labelsTr/ --out cases/

# run a trained bundle over one case, sliding window
python -m medos_trainer predict --checkpoint-dir bundle/ \
    --input cases/case_000.npz --output pred.npz --overlap 0.5 --batch-size 2

# observe the nine MEDOS_TRAINING_ENVIRONMENT keys (refuses without a GPU
# unless --allow-cpu)
python -m medos_trainer declare-environment --out environment.json

# what this installation can and cannot do, as a document
python -m medos_trainer doctor
```

The plan is a physical decision: `vanilla.plan` derives the median spacing
and shape from the fingerprint, then picks the patch size, stem stride,
batch size and schedule against named presets, and writes every choice with
the reason attached. The fit trains against that plan and never re-derives
it.

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
label, probabilities = load_predictor("bundle").predict(cases[0].image)
```

`trainer/examples/toy_pipeline.py` is the same pipeline end to end —
synthetic corpus, plan, fit, held-out prediction — in one page, runnable
with:

```sh
python trainer/examples/toy_pipeline.py
```

## The module layout

| module | what it owns |
|---|---|
| `medos_trainer/vanilla/data.py` | the `Case` format, patch sampling, batching, mirror/rotate augmentation |
| `medos_trainer/vanilla/plan.py` | the fingerprint, the plan derivation, the reasons |
| `medos_trainer/vanilla/nets.py` | the 3D UNet with deep supervision and its configuration |
| `medos_trainer/vanilla/losses.py` | the masked segmentation loss |
| `medos_trainer/vanilla/trainer.py` | the fit loop, checkpointing, the bundle |
| `medos_trainer/vanilla/infer.py` | sliding-window inference and the bundle predictor |
| `medos_trainer/standalone.py` | the autonomous entry: cases in, plan/fit/bundle out |
| `medos_trainer/environment.py` | the nine-key environment declaration, observed |
| `medos_trainer/stamp.py` | the build stamp, recorded at image build time |
| `medos_trainer/detection.py`, `evidence.py`, `overlap.py` | metric and evidence utilities |
| `medos_trainer/__main__.py` | the CLI above |

No module imports `medos` — not the platform, not the SDK, nothing under
that name. The package imports torch, numpy, scipy and — lazily, inside the
importer alone — nibabel.

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
`test_vanilla_nets.py`, `test_vanilla_data.py`, `test_vanilla_infer.py`),
the autonomous entry (`test_standalone.py`) and the metric utilities
(`test_detection.py`, `test_evidence.py`, `test_overlap.py`). It needs
torch, numpy and scipy — the pins in `requirements.txt` — and nothing else;
nibabel-gated importer tests skip when nibabel is absent.

**The rule for this directory: a test here may read `trainer/` and nothing
else.** What the rest of the monorepo asserts about this tree — that it
imports no `medos` module at all — lives in the platform's suite, because
those are facts about two things rather than about this one.
