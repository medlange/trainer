# Medlange Trainer

**`medos-trainer` — the image that actually touches the pixels.**

Part of the [Medlange](https://github.com/medlange) umbrella: Medlange Trainer fits models in the
lineage of nnU-Net and MONAI and writes the medlange.modelcard/1 card that
[Medlange Core](https://github.com/medlange/core/blob/main/medos/medos/sdk/README.md) reads to rebuild each model's
preprocessing pipeline.

This directory is a **second deployable**. `medos/medos` — the API, the worker, the
gateway — carries no torch, no MONAI and no nnU-Net, and that separation is load-bearing
rather than tidy: `medos/medos/sdk/chain.py` generates MONAI Bundle configs as *data* and
never imports MONAI; `MOS-TRAIN-225` forbids the nnU-Net planner and `nnUNetPlansManager`
from the serving image's import closure *by name*; `MOS-REL-108` forbids in-process
plugin loading. `tests/integration/test_trainer_boundary.py` holds all of it.

The platform starts this image's entrypoint and talks to it over **files and an exit
code**. Nothing else.

**Position in the product map: the trainer is a training pipeline in the lineage of
nnU-Net and MONAI, not an application.** It ships no web surface and never did — no HTTP
server, no UI, no socket of any kind. Operators and scripts drive it through the
platform's training-plane API (`/api/v1/training-runs` and the curation routes of chapter
10); the no-code console that once wrapped that API was withdrawn at specification 0.4.0
(`MOS-UI-100` CUT, register entry 150).

## Tests

In `tests/`, beside the code they read — four modules, 54 tests, which run with no
repository around them:

```sh
cd trainer && pytest tests
```

Verified by copying this directory alone into an empty folder with no `pyproject.toml`
above it, no root `conftest.py` and no `tests/_support` anywhere on disk. They need
`torch`, `numpy` and `nnunetv2` (the pins in `requirements.txt`) and nothing from the
platform.

**The rule for this directory: a test here may read `trainer/` and nothing else.** What
the PLATFORM asserts about this image — that the exchange contract agrees on both sides,
that the serving image never imports torch, that the trainer's reach into `medos.` stays
inside a declared allow-list — lives in [`https://github.com/medlange/core/tree/main/tests`](https://github.com/medlange/core/tree/main/tests), because those are facts about two
things rather than about this one. The direction is one-way: the platform may read the
trainer, the trainer may not read the platform.

## What this directory imports, and what is left

**Exactly one module imports `medos`, and it is `__main__.py`.** That is the property,
and `tests/unit/test_trainer_import_boundary.py` is where it is asserted --
`test_no_fitter_file_imports_the_platform_at_all`, with `__main__.py` exempt because the
cut runs through its `execute` branch. This sentence used to count the package's modules
instead, and the count had drifted; register entry 137 records what it said and what was
measured. A headcount goes stale every time somebody adds a file. The boundary does not.
`plan`, `fit`,
`declare-environment` and `doctor` — the fitter — take `--run-dir`, read a directory, fit
a model and write a bundle, reaching nothing outside themselves but
`medos.sdk`, the SDK inside the `medos` distribution that both images install. That is
what makes this tree installable by somebody with no MedicalOS.

It reached **ten** platform modules when the separation started.

The ninth module is `__main__.py`, and the remainder is its `execute` branch: the
supervisor, which opens `MEDOS_DATABASE_URL`, polls `training_runs` for `PENDING` and
writes run state back. Its body is not here any more — it is
`medos/medos/training/supervisor.py`, on the platform side where its database is. What stays is
the subcommand that calls it, because `MOS-REL-039` allows one deployable per trust
boundary and `LocalProcessOrchestrator` starts the fit as a CHILD PROCESS of this image
rather than through an orchestrator API, which chapter 15 forbids from application code.
So the supervisor runs here and lives there, and those were always two questions.

`https://github.com/medlange/core/blob/main/tests/unit/test_trainer_import_boundary.py` holds the line. Its allow-list is written
to SHRINK — an entry nobody imports any more is a failure there, not a formality — and it
has gone 10 → 4 → 3 by failing on exactly that clause each time.

Register entry 110 of `https://github.com/medlange/core/blob/main/docs/spec/99-known-inconsistencies.md` records the measurement.

---

## 1. Build

```sh
trainer/build.sh            # medicalos/trainer:0.3.0.dev0
```

`docker build` by hand is refused. `code_commit` and `code_dirty` are build arguments the
script reads out of git, and the Dockerfile fails without them — see §4.

---

## 2. Why nnU-Net and not Auto3DSeg

`MOS-UI-148` requires a dataset-fingerprint auto-configuring backend for the no-code
console — `nnunet` or `auto3dseg` — and forbids offering `monai_supervised`.
`MOS-TRAIN-211` makes the same choice the default for a `label` capability. Between the
two permitted ones:

| | nnU-Net v2 | Auto3DSeg |
|---|---|---|
| `MOS-TRAIN-223`'s mapping table | `medos/medos/sdk/autoconfig.py::DERIVED_QUANTITIES` already names v2 `plans.json` keys — `configurations.3d_fullres.spacing`, `transpose_forward`, `foreground_intensity_properties_per_channel.0.percentile_00_5` | its column points at `hyper_parameters.canonical_axis_order` and `hyper_parameters.crop_mode`, which `AutoRunner` does not emit under those names |
| fingerprint document | one file, `plans.json`, one digest | `datastats.yaml` **plus** a per-algorithm `hyper_parameters.yaml`; `fingerprint_digest` is one `sha256_digest` column |
| natural output | one network at one fold | N algorithms and a combination rule — `MOS-TRAIN-227`'s ensemble machinery before the first registrable artifact |
| spec specificity | `MOS-TRAIN-135`/`-136`/`-137` name it and its freeze explicitly | referenced, never pinned to a key |

The exporter is deliberately the component that *refuses rather than substitutes*
(`MOS-TRAIN-223`). Choosing Auto3DSeg would mean it refuses on arrival. An Auto3DSeg
image is a **sibling** of this one, not a mode of it: two backends in one image means two
fingerprint documents behind one `pip freeze`.

---

## 3. The entrypoint contract

`medos/medos/training/orchestrator.py::RUN_DIRECTORY` is the normative spelling;
`medos/medos/sdk/contract.py` holds it for both sides, and
`tests/unit/test_trainer_contract.py` asserts the two agree.

```
<run-dir>/
  request.json          in   the run's identity, backend, seeds, determinism, bound spec
  preprocessing.json    in   the bound PreprocessingSpec document, byte for byte
  cohort/fit.jsonl      in   the FIT partition's cases, resolved by the platform
  cohort/select.jsonl   in   the SELECT partition's cases
  images/<case>/…       in   the staged volumes the two files above point into
  fingerprint.json      out  phase 1: nnU-Net's plans.json, verbatim (MOS-TRAIN-223)
  plan.json             out  phase 1: the exporter's transcription of it
  bundle/               out  phase 2: MOS-TRAIN-129's layout
  result.json           out  both: the terminal record, success or refusal
  run.log               out  the orchestrator's capture of stdout/stderr
  work/                 out  nnU-Net's raw / preprocessed / results scratch
```

**Two phases, two processes.** `MOS-TRAIN-135` requires the derived plan to be "frozen at
run start", `medos.training.runs.start` refuses an auto-configured run without a
`fingerprint_digest`, and `0013_training.up.sql::training_runs_guard()` then refuses any
later change. So the fingerprint has to exist *before* the row leaves `PENDING` — which
means the deriving process must have exited before the fitting one begins. One process
writing the fingerprint partway through would make the freeze a race with the poller.

```
stage → medos-trainer plan → runs.start(fingerprint_digest=…)
      → medos-trainer fit  → runs.succeed(bundle_digest=…)
```

**Success** is exit 0 plus `result.json` with `"status": "SUCCEEDED"`. **Failure** is a
non-zero exit plus `result.json` with `"status": "FAILED"` and a `reason` —
`medos.training.runs.fail` refuses an empty one (`MOS-REL-051`), and the reason has to
survive the process, so every exit writes the file first.

**The artifact form is not new.** `medos.training.bundle.write_bundle` writes the bundle
and `medos.training.bundle.verify` reads it back — the same two functions the platform
calls. `medos.training.autoconfig.export_spec_fields` does the fingerprint
transcription. The trainer installs `medos` (with `--no-deps`) precisely so there is one
implementation of each.

**Nothing takes a cohort argument.** No `--split`, no `--partition`, no `--bucket`, no
`--glob`. `--run-dir` names a directory the *platform* created. Chapter 17 acceptance
check 15 requires the training container's resolver to reject a path, a prefix and a
glob; the strongest form of that is having nowhere to put one, and `staged_path()`
refuses anything resolving outside `images/`.

---

## 4. Register entry 82: the environment, stamped at build

Entry 82: *"`code_commit` and `image_digest` … change with every build, so a value written
into `docker-compose.yml` is correct exactly until the next `docker compose build` and
silently false afterwards — and it is silently false in the provenance record of every
training run."*

| key | where it comes from |
|---|---|
| `code_commit` | build stamp — `build.sh` reads git, the Dockerfile refuses an empty one |
| `code_dirty` | build stamp — `git status --porcelain`, tracked **and** untracked |
| `image_digest` | build stamp — the image's own content inventory (see below) |
| `backend_versions` | `importlib.metadata`, from the installed nnU-Net |
| `framework_versions` | `importlib.metadata` — `monai` records `absent`, because it is |
| `hardware` | `torch.cuda` + `nvidia-smi`, on the device compose granted |
| `seeds` | the trainer's own policy, applied by `backend.apply_determinism` |
| `determinism` | likewise |
| `preprocessing` | `preprocessing-bindings.json` names the spec; the **digest is computed** from the document |

Nobody edits anything. `medos-trainer-environment` runs `declare-environment` on every
`docker compose up`, writes the nine keys into a volume, and exits; `medos-api` waits for
it and reads `MEDOS_TRAINING_ENVIRONMENT=@/run/medos-training/training-environment.json`.

**`image_digest` is the image's content inventory digest, not the OCI manifest digest,
and that is a deliberate reading of `MOS-TRAIN-124` reported as such.** Three reasons,
the third decisive: a container cannot read its own manifest digest without a route to
the daemon (`MOS-REL-039`); writing the final image's digest into the final image is
circular; and **an OCI image id is not reproducible** — it carries layer timestamps, so
recording it would make the same experiment, rebuilt from the same commit with the same
pins, produce a different `run_digest` and stop colliding with itself under
`training_runs_run_digest_uk`. The inventory covers the interpreter, the platform, the
resolved distribution set and the content hash of every source file the build copied in;
it is stored *beside* the digest in the stamp so a reviewer recomputes rather than
trusts. `build.sh` prints the OCI image id next to it so the two can be correlated.

---

## 5. GPU, and what happens without one

Compose reserves the device on both trainer services. **Without a GPU the trainer
refuses**: `environment.observe_hardware` raises, `medos-trainer-environment` exits
non-zero, `medos-api` never starts, and the operator is told why. A trainer that silently
falls back to CPU and takes four days is worse than one that refuses — and a `hardware`
block claiming a GPU that is not attached is worse than both, because it is false inside
`MOS-TRAIN-124`'s binding.

`MEDOS_TRAINER_ALLOW_CPU=1` records a CPU deployment **as** one: `gpu_count: 0`,
`gpu_model: absent`, `accelerator: cpu`, `cpu_run_explicitly_permitted: true`.

`medos-trainer doctor` answers "is the GPU attached" without starting a run.

---

## 6. Where the pixels come from

`MOS-TRAIN-068`/`MOS-TRAIN-199` permit the pipeline to acquire imaging **only** from the
de-identified side of the Gateway, as the `dataset_export` consumer class.
`MEDOS_TRAINER_STAGER=gateway` is the default and is that route.

**On the shipped deployment it cannot produce a volume**, and the run fails saying so.
`medos/medos/gateway/app.py` answers `503 DEID_NOT_IMPLEMENTED` for that consumer class because
the de-identification stage does not exist and `MOS-DATA-037` requires the egress to fail
closed; `medos/deploy/compose/gateway-principals.json` declares no `dataset_export` principal
either. That refusal is a working control, not a bug in this service. The alternative —
presenting the worker's credential, which the Gateway resolves as `platform_writer` with
no de-identification on egress — is the one `medos/medos/training/retrieval.py` calls "the worst
combination available: a seal that succeeds and is wrong".

`MEDOS_TRAINER_STAGER=directory` with `MEDOS_TRAINER_IMAGE_ROOT` is for a site that
exported its corpus out of band under its own de-identification. It asserts **nothing**
about provenance: `MOS-EVID-021`'s de-identification status is recorded on the
`DatasetVersion` at seal time by `medos/medos/training/seal.py`, and nothing here can add to it.

---

## 7. Known gaps, recorded rather than worked around

1. **`MOS-TRAIN-224` cannot be satisfied on this deployment.** The requirement puts the
   derived *training* batch size in `TrainingRun.hyperparameters`.
   `medos/medos/api/routes_training.py` inserts `hyperparameters = {}` at submit and
   `0013_training.up.sql::training_runs_guard()` seals that column against every later
   update. So there is no writable location for it. The trainer records it in
   `plan.json` and `result.json`; the half of the requirement that *is* enforceable —
   it must never reach `PreprocessingSpec.patch.batch_size` — is enforced in
   `packaging.derive_spec_document`.
2. **The shipped exporter cannot read a real nnU-Net `plans.json`.**
   `medos/medos/sdk/autoconfig.py` maps `foreground_crop` from
   `configurations.3d_fullres.use_mask_for_norm` and `_coerce` handles a `bool`; nnU-Net
   v2 writes a **list of bools, one per channel**, which falls through to the string
   branch and is refused as `crop_mode_not_mapped`. `backend.single_channel_view`
   flattens that one key for a single-channel dataset and **refuses** for more than one
   rather than taking the first channel's value. The fix belongs in
   `medos/medos/sdk/autoconfig.py`, which this change does not own.
3. **`MOS-TRAIN-223`'s "that spec MUST be the registered one" is satisfied downstream,
   not by the run binding.** The run binds the *declared* spec's digest at submit; the
   derived spec exists only after the planner has run. The bundle carries the derived
   spec as `configs/preprocessing.json` and `result.json` carries its digest; pinning it
   through `spec.preprocessing_spec_ref` is `medos/medos/training/candidate.py`'s job at
   registration.
4. **`framework_versions.monai` is `absent`.** `MOS-TRAIN-124` requires the key and this
   image does not install MONAI. `configs/metadata.json`'s `monai_version` is a different
   quantity — which MONAI can *load* the generated bundle — and is
   `packaging.MONAI_BUNDLE_TARGET`.
5. **`preprocessing.version` is the spec's major.** `routes_training` reads it as an
   `int` and the shipped spec documents version themselves `"1.0.0"`.
6. **The exporter's axis alphabet is not the parser's, and no real plan can be
   transcribed on the shipped code.** `medos/medos/sdk/autoconfig.py::_coerce` maps
   nnU-Net's `transpose_forward` to `z`/`y`/`x`; `medos/medos/sdk/spec.py::parse_spec`
   refuses anything that is not a permutation of `k`/`j`/`i`. Found by running it: the
   first real plan produced `axis_order: ["x","z","y"]` and
   `ChainRefused: axis_order: must be a permutation of ['k','j','i']`.
   `packaging._AXIS_ALPHABET` translates the two spellings of the same three axes —
   exactly, and only for that field. The fix belongs in `autoconfig.py`.
7. **A rebuild between submit and execute is not detected.** The run binds
   `image_digest` at submit; the trainer that later executes it may be a different
   image, and `request.json` carries no `image_digest` for the phase to compare against
   its own stamp. `MOS-TRAIN-124`'s binding would then describe an image that did not
   do the work. One member and one comparison would close it; it is named here rather
   than added quietly at the end of a change that is already large.
8. **`nnUNet_compile` is on and is not in the `determinism` block.** `MOS-TRAIN-124`'s
   four reproducibility blocks have no slot for it, and TorchInductor's kernel
   selection is a run-to-run variable exactly like `cudnn_benchmark`, which the block
   *does* carry. Recorded here; the block is `medos/medos/training/runs.py`'s.

---

## 8. Measured on this deployment

RTX 4070, 12282 MiB, driver 610.62, torch 2.5.1+cu124, nnU-Net 2.5.1.

| | |
|---|---|
| cohort | 11 synthetic phantom cases, 8 `train` / 3 `tune`, 48×72×72 at 2.0/1.0/1.0 mm |
| planner | `3d_fullres`, patch 48×80×80, batch 2, `CTNormalization`, `use_mask_for_norm [False]` |
| short fit | 3 epochs × 20 iterations: epoch 0 **109.6 s** (TorchInductor compile), epochs 1–2 **2.68 s** and **2.35 s** |
| whole run | `PENDING → SUCCEEDED` in **162 s**, both phases, staging included |
| bundle | 7 files, `models/model.ts` TorchScript, digest recomputed by `medos.training.bundle.verify` and equal to the run row's |
| full run | nnU-Net's default 1000 × 250 was measured by accident on this cohort at **23.8 s/epoch** → ≈ **6.6 hours** for one fold. A real chest-CT cohort at a larger patch size is longer; nnU-Net's own guidance is one to two days per fold on one consumer GPU. |

The first epoch costs two minutes of TorchInductor compilation and every later epoch
costs two seconds, which is the entire argument for putting `gcc` in the image (§ the
Dockerfile's apt layer) rather than setting `nnUNet_compile=f`.
