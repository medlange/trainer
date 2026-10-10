# SPDX-License-Identifier: Apache-2.0
"""The vanilla training loop.

Deliberately a loop, not a framework: `VanillaTrainer.fit` is readable end to
end in one sitting, because "the trainer" is the product and its behaviour
must be reviewable by the people whose name is on the model card.

WHAT IT DOES, in order: SGD (momentum 0.99, nesterov — the optimizer this
architecture family converges with), a fixed number of steps per epoch over
foreground-biased patches, augmentation (mirror/rotate always; the
scale/elastic resampling pair when the plan asks for it; the intensity trio
— brightness/contrast/gamma — when the plan asks for that, always after the
geometric tiers), a per-epoch SELECTION SCORE the checkpoint is chosen by
(the plan's `selection`: masked soft Dice on random patches — the historical
proxy, "patch_dice" — or FULL-VOLUME foreground Dice on the validation
split, the deployment metric, "volume_dice"; see vanilla/selection.py for
why the benchmark replaced the proxy), a checkpoint kept for the BEST
selection score (HIGHER is better in both modes — the patch mode stores the
negated loss), and a learning-rate law
the plan chooses: "plateau" steps ReduceLROnPlateau (tracking the same
selection number: mode="min" on the patch loss, mode="max" on volume Dice)
when validation stalls, while "poly" rewrites the lr EVERY
TRAINING STEP as `learning_rate * (1 - progress)^0.9` over the run's
progress — nnU-Net's PolyLRScheduler shape. Every best checkpoint is written
beside its resume record (`training_state.pt` — net, optimizer, scheduler,
epoch), so a run can continue from its best rather than from its end; and
when the plan keeps more than one (`keep_checkpoints` >= 2), every improving
epoch ALSO persists a full inference bundle under
`out_dir/checkpoints/epoch-<n>/` with an `index.json` ranking them — a
selector that can be wrong must never leave the best model as one
overwriteable file (the PulmoAI benchmark lost a 0.539 bundle exactly that
way). Determinism is the caller's job (`torch.manual_seed`, existing
`environment.apply_determinism`); the loop receives a generator and uses it,
and the resume record deliberately holds NO generator state: a resumed run
continues the weights and the schedule, not the exact stream of patches —
fresh-from-seed reproducibility is the only reproducibility promise.

AMP: when the plan sets `use_amp` and the device is a CUDA one, the forward
runs under `torch.autocast` with a `GradScaler` step. CPU plans never build
a scaler, so the CPU path is the same arithmetic it always was.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from medos_trainer.vanilla.data import (
    Case,
    PatchSampler,
    augment_intensity,
    augment_mirror_rotate,
    augment_scale_elastic,
    augment_texture,
    make_batch,
)
from medos_trainer.vanilla.distributed import (
    is_main_process,
    maybe_init_distributed,
)
from medos_trainer.vanilla.losses import MaskedSegmentationLoss
from medos_trainer.vanilla.nets import VanillaUNet
from medos_trainer.vanilla.prefetch import BatchPrefetcher


@dataclass(frozen=True)
class FitPlan:
    """The training-time slice of the plan (T5 decides these numbers).

    `augment_resample` switches the scale/elastic pair on top of the always-on
    mirror/rotate — default False so that hand-written plans and every
    existing test keep byte-identical training dynamics; `PlannedRun.fit_plan`
    sets it True because a real plan wants the full augmentation family.
    `augment_intensity` switches the brightness/contrast/gamma trio, the
    nnU-Net-parity intensity tier (data.augment_intensity) — same default
    logic: off for hand-written plans, on for planned runs. Intensity is
    image-only and runs AFTER the geometric tiers in both the inline loop
    and the prefetch producer, and the two sites must stay in the same order
    (see `_batch_factory`).
    `augment_texture` switches the texture trio — gaussian noise, gaussian
    blur, low-resolution simulation (data.augment_texture) — the last
    nnU-Net-parity tier, completing the family with geometric + intensity:
    same default logic (off for hand-written plans, on for planned runs),
    same image-only contract, and it runs AFTER intensity in both the inline
    loop and the prefetch producer; the two sites must stay in the same
    order (see `_batch_factory`).
    `use_amp` enables autocast+GradScaler on CUDA devices; on CPU it is
    ignored (there is nothing to accelerate and the scaler would only add
    dtype noise).
    `lr_schedule` picks the learning-rate law: "plateau" (the default —
    ReduceLROnPlateau on the validation loss, the behaviour every existing
    plan and test pins) or "poly" — nnU-Net's PolyLRScheduler shape,
    `lr = learning_rate * (1 - progress)^0.9` with
    `progress = (epoch + step/steps_per_epoch) / epochs`, rewritten onto the
    optimizer EVERY TRAINING STEP. Validation happens in `__post_init__`
    because the dataclass is frozen: an unknown value is refused at
    construction (naming the choices), never patched in afterwards.
    `prefetch_batches` is the data-pipeline overlap: 0 keeps the current
    INLINE sampling loop — the byte-identical, fresh-from-seed-reproducible
    path every existing test pins — and N > 0 runs sampling, augmentation and
    batch stacking on ONE daemon producer thread (see
    vanilla/prefetch.py), N batches ahead, so on GPU the patch pipeline
    overlaps the CUDA work instead of blocking it. PREFETCHING CHANGES THE
    SAMPLE STREAM: the producer owns a generator spawned from the fit's rng
    (documented in `fit`), so same-seed inline and prefetch runs see
    different patches — the same honesty nnU-Net's dataloader workers carry,
    and why prefetch off is the path tests and reproducibility claims use.
    `selection` picks what the checkpoint is chosen by: "patch_dice" (the
    default — masked soft Dice LOSS on random patches via `validate`,
    byte-identical to every pre-existing plan and test; the record keeps the
    loss under `val_masked_dice_loss` and the index/score under the negated
    `selection_score`) or "volume_dice" — full-volume foreground Dice on the
    validation split (vanilla/selection.py), the deployment metric, which
    planned runs select by because the PulmoAI benchmark caught the patch
    proxy anti-correlating with it (epoch 121's better proxy val scored 0.438
    fg Dice and had overwritten epoch 111's 0.539 —
    docs/benchmark-pulmo-2026-10-07.md). Volume selection costs minutes per
    epoch at real CT sizes; the toy volumes tests use make it seconds.
    `selection_cases` caps how many validation cases the volume selector
    scores per epoch. `keep_checkpoints` is the top-K retention: 1 is the
    historical layout exactly (the single live bundle, no `checkpoints/`
    directory); K >= 2 additionally snapshots every improving epoch as a
    full inference bundle under `out_dir/checkpoints/` and evicts the worst
    beyond K — 0 is refused, because "keep none" would mean the best model
    is nothing at all.
    """

    patch_size: tuple[int, int, int]
    batch_size: int = 2
    steps_per_epoch: int = 20
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 3e-5
    foreground_prob: float = 1 / 3
    augment_resample: bool = False
    augment_intensity: bool = False
    augment_texture: bool = False
    use_amp: bool = False
    lr_schedule: str = "plateau"
    prefetch_batches: int = 0
    selection: str = "patch_dice"
    selection_cases: int = 4
    keep_checkpoints: int = 3

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("plateau", "poly"):
            raise ValueError(
                f"lr_schedule must be one of ('plateau', 'poly'), "
                f"got {self.lr_schedule!r}"
            )
        if self.prefetch_batches < 0:
            raise ValueError(
                f"prefetch_batches is a queue depth (0 = inline sampling), "
                f"not a negative number: {self.prefetch_batches}"
            )
        if self.selection not in ("patch_dice", "volume_dice"):
            raise ValueError(
                f"selection must be one of ('patch_dice', 'volume_dice'), "
                f"got {self.selection!r}"
            )
        if self.selection_cases < 1:
            raise ValueError(
                f"selection_cases is a positive count of validation cases, "
                f"got {self.selection_cases}"
            )
        if self.keep_checkpoints < 1:
            raise ValueError(
                f"keep_checkpoints is the snapshot retention count "
                f"(1 = the single best only, the historical layout); "
                f"0 is refused: {self.keep_checkpoints}"
            )


def _batch_factory(
    cases: list[Case], plan: FitPlan, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """THE PREFETCH PRODUCER'S BODY: an endless stream of training batches.

    Runs on the BatchPrefetcher's daemon thread, never on the training
    thread. THREAD-SAFETY CONTRACT: it reads ONLY `cases` and `plan` — it
    builds its OWN PatchSampler because the trainer's sampler is shared with
    `validate`, which runs on the main thread between epochs, and two
    threads touching one sampler's foreground cache would be a race. It must
    never touch the net, the optimizer, or any other training-thread state.

    The draw ORDER replicates the inline loop exactly — case index, patch
    sample, mirror/rotate, then the scale/elastic pair when the plan asks,
    then intensity when the plan asks, then texture when the plan asks — so
    the two paths differ only in WHICH generator the draws come from, never
    in what a draw means. Only `rng` differs: the producer owns its own
    generator, spawned from the fit's rng by `fit` (documented there).

    THE AUGMENTATION ORDER IS PINNED IN TWO PLACES: this factory and the
    inline loop in `VanillaTrainer.fit` must apply the same tiers in the
    same order (geometric first, then intensity, then texture last) — keep
    them in sync.
    """
    sampler = PatchSampler(plan.patch_size, foreground_prob=plan.foreground_prob)
    while True:
        patches = [
            sampler.sample(cases[int(rng.integers(0, len(cases)))], rng)
            for _ in range(plan.batch_size)
        ]
        patches = [augment_mirror_rotate(p, rng) for p in patches]
        if plan.augment_resample:
            patches = [augment_scale_elastic(p, rng) for p in patches]
        if plan.augment_intensity:
            patches = [augment_intensity(p, rng) for p in patches]
        if plan.augment_texture:
            patches = [augment_texture(p, rng) for p in patches]
        yield make_batch(patches)


class VanillaTrainer:
    def __init__(
        self, net: VanillaUNet, num_classes: int, plan: FitPlan, device: str = "cpu"
    ) -> None:
        self.num_classes = num_classes
        self.plan = plan
        self.device = device
        # THE DDP SEAM, BEFORE ANYTHING ELSE TOUCHES THE NET: under torchrun
        # (WORLD_SIZE > 1) the world is real, the net is wrapped, and every
        # rank trains on its own patch stream; single-process, maybe_init_
        # distributed returns False and NOTHING here changes — no group, no
        # wrapper, byte-identical arithmetic. Checkpoints and the resume
        # record always go through `_bare_net`, so a DDP run's files carry no
        # "module." prefix and load exactly like a single-process run's.
        self.distributed = maybe_init_distributed(device)
        self.net = net.to(device)
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel

            self.net = DistributedDataParallel(self.net)
        # TF32 ON CUDA — THE FREE 3-5x nnU-Net ALREADY TAKES. PyTorch ships
        # with cudnn.allow_tf32=False; nnU-Net's trainer enables it, and on
        # Ampere+ cards conv3d in TF32 is 3-5x faster with no measurable
        # quality change at this scale. We leave fp32 math untouched on CPU
        # (flag is CUDA-only) and document that a run is TF32 so a reviewer
        # reproducing bit-exact numbers knows where the last ulp went.
        if device.startswith("cuda"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        self.criterion = MaskedSegmentationLoss(num_classes=num_classes)
        self.optimizer = torch.optim.SGD(
            net.parameters(), lr=plan.learning_rate,
            momentum=0.99, nesterov=True, weight_decay=plan.weight_decay,
        )
        # THE PLATEAU SCHEDULER EXISTS ONLY FOR THE PLATEAU LAW. "poly" is
        # stateless — the lr is a pure function of (epoch, step) — so there is
        # no scheduler object to own; `fit` rewrites the lr per step and the
        # record below is the only place the law is written down. THE MODE
        # FOLLOWS THE SELECTOR: the plateau law watches the same per-epoch
        # number checkpoint selection compares — mode="min" on the patch-dice
        # loss, mode="max" on volume Dice (higher is better there).
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if plan.selection == "volume_dice" else "min",
                factor=0.2, patience=2,
            )
            if plan.lr_schedule == "plateau"
            else None
        )
        self.sampler = PatchSampler(plan.patch_size, foreground_prob=plan.foreground_prob)
        # AMP EXISTS ONLY ON CUDA: a GradScaler on a CPU build would either
        # refuse or silently disable itself, and neither is a plan. The CPU
        # path keeps the exact arithmetic every existing test pins.
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if plan.use_amp and device.startswith("cuda")
            else None
        )

    def _set_poly_lr(self, epoch: int, step: int) -> None:
        """nnU-Net's PolyLRScheduler, inline: one lr per TRAINING STEP.

        `progress` is the fraction of the whole run the just-taken step sits
        at, so the first step of the first epoch is exactly `learning_rate`
        and the law decays monotonically to `(1 - progress)^0.9` of it. The
        plateau law does not call this — its lr moves only when validation
        stalls, and `scheduler.step(val)` below is the only mutation it gets.
        """
        progress = (epoch + step / self.plan.steps_per_epoch) / self.plan.epochs
        lr = self.plan.learning_rate * (1.0 - progress) ** 0.9
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _bare_net(self) -> VanillaUNet:
        """The net without a DDP wrapper — what state dicts are saved from and
        loaded into, so bundles from a distributed run and a single-process
        run are the same artifact shape."""
        module = getattr(self.net, "module", None)
        return module if module is not None else self.net

    def train_step(self, batch_images: torch.Tensor, batch_labels: torch.Tensor,
                   batch_mask: torch.Tensor | None) -> float:
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            with torch.autocast("cuda"):
                outputs = self.net(batch_images)
                loss = self.criterion(outputs, batch_labels, batch_mask)
            self.scaler.scale(loss).backward()
            # scaler.step unscales before stepping and skips the step on
            # inf/nan — the fused version of the check a hand-rolled AMP
            # loop would have to spell out.
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.net(batch_images)
            loss = self.criterion(outputs, batch_labels, batch_mask)
            loss.backward()
            self.optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def validate(self, cases: list[Case], rng: np.random.Generator,
                 patches_per_case: int = 2) -> float:
        """Masked soft Dice on random patches — the number the PATCH selector
        ("patch_dice") selects the checkpoint by, and the number the plateau
        law watches in that mode (poly plans move by the clock, not by this;
        volume plans select on full-volume Dice, not on this loss)."""
        from medos_trainer.vanilla.losses import masked_soft_dice

        self.net.eval()
        scores: list[float] = []
        for case in cases:
            for _ in range(patches_per_case):
                patch = self.sampler.sample(case, rng)
                images, labels, masks = make_batch([patch])
                x = torch.as_tensor(images, device=self.device)
                if self.scaler is not None:
                    with torch.autocast("cuda"):
                        out = self.net(x)
                else:
                    out = self.net(x)
                if isinstance(out, tuple):
                    out = out[-1]
                scores.append(float(masked_soft_dice(
                    out, torch.as_tensor(labels, device=self.device),
                    None if masks is None else torch.as_tensor(masks, device=self.device),
                    self.num_classes,
                )))
        return float(np.mean(scores)) if scores else 0.0

    def volume_selection_score(self, val_cases: list[Case]) -> float:
        """THE VOLUME SELECTOR'S SEAM: full-volume foreground Dice over up to
        `plan.selection_cases` validation cases — the deployment metric
        (vanilla/selection.py), HIGHER is better. Deliberately a method and
        not an inline call: tests monkeypatch THIS to drive selection
        deterministically, and the trainer's own docstring points here. Runs
        the bare net under autocast when the fit uses AMP, exactly like
        `validate`."""
        from medos_trainer.vanilla.selection import volume_dice

        net = self._bare_net()
        if self.scaler is not None:
            with torch.autocast("cuda"):
                return volume_dice(
                    net, val_cases, self.device, self.plan.selection_cases,
                    patch_size=self.plan.patch_size,
                )
        return volume_dice(
            net, val_cases, self.device, self.plan.selection_cases,
            patch_size=self.plan.patch_size,
        )

    def fit(self, train_cases: list[Case], val_cases: list[Case],
            rng: np.random.Generator, out_dir: str | Path | None = None,
            resume: dict | None = None) -> dict:
        """Train `plan.epochs` epochs; `resume` continues a saved best state.

        `resume` is the dict `load_state` returns: training starts at epoch
        `resume["epoch"] + 1`, and the checkpoint is only overwritten when the
        selection score beats the RUN-WIDE best (`resume["best_selection_score"]`,
        restored from the bundle's top-K index head by `load_state`), so a
        resumed run can never regress the artifact. Legacy resume dicts built
        from a bundle's `checkpoint.json` alone — carrying only
        `best_val_masked_dice_loss` — are still accepted; the loss is negated
        into the score orientation.
        SELECTION (plan.selection): "patch_dice" validates as the masked soft-
        Dice loss on random patches (the historical proxy, byte-identical
        arithmetic); "volume_dice" scores full-volume foreground Dice on up to
        `plan.selection_cases` val cases — minutes per epoch at real CT sizes,
        seconds on the toy volumes tests use (vanilla/selection.py documents
        the trade). In both modes the record carries `selection_score`
        (HIGHER is better; the patch mode negates the loss), and
        `keep_checkpoints` >= 2 snapshots every improving epoch under
        `out_dir/checkpoints/` with an `index.json` ranking — the best model
        is never a single overwriteable file.
        THE GENERATOR IS NOT RESUMED — a fresh-from-seed run replays exactly;
        a resumed run continues the schedule with a new patch stream. That
        asymmetry is deliberate and documented rather than hidden.

        UNDER DDP (the trainer was constructed in a torchrun world) rank 0
        validates, selects checkpoints and owns the returned history; every
        rank trains, and the caller seeds each rank's generator `seed+rank`
        so the patch streams differ — the sampler draws WITH REPLACEMENT, so
        overlapping draws between ranks are expected and harmless. The lr is
        identical on every rank at every epoch boundary: poly because it is a
        pure function of progress, plateau because rank 0 broadcasts the
        scheduled value with the gradients.

        PREFETCH (plan.prefetch_batches > 0): sampling, augmentation and
        batch stacking move onto one daemon producer thread that stays
        `prefetch_batches` deep ahead of the training loop (the WHY is
        prefetch.py's docstring). THE SAMPLE STREAM CHANGES, honestly and on
        purpose: the producer cannot share the caller's generator with the
        inline draws (it runs on another thread, and a generator is not
        thread-safe), so `fit` draws ONE entropy integer from `rng` and spawns
        the producer's generator from it. Same seed + prefetch off replays
        exactly as it always did; same seed + prefetch on is a DIFFERENT
        honest run — the same trade nnU-Net's dataloader workers make — while
        validation keeps drawing from `rng` itself on this thread. Everything
        else — train_step, the lr laws, checkpoint selection, DDP semantics —
        is identical between the two paths; under DDP each rank's producer
        inherits its rank-seeded `rng` exactly like the inline path does.
        """
        start_epoch = 0
        # THE SELECTION BAR, run-wide: the best selection score so far,
        # HIGHER is better in both selector modes. The patch mode's score is
        # the negated validation loss (so `score > best` below selects exactly
        # the epochs the historical `val < best` did); the volume mode's is
        # foreground Dice.
        best = float("-inf")
        if resume is not None:
            start_epoch = int(resume["epoch"]) + 1
            if "best_selection_score" in resume:
                best = float(resume["best_selection_score"])
            else:
                # LEGACY RESUME DICT (pre-top-K bundles): only the patch loss
                # is on it; negate into the score orientation.
                best = -float(resume["best_val_masked_dice_loss"])
        history: list[dict] = []
        prefetcher: BatchPrefetcher | None = None
        if self.plan.prefetch_batches > 0:
            producer_rng = np.random.default_rng(
                np.random.SeedSequence(int(rng.integers(0, 2**31))).spawn(1)[0]
            )
            prefetcher = BatchPrefetcher(
                lambda: _batch_factory(train_cases, self.plan, producer_rng),
                queue_size=self.plan.prefetch_batches,
            )
        try:
            for epoch in range(start_epoch, self.plan.epochs):
                losses = []
                for step in range(self.plan.steps_per_epoch):
                    if prefetcher is not None:
                        images, labels, masks = next(prefetcher)
                    else:
                        cases = train_cases
                        patches = [
                            self.sampler.sample(cases[int(rng.integers(0, len(cases)))], rng)
                            for _ in range(self.plan.batch_size)
                        ]
                        patches = [augment_mirror_rotate(p, rng) for p in patches]
                        if self.plan.augment_resample:
                            patches = [augment_scale_elastic(p, rng) for p in patches]
                        # INTENSITY THEN TEXTURE LAST, mirroring
                        # _batch_factory — the two sites must stay in the
                        # same order (geometric first, then intensity, then
                        # texture after); see the producer's docstring.
                        if self.plan.augment_intensity:
                            patches = [augment_intensity(p, rng) for p in patches]
                        if self.plan.augment_texture:
                            patches = [augment_texture(p, rng) for p in patches]
                        images, labels, masks = make_batch(patches)
                    losses.append(self.train_step(
                        torch.as_tensor(images, device=self.device),
                        torch.as_tensor(labels, device=self.device),
                        None if masks is None else torch.as_tensor(masks, device=self.device),
                    ))
                    # THE POLY LAW MOVES EVERY STEP; the plateau law never touches
                    # the lr here — its step happens once per epoch, below, fed by
                    # the same selection number checkpoint selection compares.
                    if self.plan.lr_schedule == "poly":
                        self._set_poly_lr(epoch, step)
                # RANK 0 OWNS VALIDATION, because every rank's net holds identical
                # weights (DDP all-reduce guarantees it) — validating on all ranks
                # would run the same patches the same number of times and report
                # the same number, quadrupled work for zero information. The lr
                # law is unaffected: poly is a pure function of progress and the
                # plateau scheduler lives behind the same main-process guard it
                # always lived behind (it was always fed by validation).
                if is_main_process():
                    if self.plan.selection == "volume_dice":
                        # FULL-VOLUME FOREGROUND DICE — what deployment scores
                        # (the benchmark lesson: the patch proxy anti-correlated
                        # with this number and overwrote a better bundle).
                        score = self.volume_selection_score(val_cases)
                        if self.scheduler is not None:
                            self.scheduler.step(score)
                        record = {
                            "epoch": epoch, "loss": float(np.mean(losses)),
                            "selection_score": score, "volume_dice": score,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    else:
                        # THE HISTORICAL PROXY, BYTE-IDENTICAL: masked soft-Dice
                        # LOSS on random patches; the record keeps the loss and
                        # carries the negated value as the selection score so one
                        # comparison direction serves both selectors.
                        val = self.validate(val_cases, rng)
                        if self.scheduler is not None:
                            self.scheduler.step(val)
                        score = -val
                        record = {
                            "epoch": epoch, "loss": float(np.mean(losses)),
                            "val_masked_dice_loss": val,
                            "selection_score": score,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    history.append(record)
                    if score > best:
                        best = score
                        if out_dir is not None:
                            self.save_checkpoint(out_dir, record)
                if self.distributed:
                    import torch.distributed as dist

                    # THE PLATEAU LAW UNDER DDP: rank 0 is the only rank that sees
                    # validation, so it is the only rank whose scheduler moves —
                    # but DDP keeps weights identical only while every rank's
                    # optimizer steps with the SAME lr. The scheduled lr therefore
                    # travels with the gradients: rank 0's value is broadcast and
                    # every rank adopts it before the next epoch. Poly needs no
                    # such courier — its lr is the same pure function of progress
                    # on every rank — but the broadcast is uniform and cheap.
                    lr_now = torch.tensor(
                        [self.optimizer.param_groups[0]["lr"]], device=self.device
                    )
                    dist.broadcast(lr_now, src=0)
                    for group in self.optimizer.param_groups:
                        group["lr"] = float(lr_now.item())
                    # Keep the ranks in lockstep: with identical step counts the
                    # loop cannot skew, but a barrier makes "cannot" a guarantee
                    # (and covers the epoch where validation time differs most).
                    dist.barrier()
        finally:
            # The producer is a daemon (process exit is safe without this),
            # but a fit that ends — normally, on an exception, or on the
            # consumer side of a failed factory — should release its thread
            # and its queued batches promptly in a long-lived process.
            if prefetcher is not None:
                prefetcher.close()
        return {
            "best_selection_score": best,
            # The historical key, kept for the patch selector only: the min
            # masked val loss. Volume runs never compute it — None, not a
            # made-up number.
            "best_val_masked_dice_loss": (
                -best if self.plan.selection == "patch_dice" else None
            ),
            "history": history,
        }

    def save_checkpoint(self, out_dir: str | Path, record: dict) -> Path:
        from medos_trainer.vanilla.infer import save_inference_bundle

        save_inference_bundle(
            out_dir, self._bare_net(), record, patch_size=self.plan.patch_size
        )
        self.save_state(out_dir, epoch=int(record["epoch"]))
        if self.plan.keep_checkpoints >= 2:
            self._write_snapshot(out_dir, record)
        return Path(out_dir) / "model.pt"

    def _write_snapshot(self, out_dir: str | Path, record: dict) -> None:
        """TOP-K RETENTION, the benchmark's second lesson (docs/benchmark-
        pulmo-2026-10-07.md): a selector that can be wrong must never leave
        the best model as one overwriteable file. Every improving epoch
        persists a FULL inference bundle under
        `out_dir/checkpoints/epoch-<n>/` and rewrites the index —
        [{epoch, score, dir}] sorted by `score` DESCENDING, where `score` is
        the selection score (HIGHER is better in both modes; the patch mode
        stores the negated loss). Snapshots past `keep_checkpoints` are
        evicted from DISK, not just from the index: retention is a disk
        contract, ~K times the model size.
        """
        from medos_trainer.vanilla.infer import save_inference_bundle

        root = Path(out_dir) / "checkpoints"
        epoch = int(record["epoch"])
        snapshot_dir = root / f"epoch-{epoch}"
        save_inference_bundle(
            snapshot_dir, self._bare_net(), record, patch_size=self.plan.patch_size
        )
        index_path = root / "index.json"
        entries: list[dict] = []
        if index_path.is_file():
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        entries = [e for e in entries if int(e["epoch"]) != epoch]
        entries.append({
            "epoch": epoch,
            "score": float(record["selection_score"]),
            "dir": snapshot_dir.name,
        })
        entries.sort(key=lambda e: e["score"], reverse=True)
        for evicted in entries[self.plan.keep_checkpoints :]:
            _remove_tree(root / evicted["dir"])
        entries = entries[: self.plan.keep_checkpoints]
        index_path.write_text(
            json.dumps(entries, indent=2) + "\n", encoding="utf-8"
        )

    def save_state(self, out_dir: str | Path, epoch: int) -> Path:
        """THE RESUME RECORD, written beside every best checkpoint.

        Holds the net, the optimizer and the scheduler state dicts and the
        epoch this best was reached at. Deliberately NO generator state and
        no history: `fit`'s docstring states the reproducibility promise
        (fresh-from-seed only) rather than smuggling a stronger one in here.
        A "poly" plan has no scheduler object — the law is stateless — so the
        record carries None there and `load_state` restores nothing, which is
        exactly the amount of scheduler a poly run owns. The net is saved
        through `_bare_net`, so a DDP run's record loads into an unwrapped
        net without a "module." prefix to explain.
        """
        path = Path(out_dir) / "training_state.pt"
        torch.save(
            {
                "net": self._bare_net().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "epoch": int(epoch),
            },
            path,
        )
        return path

    def load_state(self, bundle_dir: str | Path) -> dict:
        """Inverse of `save_state`: restores net, optimizer and scheduler
        from a bundle's training_state.pt and returns the saved dict so the
        caller can build `fit`'s `resume` argument. The returned dict ALSO
        carries `best_selection_score` — the run-wide best (higher is
        better) restored from the top-K index head when the bundle has one;
        else from `checkpoint.json` (new records carry `selection_score`;
        legacy patch-only bundles carry `val_masked_dice_loss`, negated). A
        directory holding `training_state.pt` but neither record (a bare
        `save_state` target) gets -inf: the resumed run re-selects from its
        first epoch. A None scheduler record is the
        poly plan's honest state: there is nothing to load, and the step
        counter the law reads is the epoch loop's own."""
        state = torch.load(
            Path(bundle_dir) / "training_state.pt", map_location=self.device
        )
        self._bare_net().load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state["scheduler"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        state["best_selection_score"] = self._read_best_selection_score(bundle_dir)
        return state

    def _read_best_selection_score(self, bundle_dir: str | Path) -> float:
        root = Path(bundle_dir)
        index_path = root / "checkpoints" / "index.json"
        if index_path.is_file():
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            if entries:
                return float(entries[0]["score"])
        record_path = root / "checkpoint.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if "selection_score" in record:
                return float(record["selection_score"])
            return -float(record["val_masked_dice_loss"])
        return float("-inf")


def _remove_tree(path: Path) -> None:
    """`shutil.rmtree`'s job in pathlib/os — the vanilla purity gate's
    allowed import set has no shutil, and a snapshot directory holds files
    only. Missing paths are fine: eviction must be idempotent."""
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()
