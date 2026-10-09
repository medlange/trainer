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
geometric tiers), validation each
epoch as masked soft Dice LOSS (lower is better — the number is 1 − dice),
a checkpoint kept for the LOWEST validation loss, and a learning-rate law
the plan chooses: "plateau" steps ReduceLROnPlateau (mode="min", tracking
the same loss) when validation stalls, while "poly" rewrites the lr EVERY
TRAINING STEP as `learning_rate * (1 - progress)^0.9` over the run's
progress — nnU-Net's PolyLRScheduler shape. Every best checkpoint is written
beside its resume record (`training_state.pt` — net, optimizer, scheduler,
epoch), so a run can continue from its best rather than from its end.
Determinism is the caller's job (`torch.manual_seed`, existing
`environment.apply_determinism`); the loop receives a generator and uses it,
and the resume record deliberately holds NO generator state: a resumed run
continues the weights and the schedule, not the exact stream of patches —
fresh-from-seed reproducibility is the only reproducibility promise.

AMP: when the plan sets `use_amp` and the device is a CUDA one, the forward
runs under `torch.autocast` with a `GradScaler` step. CPU plans never build
a scaler, so the CPU path is the same arithmetic it always was.
"""

from __future__ import annotations

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
    use_amp: bool = False
    lr_schedule: str = "plateau"
    prefetch_batches: int = 0

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
    then intensity when the plan asks — so the two paths differ only in
    WHICH generator the draws come from, never in what a draw means. Only
    `rng` differs: the producer owns its own generator, spawned from the
    fit's rng by `fit` (documented there).

    THE AUGMENTATION ORDER IS PINNED IN TWO PLACES: this factory and the
    inline loop in `VanillaTrainer.fit` must apply the same tiers in the
    same order (geometric first, intensity last) — keep them in sync.
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
        # record below is the only place the law is written down.
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2
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
        """Masked soft Dice on random patches — the number the checkpoint is
        selected by, and the number the plateau law watches when a plateau
        plan runs (poly plans move by the clock, not by this)."""
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

    def fit(self, train_cases: list[Case], val_cases: list[Case],
            rng: np.random.Generator, out_dir: str | Path | None = None,
            resume: dict | None = None) -> dict:
        """Train `plan.epochs` epochs; `resume` continues a saved best state.

        `resume` is the dict `load_state` returns plus the fresh run's best
        score under "best_val_masked_dice_loss" (the caller reads it from the
        bundle's checkpoint.json): training starts at epoch `resume["epoch"]
        + 1`, and the checkpoint is only overwritten when validation beats
        the RUN-WIDE best, so a resumed run can never regress the artifact.
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
        best = float("inf")
        if resume is not None:
            start_epoch = int(resume["epoch"]) + 1
            best = float(resume["best_val_masked_dice_loss"])
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
                        # INTENSITY LAST, mirroring _batch_factory — the two
                        # sites must stay in the same order (geometric first,
                        # intensity after); see the producer's docstring.
                        if self.plan.augment_intensity:
                            patches = [augment_intensity(p, rng) for p in patches]
                        images, labels, masks = make_batch(patches)
                    losses.append(self.train_step(
                        torch.as_tensor(images, device=self.device),
                        torch.as_tensor(labels, device=self.device),
                        None if masks is None else torch.as_tensor(masks, device=self.device),
                    ))
                    # THE POLY LAW MOVES EVERY STEP; the plateau law never touches
                    # the lr here — its step happens once per epoch, below, fed by
                    # the validation loss it exists to watch.
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
                    val = self.validate(val_cases, rng)
                    if self.scheduler is not None:
                        self.scheduler.step(val)
                    record = {"epoch": epoch, "loss": float(np.mean(losses)),
                              "val_masked_dice_loss": val,
                              "lr": self.optimizer.param_groups[0]["lr"]}
                    history.append(record)
                    if val < best:
                        best = val
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
        return {"best_val_masked_dice_loss": best, "history": history}

    def save_checkpoint(self, out_dir: str | Path, record: dict) -> Path:
        from medos_trainer.vanilla.infer import save_inference_bundle

        save_inference_bundle(
            out_dir, self._bare_net(), record, patch_size=self.plan.patch_size
        )
        self.save_state(out_dir, epoch=int(record["epoch"]))
        return Path(out_dir) / "model.pt"

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
        caller can build `fit`'s `resume` argument (with the run-wide best
        from the bundle's checkpoint.json). A None scheduler record is the
        poly plan's honest state: there is nothing to load, and the step
        counter the law reads is the epoch loop's own."""
        state = torch.load(
            Path(bundle_dir) / "training_state.pt", map_location=self.device
        )
        self._bare_net().load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state["scheduler"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        return state
