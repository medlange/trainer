# SPDX-License-Identifier: Apache-2.0
"""The vanilla training loop.

Deliberately a loop, not a framework: `VanillaTrainer.fit` is readable end to
end in one sitting, because "the trainer" is the product and its behaviour
must be reviewable by the people whose name is on the model card.

WHAT IT DOES, in order: SGD (momentum 0.99, nesterov — the optimizer this
architecture family converges with), a fixed number of steps per epoch over
foreground-biased patches, augmentation (mirror/rotate always; the
scale/elastic resampling pair when the plan asks for it), validation each
epoch as masked soft Dice LOSS (lower is better — the number is 1 − dice),
a checkpoint kept for the LOWEST validation loss, and ReduceLROnPlateau
(mode="min", tracking the same loss) stepping the learning rate when
validation stalls. Every best checkpoint is written beside its resume record
(`training_state.pt` — net, optimizer, scheduler, epoch), so a run can
continue from its best rather than from its end. Determinism is the caller's
job (`torch.manual_seed`, existing `environment.apply_determinism`); the
loop receives a generator and uses it, and the resume record deliberately
holds NO generator state: a resumed run continues the weights and the
schedule, not the exact stream of patches — fresh-from-seed reproducibility
is the only reproducibility promise.

AMP: when the plan sets `use_amp` and the device is a CUDA one, the forward
runs under `torch.autocast` with a `GradScaler` step. CPU plans never build
a scaler, so the CPU path is the same arithmetic it always was.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from medos_trainer.vanilla.data import (
    Case,
    PatchSampler,
    augment_mirror_rotate,
    augment_scale_elastic,
    make_batch,
)
from medos_trainer.vanilla.losses import MaskedSegmentationLoss
from medos_trainer.vanilla.nets import VanillaUNet


@dataclass(frozen=True)
class FitPlan:
    """The training-time slice of the plan (T5 decides these numbers).

    `augment_resample` switches the scale/elastic pair on top of the always-on
    mirror/rotate — default False so that hand-written plans and every
    existing test keep byte-identical training dynamics; `PlannedRun.fit_plan`
    sets it True because a real plan wants the full augmentation family.
    `use_amp` enables autocast+GradScaler on CUDA devices; on CPU it is
    ignored (there is nothing to accelerate and the scaler would only add
    dtype noise).
    """

    patch_size: tuple[int, int, int]
    batch_size: int = 2
    steps_per_epoch: int = 20
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 3e-5
    foreground_prob: float = 1 / 3
    augment_resample: bool = False
    use_amp: bool = False


class VanillaTrainer:
    def __init__(
        self, net: VanillaUNet, num_classes: int, plan: FitPlan, device: str = "cpu"
    ) -> None:
        self.net = net.to(device)
        self.num_classes = num_classes
        self.plan = plan
        self.device = device
        self.criterion = MaskedSegmentationLoss(num_classes=num_classes)
        self.optimizer = torch.optim.SGD(
            net.parameters(), lr=plan.learning_rate,
            momentum=0.99, nesterov=True, weight_decay=plan.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=2
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
        """Masked soft Dice on random patches — the same number the LR plateau
        watches and the checkpoint is selected by."""
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
        """
        start_epoch = 0
        best = float("inf")
        if resume is not None:
            start_epoch = int(resume["epoch"]) + 1
            best = float(resume["best_val_masked_dice_loss"])
        history: list[dict] = []
        for epoch in range(start_epoch, self.plan.epochs):
            losses = []
            for _ in range(self.plan.steps_per_epoch):
                cases = train_cases
                patches = [
                    self.sampler.sample(cases[int(rng.integers(0, len(cases)))], rng)
                    for _ in range(self.plan.batch_size)
                ]
                patches = [augment_mirror_rotate(p, rng) for p in patches]
                if self.plan.augment_resample:
                    patches = [augment_scale_elastic(p, rng) for p in patches]
                images, labels, masks = make_batch(patches)
                losses.append(self.train_step(
                    torch.as_tensor(images, device=self.device),
                    torch.as_tensor(labels, device=self.device),
                    None if masks is None else torch.as_tensor(masks, device=self.device),
                ))
            val = self.validate(val_cases, rng)
            self.scheduler.step(val)
            record = {"epoch": epoch, "loss": float(np.mean(losses)),
                      "val_masked_dice_loss": val,
                      "lr": self.optimizer.param_groups[0]["lr"]}
            history.append(record)
            if val < best:
                best = val
                if out_dir is not None:
                    self.save_checkpoint(out_dir, record)
        return {"best_val_masked_dice_loss": best, "history": history}

    def save_checkpoint(self, out_dir: str | Path, record: dict) -> Path:
        from medos_trainer.vanilla.infer import save_inference_bundle

        save_inference_bundle(
            out_dir, self.net, record, patch_size=self.plan.patch_size
        )
        self.save_state(out_dir, epoch=int(record["epoch"]))
        return Path(out_dir) / "model.pt"

    def save_state(self, out_dir: str | Path, epoch: int) -> Path:
        """THE RESUME RECORD, written beside every best checkpoint.

        Holds the net, the optimizer and the scheduler state dicts and the
        epoch this best was reached at. Deliberately NO generator state and
        no history: `fit`'s docstring states the reproducibility promise
        (fresh-from-seed only) rather than smuggling a stronger one in here.
        """
        path = Path(out_dir) / "training_state.pt"
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": int(epoch),
            },
            path,
        )
        return path

    def load_state(self, bundle_dir: str | Path) -> dict:
        """Inverse of `save_state`: restores net, optimizer and scheduler
        from a bundle's training_state.pt and returns the saved dict so the
        caller can build `fit`'s `resume` argument (with the run-wide best
        from the bundle's checkpoint.json)."""
        state = torch.load(
            Path(bundle_dir) / "training_state.pt", map_location=self.device
        )
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        return state
