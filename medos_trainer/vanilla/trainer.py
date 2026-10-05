# SPDX-License-Identifier: Apache-2.0
"""The vanilla training loop.

Deliberately a loop, not a framework: `VanillaTrainer.fit` is readable end to
end in one sitting, because "the trainer" is the product and its behaviour
must be reviewable by the people whose name is on the model card.

WHAT IT DOES, in order: SGD (momentum 0.99, nesterov — the optimizer this
architecture family converges with), a fixed number of steps per epoch over
foreground-biased patches, validation each epoch as masked soft Dice, a
checkpoint kept for the BEST validation, and ReduceLROnPlateau stepping the
learning rate when validation stalls. Determinism is the caller's job
(`torch.manual_seed`, existing `environment.apply_determinism`); the loop
receives a generator and uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from medos_trainer.vanilla.data import Case, PatchSampler, augment_mirror_rotate, make_batch
from medos_trainer.vanilla.losses import MaskedSegmentationLoss
from medos_trainer.vanilla.nets import VanillaUNet


@dataclass(frozen=True)
class FitPlan:
    """The training-time slice of the plan (T5 decides these numbers)."""

    patch_size: tuple[int, int, int]
    batch_size: int = 2
    steps_per_epoch: int = 20
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 3e-5
    foreground_prob: float = 1 / 3


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
            self.optimizer, mode="max", factor=0.2, patience=2
        )
        self.sampler = PatchSampler(plan.patch_size, foreground_prob=plan.foreground_prob)

    def train_step(self, batch_images: torch.Tensor, batch_labels: torch.Tensor,
                   batch_mask: torch.Tensor | None) -> float:
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
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
                out = self.net(torch.as_tensor(images, device=self.device))
                if isinstance(out, tuple):
                    out = out[-1]
                scores.append(float(masked_soft_dice(
                    out, torch.as_tensor(labels, device=self.device),
                    None if masks is None else torch.as_tensor(masks, device=self.device),
                    self.num_classes,
                )))
        return float(np.mean(scores)) if scores else 0.0

    def fit(self, train_cases: list[Case], val_cases: list[Case],
            rng: np.random.Generator, out_dir: str | Path | None = None) -> dict:
        best = -1.0
        history: list[dict] = []
        for epoch in range(self.plan.epochs):
            losses = []
            for _ in range(self.plan.steps_per_epoch):
                cases = train_cases
                patches = [
                    self.sampler.sample(cases[int(rng.integers(0, len(cases)))], rng)
                    for _ in range(self.plan.batch_size)
                ]
                patches = [augment_mirror_rotate(p, rng) for p in patches]
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
            if val > best:
                best = val
                if out_dir is not None:
                    self.save_checkpoint(out_dir, record)
        return {"best_val_masked_dice_loss": best, "history": history}

    def save_checkpoint(self, out_dir: str | Path, record: dict) -> Path:
        from medos_trainer.vanilla.infer import save_inference_bundle

        save_inference_bundle(
            out_dir, self.net, record, patch_size=self.plan.patch_size
        )
        return Path(out_dir) / "model.pt"
