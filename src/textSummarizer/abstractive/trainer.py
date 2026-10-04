"""Training loop: Adam, gradient clipping, LR decay on plateau, early stopping."""

import csv
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from textSummarizer.abstractive.config import TrainingConfig
from textSummarizer.abstractive.model import PointerGenerator

logger = logging.getLogger(__name__)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_nll: float
    train_coverage: float
    val_loss: float
    val_nll: float
    val_coverage: float
    val_perplexity: float
    learning_rate: float
    seconds: float


class Trainer:
    def __init__(
        self,
        model: PointerGenerator,
        config: TrainingConfig,
        device: torch.device,
        log_path: Path,
        on_improve: Callable[[EpochLog], None],
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.log_path = log_path
        self.on_improve = on_improve
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[EpochLog]:
        history: list[EpochLog] = []
        best, stale = math.inf, 0
        for epoch in range(1, self.config.epochs + 1):
            start = time.perf_counter()
            train = self._run_epoch(train_loader, train=True)
            with torch.no_grad():
                val = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val["loss"])

            log = EpochLog(
                epoch=epoch,
                train_loss=train["loss"],
                train_nll=train["nll"],
                train_coverage=train["coverage"],
                val_loss=val["loss"],
                val_nll=val["nll"],
                val_coverage=val["coverage"],
                val_perplexity=math.exp(min(val["nll"], 50)),
                learning_rate=self.optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - start,
            )
            history.append(log)
            self._append_log(log)
            improved = log.val_loss < best - 1e-4
            logger.info(
                "epoch %d  train %.4f  val %.4f (nll %.4f, ppl %.1f)  lr %.1e  %.0fs%s",
                epoch, log.train_loss, log.val_loss, log.val_nll, log.val_perplexity,
                log.learning_rate, log.seconds, "  *best*" if improved else "",
            )  # fmt: skip

            if improved:
                best, stale = log.val_loss, 0
                self.on_improve(log)
            else:
                stale += 1
                if stale >= self.config.patience:
                    logger.info("early stopping: no val improvement for %d epochs", stale)
                    break
        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "nll": 0.0, "coverage": 0.0}
        n = 0
        for batch in loader:
            batch = batch.to(self.device)
            losses = self.model(batch, coverage_loss_weight=self.config.coverage_loss_weight)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            size = batch.src.size(0)
            n += size
            for key in totals:
                if key in losses:
                    totals[key] += losses[key].item() * size
        return {k: v / max(n, 1) for k, v in totals.items()}

    def _append_log(self, log: EpochLog) -> None:
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(log)))
            if new:
                writer.writeheader()
            writer.writerow(asdict(log))
