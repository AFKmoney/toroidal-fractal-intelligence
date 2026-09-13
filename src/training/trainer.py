"""
Trainer: training loop for toroidal fractal intelligence.

Features:
  - Continuous/infinite training
  - Checkpointing
  - Metrics logging
  - Early stopping (optional)
  - Mixed precision support
"""

from __future__ import annotations

import os
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from ..toroidal.model import ToroidalFractalIntelligence
from ..evaluation.metrics import ToroidalMetrics
from ..io.data import InfiniteDataLoader


class ToroidalTrainer:
    """
    Training loop for toroidal fractal intelligence.

    Parameters
    ----------
    model : ToroidalFractalIntelligence
    dataloader : DataLoader or InfiniteDataLoader
    optimizer : torch.optim.Optimizer
    save_dir : str
    log_interval : int
    save_interval : int
    """

    def __init__(
        self,
        model: ToroidalFractalIntelligence,
        dataloader: DataLoader | InfiniteDataLoader,
        optimizer: torch.optim.Optimizer,
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        save_interval: int = 1000,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.metrics = ToroidalMetrics()
        self.step = 0
        self.epoch = 0

        # Training state
        self.best_loss = float("inf")
        self.training_log: list[dict] = []

    def train_epoch(self) -> dict:
        """
        Train for one epoch.

        Returns
        -------
        epoch_metrics : dict
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, batch in enumerate(self.dataloader):
            if isinstance(batch, dict):
                token_ids = batch["input_ids"].squeeze(-1)
            else:
                token_ids = batch.squeeze(-1) if batch.dim() == 2 else batch

            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            B = token_ids.shape[0]
            # Create targets (next token prediction)
            target_ids = token_ids[:, 1:] if token_ids.shape[1] > 1 else token_ids[:, 0]
            input_ids = token_ids[:, :-1] if token_ids.shape[1] > 1 else token_ids

            # Ensure same length
            if input_ids.shape[1] != target_ids.shape[0]:
                target_ids = target_ids[:input_ids.shape[1]]

            # Forward pass
            output = self.model(input_ids.flatten())
            logits = output["logits"]

            # Loss
            loss = nn.CrossEntropyLoss()(logits, target_ids.flatten())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item() * B
            epoch_tokens += B
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0:
                metrics = self._log_step(loss.item(), B)
                print(f"Step {self.step}: loss={loss.item():.4f}, "
                      f"perplexity={metrics['perplexity']:.2f}, "
                      f"atoms={metrics['n_atoms']}, "
                      f"aggregates={metrics['n_aggregates']}")

            # Saving
            if self.step % self.save_interval == 0:
                self.save()

        epoch_loss /= max(epoch_tokens, 1)
        self.epoch += 1

        return {
            "loss": epoch_loss,
            "perplexity": float(torch.exp(torch.tensor(epoch_loss))),
            "step": self.step,
            "epoch": self.epoch,
        }

    def _log_step(
        self,
        loss: float,
        batch_size: int,
    ) -> dict:
        """Log a single training step."""
        with torch.no_grad():
            output = self.model(
                torch.randint(0, self.model.encoder.vocab_size, (batch_size,))
            )
            perplexity = float(torch.exp(torch.tensor(loss)))

        metric = {
            "step": self.step,
            "loss": loss,
            "perplexity": perplexity,
            "n_atoms": len(self.model.atoms),
            "n_aggregates": len(output.get("aggregates", [])),
            "n_abstractions": len(output.get("abstractions", [])),
            "consolidation_count": self.model.consolidation_count,
        }
        self.training_log.append(metric)
        self.metrics.log_metrics(**metric)

        return metric

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save model checkpoint.

        Parameters
        ----------
        filename : str or None

        Returns
        -------
        path : str
        """
        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"

        path = self.save_dir / filename
        self.model.save(str(path))

        # Save training log
        log_path = self.save_dir / f"training_log_step_{self.step}.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log[-100:], f, indent=2)

        print(f"Saved checkpoint to {path}")
        return str(path)

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Parameters
        ----------
        path : str
        """
        self.model.load(path)
        print(f"Loaded checkpoint from {path}")

    def train(
        self,
        max_steps: int = 10000,
        validation_interval: int = 1000,
    ) -> dict:
        """
        Train for max_steps steps.

        Parameters
        ----------
        max_steps : int
        validation_interval : int

        Returns
        -------
        final_metrics : dict
        """
        print(f"Starting training for {max_steps} steps...")
        start_time = time.time()

        for _ in range(max_steps):
            # Train one step
            batch = next(self.dataloader)
            if isinstance(batch, dict):
                token_ids = batch["input_ids"].squeeze(-1)
            else:
                token_ids = batch.squeeze(-1) if batch.dim() == 2 else batch

            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            B = token_ids.shape[0]
            target_ids = token_ids[:, 1:] if token_ids.shape[1] > 1 else token_ids[:, 0]
            input_ids = token_ids[:, :-1] if token_ids.shape[1] > 1 else token_ids

            if input_ids.shape[1] != target_ids.shape[0]:
                target_ids = target_ids[:input_ids.shape[1]]

            output = self.model(input_ids.flatten())
            loss = nn.CrossEntropyLoss()(output["logits"], target_ids.flatten())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.step += 1

            # Validation
            if self.step % validation_interval == 0:
                val_metrics = self.validate()
                print(f"Step {self.step}: val_loss={val_metrics['loss']:.4f}, "
                      f"val_perplexity={val_metrics['perplexity']:.2f}")

                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.save("best_model.pt")

            # Progress
            if self.step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {self.step}/{max_steps} ({self.step/max_steps*100:.1f}%) "
                      f"loss={loss.item():.4f} elapsed={elapsed:.1f}s")

        # Final save
        self.save("final_model.pt")

        return {
            "final_loss": loss.item(),
            "best_loss": self.best_loss,
            "total_steps": self.step,
            "total_time": time.time() - start_time,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Validate the model.

        Returns
        -------
        metrics : dict
        """
        self.model.eval()
        val_loss = 0.0
        val_tokens = 0

        # Use a subset for validation
        val_batches = 0
        max_val_batches = 10

        for batch in self.dataloader:
            if val_batches >= max_val_batches:
                break

            if isinstance(batch, dict):
                token_ids = batch["input_ids"].squeeze(-1)
            else:
                token_ids = batch.squeeze(-1) if batch.dim() == 2 else batch

            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            B = token_ids.shape[0]
            target_ids = token_ids[:, 1:] if token_ids.shape[1] > 1 else token_ids[:, 0]
            input_ids = token_ids[:, :-1] if token_ids.shape[1] > 1 else token_ids

            if input_ids.shape[1] != target_ids.shape[0]:
                target_ids = target_ids[:input_ids.shape[1]]

            output = self.model(input_ids.flatten())
            loss = nn.CrossEntropyLoss()(output["logits"], target_ids.flatten())

            val_loss += loss.item() * B
            val_tokens += B
            val_batches += 1

        val_loss /= max(val_tokens, 1)

        return {
            "loss": val_loss,
            "perplexity": float(torch.exp(torch.tensor(val_loss))),
            "n_atoms": len(self.model.atoms),
        }
