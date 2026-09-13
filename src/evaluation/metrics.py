"""
Evaluation metrics for toroidal fractal intelligence.

Metrics:
  - Perplexity (language modeling)
  - Structure complexity (number of atoms/aggregates)
  - Consolidation ratio
  - Abstraction reuse rate
  - Energy efficiency (information per FLOP)
"""

from __future__ import annotations

import torch
import numpy as np


class ToroidalMetrics:
    """
    Comprehensive evaluation metrics for toroidal models.
    """

    def __init__(self) -> None:
        self.history: list[dict] = []

    def compute_perplexity(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Compute perplexity from logits and target IDs.

        Parameters
        ----------
        logits : torch.Tensor [vocab_size]
        target_ids : torch.Tensor [B]

        Returns
        -------
        perplexity : float
        """
        import torch.nn.functional as F
        loss = F.cross_entropy(logits.unsqueeze(0), target_ids)
        return np.exp(loss.item())

    def compute_structure_complexity(
        self,
        n_atoms: int,
        n_aggregates: int,
        n_abstractions: int,
    ) -> dict:
        """
        Compute structure complexity metrics.

        Parameters
        ----------
        n_atoms : int
        n_aggregates : int
        n_abstractions : int

        Returns
        -------
        metrics : dict
        """
        return {
            "n_atoms": n_atoms,
            "n_aggregates": n_aggregates,
            "n_abstractions": n_abstractions,
            "atom_to_aggregate_ratio": n_aggregates / max(n_atoms, 1),
            "structure_density": (n_atoms + n_aggregates) / max(n_abstractions, 1),
        }

    def compute_energy_efficiency(
        self,
        n_parameters: int,
        n_atoms: int,
        loss: float,
        flops: int,
    ) -> dict:
        """
        Compute energy efficiency metrics.

        Parameters
        ----------
        n_parameters : int
        n_atoms : int
        loss : float
        flops : int

        Returns
        -------
        metrics : dict
        """
        return {
            "loss_per_parameter": loss / max(n_parameters, 1),
            "atoms_per_parameter": n_atoms / max(n_parameters, 1),
            "loss_per_flop": loss / max(flops, 1),
            "information_per_flop": (1.0 / max(loss, 1e-6)) / max(flops, 1),
        }

    def log_metrics(
        self,
        perplexity: float,
        n_atoms: int,
        n_aggregates: int,
        n_abstractions: int,
        consolidation_count: int,
        loss: float,
        n_parameters: int,
    ) -> None:
        """
        Log a complete metrics snapshot.

        Parameters
        ----------
        perplexity : float
        n_atoms : int
        n_aggregates : int
        n_abstractions : int
        consolidation_count : int
        loss : float
        n_parameters : int
        """
        metric = {
            "perplexity": perplexity,
            **self.compute_structure_complexity(n_atoms, n_aggregates, n_abstractions),
            "consolidation_count": consolidation_count,
            "loss": loss,
            "n_parameters": n_parameters,
            "epoch": len(self.history),
        }
        self.history.append(metric)

    def get_summary(self) -> dict:
        """
        Get summary of all logged metrics.

        Returns
        -------
        summary : dict
        """
        if not self.history:
            return {}

        import numpy as np
        keys = ["perplexity", "loss", "n_atoms", "n_aggregates", "n_abstractions"]
        summary = {}
        for key in keys:
            values = [h[key] for h in self.history]
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_final"] = values[-1]

        return summary
