"""
ConsolidationEngine: transforms temporary structures into persistent ones.

Design choice: consolidation is based on:
  - Repetition frequency (how often a structure is activated)
  - Importance (energy level)
  - Stability (consistency over time)
  - Predictive utility (how well it helps predict future tokens)

Consolidated structures are stored separately and have lower update
frequency, saving computation on non-critical parts of the state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsolidationEngine(nn.Module):
    """
    Consolidates temporary structures into persistent memory.

    Attributes
    ----------
    d_model : int
    consolidation_threshold : float
        Minimum stability score for consolidation.
    persistence_decay : float
        Rate at which unconsolidated structures decay.
    """

    def __init__(
        self,
        d_model: int = 256,
        consolidation_threshold: float = 0.7,
        persistence_decay: float = 0.995,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.consolidation_threshold = consolidation_threshold
        self.persistence_decay = persistence_decay

        # Persistence scores (one per mode)
        self.persistence = nn.Parameter(torch.zeros(self.d_model))

        # Consolidation gate
        self.consolidation_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def compute_stability(
        self,
        E: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute stability score based on energy consistency over time.

        Parameters
        ----------
        E : torch.Tensor [d_model]
            Current energy.
        history : list of torch.Tensor [d_model]
            Past energy values.

        Returns
        -------
        stability : torch.Tensor [d_model]
        """
        if not history:
            return E.new_ones(E.shape[0]) * 0.5

        # Stack history
        hist = torch.stack(history[-10:], dim=0)  # Last 10 steps
        mean_E = hist.mean(dim=0)
        std_E = hist.std(dim=0) + 1e-6

        # Stability = inverse of coefficient of variation
        cv = std_E / mean_E.clamp(min=1e-6)
        stability = torch.exp(-cv)

        # Boost by current energy
        stability = stability * E.sigmoid()

        return stability

    def select_for_consolidation(
        self,
        E: torch.Tensor,
        stability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select modes/atoms for consolidation.

        Parameters
        ----------
        E : torch.Tensor [d_model]
        stability : torch.Tensor [d_model]

        Returns
        -------
        consolidate_mask : torch.Tensor [d_model] bool
        """
        score = E * stability
        threshold = self.consolidation_threshold
        return score > threshold

    def consolidate(
        self,
        alpha: torch.Tensor,
        E: torch.Tensor,
        stability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Consolidate selected structures into persistent memory.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
        E : torch.Tensor [n_modes, d_model]
        stability : torch.Tensor [n_modes, d_model]

        Returns
        -------
        persistent_alpha : torch.Tensor [n_persistent, d_model]
        consolidate_mask : torch.Tensor [n_modes] bool
        """
        # Flatten for easier handling
        alpha_flat = alpha.reshape(-1, self.d_model)
        E_flat = E.reshape(-1, self.d_model)
        stability_flat = stability.reshape(-1, self.d_model)

        # Select for consolidation
        consolidate_mask = self.select_for_consolidation(E_flat, stability_flat)

        # Update persistence
        with torch.no_grad():
            self.persistence.data = (
                self.persistence_decay * self.persistence.data
                + (1 - self.persistence_decay) * E_flat[consolidate_mask].mean(dim=0)
            )

        # Consolidate: store in persistent memory
        persistent_alpha = alpha_flat[consolidate_mask].mean(dim=0, keepdim=True)

        return persistent_alpha, consolidate_mask

    def get_persistent_state(self) -> torch.Tensor:
        """Get the persistent (consolidated) state."""
        return self.persistence.detach()
