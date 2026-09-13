"""
AbstractionEngine: extracts higher-level patterns from aggregates.

Design choice: abstraction is performed by:
  - Finding common properties across aggregates
  - Creating a compressed representation that captures the regularity
  - Storing abstractions as reusable templates

Abstractions are *contextual* and *dynamic* — they can be created and
discarded based on utility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractionEngine(nn.Module):
    """
    Extracts abstract patterns from aggregated structures.

    Attributes
    ----------
    d_model : int
    abstraction_capacity : int
        Maximum number of concurrent abstractions.
    """

    def __init__(self, d_model: int = 256, abstraction_capacity: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.abstraction_capacity = abstraction_capacity

        # Abstraction memory (learnable templates)
        self.abstraction_memory = nn.Parameter(
            torch.randn(abstraction_capacity, d_model) * 0.1
        )

        # Encoder for creating new abstractions
        self.abstraction_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Similarity matcher
        self.similarity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def compute_pattern(
        self,
        aggregates: list[dict],
    ) -> torch.Tensor | None:
        """
        Compute an abstract pattern from a list of aggregates.

        Parameters
        ----------
        aggregates : list of dict
            Each dict has keys: r, phi, omega, E, kappa, M, tau, rho

        Returns
        -------
        pattern : torch.Tensor [d_model] or None
        """
        if not aggregates:
            return None

        # Extract common properties
        props = {}
        for key in ["r", "phi", "omega", "E", "kappa", "M", "tau"]:
            values = torch.stack([a[key] for a in aggregates])
            props[key] = values.mean(dim=0)

        # Compute pattern from common properties
        combined = torch.cat([
            props["r"], props["phi"], props["omega"],
            props["E"], props["M"],
        ])

        pattern = self.abstraction_encoder(combined)
        return F.normalize(pattern, dim=-1)

    def match_abstraction(
        self,
        pattern: torch.Tensor,
    ) -> tuple[int, float]:
        """
        Find the best matching abstraction in memory.

        Parameters
        ----------
        pattern : torch.Tensor [d_model]

        Returns
        -------
        best_idx : int
        similarity : float
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        # Compute similarities
        similarities = F.cosine_similarity(
            pattern, self.abstraction_memory, dim=-1
        )

        best_idx = similarities.argmax().item()
        best_sim = similarities.max().item()

        return best_idx, best_sim

    def create_or_update_abstraction(
        self,
        aggregates: list[dict],
        threshold: float = 0.8,
    ) -> int:
        """
        Create a new abstraction or update an existing one.

        Parameters
        ----------
        aggregates : list of dict
        threshold : float
            Similarity threshold for reusing an existing abstraction.

        Returns
        -------
        abstraction_id : int
        """
        pattern = self.compute_pattern(aggregates)
        if pattern is None:
            return -1

        best_idx, best_sim = self.match_abstraction(pattern)

        if best_sim > threshold:
            # Update existing abstraction (exponential moving average)
            with torch.no_grad():
                self.abstraction_memory.data[best_idx] = (
                    0.9 * self.abstraction_memory.data[best_idx]
                    + 0.1 * pattern.squeeze(0)
                )
            return best_idx
        else:
            # Create new abstraction (circular buffer)
            next_idx = len(self.abstraction_memory) % self.abstraction_capacity
            with torch.no_grad():
                self.abstraction_memory.data[next_idx] = pattern.squeeze(0)
            return next_idx

    def get_active_abstractions(
        self,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """
        Get all abstractions above a similarity threshold.

        Returns
        -------
        abstractions : list of dict
        """
        abstractions = []
        for i in range(self.abstraction_capacity):
            vec = self.abstraction_memory[i]
            if vec.norm() > min_similarity:
                abstractions.append({
                    "id": i,
                    "pattern": vec.detach().cpu(),
                })
        return abstractions
