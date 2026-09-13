"""
ProductionHead: generates output from the current toroidal state.

Design choice: the production head reads the current superposition
state and persistent memory to generate tokens or other outputs.
It uses the toroidal structure to guide generation:
  - Phase coherence guides token selection
  - Energy levels guide confidence
  - Abstractions provide high-level context
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductionHead(nn.Module):
    """
    Generates output from the toroidal state.

    Attributes
    ----------
    d_model : int
    vocab_size : int
    """

    def __init__(self, d_model: int = 256, vocab_size: int = 32000) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Production decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )

        # Confidence calibration
        self.confidence_head = nn.Linear(d_model, 1)

    def produce(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None = None,
        abstractions: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate output from the current state.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
            Current superposition field.
        persistent_state : torch.Tensor [d_model] or None
            Consolidated persistent memory.
        abstractions : list of dict or None
            Active abstractions.

        Returns
        -------
        logits : torch.Tensor [vocab_size]
        confidence : torch.Tensor [1]
        """
        # Combine current state with persistent memory
        # alpha is [n_modes, d_model], we need [d_model]
        if alpha.dim() > 1:
            state = alpha.mean(dim=0)
        else:
            state = alpha
        if persistent_state is not None:
            if persistent_state.dim() > 1:
                state = state + persistent_state.mean(dim=0)
            else:
                state = state + persistent_state

        # Incorporate abstractions
        if abstractions:
            abs_reps = []
            for a in abstractions:
                p = a["pattern"]
                if p.dim() > 1:
                    p = p.mean(dim=0)
                abs_reps.append(p)
            abs_repr = torch.stack(abs_reps).mean(dim=0)
            state = state + 0.1 * abs_repr

        # Decode to logits
        # state should be [d_model] after mean reduction
        if state.dim() > 1:
            state_flat = state.reshape(-1)
        else:
            state_flat = state
        logits = self.decoder(state_flat)
        # logits is [vocab_size], add batch dim
        logits = logits.unsqueeze(0)  # [1, vocab_size]

        # Compute confidence
        confidence = torch.sigmoid(self.confidence_head(state)).unsqueeze(0)

        return logits, confidence

    def sample(
        self,
        alpha: torch.Tensor,
        persistent_state: torch.Tensor | None = None,
        abstractions: list[dict] | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample a token from the output distribution.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
        persistent_state : torch.Tensor [d_model] or None
        abstractions : list of dict or None
        temperature : float
        top_k : int

        Returns
        -------
        token_id : int
        logits : torch.Tensor [vocab_size]
        confidence : torch.Tensor [1]
        """
        logits, confidence = self.produce(alpha, persistent_state, abstractions)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k < logits.shape[0]:
            top_values, top_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits = logits.scatter(0, top_indices, top_values)

        # Sample
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()

        return token_id, logits, confidence
