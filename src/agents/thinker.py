"""
ThinkerAgent: internal monologue and reasoning agent.

This agent uses the toroidal state to perform internal reasoning
before producing external output. It creates "thought atoms" that
evolve in the superposition before influencing the final response.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..toroidal.model import ToroidalFractalIntelligence


class ThinkerAgent(nn.Module):
    """
    Internal reasoning agent that creates thought structures.

    Parameters
    ----------
    model : ToroidalFractalIntelligence
        The main toroidal intelligence model.
    n_thoughts : int
        Number of parallel thought atoms to maintain.
    """

    def __init__(self, model: ToroidalFractalIntelligence, n_thoughts: int = 8) -> None:
        super().__init__()
        self.model = model
        self.n_thoughts = n_thoughts

        # Thought initialization network
        self.thought_init = nn.Linear(model.encoder.d_model, model.encoder.d_model)

        # Thought evolution (separate dynamics for thoughts)
        self.thought_dynamics = nn.LSTM(
            input_size=model.encoder.d_model,
            hidden_size=model.encoder.d_model,
            num_layers=2,
            batch_first=True,
        )

    def think(
        self,
        context: torch.Tensor,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """
        Generate internal thoughts given context.

        Parameters
        ----------
        context : torch.Tensor [seq_len, d_model]
        n_steps : int
            Number of thought evolution steps.

        Returns
        -------
        thoughts : torch.Tensor [n_thoughts, d_model]
        """
        # Initialize thoughts from context
        context_repr = context.mean(dim=0)  # [d_model]
        thoughts = self.thought_init(context_repr).unsqueeze(0).repeat(self.n_thoughts, 1)

        # Evolve thoughts
        for _ in range(n_steps):
            thoughts, _ = self.thought_dynamics(thoughts.unsqueeze(0))
            thoughts = thoughts.squeeze(0)

        return thoughts

    def integrate_thoughts(
        self,
        thoughts: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate thoughts into the main superposition state.

        Parameters
        ----------
        thoughts : torch.Tensor [n_thoughts, d_model]
        alpha : torch.Tensor [n_modes, d_model]

        Returns
        -------
        alpha_enhanced : torch.Tensor [n_modes + n_thoughts, d_model]
        """
        # Project thoughts onto state space
        thought_projection = thoughts @ alpha.mean(dim=0).unsqueeze(0)
        return torch.cat([alpha, thought_projection], dim=0)
