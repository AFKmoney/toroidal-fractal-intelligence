"""
FractalSuperpositionState: the collective state S = sum_i alpha_i A_i.

Design choice: the superposition is stored as a *field* (dense tensor)
rather than a list of N independent atoms, so that the cost of
representing N structures does not grow linearly with N.

We use a spectral/toroidal field representation:
  - A set of basis modes (fixed number)
  - Each atom contributes to a subset of modes via its (r, phi, omega)
  - The field is the superposition of all mode contributions

This gives:
  - O(modes) storage instead of O(N * d_model)
  - Natural frequency/phase separation
  - Efficient RK4 integration on the field
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalSuperpositionState(nn.Module):
    """
    The collective fractal superposition S.

    Attributes
    ----------
    n_modes : int
        Number of spectral basis modes (fixed, compact).
    d_model : int
        Dimension of each mode's representation.
    alpha : torch.Tensor [n_modes, d_model]
        Mode coefficients (the actual state field).
    """

    def __init__(self, n_modes: int = 256, d_model: int = 256) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.d_model = d_model

        # The field: superposition coefficients
        self.alpha = nn.Parameter(torch.zeros(n_modes, d_model))

        # Current time accumulator
        self.t = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def add_atom_contribution(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
    ) -> None:
        """
        Project a new atom onto the spectral field.

        Parameters
        ----------
        r : torch.Tensor [d_model] or [B, d_model]
        phi : torch.Tensor [d_model] or [B, d_model]
        omega : torch.Tensor [d_model] or [B, d_model]
        E : torch.Tensor [d_model] or [B, d_model]
        kappa : torch.Tensor [d_model] or [B, d_model]
        """
        # Ensure 2D
        if r.dim() == 1:
            r = r.unsqueeze(0)
            phi = phi.unsqueeze(0)
            omega = omega.unsqueeze(0)
            E = E.unsqueeze(0)
            kappa = kappa.unsqueeze(0)

        B = r.shape[0]

        # Compute contribution: each atom adds to all modes
        # Simplified: use mean of atom properties as scalar contribution
        atom_strength = (E * kappa).mean(dim=-1).unsqueeze(-1)  # [B, 1]

        # Spread contribution across modes based on omega (frequency)
        # Higher omega -> higher modes
        mode_weights = torch.zeros(B, self.n_modes, device=r.device)
        for b in range(B):
            # Map omega to mode index
            omega_val = omega[b, 0].item() if omega[b, 0].numel() == 1 else omega[b].mean().item()
            if torch.isnan(torch.tensor(omega_val)) or torch.isinf(torch.tensor(omega_val)):
                omega_val = 1.0  # Default to middle mode
            mode_idx = int((omega_val / (omega_val + 1.0)) * self.n_modes)
            mode_idx = min(max(mode_idx, 0), self.n_modes - 1)
            mode_weights[b, mode_idx] = atom_strength[b, 0]

        # Add to field: each mode gets a contribution
        # mode_weights: [B, n_modes], E[b].mean(): scalar
        # Result: add to each mode's d_model vector
        for b in range(B):
            for m in range(self.n_modes):
                if mode_weights[b, m] > 0:
                    self.alpha.data[m] += mode_weights[b, m] * E[b].mean()

    def remove_atom_contribution(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
    ) -> None:
        """Reverse of add_atom_contribution."""
        if r.dim() == 1:
            r = r.unsqueeze(0)
            phi = phi.unsqueeze(0)
            omega = omega.unsqueeze(0)
            E = E.unsqueeze(0)
            kappa = kappa.unsqueeze(0)

        B = r.shape[0]
        atom_strength = (E * kappa).mean(dim=-1).unsqueeze(-1)

        mode_weights = torch.zeros(B, self.n_modes, device=r.device)
        for b in range(B):
            omega_val = omega[b, 0].item() if omega[b, 0].numel() == 1 else omega[b].mean().item()
            if torch.isnan(torch.tensor(omega_val)) or torch.isinf(torch.tensor(omega_val)):
                omega_val = 1.0
            mode_idx = int((omega_val / (omega_val + 1.0)) * self.n_modes)
            mode_idx = min(max(mode_idx, 0), self.n_modes - 1)
            mode_weights[b, mode_idx] = atom_strength[b, 0]

        # Remove from field
        for b in range(B):
            for m in range(self.n_modes):
                if mode_weights[b, m] > 0:
                    self.alpha.data[m] -= mode_weights[b, m] * E[b].mean()

    def get_field(self) -> torch.Tensor:
        """Return the current superposition field."""
        return self.alpha.detach()

    def get_state_dict(self) -> dict:
        return {
            "alpha": self.alpha.cpu().clone(),
            "t": self.t.cpu().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha.data = state["alpha"].to(self.alpha.device)
        self.t.data = state["t"].to(self.t.device)

    def to(self, device):
        self.alpha = self.alpha.to(device)
        self.t = self.t.to(device)
        return self
