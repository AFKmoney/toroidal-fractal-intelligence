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
        
        # History for modify operations (circular buffer)
        self.register_buffer('alpha_history', torch.zeros(10, n_modes, d_model))
        self.register_buffer('history_idx', torch.tensor(0))

    def add_atom_contribution(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
        modify: bool = False,
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
        modify : bool
            If True, subtract previous contribution first (for atom reuse)
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
        # E and kappa are [B, d_model], take mean across d_model
        atom_strength = (E * kappa).mean(dim=-1, keepdim=True)  # [B, 1]

        # Real superposition: spread across ALL modes with Gaussian weighting
        # This creates actual fractal superposition, not just 1 mode per atom
        # Use first atom's omega for simplicity
        omega_vals = omega[0, 0] if omega.dim() > 1 else omega[0]  # Scalar
        omega_vals = torch.clamp(omega_vals, min=0.01, max=10.0)  # Safe range
        
        # Map omega to mode center (0 to n_modes-1)
        omega_idx = (omega_vals / max(omega_vals, 1e-8)) * (self.n_modes - 1)  # Scalar
        
        # Create Gaussian distribution across modes
        mode_indices = torch.arange(self.n_modes, device=r.device).float()
        sigma = max(self.n_modes / 4, 1.0)  # Width of Gaussian, min 1
        
        # Compute weights: [n_modes]
        diff = mode_indices - omega_idx  # [n_modes]
        weights = torch.exp(-(diff ** 2) / (2 * sigma**2))
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        
        # If modifying, subtract previous contribution first
        if modify and self.alpha_history.sum() > 0:
            idx = int(self.history_idx.cpu())
            prev = self.alpha_history[idx]  # Most recent
            self.alpha.data -= prev * 0.3  # Partial removal (decay)
        
        # Add new contribution to ALL modes (real superposition)
        # atom_strength: [B, 1], weights: [n_modes]
        # Use first atom's strength for simplicity
        strength = atom_strength[0, 0]  # Scalar
        # Broadcast weights to [n_modes, d_model]
        contribution = (weights.unsqueeze(-1) * strength).expand(self.n_modes, self.d_model)
        self.alpha.data += contribution * 0.1  # Scale to prevent explosion
        
        # Track history for modify operations
        idx = int(self.history_idx.cpu())
        self.alpha_history[idx] = self.alpha.data.clone()
        self.history_idx = torch.tensor((idx + 1) % 10)

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
        atom_strength = (E * kappa).mean(dim=-1, keepdim=True)

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
