"""
RK4DynamicsEngine: integrates the fractal superposition state forward in time.

Design choice: the dynamics function F(S, x, C, Theta) is a learned
neural ODE that respects the toroidal structure:
  - Phase coupling terms
  - Energy transfer between modes
  - Attractor dynamics (modes tend toward stable fixed points)
  - Sparsity enforcement (unimportant modes decay)

The RK4 integrator is the default, but Euler and Heun are also available
for ablation studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToroidalDynamics(nn.Module):
    """
    Learned dynamics function F(S, x, C; Theta).

    Parameters
    ----------
    d_model : int
        Dimension of the state field.
    n_modes : int
        Number of spectral modes.
    """

    def __init__(self, d_model: int = 256, n_modes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes

        # Shared parametersTheta
        self.coupling_scale = nn.Parameter(torch.tensor(1.0))
        self.energy_decay = nn.Parameter(torch.tensor(0.99))
        self.phase_sync = nn.Parameter(torch.tensor(0.5))

        # Dynamics MLP
        self.dynamics_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Attractor potential (tends to pull modes toward learned centers)
        self.attractor_centers = nn.Parameter(
            torch.randn(n_modes, d_model) * 0.1
        )

    def forward(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute dS/dt = F(S, x, C; Theta).

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
            Current superposition field.
        input_token : torch.Tensor [d_model] or None
            New token embedding.
        context : torch.Tensor [d_model] or None
            Context embedding.

        Returns
        -------
        d_alpha_dt : torch.Tensor [n_modes, d_model]
        """
        # Base dynamics from MLP
        base = self.dynamics_net(alpha.reshape(-1, self.d_model))
        base = base.reshape(alpha.shape)

        # Phase coupling term (self-interaction)
        phase_coupling = self.phase_sync * torch.sin(
            self._compute_phase_laplacian(alpha)
        )

        # Attractor term
        dist_sq = torch.cdist(alpha, self.attractor_centers) ** 2
        # Clip to avoid overflow
        dist_sq = torch.clamp(dist_sq, max=100.0)
        attractor = -F.relu(dist_sq).mean(dim=-1, keepdim=True) * alpha

        # Input driving term
        drive = torch.zeros_like(alpha)
        if input_token is not None:
            # input_token is [B, d_model] from encoder, we need [n_modes, d_model]
            # Use mean across batch and broadcast to all modes
            input_mean = input_token.mean(dim=0)  # [d_model]
            drive = input_mean.unsqueeze(0).expand(self.n_modes, -1) * self.coupling_scale

        # Energy decay (stability)
        decay = (self.energy_decay - 1.0) * alpha

        d_alpha_dt = base + phase_coupling + attractor + drive + decay
        return d_alpha_dt

    def _mode_freqs(self, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_modes, device=device).float() / self.n_modes * 2 * torch.pi

    def _compute_phase_laplacian(self, alpha: torch.Tensor) -> torch.Tensor:
        """Simple nearest-neighbor phase coupling in mode space."""
        # Shifted versions for finite-difference approximation
        alpha_shifted = torch.roll(alpha, shifts=1, dims=0)
        return torch.sin(alpha - alpha_shifted)

    def get_shared_params(self) -> dict:
        return {
            "coupling_scale": self.coupling_scale.detach().cpu(),
            "energy_decay": self.energy_decay.detach().cpu(),
            "phase_sync": self.phase_sync.detach().cpu(),
            "attractor_centers": self.attractor_centers.detach().cpu(),
        }

    def load_shared_params(self, params: dict) -> None:
        for k, v in params.items():
            if hasattr(self, k):
                getattr(self, k).data.copy_(v.to(getattr(self, k).device))


class RK4DynamicsEngine(nn.Module):
    """
    RK4 integrator for the toroidal dynamics.

    Parameters
    ----------
    dynamics : ToroidalDynamics
        The learned dynamics function.
    dt : float
        Time step for integration.
    n_steps : int
        Number of RK4 steps per update.
    """

    def __init__(
        self,
        dynamics: ToroidalDynamics,
        dt: float = 0.1,
        n_steps: int = 4,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.dt = dt
        self.n_steps = n_steps

    def evolve(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform one RK4 integration step.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
            Current state.
        input_token : torch.Tensor [d_model] or None
        context : torch.Tensor [d_model] or None

        Returns
        -------
        alpha_new : torch.Tensor [n_modes, d_model]
        """
        def f(s):
            return self.dynamics(s, input_token, context)

        k1 = f(alpha)
        k2 = f(alpha + self.dt / 2 * k1)
        k3 = f(alpha + self.dt / 2 * k2)
        k4 = f(alpha + self.dt * k3)

        alpha_new = alpha + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Clip to prevent explosion
        alpha_new = torch.clamp(alpha_new, min=-10.0, max=10.0)
        return alpha_new

    def evolve_multi(
        self,
        alpha: torch.Tensor,
        input_token: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Evolve for multiple steps."""
        steps = n_steps or self.n_steps
        current = alpha
        for _ in range(steps):
            current = self.evolve(current, input_token, context)
        return current
