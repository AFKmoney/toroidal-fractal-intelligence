"""
ToroidalInteraction: computes pairwise interactions between atoms.

Design choice: interactions are computed in the spectral field domain
rather than O(N^2) pairwise, using the superposition structure.
This gives O(n_modes * d_model) complexity instead of O(N^2 * d_model).

For cases where explicit pairwise interaction is needed (e.g., for
aggregation decisions), a sparse attention mechanism is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToroidalInteraction(nn.Module):
    """
    Pairwise interaction engine for toroidal atoms.

    Uses spectral field for global interactions and sparse attention
    for local pairwise computations.
    """

    def __init__(self, d_model: int = 256, n_modes: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes

        # Sparse attention for pairwise interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
        )

        # Interaction prediction head
        self.interaction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def field_interaction(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute global interaction energy from the spectral field.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]

        Returns
        -------
        interaction_energy : torch.Tensor [n_modes]
        """
        # Phase coupling in spectral domain
        phase = torch.angle(
            alpha[:, 0].float() + 1j * alpha[:, 1].float()
        ) if alpha.shape[-1] >= 2 else torch.zeros(alpha.shape[0])

        # Laplacian-like interaction
        alpha_shifted = torch.roll(alpha, shifts=1, dims=0)
        diff = alpha - alpha_shifted
        interaction_energy = (diff ** 2).sum(dim=-1).sqrt()

        return interaction_energy

    def pairwise_interaction(
        self,
        atoms_r: torch.Tensor,
        atoms_phi: torch.Tensor,
        atoms_kappa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise coupling matrix K_ij = kappa_i * kappa_j * cos(phi_i - phi_j).

        Parameters
        ----------
        atoms_r : torch.Tensor [N, d_model]
        atoms_phi : torch.Tensor [N, d_model]
        atoms_kappa : torch.Tensor [N, d_model]

        Returns
        -------
        K : torch.Tensor [N, N]
        """
        N = atoms_r.shape[0]
        if N == 0:
            return torch.zeros(0, 0, device=atoms_r.device)

        # Efficient computation using batched operations
        phi_diff = atoms_phi.unsqueeze(1) - atoms_phi.unsqueeze(0)  # [N, N, d]
        cos_diff = torch.cos(phi_diff)  # [N, N, d]
        kappa_prod = atoms_kappa.unsqueeze(1) * atoms_kappa.unsqueeze(0)  # [N, N, d]

        K = (kappa_prod * cos_diff).mean(dim=-1)  # [N, N]
        return K

    def sparse_attention_interactions(
        self,
        alpha: torch.Tensor,
        k: int = 16,
    ) -> torch.Tensor:
        """
        Compute interactions using sparse k-NN attention.

        Parameters
        ----------
        alpha : torch.Tensor [n_modes, d_model]
        k : int
            Number of nearest neighbors to attend to.

        Returns
        -------
        interactions : torch.Tensor [n_modes, d_model]
        """
        if alpha.shape[0] <= k:
            # Full attention
            alpha_seq = alpha.unsqueeze(0)  # [1, n_modes, d_model]
            attn_out, _ = self.attention(alpha_seq, alpha_seq, alpha_seq)
            return attn_out.squeeze(0)

        # Approximate k-NN attention
        # Compute pairwise distances
        dist = torch.cdist(alpha, alpha)  # [n_modes, n_modes]
        # Get k nearest neighbors (exclude self)
        _, indices = dist.topk(k + 1, dim=-1, largest=False)
        indices = indices[:, 1:]  # Remove self

        # Gather neighbor representations
        neighbors = torch.gather(alpha, 0, indices.unsqueeze(-1).expand(-1, -1, self.d_model))
        # Compute interaction with neighbors
        neighbor_mean = neighbors.mean(dim=1)  # [n_modes, d_model]
        interaction = self.interaction_head(
            torch.cat([alpha, neighbor_mean], dim=-1)
        ).squeeze(-1)

        return interaction

    def get_interaction_matrix(
        self,
        atoms_phi: torch.Tensor,
        atoms_kappa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the full interaction matrix for aggregation decisions.

        Parameters
        ----------
        atoms_phi : torch.Tensor [N, d_model]
        atoms_kappa : torch.Tensor [N, d_model]

        Returns
        -------
        K : torch.Tensor [N, N]
        """
        return self.pairwise_interaction(
            torch.zeros_like(atoms_phi), atoms_phi, atoms_kappa
        )
