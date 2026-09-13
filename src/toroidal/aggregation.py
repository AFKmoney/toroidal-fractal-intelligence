"""
AggregationEngine: merges coherent atoms into higher-level structures.

Design choice: aggregation is driven by:
  - Phase coherence (atoms with similar phase tend to merge)
  - Spectral proximity (atoms in similar frequency bands)
  - Energy threshold (only atoms above a certain energy are candidates)
  - Contextual correlation (atoms activated together)

The result is a hierarchical structure: atom -> motif -> structure -> meta-structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationEngine(nn.Module):
    """
    Merges coherent atoms into aggregate structures.

    Attributes
    ----------
    d_model : int
    phase_coherence_threshold : float
    energy_threshold : float
    max_depth : int
    """

    def __init__(
        self,
        d_model: int = 256,
        phase_coherence_threshold: float = 0.7,
        energy_threshold: float = 0.3,
        max_depth: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.phase_coherence_threshold = phase_coherence_threshold
        self.energy_threshold = energy_threshold
        self.max_depth = max_depth

        # Aggregate representation projector
        self.aggregate_projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_coherence(
        self,
        phi: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase coherence matrix.

        Parameters
        ----------
        phi : torch.Tensor [N, d_model]
        E : torch.Tensor [N, d_model]

        Returns
        -------
        coherence : torch.Tensor [N, N]
        """
        # Phase difference (wrapped)
        phi_diff = torch.remainder(phi.unsqueeze(1) - phi.unsqueeze(0), 2 * torch.pi)
        # Coherence = cos(phase_diff) * geometric_mean(energy)
        cos_diff = torch.cos(phi_diff)
        E_geom = (E.unsqueeze(1) * E.unsqueeze(0)).sqrt()
        coherence = cos_diff * E_geom
        return coherence

    def find_clusters(
        self,
        coherence: torch.Tensor,
        min_size: int = 2,
    ) -> list[list[int]]:
        """
        Find clusters of highly coherent atoms.

        Parameters
        ----------
        coherence : torch.Tensor [N, N]
        min_size : int

        Returns
        -------
        clusters : list of list of int
        """
        N = coherence.shape[0]
        if N == 0:
            return []

        # Threshold coherence
        mask = coherence > self.phase_coherence_threshold
        torch.diagonal(mask).fill_(False)  # Exclude self-loops

        # Connected components (simple BFS)
        visited = torch.zeros(N, dtype=torch.bool)
        clusters = []

        for i in range(N):
            if visited[i] or not mask[i].any():
                continue

            # BFS
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                node = queue.pop(0)
                cluster.append(node.item())

                neighbors = mask[node] & ~visited
                visited[neighbors] = True
                queue.extend(neighbors.nonzero().squeeze(-1).tolist())

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    def aggregate(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
        M: torch.Tensor,
        tau: torch.Tensor,
        rho: torch.Tensor,
    ) -> tuple:
        """
        Perform aggregation on atoms.

        Parameters
        ----------
        r, phi, omega, E, kappa, M, tau, rho : torch.Tensor [N, d_model] or [d_model]
            Atom properties.

        Returns
        -------
        aggregates : list of dict
            Each aggregate is a dict with merged properties.
        cluster_indices : list of list of int
        """
        # Ensure 2D
        if r.dim() == 1:
            r = r.unsqueeze(0)
            phi = phi.unsqueeze(0)
            omega = omega.unsqueeze(0)
            E = E.unsqueeze(0)
            kappa = kappa.unsqueeze(0)
            M = M.unsqueeze(0)
            tau = tau.unsqueeze(0)
            rho = rho.unsqueeze(0)

        N = r.shape[0]
        if N < 2:
            return [], []

        # Compute coherence
        coherence = self.compute_coherence(phi, E)

        # Find clusters
        clusters = self.find_clusters(coherence)

        # Create aggregates
        aggregates = []
        for cluster in clusters:
            cluster_tensors = {
                "r": r[cluster].mean(dim=0),
                "phi": torch.remainder(phi[cluster].mean(dim=0), 2 * torch.pi),
                "omega": omega[cluster].mean(dim=0),
                "E": E[cluster].sum(dim=0).clamp(0, 1),
                "kappa": kappa[cluster].mean(dim=0),
                "M": M[cluster].mean(dim=0),
                "tau": tau[cluster].mean(dim=0),
                "rho": rho[cluster].mode().values + 1,  # Higher level
            }
            aggregates.append(cluster_tensors)

        return aggregates, clusters

    def hierarchical_aggregate(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        omega: torch.Tensor,
        E: torch.Tensor,
        kappa: torch.Tensor,
        M: torch.Tensor,
        tau: torch.Tensor,
        rho: torch.Tensor,
        depth: int = 3,
    ) -> list[list[dict]]:
        """
        Perform hierarchical aggregation up to `depth` levels.

        Returns
        -------
        hierarchy : list of list of dict
            Each inner list is a level in the hierarchy.
        """
        hierarchy = []
        current_r, current_phi = r.clone(), phi.clone()
        current_omega, current_E = omega.clone(), E.clone()
        current_kappa, current_M = kappa.clone(), M.clone()
        current_tau, current_rho = tau.clone(), rho.clone()

        for d in range(depth):
            aggregates, clusters = self.aggregate(
                current_r, current_phi, current_omega, current_E,
                current_kappa, current_M, current_tau, current_rho,
            )
            if not aggregates:
                break

            hierarchy.append(aggregates)

            # Prepare for next level
            if len(aggregates) < 2:
                break

            current_r = torch.stack([a["r"] for a in aggregates])
            current_phi = torch.stack([a["phi"] for a in aggregates])
            current_omega = torch.stack([a["omega"] for a in aggregates])
            current_E = torch.stack([a["E"] for a in aggregates])
            current_kappa = torch.stack([a["kappa"] for a in aggregates])
            current_M = torch.stack([a["M"] for a in aggregates])
            current_tau = torch.stack([a["tau"] for a in aggregates])
            current_rho = torch.stack([a["rho"] for a in aggregates])

        return hierarchy
