"""
ToroidalEncoder: maps tokens (with context and current state) to toroidal atoms.

Design choice: instead of a simple linear projection token -> vector,
the encoder produces a structured atom with the 8 primitive properties
(r, phi, omega, E, kappa, M, tau, rho) in a differentiable way.

The encoder is a small feed-forward network with:
  - input token embedding (learnable) + positional/context encoding
  - a toroidal mapping head that produces the 8 primitive properties
  - a gating mechanism that can also issue MODIFY / MERGE / REINFORCE
    operations on existing atoms rather than always creating a new one.

This keeps the parameter budget small (shared encoder weights) while
the *state capacity* grows with the number of atoms in the superposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToroidalEncoder(nn.Module):
    """
    Token -> Toroidal Atom encoder.

    Attributes
    ----------
    vocab_size : int
        Size of the input token vocabulary.
    d_hidden : int
        Hidden dimension for embeddings and MLP.
    n_atoms_max : int
        Maximum number of atoms that can coexist in the superposition.
    tau_eps : float
        Small constant to avoid log(0) in the tau (time-scale) output.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        d_hidden: int = 512,
        n_atoms_max: int = 1024,
        tau_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_atoms_max = n_atoms_max
        self.tau_eps = tau_eps

        # Shared embedding table (compact)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Contextual position encoding (learned, not sinusoidal — toroidal)
        self.pos_embedding = nn.Embedding(n_atoms_max, d_model)

        # Mapping network: token_emb + pos -> atom_properties
        # COMPACT: d_hidden = d_model for efficiency
        self.mapping = nn.Sequential(
            nn.Linear(d_model, d_model),  # Compact: no expansion
            nn.GELU(),
            nn.Linear(d_model, 8 * d_model),  # 8 primitives * d_model each
        )

        # Operation gate: decides CREATE / MODIFY / MERGE / REINFORCE / ...
        self.op_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 7),  # 7 possible operations
        )

        # Shared parametersTheta — these are the compact shared rules
        self.theta_shared = nn.ParameterDict({
            "coupling_scale": nn.Parameter(torch.tensor(1.0)),
            "energy_decay": nn.Parameter(torch.tensor(0.99)),
            "phase_sync_strength": nn.Parameter(torch.tensor(0.5)),
        })
        
        # Atom reuse threshold (for MODIFY vs CREATE decision)
        self.reuse_threshold = nn.Parameter(torch.tensor(0.7))

    def forward(
        self,
        token_ids: torch.Tensor,
        context_ids: torch.Tensor | None = None,
        existing_atoms: "ToroidalAtom" | None = None,
    ) -> tuple:
        """
        Parameters
        ----------
        token_ids : torch.Tensor [B]
        context_ids : torch.Tensor [B, seq_len] or None
        existing_atoms : ToroidalAtom or None

        Returns
        -------
        new_atom : ToroidalAtom | None
        operation : torch.Tensor [B] int
        confidence : torch.Tensor [B] float
        """
        b = token_ids.shape[0]
        tok_emb = self.token_embedding(token_ids)  # [B, d_model]

        # Map to raw atom properties (direct from token embedding)
        raw = self.mapping(tok_emb)  # [B, 8*d_model]
        raw = raw.view(b, 8, -1)  # [B, 8, d_model]

        # Decode primitives (each is a vector in d_model space)
        r = F.normalize(raw[:, 0], dim=-1)        # position / radius direction
        phi = raw[:, 1]                            # phase (unnormalized angle latent)
        omega = F.softplus(raw[:, 2]) + 1e-3       # natural frequency (positive)
        E = F.sigmoid(raw[:, 3])                   # energy in [0,1]
        kappa = F.softplus(raw[:, 4])              # coupling strength (positive)
        M = F.normalize(raw[:, 5], dim=-1)         # local memory direction
        tau = F.softplus(raw[:, 6]) + self.tau_eps # time scale (positive)
        rho = torch.randint(0, 5, (b,)).to(r.device) # hierarchical level (discrete)

        from .atom import ToroidalAtom
        new_atom = ToroidalAtom(
            r=r, phi=phi, omega=omega, E=E,
            kappa=kappa, M=M, tau=tau, rho=rho,
        )

        # Operation decision
        op_logits = self.op_gate(tok_emb)
        operation = torch.argmax(op_logits, dim=-1)
        confidence = torch.softmax(op_logits, dim=-1).max(dim=-1).values

        return new_atom, operation, confidence

    def get_shared_params(self) -> dict:
        params = {k: v.clone() for k, v in self.theta_shared.items()}
        params["reuse_threshold"] = self.reuse_threshold.clone()
        return params

    def load_shared_params(self, params: dict) -> None:
        for k, v in params.items():
            if k in self.theta_shared:
                self.theta_shared[k].data.copy_(v)
            elif k == "reuse_threshold":
                self.reuse_threshold.data.copy_(v)
