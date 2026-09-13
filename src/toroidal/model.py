"""
ToroidalFractalIntelligence: the complete model integrating all components.

This is the main entry point — a single nn.Module that encapsulates:
  - ToroidalEncoder
  - FractalSuperpositionState
  - RK4DynamicsEngine
  - ToroidalInteraction
  - AggregationEngine
  - AbstractionEngine
  - ConsolidationEngine
  - ProductionHead

The model is designed for infinite trainability:
  - Atoms can be added/removed dynamically
  - Shared parametersTheta are updated via backprop
  - Persistent memory is consolidated gradually
  - No fixed training/inference boundary
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .encoder import ToroidalEncoder
from .state import FractalSuperpositionState
from .dynamics import RK4DynamicsEngine, ToroidalDynamics
from .interaction import ToroidalInteraction
from .aggregation import AggregationEngine
from .abstraction import AbstractionEngine
from .consolidation import ConsolidationEngine
from .production import ProductionHead
from .atom import ToroidalAtomCollection


class ToroidalFractalIntelligence(nn.Module):
    """
    Complete toroidal fractal intelligence system.

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_modes : int
    n_atoms_max : int
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        n_modes: int = 256,
        n_atoms_max: int = 1024,
    ) -> None:
        super().__init__()

        # Core components
        self.encoder = ToroidalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_atoms_max=n_atoms_max,
        )
        self.state = FractalSuperpositionState(
            n_modes=n_modes,
            d_model=d_model,
        )
        self.dynamics = RK4DynamicsEngine(
            dynamics=ToroidalDynamics(d_model=d_model, n_modes=n_modes),
            dt=0.1,
            n_steps=4,
        )
        self.interaction = ToroidalInteraction(
            d_model=d_model,
            n_modes=n_modes,
        )
        self.aggregation = AggregationEngine(d_model=d_model)
        self.abstraction = AbstractionEngine(d_model=d_model)
        self.consolidation = ConsolidationEngine(d_model=d_model)
        self.production = ProductionHead(d_model=d_model, vocab_size=vocab_size)

        # Atom collection (temporary structures)
        self.atoms = ToroidalAtomCollection()

        # History for stability computation
        self.energy_history: list[torch.Tensor] = []

        # Training metrics
        self.training_loss = 0.0
        self.consolidation_count = 0
        self.abstraction_count = 0

    def forward(
        self,
        token_ids: torch.Tensor,
        context_ids: torch.Tensor | None = None,
    ) -> dict:
        """
        Process a batch of tokens through the full pipeline.

        Parameters
        ----------
        token_ids : torch.Tensor [B] or [B, seq_len]
        context_ids : torch.Tensor [B, seq_len] or None

        Returns
        -------
        output : dict
            Contains logits, confidence, state info, and metrics.
        """
        # Flatten if batch of sequences
        if token_ids.dim() > 1:
            token_ids = token_ids[:, -1]  # Use last token

        # 1. Token -> Toroidal Atom
        new_atom, operation, confidence = self.encoder(
            token_ids, context_ids, self.atoms
        )

        # 2. Add to superposition
        if new_atom is not None:
            self.state.add_atom_contribution(
                new_atom.r, new_atom.phi, new_atom.omega,
                new_atom.E, new_atom.kappa,
            )
            self.atoms.add(new_atom)

        # 3. RK4 dynamics evolution
        alpha = self.state.get_field()
        alpha_new = self.dynamics.evolve(
            alpha,
            input_token=new_atom.r if new_atom is not None else None,
        )
        self.state.alpha.data = alpha_new

        # 4. Compute interactions
        interaction_energy = self.interaction.field_interaction(alpha_new)

        # 5. Aggregation (if many atoms)
        aggregates = []
        if len(self.atoms) > 10:
            aggregates, _ = self.aggregation.aggregate(
                self.atoms.r, self.atoms.phi, self.atoms.omega,
                self.atoms.E, self.atoms.kappa, self.atoms.M,
                self.atoms.tau, self.atoms.rho,
            )

        # 6. Abstraction
        abstractions = []
        if aggregates:
            for agg in aggregates:
                abs_id = self.abstraction.create_or_update_abstraction([agg])
                if abs_id >= 0:
                    abstractions.append({
                        "id": abs_id,
                        "pattern": self.abstraction.abstraction_memory[abs_id].detach(),
                    })
                    self.abstraction_count += 1

        # 7. Consolidation
        persistent_state = None
        if len(self.energy_history) > 5:
            E = self.atoms.E.mean(dim=0) if len(self.atoms) > 0 else alpha_new.mean(dim=0)
            stability = self.consolidation.compute_stability(E, self.energy_history)
            self.energy_history.append(E.detach())
            if len(self.energy_history) > 20:
                self.energy_history = self.energy_history[-20:]

            persistent_state, consolidate_mask = self.consolidation.consolidate(
                alpha_new, E.unsqueeze(0), stability.unsqueeze(0)
            )
            if consolidate_mask.any():
                self.consolidation_count += consolidate_mask.sum().item()

        # 8. Production
        # Use mean of alpha for production (single prediction for batch)
        logits, conf = self.production.produce(
            alpha_new, persistent_state, abstractions
        )
        # logits shape: [1, vocab_size]
        # Use mean prediction for all tokens in batch
        n_tokens = token_ids.shape[0]
        logits = logits.mean(dim=0, keepdim=True).expand(n_tokens, -1)  # [n_tokens, vocab_size]

        return {
            "logits": logits,
            "confidence": conf,
            "interaction_energy": interaction_energy,
            "aggregates": aggregates,
            "abstractions": abstractions,
            "persistent_state": persistent_state,
            "operation": operation,
        }

    def train_step(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        context_ids: torch.Tensor | None = None,
    ) -> dict:
        """
        Perform one training step with gradient descent.

        Parameters
        ----------
        token_ids : torch.Tensor [B] or [B, seq_len]
        target_ids : torch.Tensor [B]
        optimizer : torch.optim.Optimizer
        context_ids : torch.Tensor [B, seq_len] or None

        Returns
        -------
        metrics : dict
        """
        optimizer.zero_grad()

        output = self.forward(token_ids, context_ids)
        logits = output["logits"]

        # Handle target_ids shape
        if target_ids.dim() > 1:
            target_ids = target_ids[:, -1]

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, target_ids)

        # Regularization: encourage sparsity in state
        alpha = self.state.get_field()
        sparsity_loss = alpha.abs().mean() * 0.01

        # Regularization: encourage energy conservation
        E = self.atoms.E.mean() if len(self.atoms) > 0 else torch.tensor(0.0)
        energy_loss = (E - 0.5).abs() * 0.01

        total_loss = loss + sparsity_loss + energy_loss
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        self.training_loss = loss.item()

        return {
            "loss": loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "energy_loss": energy_loss.item(),
            "total_loss": total_loss.item(),
            "n_atoms": len(self.atoms),
            "n_aggregates": len(output["aggregates"]),
            "n_abstractions": len(output["abstractions"]),
            "consolidation_count": self.consolidation_count,
        }

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate text from a prompt.

        Parameters
        ----------
        prompt_ids : torch.Tensor [seq_len]
        max_length : int
        temperature : float
        top_k : int

        Returns
        -------
        generated : torch.Tensor [max_length]
        """
        generated = prompt_ids.clone()

        for _ in range(max_length):
            # Use last token as input
            input_token = generated[-1:].unsqueeze(0)

            # Forward pass
            output = self.forward(input_token)
            logits = output["logits"]

            # Sample
            token_id, _, _ = self.production.sample(
                self.state.get_field(),
                output["persistent_state"],
                output["abstractions"],
                temperature=temperature,
                top_k=top_k,
            )

            generated = torch.cat([generated, torch.tensor([token_id])])

            # Stop on EOS
            if token_id == 2:  # Assuming 2 is EOS
                break

        return generated

    def save(self, path: str) -> None:
        """Save the complete model state."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "state": self.state.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "interaction": self.interaction.state_dict(),
            "aggregation": self.aggregation.state_dict(),
            "abstraction": self.abstraction.state_dict(),
            "consolidation": self.consolidation.state_dict(),
            "production": self.production.state_dict(),
            "atoms": self.atoms.state_dict(),
            "energy_history": self.energy_history,
            "training_loss": self.training_loss,
            "consolidation_count": self.consolidation_count,
            "abstraction_count": self.abstraction_count,
            "config": {
                "vocab_size": self.encoder.vocab_size,
                "d_model": self.encoder.d_model,
                "n_modes": self.state.n_modes,
                "n_atoms_max": self.encoder.n_atoms_max,
            },
        }, path)

    def load(self, path: str) -> None:
        """Load the complete model state."""
        checkpoint = torch.load(path, map_location="cpu")

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.state.load_state_dict(checkpoint["state"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.interaction.load_state_dict(checkpoint["interaction"])
        self.aggregation.load_state_dict(checkpoint["aggregation"])
        self.abstraction.load_state_dict(checkpoint["abstraction"])
        self.consolidation.load_state_dict(checkpoint["consolidation"])
        self.production.load_state_dict(checkpoint["production"])
        self.atoms.load_state_dict(checkpoint["atoms"])
        self.energy_history = checkpoint.get("energy_history", [])
        self.training_loss = checkpoint.get("training_loss", 0.0)
        self.consolidation_count = checkpoint.get("consolidation_count", 0)
        self.abstraction_count = checkpoint.get("abstraction_count", 0)

    def get_shared_params_summary(self) -> dict:
        """Get a summary of shared parametersTheta."""
        return {
            "coupling_scale": self.dynamics.dynamics.coupling_scale.item(),
            "energy_decay": self.dynamics.dynamics.energy_decay.item(),
            "phase_sync": self.dynamics.dynamics.phase_sync.item(),
            "n_atoms": len(self.atoms),
            "n_modes": self.state.n_modes,
            "d_model": self.encoder.d_model,
        }
