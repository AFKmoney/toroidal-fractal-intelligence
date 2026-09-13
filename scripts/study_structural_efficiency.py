"""
ATOM AI Scaling Study: Structural Efficiency vs Compute Cost

This module implements the core hypothesis test:
"Can a small number of parameters generate and manipulate
a much larger amount of structure without exploding compute cost?"

Key metrics:
- Structural Efficiency = structures_created / FLOPs
- Learning Efficiency = validation_improvement / FLOPs

The goal is NOT to maximize compute, but to maximize
structure per unit of compute.
"""

from __future__ import annotations

import torch
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ComputeMetrics:
    """Track compute and memory usage."""
    flops_estimated: int = 0
    flops_actual: int = 0  # Will use torch.profiler in future
    memory_allocated_gb: float = 0.0
    memory_peak_gb: float = 0.0
    time_seconds: float = 0.0
    tokens_processed: int = 0
    structures_created: int = 0
    structures_consolidated: int = 0


@dataclass
class StructuralMetrics:
    """Track emergent structures."""
    atoms_created: int = 0
    atoms_active: int = 0
    aggregates_formed: int = 0
    abstractions_created: int = 0
    consolidation_events: int = 0
    max_hierarchy_depth: int = 0


@dataclass
class LearningMetrics:
    """Track learning progress."""
    training_loss: float = 0.0
    validation_loss: float = 0.0
    loss_improvement: float = 0.0
    perplexity: float = 0.0


class StructuralEfficiencyTracker:
    """
    Core metric calculator for the scaling study.

    Tracks:
    1. Structural Efficiency: structures / FLOPs
    2. Learning Efficiency: loss_improvement / FLOPs
    3. Memory Efficiency: structures / RAM_GB
    """

    def __init__(self):
        self.compute = ComputeMetrics()
        self.structural = StructuralMetrics()
        self.learning = LearningMetrics()

    def estimate_flops_per_step(self, model, batch_size: int, seq_len: int) -> int:
        """Estimate FLOPs for one training step."""
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Forward pass: ~2 * n_params * batch_size * seq_len
        forward_flops = 2 * n_params * batch_size * seq_len

        # Backward pass: ~2x forward
        backward_flops = 2 * forward_flops

        # Total
        total = forward_flops + backward_flops

        self.compute.flops_estimated += total
        self.compute.tokens_processed += batch_size * seq_len

        return total

    def record_structure_creation(self, atoms: int, aggregates: int, abstractions: int, consolidations: int):
        """Record structure creation metrics."""
        self.structural.atoms_created = atoms
        self.structural.aggregates_formed = aggregates
        self.structural.abstractions_created = abstractions
        self.structural.consolidation_events = consolidations

    def update(self, model, batch_size: int, seq_len: int) -> Dict:
        """Update all metrics after one step."""
        # Estimate FLOPs
        flops = self.estimate_flops_per_step(model, batch_size, seq_len)

        # Record structures
        n_atoms = len(model.atoms) if hasattr(model, 'atoms') else 0
        n_aggs = len(model.aggregation.hierarchy[-1]) if hasattr(model, 'aggregation') and model.aggregation.hierarchy else 0
        n_abs = len(model.abstraction.get_active_abstractions()) if hasattr(model, 'abstraction') else 0

        self.structural.atoms_active = n_atoms
        self.structural.aggregates_formed = n_aggs
        self.structural.abstractions_created = n_abs

        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        self.compute.memory_allocated_gb = current / 1e9
        self.compute.memory_peak_gb = peak / 1e9

        return {
            "flops": flops,
            "atoms": n_atoms,
            "aggregates": n_aggs,
            "abstractions": n_abs,
            "memory_gb": self.compute.memory_allocated_gb,
        }

    def compute_structural_efficiency(self) -> float:
        """
        Structural Efficiency = useful structures / FLOPs

        Useful structures = atoms + aggregates*10 + abstractions*100
        (weighted by estimated information content)
        """
        if self.compute.flops_estimated == 0:
            return 0.0

        useful_structures = (
            self.structural.atoms_active * 1 +
            self.structural.aggregates_formed * 10 +
            self.structural.abstractions_created * 100
        )

        return useful_structures / max(self.compute.flops_estimated, 1)

    def compute_learning_efficiency(self) -> float:
        """
        Learning Efficiency = validation_improvement / FLOPs

        Measures how much learning per FLOP.
        """
        if self.compute.flops_estimated == 0:
            return 0.0

        return self.learning.loss_improvement / max(self.compute.flops_estimated, 1)

    def compute_memory_efficiency(self) -> float:
        """Memory Efficiency = structures / RAM_GB"""
        if self.compute.memory_peak_gb == 0:
            return 0.0

        useful_structures = (
            self.structural.atoms_active * 1 +
            self.structural.aggregates_formed * 10 +
            self.structural.abstractions_created * 100
        )

        return useful_structures / self.compute.memory_peak_gb

    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        return {
            "compute": asdict(self.compute),
            "structural": asdict(self.structural),
            "learning": asdict(self.learning),
            "structural_efficiency": self.compute_structural_efficiency(),
            "learning_efficiency": self.compute_learning_efficiency(),
            "memory_efficiency": self.compute_memory_efficiency(),
        }


class ScalingStudy:
    """
    Main orchestrator for the structural efficiency study.

    Tests the hypothesis:
    "Small parameters can generate large structure without
    proportional compute cost."
    """

    def __init__(
        self,
        model,
        data_loader,
        tokenizer=None,
        max_steps: int = 1000,
        checkpoint_interval: int = 100,
    ):
        self.model = model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.checkpoint_interval = checkpoint_interval

        self.tracker = StructuralEfficiencyTracker()
        self.loss_history = []
        self.step = 0
        self.data_iterator = None

    def train_step(self, optimizer, batch_size: int = 32, seq_len: int = 128):
        """Single training step with metrics tracking."""
        start_time = time.time()
        tracemalloc.start()

        # Initialize iterator if needed
        if self.data_iterator is None:
            self.data_iterator = iter(self.data_loader)

        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)

        # Forward
        output = self.model(token_ids)
        logits = output["logits"]

        # Loss
        loss = torch.nn.CrossEntropyLoss()(logits, target_ids)
        self.loss_history.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        metrics = self.tracker.update(self.model, batch_size, seq_len)
        self.tracker.compute.time_seconds += time.time() - start_time
        self.step += 1

        tracemalloc.stop()

        return {
            "step": self.step,
            "loss": loss.item(),
            "metrics": metrics,
        }

    def run_study(self) -> Dict:
        """Run the complete scaling study."""
        print("=" * 70)
        print("ATOM AI — STRUCTURAL EFFICIENCY STUDY")
        print("=" * 70)
        print()

        # Initial state
        initial_atoms = len(self.model.atoms) if hasattr(self.model, 'atoms') else 0
        initial_flops = self.tracker.compute.flops_estimated

        print(f"Initial atoms: {initial_atoms}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)

        print(f"Training for {self.max_steps} steps...")
        print()

        for step in range(1, self.max_steps + 1):
            result = self.train_step(optimizer)

            # Progress
            if step % 50 == 0:
                self._log_progress(result)

            # Checkpoint
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step)

        # Final metrics
        final_metrics = self.tracker.get_summary()
        final_metrics["training_steps"] = self.max_steps
        final_metrics["final_loss"] = self.loss_history[-1] if self.loss_history else None
        final_metrics["avg_loss"] = sum(self.loss_history) / len(self.loss_history) if self.loss_history else None

        # Compute efficiency
        final_metrics["structural_efficiency"] = self.tracker.compute_structural_efficiency()
        final_metrics["memory_efficiency"] = self.tracker.compute_memory_efficiency()

        print()
        print("=" * 70)
        print("STUDY COMPLETE — RESULTS")
        print("=" * 70)
        self._print_summary(final_metrics)

        return final_metrics

    def _log_progress(self, result: Dict):
        """Log training progress."""
        flops = self.tracker.compute.flops_estimated
        atoms = result["metrics"]["atoms"]
        aggs = result["metrics"]["aggregates"]
        abs_count = result["metrics"]["abstractions"]
        efficiency = self.tracker.compute_structural_efficiency()

        print(f"Step {result['step']:4d}/{self.max_steps} | "
              f"Loss: {result['loss']:.4f} | "
              f"Atoms: {atoms:4d} | "
              f"Aggs: {aggs:3d} | "
              f"Abs: {abs_count:2d} | "
              f"Efficiency: {efficiency:.2e}")

    def _save_checkpoint(self, step: int):
        """Save checkpoint with metrics."""
        checkpoint_dir = Path("./checkpoints/study")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step": step,
            "model_state": self.model.state_dict(),
            "tracker_state": self.tracker.get_summary(),
            "loss_history": self.loss_history[-100:],
        }

        checkpoint_path = checkpoint_dir / f"step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    def _print_summary(self, metrics: Dict):
        """Print final summary."""
        print(f"\nFinal Metrics:")
        print(f"  Training steps: {metrics['training_steps']}")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Average loss: {metrics['avg_loss']:.4f}")
        print()
        print(f"  Total FLOPs (est): {metrics['compute']['flops_estimated']:,}")
        print(f"  Memory peak: {metrics['compute']['memory_peak_gb']:.2f} GB")
        print(f"  Time: {metrics['compute']['time_seconds']:.1f}s")
        print()
        print(f"  Final atoms: {metrics['structural']['atoms_active']}")
        print(f"  Final aggregates: {metrics['structural']['aggregates_formed']}")
        print(f"  Final abstractions: {metrics['structural']['abstractions_created']}")
        print()
        print(f"  Structural Efficiency: {metrics['structural_efficiency']:.2e} structures/FLOP")
        print(f"  Memory Efficiency: {metrics['memory_efficiency']:.2e} structures/GB")
        print()
        print("=" * 70)

    def run_scaling_series(
        self,
        atom_counts: List[int] = [256, 512, 1024, 2048, 4096],
        d_models: List[int] = [128, 256],
    ) -> List[Dict]:
        """
        Run scaling study across different sizes.

        Parameters
        ----------
        atom_counts : List[int]
            List of n_atoms_max values to test
        d_models : List[int]
            List of d_model values to test

        Returns
        -------
        results : List[Dict]
            Results for each configuration
        """
        print("=" * 70)
        print("ATOM AI — SCALING SERIES")
        print("=" * 70)
        print()

        all_results = []

        for n_atoms in atom_counts:
            for d_model in d_models:
                print(f"\n{'='*70}")
                print(f"Testing: n_atoms={n_atoms}, d_model={d_model}")
                print(f"{'='*70}")

                # Create model
                from src.toroidal.model import ToroidalFractalIntelligence
                model = ToroidalFractalIntelligence(
                    vocab_size=50257,
                    d_model=d_model,
                    n_modes=d_model,
                    n_atoms_max=n_atoms,
                )

                # Load data (reuse same loader for fairness)
                # In practice, would load same dataset
                from src.io.data import load_custom_text
                data = "Sample text for scaling study " * 2000
                loader = load_custom_text(data, seq_len=64)

                # Run study
                study = ScalingStudy(
                    model=model,
                    data_loader=loader,
                    max_steps=200,
                )

                result = study.run_study()
                result["config"] = {
                    "n_atoms_max": n_atoms,
                    "d_model": d_model,
                    "n_modes": d_model,
                }

                all_results.append(result)

                # Save individual result
                results_dir = Path("./results/scaling_study")
                results_dir.mkdir(parents=True, exist_ok=True)
                result_file = results_dir / f"atoms_{n_atoms}_dmodel_{d_model}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

        # Save comparison
        comparison = {
            "experiments": all_results,
            "timestamp": str(time.time()),
        }
        comparison_file = results_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n{'='*70}")
        print("SCALING SERIES COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {comparison_file}")

        return all_results


if __name__ == "__main__":
    import argparse
    from src.toroidal.model import ToroidalFractalIntelligence
    from src.io.data import load_custom_text

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "series"], default="single")
    parser.add_argument("--n-atoms", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=500)

    args = parser.parse_args()

    if args.mode == "series":
        # Use default config for series
        from src.toroidal.model import ToroidalFractalIntelligence
        from src.io.data import load_custom_text
        model = ToroidalFractalIntelligence(
            vocab_size=50257,
            d_model=128,
            n_modes=128,
            n_atoms_max=256,
        )
        data = "Sample text for scaling study " * 100
        loader = load_custom_text(data, seq_len=128)
        study = ScalingStudy(model=model, data_loader=loader)
        study.run_scaling_series()
    else:
        # Single experiment
        model = ToroidalFractalIntelligence(
            vocab_size=50257,
            d_model=args.d_model,
            n_modes=args.d_model,
            n_atoms_max=args.n_atoms,
        )

        data = "To be or not to be that is the question " * 50
        loader = load_custom_text(data, seq_len=128)

        study = ScalingStudy(
            model=model,
            data_loader=loader,
            max_steps=args.max_steps,
        )

        results = study.run_study()
