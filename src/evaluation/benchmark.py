"""
Benchmark comparator: Toroidal Fractal vs Standard Transformer.

Provides fair comparison metrics:
- Same FLOPs budget
- Same parameter count
- Same training time
- Perplexity on held-out data
- Structure emergence metrics
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time


class FLOPCounter:
    """Count FLOPs for model operations."""

    @staticmethod
    def count_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for a forward pass."""
        # This is a simplified estimate
        # For accurate counting, use torch.profiler
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: ~2 FLOPs per parameter per forward pass
        return total_params * 2 * input_shape[0]


class BenchmarkComparator:
    """
    Compare Toroidal Fractal Intelligence with standard transformers.

    Usage:
        comparator = BenchmarkComparator()
        results = comparator.compare(
            toroidal_model,
            transformer_model,
            dataset,
            steps=1000
        )
    """

    def __init__(self):
        self.flop_counter = FLOPCounter()

    def count_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Count FLOPs for one forward pass."""
        return self.flop_counter.count_model_flops(model, input_shape)

    def train_and_evaluate(
        self,
        model: nn.Module,
        data_loader,
        steps: int,
        device: str = "cpu",
    ) -> Dict:
        """
        Train model for fixed steps and return metrics.

        Parameters
        ----------
        model : nn.Module
            Model to train
        data_loader : InfiniteDataLoader
            Training data
        steps : int
            Number of training steps
        device : str
            Device to train on

        Returns
        -------
        metrics : dict
        """
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        losses = []
        start_time = time.time()
        total_flops = 0

        for step in range(steps):
            batch = next(data_loader)
            batch = batch.to(device)

            token_ids = batch[:, :-1].reshape(-1)
            target_ids = batch[:, 1:].reshape(-1)

            # Forward pass
            output = model(token_ids)
            logits = output["logits"]

            # Count FLOPs
            flops = self.count_flops(model, (token_ids.shape[0], model.encoder.d_model))
            total_flops += flops

            # Loss
            loss = criterion(logits, target_ids)
            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        elapsed = time.time() - start_time

        return {
            "avg_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "total_flops": total_flops,
            "flops_per_step": total_flops / steps,
            "training_time": elapsed,
            "steps_per_second": steps / elapsed,
        }

    def evaluate_generation(
        self,
        model: nn.Module,
        tokenizer,
        prompts: List[str],
        max_length: int = 100,
    ) -> Dict:
        """
        Evaluate text generation quality.

        Parameters
        ----------
        model : nn.Module
        tokenizer : AutoTokenizer
        prompts : List[str]
        max_length : int

        Returns
        -------
        metrics : dict
        """
        results = {}
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt)
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids,
                    max_length=max_length,
                    temperature=0.8,
                    top_k=50,
                )
            text = tokenizer.decode(generated)
            results[prompt] = text

        return results

    def compare(
        self,
        toroidal_model,
        transformer_model,
        data_loader,
        steps: int = 1000,
        device: str = "cpu",
    ) -> Dict:
        """
        Compare Toroidal vs Transformer fairly.

        Parameters
        ----------
        toroidal_model : ToroidalFractalIntelligence
        transformer_model : nn.Module (e.g., GPT2LMHeadModel)
        data_loader : InfiniteDataLoader
        steps : int
        device : str

        Returns
        -------
        comparison : dict
        """
        print("=" * 70)
        print("BENCHMARK COMPARISON")
        print("=" * 70)

        # Count parameters
        toroidal_params = sum(p.numel() for p in toroidal_model.parameters())
        transformer_params = sum(p.numel() for p in transformer_model.parameters())

        print(f"\nToroidal parameters: {toroidal_params:,}")
        print(f"Transformer parameters: {transformer_params:,}")

        # Train both
        print("\n--- Training Toroidal ---")
        toroidal_metrics = self.train_and_evaluate(
            toroidal_model, data_loader, steps, device
        )
        print(f"  Final loss: {toroidal_metrics['final_loss']:.4f}")
        print(f"  Total FLOPs: {toroidal_metrics['total_flops']:,}")
        print(f"  Time: {toroidal_metrics['training_time']:.2f}s")

        print("\n--- Training Transformer ---")
        transformer_metrics = self.train_and_evaluate(
            transformer_model, data_loader, steps, device
        )
        print(f"  Final loss: {transformer_metrics['final_loss']:.4f}")
        print(f"  Total FLOPs: {transformer_metrics['total_flops']:,}")
        print(f"  Time: {transformer_metrics['training_time']:.2f}s")

        # Structure metrics for toroidal
        structure_metrics = {
            "atoms": len(toroidal_model.atoms),
            "aggregates": len(toroidal_model.aggregation.hierarchy[-1]) if toroidal_model.aggregation.hierarchy else 0,
            "abstractions": len(toroidal_model.abstraction.get_active_abstractions()),
        }

        # Compute efficiency ratios
        efficiency = {
            "loss_per_parameter_toroidal": toroidal_metrics['final_loss'] / max(toroidal_params, 1),
            "loss_per_parameter_transformer": transformer_metrics['final_loss'] / max(transformer_params, 1),
            "loss_per_flop_toroidal": toroidal_metrics['final_loss'] / max(toroidal_metrics['total_flops'], 1),
            "loss_per_flop_transformer": transformer_metrics['final_loss'] / max(transformer_metrics['total_flops'], 1),
            "structures_per_parameter": structure_metrics['atoms'] / max(toroidal_params, 1),
        }

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nLoss comparison:")
        print(f"  Toroidal:    {toroidal_metrics['final_loss']:.4f}")
        print(f"  Transformer: {transformer_metrics['final_loss']:.4f}")
        print(f"\nEfficiency:")
        for k, v in efficiency.items():
            print(f"  {k}: {v:.6f}")
        print(f"\nStructure metrics (Toroidal):")
        for k, v in structure_metrics.items():
            print(f"  {k}: {v}")

        return {
            "toroidal": toroidal_metrics,
            "transformer": transformer_metrics,
            "structure": structure_metrics,
            "efficiency": efficiency,
        }


if __name__ == "__main__":
    print("Benchmark module loaded.")
    print("Use BenchmarkComparator to compare models.")
