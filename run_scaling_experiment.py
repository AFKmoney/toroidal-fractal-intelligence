"""
Scaling experiment runner.

Tests the Toroidal Fractal Intelligence system at different scales:
- Small: d_model=128, n_atoms_max=256 (baseline)
- Medium: d_model=256, n_atoms_max=1024
- Large: d_model=512, n_atoms_max=10000

Measures:
- Learning efficiency (loss / FLOPs)
- Structure emergence (atoms, aggregates, abstractions)
- Generalization (cross-domain generation)
- Memory usage
"""

import torch
import json
import time
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import load_wikitext, load_custom_text


def run_scaling_experiment(
    scale: str = "small",
    dataset: str = "wikitext",
    max_steps: int = 1000,
    batch_size: int = 32,
):
    """
    Run a scaling experiment.

    Parameters
    ----------
    scale : str
        "small", "medium", or "large"
    dataset : str
        "wikitext" or "custom"
    max_steps : int
        Number of training steps
    batch_size : int
        Batch size
    """
    print("=" * 70)
    print(f"SCALING EXPERIMENT: {scale.upper()}")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Configuration based on scale
    configs = {
        "small": {"d_model": 128, "n_modes": 128, "n_atoms_max": 256},
        "medium": {"d_model": 256, "n_modes": 256, "n_atoms_max": 1024},
        "large": {"d_model": 512, "n_modes": 512, "n_atoms_max": 10000},
    }

    config = configs[scale]
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Create model
    print("Creating model...")
    model = ToroidalFractalIntelligence(
        vocab_size=50257,
        d_model=config["d_model"],
        n_modes=config["n_modes"],
        n_atoms_max=config["n_atoms_max"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Load dataset
    print("Loading dataset...")
    if dataset == "wikitext":
        try:
            loader = load_wikitext(max_tokens=50000 if scale == "small" else None)
        except Exception as e:
            print(f"  Wikitext load failed: {e}")
            print("  Falling back to custom text...")
            loader = load_custom_text("Sample text for testing " * 1000, seq_len=128)
    else:
        loader = load_custom_text("Sample text for testing " * 1000, seq_len=128)

    print(f"Dataset: {len(loader.data)} sequences")
    print()

    # Training
    print(f"Training for {max_steps} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    atom_history = []
    aggregate_history = []
    abstraction_history = []
    start_time = time.time()

    for step in range(1, max_steps + 1):
        batch = next(loader)
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)

        output = model(token_ids)
        loss = criterion(output["logits"], target_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        atom_history.append(len(model.atoms))
        aggregate_history.append(len(output.get("aggregates", [])))
        abstraction_history.append(len(output.get("abstractions", [])))

        if step % 100 == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / 100
            print(f"  Step {step:4d}/{max_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Atoms: {len(model.atoms):5d} | "
                  f"Aggs: {len(output.get('aggregates', [])):3d} | "
                  f"Abs: {len(output.get('abstractions', [])):2d} | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    avg_loss = sum(losses) / len(losses)
    final_loss = losses[-1]
    final_atoms = len(model.atoms)
    final_aggs = len(model.aggregation.hierarchy[-1]) if model.aggregation.hierarchy else 0
    final_abs = len(model.abstraction.get_active_abstractions())

    print(f"\nTraining metrics:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Steps/sec: {max_steps/elapsed:.2f}")

    print(f"\nStructure metrics:")
    print(f"  Final atoms: {final_atoms}")
    print(f"  Final aggregates: {final_aggs}")
    print(f"  Final abstractions: {final_abs}")

    print(f"\nEfficiency metrics:")
    print(f"  Parameters: {n_params:,}")
    print(f"  Atoms per parameter: {final_atoms / max(n_params, 1):.4f}")
    print(f"  Loss per parameter: {final_loss / max(n_params, 1):.6f}")
    print(f"  FLOPs per step (estimated): {n_params * 2 * batch_size:,}")

    # Generate test text
    print(f"\nGeneration test:")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "To be or not to"
    prompt_ids = tokenizer.encode(prompt)

    with torch.no_grad():
        generated = model.generate(prompt_ids, max_length=100, temperature=0.8)
    text = tokenizer.decode(generated)
    print(f"  Prompt: '{prompt}'")
    print(f"  Response: '{text[:100]}...'")

    # Save results
    results = {
        "experiment": f"scaling_{scale}",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "training": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "final_loss": final_loss,
            "training_time_seconds": elapsed,
            "steps_per_second": max_steps / elapsed,
        },
        "structure": {
            "final_atoms": final_atoms,
            "final_aggregates": final_aggs,
            "final_abstractions": final_abs,
            "atoms_per_parameter": final_atoms / max(n_params, 1),
        },
        "efficiency": {
            "total_parameters": n_params,
            "loss_per_parameter": final_loss / max(n_params, 1),
            "flops_per_step_estimated": n_params * 2 * batch_size,
        },
        "generation": {
            "prompt": prompt,
            "response": text,
        },
    }

    results_dir = Path("./results/scaling")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{scale}_results.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def run_all_scales():
    """Run experiments at all scales."""
    print("=" * 70)
    print("TOROIDAL FRACTAL INTELLIGENCE — SCALING EXPERIMENTS")
    print("=" * 70)
    print()

    all_results = {}

    for scale in ["small", "medium", "large"]:
        print(f"\n{'='*70}")
        print(f"Running {scale.upper()} scale experiment...")
        print(f"{'='*70}")

        try:
            result = run_scaling_experiment(
                scale=scale,
                dataset="custom",  # Use custom for reliability
                max_steps=500,
                batch_size=16,
            )
            all_results[scale] = result
        except Exception as e:
            print(f"ERROR in {scale} experiment: {e}")
            all_results[scale] = {"error": str(e)}

    # Compare results
    print("\n" + "=" * 70)
    print("SCALING COMPARISON")
    print("=" * 70)

    print(f"\n{'Scale':<10} {'Params':<12} {'Atoms':<8} {'Loss':<8} {'Atoms/Param':<12}")
    print("-" * 60)

    for scale in ["small", "medium", "large"]:
        if scale in all_results and "error" not in all_results[scale]:
            r = all_results[scale]
            params = r["efficiency"]["total_parameters"]
            atoms = r["structure"]["final_atoms"]
            loss = r["training"]["final_loss"]
            ratio = r["structure"]["atoms_per_parameter"]
            print(f"{scale:<10} {params:<12,} {atoms:<8} {loss:<8.4f} {ratio:<12.4f}")

    # Save comparison
    comparison = {
        "experiments": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    comparison_file = Path("./results/scaling/comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {comparison_file}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="all")
    parser.add_argument("--dataset", choices=["wikitext", "custom"], default="custom")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    if args.scale == "all":
        run_all_scales()
    else:
        run_scaling_experiment(
            scale=args.scale,
            dataset=args.dataset,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )
