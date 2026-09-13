"""
Experiment runner for toroidal fractal intelligence.

Runs a complete training experiment and logs results.
"""

import torch
import json
from pathlib import Path
from datetime import datetime

from toroidal_fractal_intelligence import create_model, train
from toroidal_fractal_intelligence.io.tokenizer import ToroidalTokenizer
from toroidal_fractal_intelligence.evaluation.metrics import ToroidalMetrics


def run_experiment(
    exp_name: str,
    d_model: int = 256,
    n_modes: int = 256,
    n_atoms_max: int = 1024,
    max_steps: int = 5000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
):
    """
    Run a complete training experiment.

    Parameters
    ----------
    exp_name : str
        Experiment name for logging
    d_model : int
        Hidden dimension
    n_modes : int
        Number of spectral modes
    n_atoms_max : int
        Maximum concurrent atoms
    max_steps : int
        Training steps
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    """
    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"d_model={d_model}, n_modes={n_modes}, n_atoms_max={n_atoms_max}")
    print(f"max_steps={max_steps}, batch_size={batch_size}, lr={learning_rate}")
    print("=" * 70)

    # Create model
    model = create_model(
        vocab_size=32000,
        d_model=d_model,
        n_modes=n_modes,
        n_atoms_max=n_atoms_max,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create tokenizer
    tokenizer = ToroidalTokenizer("gpt2")

    # Run training
    result = train(
        model=model,
        dataset_name="wikitext",
        dataset_subset="wikitext-2-raw-v1",
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        save_dir=f"./checkpoints/{exp_name}",
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Test generation
    test_prompts = [
        "The future of AI is",
        "Toroidal fractal intelligence",
        "In the beginning was",
    ]

    generations = {}
    for prompt in test_prompts:
        response = model.generate(
            tokenizer.encode(prompt, return_tensor=True),
            max_length=100,
            temperature=0.8,
        )
        text = tokenizer.decode(response)
        generations[prompt] = text
        print(f"\nPrompt: {prompt}")
        print(f"Response: {text[:200]}...")

    # Compute metrics
    n_params = sum(p.numel() for p in model.parameters())
    n_atoms = len(model.atoms)
    efficiency = {
        "atoms_per_parameter": n_atoms / max(n_params, 1),
        "total_parameters": n_params,
        "final_atoms": n_atoms,
    }

    # Save results
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "experiment": exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "d_model": d_model,
            "n_modes": n_modes,
            "n_atoms_max": n_atoms_max,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "training": result,
        "generations": generations,
        "efficiency": efficiency,
        "model_summary": model.get_shared_params_summary(),
    }

    results_file = results_dir / f"{exp_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Total parameters: {n_params:,}")
    print(f"Final atoms: {n_atoms}")
    print(f"Atoms per parameter: {efficiency['atoms_per_parameter']:.4f}")
    print(f"Training time: {result['total_time']:.1f}s")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-modes", type=int, default=256)
    parser.add_argument("--n-atoms-max", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    run_experiment(
        exp_name=args.name,
        d_model=args.d_model,
        n_modes=args.n_modes,
        n_atoms_max=args.n_atoms_max,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
