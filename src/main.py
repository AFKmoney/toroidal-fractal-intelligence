"""
Main entry point for toroidal fractal intelligence.

Provides:
  - Training interface
  - Chat/inference interface
  - Model loading/saving
"""

from __future__ import annotations

import argparse
import torch
import json
from pathlib import Path

from .toroidal.model import ToroidalFractalIntelligence
from .training.trainer import ToroidalTrainer
from .io.tokenizer import ToroidalTokenizer
from .io.data import create_training_dataloader


def create_model(
    vocab_size: int = 32000,
    d_model: int = 256,
    n_modes: int = 256,
    n_atoms_max: int = 1024,
    device: str = "cpu",
) -> ToroidalFractalIntelligence:
    """Create a new toroidal fractal intelligence model."""
    model = ToroidalFractalIntelligence(
        vocab_size=vocab_size,
        d_model=d_model,
        n_modes=n_modes,
        n_atoms_max=n_atoms_max,
    )
    return model.to(device)


def train(
    model: ToroidalFractalIntelligence,
    dataset_name: str = "wikitext",
    dataset_subset: str = "wikitext-2-raw-v1",
    batch_size: int = 32,
    max_steps: int = 10000,
    learning_rate: float = 3e-4,
    save_dir: str = "./checkpoints",
    device: str = "cpu",
) -> dict:
    """Train the model."""
    # Create dataloader
    dataloader = create_training_dataloader(
        dataset_name=dataset_name,
        subset=dataset_subset,
        batch_size=batch_size,
        infinite=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Create trainer
    trainer = ToroidalTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        save_dir=save_dir,
    )

    # Train
    print("=" * 60)
    print("TOROIDAL FRACTAL INTELLIGENCE — TRAINING")
    print("=" * 60)
    print(f"Model config: d_model={model.encoder.d_model}, "
          f"n_modes={model.state.n_modes}, "
          f"n_atoms_max={model.encoder.n_atoms_max}")
    print(f"Dataset: {dataset_name}/{dataset_subset}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    result = trainer.train(max_steps=max_steps)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Total time: {result['total_time']:.1f}s")
    print("=" * 60)

    return result


def chat(
    model: ToroidalFractalIntelligence,
    tokenizer: ToroidalTokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate text from a prompt."""
    model.eval()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensor=True)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode
    return tokenizer.decode(generated)


def interactive_chat(model: ToroidalFractalIntelligence, tokenizer: ToroidalTokenizer) -> None:
    """Interactive chat loop."""
    print("=" * 60)
    print("TOROIDAL FRACTAL INTELLIGENCE — INTERACTIVE MODE")
    print("Type 'quit' to exit, 'status' for model info, 'save' to save")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "status":
                summary = model.get_shared_params_summary()
                print(f"\nModel status:")
                print(f"  Atoms: {summary['n_atoms']}")
                print(f"  Modes: {summary['n_modes']}")
                print(f"  d_model: {summary['d_model']}")
                print(f"  Coupling scale: {summary['coupling_scale']:.4f}")
                print(f"  Energy decay: {summary['energy_decay']:.4f}")
                print(f"  Phase sync: {summary['phase_sync']:.4f}")
                continue
            if user_input.lower() == "save":
                model.save("interactive_model.pt")
                print("Model saved to interactive_model.pt")
                continue

            if user_input:
                print("\nThoughts:", end=" ")
                response = chat(model, tokenizer, user_input)
                print(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toroidal Fractal Intelligence")
    parser.add_argument("--mode", choices=["train", "chat", "interactive"], default="train")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-subset", default="wikitext-2-raw-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-modes", type=int, default=256)
    parser.add_argument("--n-atoms-max", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    # Create or load model
    if args.checkpoint:
        model = create_model(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_modes=args.n_modes,
            n_atoms_max=args.n_atoms_max,
            device=args.device,
        )
        model.load(args.checkpoint)
    else:
        model = create_model(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_modes=args.n_modes,
            n_atoms_max=args.n_atoms_max,
            device=args.device,
        )

    tokenizer = ToroidalTokenizer("gpt2")

    if args.mode == "train":
        train(
            model=model,
            dataset_name=args.dataset,
            dataset_subset=args.dataset_subset,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            device=args.device,
        )
    elif args.mode == "chat":
        response = chat(model, tokenizer, args.prompt)
        print(f"\nResponse: {response}")
    elif args.mode == "interactive":
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
