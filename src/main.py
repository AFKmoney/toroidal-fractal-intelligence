"""Command-line entry point for the ATOM-native Toroidal Fractal Intelligence path.

The active text path is byte/atom native:
    raw UTF-8 -> Atomizer -> AtomPacket -> AtomCompiler -> toroidal core

No GPT-2 tokenizer or token-ID vocabulary is used by this entry point.
"""
from __future__ import annotations

import argparse

from .atom_native import AtomNativeModel
from .io.atomizer import Atomizer


def create_model(
    d_model: int = 8,
    n_modes: int = 8,
    n_atoms_max: int = 128,
    max_span_bytes: int = 32,
    device: str = "cpu",
) -> AtomNativeModel:
    model = AtomNativeModel(
        d_model=d_model,
        n_modes=n_modes,
        n_atoms_max=n_atoms_max,
        max_payload_bytes=max_span_bytes,
        atomizer=Atomizer(max_span_bytes=max_span_bytes),
    )
    return model.to(device)


def chat(
    model: AtomNativeModel,
    prompt: str,
    max_packets: int = 8,
    max_length: int | None = 200,
    temperature: float = 1.0,
    top_k: int | None = 50,
) -> str:
    generated = model.generate_packets(
        prompt,
        max_packets=max_packets,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        deterministic=False,
    )
    return generated.decode("utf-8", errors="replace")


def interactive_chat(model: AtomNativeModel) -> None:
    print("=" * 60)
    print("TOROIDAL FRACTAL INTELLIGENCE — ATOM-NATIVE MODE")
    print("Raw UTF-8 -> Atomizer -> AtomPacket -> Toroidal core")
    print("Type 'quit' to exit, 'status' for model info, 'save' to save")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "status":
                print(f"\nAtoms: {len(model.core.atoms)}")
                print(f"Modes: {model.core.state.n_modes}")
                print(f"d_model: {model.core.encoder.d_model}")
                print(f"Atomizer: {model.atomizer.VERSION}")
                print(f"Feature dim: {model.atomizer.feature_dim}")
                continue
            if user_input.lower() == "save":
                model.save("atom_native_model.pt")
                print("Model saved to atom_native_model.pt")
                continue
            if user_input:
                print("\nResponse:", chat(model, user_input))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as exc:
            print(f"\nError: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toroidal Fractal Intelligence — ATOM native")
    parser.add_argument("--mode", choices=["chat", "interactive"], default="chat")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--n-atoms-max", type=int, default=128)
    parser.add_argument("--max-span-bytes", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = create_model(
        d_model=args.d_model,
        n_modes=args.n_modes,
        n_atoms_max=args.n_atoms_max,
        max_span_bytes=args.max_span_bytes,
        device=args.device,
    )
    if args.checkpoint:
        model.load(args.checkpoint)

    if args.mode == "chat":
        print(chat(model, args.prompt, args.max_packets, args.max_length, args.temperature, args.top_k))
    else:
        interactive_chat(model)


if __name__ == "__main__":
    main()
