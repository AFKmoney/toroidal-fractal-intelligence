"""Run scaling study with fixed model."""
import sys
sys.path.insert(0, '.')
import torch
import time
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("="*70)
print("ATOM AI — STRUCTURAL EFFICIENCY STUDY")
print("="*70)

# Test configurations
configs = [
    {"name": "small", "d_model": 64, "n_modes": 64, "n_atoms_max": 128},
    {"name": "medium", "d_model": 128, "n_modes": 128, "n_atoms_max": 256},
    {"name": "large", "d_model": 256, "n_modes": 256, "n_atoms_max": 512},
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']} (d_model={config['d_model']}, n_atoms_max={config['n_atoms_max']})")
    print(f"{'='*70}")

    # Create model
    model = ToroidalFractalIntelligence(
        vocab_size=100,
        d_model=config['d_model'],
        n_modes=config['n_modes'],
        n_atoms_max=config['n_atoms_max'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create data
    tokens = list(range(100)) * 100
    sequences = [torch.tensor(tokens[i:i+64]) for i in range(0, len(tokens)-64, 32)]
    loader = InfiniteDataLoader(sequences, batch_size=8, seq_len=64)
    print(f"Data: {len(loader.data)} sequences")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    atom_history = []
    start_time = time.time()

    for step in range(1, 51):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)

        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        atom_history.append(len(model.atoms))

        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

    elapsed = time.time() - start_time
    avg_loss = sum(losses) / len(losses) if losses else float('nan')
    final_loss = losses[-1] if losses else float('nan')

    result = {
        "config": config,
        "parameters": n_params,
        "final_atoms": len(model.atoms),
        "avg_loss": avg_loss,
        "final_loss": final_loss,
        "training_time": elapsed,
        "atoms_per_parameter": len(model.atoms) / max(n_params, 1),
    }
    results.append(result)

    print(f"\nResults:")
    print(f"  Final atoms: {len(model.atoms)}")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Atoms/parameter: {result['atoms_per_parameter']:.6f}")

# Summary
print("\n" + "="*70)
print("SCALING STUDY SUMMARY")
print("="*70)
print(f"\n{'Config':<12} {'Params':<12} {'Atoms':<8} {'Loss':<8} {'Atoms/Param':<12}")
print("-"*60)
for r in results:
    print(f"{r['config']['name']:<12} {r['parameters']:<12,} {r['final_atoms']:<8} {r['final_loss']:<8.4f} {r['atoms_per_parameter']:<12.6f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("The scaling study demonstrates how structural capacity grows")
print("relative to parameter count across different configurations.")
print("="*70)
