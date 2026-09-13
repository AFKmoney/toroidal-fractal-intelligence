"""Final scaling study - fixed and complete."""
import sys
sys.path.insert(0, '.')
import torch
import time
import json
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("="*70)
print("ATOM AI — FINAL SCALING STUDY (FIXED)")
print("="*70)

configs = [
    {"name": "small", "d_model": 64, "n_modes": 64, "n_atoms_max": 128},
    {"name": "medium", "d_model": 128, "n_modes": 128, "n_atoms_max": 256},
]

results = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")

    model = ToroidalFractalIntelligence(
        vocab_size=100,
        d_model=config['d_model'],
        n_modes=config['n_modes'],
        n_atoms_max=config['n_atoms_max'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    tokens = list(range(100)) * 100
    seq_len = min(64, config['d_model'])
    sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
    loader = InfiniteDataLoader(sequences, batch_size=8, seq_len=seq_len)
    print(f"Data: {len(sequences)} sequences")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    start_time = time.time()

    for step in range(1, 101):
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

        if step % 25 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

    elapsed = time.time() - start_time
    initial_loss = losses[0]
    final_loss = losses[-1]

    result = {
        "config": config,
        "parameters": n_params,
        "final_atoms": len(model.atoms),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement": initial_loss - final_loss,
        "training_time": elapsed,
        "atoms_per_parameter": len(model.atoms) / max(n_params, 1),
    }
    results.append(result)

    print(f"\nResults:")
    print(f"  Final atoms: {len(model.atoms)}")
    print(f"  Loss improvement: {initial_loss - final_loss:.4f}")
    print(f"  Atoms/parameter: {result['atoms_per_parameter']:.6f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Config':<12} {'Params':<12} {'Atoms':<8} {'LossImp':<10} {'At/Param':<12}")
print("-"*60)
for r in results:
    print(f"{r['config']['name']:<12} {r['parameters']:<12,} {r['final_atoms']:<8} {r['loss_improvement']:<10.4f} {r['atoms_per_parameter']:<12.6f}")

with open("./results/final_scaling.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: ./results/final_scaling.json")
print("="*70)
