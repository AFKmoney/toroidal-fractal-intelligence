"""Complete scaling study - 100 steps per config."""
import sys
sys.path.insert(0, '.')
import torch
import time
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("ATOM AI — FINAL SCALING STUDY")
print("="*70)

configs = [
    ("small", 64, 128),
    ("medium", 128, 256),
]

all_results = []

for name, d_model, n_atoms in configs:
    print(f"\n--- {name.upper()} ---")
    
    model = ToroidalFractalIntelligence(vocab_size=100, d_model=d_model, n_modes=d_model, n_atoms_max=n_atoms)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    
    tokens = list(range(100)) * 100
    seq_len = min(64, d_model)
    sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
    loader = InfiniteDataLoader(sequences, batch_size=8, seq_len=seq_len)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    start = time.time()
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
        
        if step % 25 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")
    
    elapsed = time.time() - start
    result = {
        "name": name,
        "params": n_params,
        "atoms": len(model.atoms),
        "loss": loss.item(),
        "time": elapsed,
        "atoms_per_param": len(model.atoms) / n_params,
    }
    all_results.append(result)
    print(f"  Final: atoms={len(model.atoms)}, loss={loss.item():.4f}, time={elapsed:.1f}s")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\n{'Config':<10} {'Params':<12} {'Atoms':<8} {'Loss':<8} {'At/Param':<12}")
print("-"*55)
for r in all_results:
    print(f"{r['name']:<10} {r['params']:<12,} {r['atoms']:<8} {r['loss']:<8.4f} {r['atoms_per_param']:<12.6f}")

print("\n" + "="*70)
print("SUCCESS — Scaling study complete!")
print("="*70)
