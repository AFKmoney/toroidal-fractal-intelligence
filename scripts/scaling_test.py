"""Scaling study for ATOM."""
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Running scaling study...")

configs = [
    ("small", 32, 64),
    ("medium", 64, 128),
    ("large", 128, 256),
]

results = []

for name, d_model, n_atoms in configs:
    print(f"\n--- {name.upper()} ---")
    
    model = ToroidalFractalIntelligence(vocab_size=100, d_model=d_model, n_modes=d_model, n_atoms_max=n_atoms)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    
    np.random.seed(42)
    tokens = np.random.randint(0, 100, 2000)
    seq_len = min(32, d_model)
    sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
    loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=seq_len)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    start = time.time()
    for step in range(1, 101):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start
    result = {
        "config": name,
        "params": n_params,
        "atoms": len(model.atoms),
        "loss": loss.item(),
        "time": elapsed,
        "atoms_per_param": len(model.atoms) / n_params,
    }
    results.append(result)
    print(f"Result: atoms={len(model.atoms)}, loss={loss.item():.4f}, time={elapsed:.2f}s")

print("\n=== SCALING STUDY RESULTS ===")
print(f"{'Config':<10} {'Params':<12} {'Atoms':<8} {'Loss':<8} {'At/Param':<12}")
print("-"*55)
for r in results:
    print(f"{r['config']:<10} {r['params']:<12,} {r['atoms']:<8} {r['loss']:<8.4f} {r['atoms_per_param']:<12.6f}")

print("\nScaling study complete!")
