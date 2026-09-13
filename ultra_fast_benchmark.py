"""
ATOM AI — Ultra Fast Benchmark (100 steps, 1 seed)
"""
import sys
sys.path.insert(0, '.')
import torch
import time
import json
import numpy as np
from pathlib import Path
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("ATOM AI — ULTRA FAST BENCHMARK")
print("="*60)

# Config
vocab_size = 500
d_model = 32
n_atoms_max = 64
batch_size = 4
seq_len = 16
max_steps = 100
lr = 3e-4
seed = 42

np.random.seed(seed)
tokens = np.random.randint(0, vocab_size, 10000)
sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
loader = InfiniteDataLoader(sequences, batch_size=batch_size, seq_len=seq_len)

# ATOM
print("\n--- ATOM ---")
torch.manual_seed(seed)
model = ToroidalFractalIntelligence(vocab_size=vocab_size, d_model=d_model, n_modes=d_model, n_atoms_max=n_atoms_max)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

start = time.time()
for step in range(1, max_steps + 1):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 25 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start
atom_loss = loss.item()
atom_atoms = len(model.atoms)
print(f"ATOM: loss={atom_loss:.4f}, atoms={atom_atoms}, time={elapsed:.2f}s")

# Transformer
print("\n--- Transformer ---")
torch.manual_seed(seed)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
model2 = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, d_model),
    TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64),
    torch.nn.Linear(d_model, vocab_size),
)
n_params2 = sum(p.numel() for p in model2.parameters())
print(f"Params: {n_params2:,}")

optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr)

start = time.time()
for step in range(1, max_steps + 1):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    emb = model2[0](token_ids).unsqueeze(0)
    out = model2[1](emb)
    out = out.squeeze(0)
    logits = model2[2](out.reshape(-1, d_model))
    
    loss = criterion(logits, target_ids)
    
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

elapsed2 = time.time() - start
trans_loss = loss.item()
print(f"Transformer: loss={trans_loss:.4f}, time={elapsed2:.2f}s")

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\n{'Metric':<25} {'ATOM':<15} {'Transformer':<15}")
print("-"*55)
print(f"{'Parameters':<25} {n_params:<15,} {n_params2:<15,}")
print(f"{'Final Loss':<25} {atom_loss:<15.4f} {trans_loss:<15.4f}")
print(f"{'Training Time':<25} {elapsed:<15.2f}s {elapsed2:<15.2f}s")
print(f"{'Atoms/Structures':<25} {atom_atoms:<15} {'N/A':<15}")
print()

# Efficiency
atom_eff = atom_atoms / max(n_params, 1)
print(f"{'Atoms/Parameter':<25} {atom_eff:<15.6f}")
print()

# Save
results = {
    "atom": {"params": n_params, "loss": atom_loss, "atoms": atom_atoms, "time": elapsed},
    "transformer": {"params": n_params2, "loss": trans_loss, "time": elapsed2},
}

results_dir = Path("./results/ultra_fast_benchmark")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {results_dir}/results.json")
print("="*60)
