"""Test fixed ATOM with atom reuse and compact encoder."""
import sys
sys.path.insert(0, '.')
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("="*60)
print("ATOM FIX TEST — Atom Reuse + Compact Encoder")
print("="*60)

# Config
vocab_size = 200
d_model = 32
n_atoms_max = 64
batch_size = 4
seq_len = 16
max_steps = 200
lr = 3e-4

np.random.seed(42)
tokens = np.random.randint(0, vocab_size, 10000)
sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
loader = InfiniteDataLoader(sequences, batch_size=batch_size, seq_len=seq_len)

# Create model
model = ToroidalFractalIntelligence(
    vocab_size=vocab_size,
    d_model=d_model,
    n_modes=d_model,
    n_atoms_max=n_atoms_max,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params: {n_params:,}")
print(f"Target: <100K params, >50 atoms after 200 steps")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

losses = []
atoms_history = []
start = time.time()

for step in range(1, max_steps + 1):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    losses.append(loss.item())
    atoms_history.append(len(model.atoms))
    
    if step % 50 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start

# Results
final_atoms = len(model.atoms)
final_loss = losses[-1]
atoms_per_param = final_atoms / n_params

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Final atoms: {final_atoms}")
print(f"Final loss: {final_loss:.4f}")
print(f"Atoms/parameter: {atoms_per_param:.6f}")
print(f"Training time: {elapsed:.2f}s")
print(f"Params: {n_params:,}")

# Target check
print(f"\n{'='*60}")
print("TARGET CHECK")
print(f"{'='*60}")
if n_params < 100000:
    print(f"[PASS] Params < 100K: {n_params:,}")
else:
    print(f"[FAIL] Params > 100K: {n_params:,}")

if final_atoms >= 50:
    print(f"[PASS] Atoms >= 50: {final_atoms}")
else:
    print(f"[FAIL] Atoms < 50: {final_atoms}")

if atoms_per_param > 0.001:
    print(f"[PASS] Atoms/Param > 0.001: {atoms_per_param:.6f}")
else:
    print(f"[FAIL] Atoms/Param < 0.001: {atoms_per_param:.6f}")

print(f"\n{'='*60}")
if n_params < 100000 and final_atoms >= 50 and atoms_per_param > 0.001:
    print("SUCCESS! Fixes working!")
else:
    print("NEEDS MORE WORK")
print(f"{'='*60}")
