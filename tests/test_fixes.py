"""Quick test of ATOM fixes."""
import sys
sys.path.insert(0, '.')
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("ATOM Quick Test")
print("="*50)

# Config
vocab_size = 200
d_model = 32
n_atoms_max = 64
batch_size = 4
seq_len = 16
max_steps = 100

np.random.seed(42)
tokens = np.random.randint(0, vocab_size, 5000)
sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
loader = InfiniteDataLoader(sequences, batch_size=batch_size, seq_len=seq_len)

# Model
model = ToroidalFractalIntelligence(
    vocab_size=vocab_size,
    d_model=d_model,
    n_modes=d_model,
    n_atoms_max=n_atoms_max,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
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
        print(f"Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start

# Results
print(f"\nResults:")
print(f"  Final atoms: {len(model.atoms)}")
print(f"  Final loss: {loss.item():.4f}")
print(f"  Atoms/param: {len(model.atoms)/n_params:.6f}")
print(f"  Time: {elapsed:.2f}s")
print(f"\n{'='*50}")
if len(model.atoms) >= 50 and n_params < 100000:
    print("SUCCESS!")
else:
    print("NEEDS WORK")
print(f"{'='*50}")
