"""Quick final test - small config only."""
import sys
sys.path.insert(0, '.')
import torch
import time
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Running small config (200 steps)...")

model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

tokens = list(range(100)) * 100
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=8, seq_len=32)
print(f"Data: {len(sequences)} sequences")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

losses = []
start = time.time()

for step in range(1, 201):
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

    if step % 50 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start
initial_loss = losses[0]
final_loss = losses[-1]

print(f"\nResults:")
print(f"  Final atoms: {len(model.atoms)}")
print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Final loss: {final_loss:.4f}")
print(f"  Loss improvement: {initial_loss - final_loss:.4f}")
print(f"  Time: {elapsed:.2f}s")
print(f"  Atoms/parameter: {len(model.atoms)/n_params:.6f}")
print(f"\nSUCCESS!")
