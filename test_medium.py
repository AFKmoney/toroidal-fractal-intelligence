"""Single config scaling test."""
import sys
sys.path.insert(0, '.')
import torch
import time
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Testing MEDIUM config...")

model = ToroidalFractalIntelligence(vocab_size=100, d_model=128, n_modes=128, n_atoms_max=256)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+128]) for i in range(0, len(tokens)-128, 64)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=128)
print(f"Data: {len(sequences)} sequences")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

losses = []
start = time.time()

for step in range(1, 31):
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

    if step % 5 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start
avg_loss = sum(losses) / len(losses)

print(f"\nResults:")
print(f"  Final atoms: {len(model.atoms)}")
print(f"  Avg loss: {avg_loss:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Time: {elapsed:.2f}s")
print(f"  Atoms/parameter: {len(model.atoms)/n_params:.6f}")
print(f"\nSUCCESS!")
