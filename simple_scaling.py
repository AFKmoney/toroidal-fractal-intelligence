"""Run scaling study - simplified."""
import sys
sys.path.insert(0, '.')
import torch
import time
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Starting scaling study...")

# Test small config
print("\n=== SMALL CONFIG ===")
model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)
print(f"Data: {len(loader.data)} sequences")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

start = time.time()
for step in range(1, 21):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1)
    target_ids = batch[:, 1:].reshape(-1)

    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start
print(f"Time: {elapsed:.2f}s")
print(f"Final atoms: {len(model.atoms)}")
print(f"Atoms/param: {len(model.atoms)/n_params:.6f}")

print("\n=== SUCCESS ===")
print("Scaling study completed!")
