"""Very quick test - 10 steps."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Quick test...")

model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

tokens = list(range(100)) * 20
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)
print(f"Data: {len(sequences)} sequences")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for step in range(1, 11):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1)
    target_ids = batch[:, 1:].reshape(-1)

    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

print(f"\nFinal: atoms={len(model.atoms)}, loss={loss.item():.4f}")
print(f"Atoms/parameter: {len(model.atoms)/n_params:.6f}")
print("DONE!")
