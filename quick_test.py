"""
Quick test of the ATOM system.
"""
import sys
sys.path.insert(0, '.')

import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Creating model...")
model = ToroidalFractalIntelligence(
    vocab_size=100,
    d_model=32,
    n_modes=32,
    n_atoms_max=64,
)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

print("\nCreating data...")
tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)
print(f"Data created: {len(loader.data)} sequences")

print("\nTraining...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1, 11):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1)
    target_ids = batch[:, 1:].reshape(-1)

    output = model(token_ids)
    loss = torch.nn.CrossEntropyLoss()(output["logits"], target_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

print("\nFinal state:")
print(f"  Atoms: {len(model.atoms)}")
print(f"  Final loss: {loss.item():.4f}")
print("\nSUCCESS!")
