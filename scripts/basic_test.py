"""Test Toroidal Fractal Intelligence."""
import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Testing ATOM...")

model = ToroidalFractalIntelligence(vocab_size=100, d_model=32, n_modes=32, n_atoms_max=64)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Create data
import numpy as np
np.random.seed(42)
tokens = np.random.randint(0, 100, 1000)
sequences = [torch.tensor(tokens[i:i+16]) for i in range(0, len(tokens)-16, 8)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=16)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for step in range(1, 51):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

print(f"\nTest complete! Final atoms: {len(model.atoms)}")
