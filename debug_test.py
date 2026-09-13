"""Quick debug test."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Creating model...")
model = ToroidalFractalIntelligence(vocab_size=100, d_model=32, n_modes=32, n_atoms_max=64)
print(f"Model params: {sum(p.numel() for p in model.parameters())}")

print("\nCreating data...")
tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)
print(f"Data: {len(loader.data)} sequences")

print("\nTraining step 1...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch = next(iter(loader))
token_ids = batch[:, :-1].reshape(-1)
target_ids = batch[:, 1:].reshape(-1)
print(f"token_ids shape: {token_ids.shape}")
print(f"target_ids shape: {target_ids.shape}")

output = model(token_ids)
print(f"output keys: {output.keys()}")
print(f"logits shape: {output['logits'].shape}")

loss = torch.nn.CrossEntropyLoss()(output['logits'], target_ids)
print(f"Loss: {loss.item()}")

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Step 1 complete, atoms: {len(model.atoms)}")

print("\nStep 2...")
batch = next(iter(loader))
token_ids = batch[:, :-1].reshape(-1)
target_ids = batch[:, 1:].reshape(-1)
output = model(token_ids)
loss = torch.nn.CrossEntropyLoss()(output['logits'], target_ids)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Step 2 complete, atoms: {len(model.atoms)}")

print("\nSUCCESS!")
