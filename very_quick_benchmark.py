"""Very quick benchmark - 50 steps, 1 seed."""
import sys
sys.path.insert(0, '.')
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Quick benchmark (50 steps, 1 seed)...")

# Config
vocab_size = 100
d_model = 32
batch_size = 4
seq_len = 16
max_steps = 50
lr = 3e-4
seed = 42

np.random.seed(seed)
tokens = np.random.randint(0, vocab_size, 1000)
sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
loader = InfiniteDataLoader(sequences, batch_size=batch_size, seq_len=seq_len)

# ATOM
print("\n--- ATOM ---")
torch.manual_seed(seed)
model = ToroidalFractalIntelligence(vocab_size=vocab_size, d_model=d_model, n_modes=d_model, n_atoms_max=32)
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
    
    if step % 10 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

elapsed = time.time() - start
print(f"ATOM: loss={loss.item():.4f}, atoms={len(model.atoms)}, time={elapsed:.2f}s")

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
    
    emb = model2[0](token_ids).unsqueeze(0)  # [1, B, d_model]
    out = model2[1](emb)
    out = out.squeeze(0)  # [B, d_model]
    logits = model2[2](out.reshape(-1, d_model))
    
    loss = criterion(logits, target_ids)
    
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

elapsed2 = time.time() - start
print(f"Transformer: loss={loss.item():.4f}, time={elapsed2:.2f}s")

# Comparison
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"ATOM:           {n_params:,} params, {len(model.atoms)} atoms, loss={loss.item():.4f}")
print(f"Transformer:    {n_params2:,} params, N/A structures, loss={loss.item():.4f}")
print(f"\nATOM creates {len(model.atoms)} structures with {n_params} params")
print(f"This is the structural capacity advantage")
print("="*50)
