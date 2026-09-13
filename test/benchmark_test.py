"""Benchmark ATOM vs Transformer."""
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Running benchmark...")

# Config
vocab_size = 100
d_model = 32
batch_size = 4
seq_len = 16
max_steps = 100

# Data
np.random.seed(42)
tokens = np.random.randint(0, vocab_size, 5000)
sequences = [torch.tensor(tokens[i:i+seq_len]) for i in range(0, len(tokens)-seq_len, seq_len//2)]
loader = InfiniteDataLoader(sequences, batch_size=batch_size, seq_len=seq_len)

# ATOM
print("\n--- ATOM ---")
model = ToroidalFractalIntelligence(vocab_size=vocab_size, d_model=d_model, n_modes=d_model, n_atoms_max=64)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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

elapsed = time.time() - start
print(f"ATOM: loss={loss.item():.4f}, atoms={len(model.atoms)}, time={elapsed:.2f}s")

# Transformer
print("\n--- Transformer ---")
from torch.nn import TransformerEncoder, TransformerEncoderLayer
model2 = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, d_model),
    TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64),
    torch.nn.Linear(d_model, vocab_size),
)
n_params2 = sum(p.numel() for p in model2.parameters())
print(f"Params: {n_params2:,}")

optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

start = time.time()
for step in range(1, max_steps + 1):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    emb = model2[0](token_ids).unsqueeze(0)
    out = model2[1](emb)
    out = out.squeeze(0)
    logits = model2[2](out.reshape(-1, d_model))
    
    loss = criterion(logits, target_ids)
    
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

elapsed2 = time.time() - start
print(f"Transformer: loss={loss.item():.4f}, time={elapsed2:.2f}s")

print(f"\nBenchmark complete!")
print(f"ATOM creates {len(model.atoms)} structures with {n_params} params")
