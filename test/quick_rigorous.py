"""Quick rigorous benchmark - 100 steps, 2 seeds."""
import sys
sys.path.insert(0, '.')
import torch
import time
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Quick Rigorous Benchmark (100 steps, 2 seeds)")
print("="*60)

config = {
    "vocab_size": 200,
    "d_model": 32,
    "n_atoms_max": 64,
    "batch_size": 4,
    "seq_len": 16,
    "max_steps": 100,
    "lr": 3e-4,
    "seeds": [42, 123],
}

results = {"atom": [], "transformer": []}

for seed in config["seeds"]:
    print(f"\n--- Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # ATOM
    model = ToroidalFractalIntelligence(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_modes=config["d_model"],
        n_atoms_max=config["n_atoms_max"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    
    np.random.seed(seed)
    tokens = np.random.randint(0, config["vocab_size"], 2000)
    sequences = [torch.tensor(tokens[i:i+config["seq_len"]]) 
                 for i in range(0, len(tokens)-config["seq_len"], config["seq_len"]//2)]
    loader = InfiniteDataLoader(sequences, batch_size=config["batch_size"], seq_len=config["seq_len"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start
    results["atom"].append({
        "seed": seed,
        "params": n_params,
        "loss": loss.item(),
        "atoms": len(model.atoms),
        "time": elapsed,
    })
    print(f"  ATOM: loss={loss.item():.4f}, atoms={len(model.atoms)}, time={elapsed:.2f}s")
    
    # Transformer
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    model2 = torch.nn.Sequential(
        torch.nn.Embedding(config["vocab_size"], config["d_model"]),
        TransformerEncoderLayer(d_model=config["d_model"], nhead=2, dim_feedforward=64),
        torch.nn.Linear(config["d_model"], config["vocab_size"]),
    )
    n_params2 = sum(p.numel() for p in model2.parameters())
    
    loader2 = InfiniteDataLoader(sequences, batch_size=config["batch_size"], seq_len=config["seq_len"])
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config["lr"])
    
    start = time.time()
    for step in range(1, config["max_steps"] + 1):
        batch = next(iter(loader2))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        emb = model2[0](token_ids).unsqueeze(0)
        out = model2[1](emb)
        out = out.squeeze(0)
        logits = model2[2](out.reshape(-1, config["d_model"]))
        
        loss = criterion(logits, target_ids)
        
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    
    elapsed2 = time.time() - start
    results["transformer"].append({
        "seed": seed,
        "params": n_params2,
        "loss": loss.item(),
        "time": elapsed2,
    })
    print(f"  TRANS: loss={loss.item():.4f}, time={elapsed2:.2f}s")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\n{'Model':<12} {'Params':<12} {'Loss':<10} {'Atoms':<8} {'Time':<10}")
print("-"*55)
for r in results["atom"]:
    print(f"{'ATOM':<12} {r['params']:<12,} {r['loss']:<10.4f} {r['atoms']:<8} {r['time']:<10.2f}s")
for r in results["transformer"]:
    print(f"{'Transformer':<12} {r['params']:<12,} {r['loss']:<10.4f} {'N/A':<8} {r['time']:<10.2f}s")

print("\n" + "="*60)
print("Benchmark complete!")
print("="*60)
