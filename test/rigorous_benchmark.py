"""Run rigorous benchmark with 3 seeds."""
import sys
sys.path.insert(0, '.')
import torch
import time
import json
import numpy as np
from pathlib import Path
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("="*70)
print("ATOM AI — RIGOROUS BENCHMARK (3 SEEDS)")
print("="*70)

# Config
config = {
    "vocab_size": 500,
    "d_model": 64,
    "n_atoms_max": 128,
    "batch_size": 8,
    "seq_len": 32,
    "max_steps": 500,
    "lr": 3e-4,
    "seeds": [42, 123, 456],
}

results = {"atom": [], "transformer": [], "config": config}

# Create data (same for all)
np.random.seed(42)
tokens = np.random.randint(0, config["vocab_size"], 20000)
sequences = [torch.tensor(tokens[i:i+config["seq_len"]]) 
             for i in range(0, len(tokens)-config["seq_len"], config["seq_len"]//2)]
base_loader = InfiniteDataLoader(sequences, batch_size=config["batch_size"], seq_len=config["seq_len"])

# ========== ATOM EXPERIMENTS ==========
print("\n--- Running ATOM Experiments ---")
for seed in config["seeds"]:
    print(f"\nSeed {seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ToroidalFractalIntelligence(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_modes=config["d_model"],
        n_atoms_max=config["n_atoms_max"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    
    loader = InfiniteDataLoader(
        [torch.tensor(t.tolist()) for t in sequences],
        batch_size=config["batch_size"],
        seq_len=config["seq_len"]
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start = time.time()
    
    for step in range(1, config["max_steps"] + 1):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")
    
    elapsed = time.time() - start
    
    result = {
        "seed": seed,
        "params": n_params,
        "final_loss": losses[-1],
        "avg_loss": sum(losses)/len(losses),
        "atoms": len(model.atoms),
        "time": elapsed,
    }
    results["atom"].append(result)
    print(f"  Done: atoms={len(model.atoms)}, loss={losses[-1]:.4f}")

# ========== TRANSFORMER BASELINE ==========
print("\n--- Running Transformer Baseline ---")
for seed in config["seeds"]:
    print(f"\nSeed {seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    model = torch.nn.Sequential(
        torch.nn.Embedding(config["vocab_size"], config["d_model"]),
        TransformerEncoderLayer(d_model=config["d_model"], nhead=4, dim_feedforward=128),
        torch.nn.Linear(config["d_model"], config["vocab_size"]),
    )
    n_params = sum(p.numel() for p in model.parameters())
    
    loader = InfiniteDataLoader(
        [torch.tensor(t.tolist()) for t in sequences],
        batch_size=config["batch_size"],
        seq_len=config["seq_len"]
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start = time.time()
    
    for step in range(1, config["max_steps"] + 1):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        emb = model[0](token_ids).unsqueeze(0)
        out = model[1](emb)
        out = out.squeeze(0)
        logits = model[2](out.reshape(-1, config["d_model"]))
        
        loss = criterion(logits, target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    
    elapsed = time.time() - start
    
    result = {
        "seed": seed,
        "params": n_params,
        "final_loss": losses[-1],
        "avg_loss": sum(losses)/len(losses),
        "time": elapsed,
    }
    results["transformer"].append(result)
    print(f"  Done: loss={losses[-1]:.4f}")

# ========== SUMMARY ==========
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

atom_losses = [r["final_loss"] for r in results["atom"]]
trans_losses = [r["final_loss"] for r in results["transformer"]]

print(f"\n{'Model':<15} {'Params':<12} {'Avg Loss':<12} {'Final Loss':<12} {'Time':<10}")
print("-"*65)
print(f"{'ATOM':<15} {results['atom'][0]['params']:<12,} {np.mean(atom_losses):<12.4f} {np.mean([r['final_loss'] for r in results['atom']]):<12.4f} {np.mean([r['time'] for r in results['atom']]):<10.1f}s")
print(f"{'Transformer':<15} {results['transformer'][0]['params']:<12,} {np.mean(trans_losses):<12.4f} {np.mean([r['final_loss'] for r in results['transformer']]):<12.4f} {np.mean([r['time'] for r in results['transformer']]):<10.1f}s")

print(f"\nATOM Atoms Created: {np.mean([r['atoms'] for r in results['atom']]):.0f}")
print(f"ATOM Atoms/Param: {np.mean([r['atoms'] for r in results['atom']]) / results['atom'][0]['params']:.6f}")

# Save
results_dir = Path("./results/rigorous_benchmark")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_dir}/benchmark_results.json")
print("="*70)
