"""
ATOM AI — Final Rigorous Benchmark
"""
import sys
sys.path.insert(0, '.')
import torch
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

# Use synthetic data for speed
print("Using synthetic data (faster than WikiText)")
print()

@dataclass
class Config:
    vocab_size: int = 1000
    d_model: int = 64
    n_atoms_max: int = 128
    batch_size: int = 8
    seq_len: int = 32
    max_steps: int = 300
    learning_rate: float = 3e-4
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]


def estimate_flops(model, batch_size: int, seq_len: int) -> int:
    n_params = sum(p.numel() for p in model.parameters())
    return int(6 * n_params * batch_size * seq_len)


def count_structures(model) -> dict:
    return {
        "atoms": len(model.atoms),
        "aggregates": 0,  # Simplified
        "abstractions": 0,
        "consolidations": model.consolidation_count,
    }


def create_data_loader(config: Config):
    """Create synthetic data."""
    np.random.seed(42)
    tokens = np.random.randint(0, config.vocab_size, 50000)
    sequences = [torch.tensor(tokens[i:i+config.seq_len]) 
                 for i in range(0, len(tokens)-config.seq_len, config.seq_len//2)]
    return InfiniteDataLoader(sequences, batch_size=config.batch_size, seq_len=config.seq_len)


def run_atom(config: Config, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ToroidalFractalIntelligence(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_modes=config.d_model,
        n_atoms_max=config.n_atoms_max,
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    loader = create_data_loader(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start_time = time.time()
    total_flops = 0
    
    for step in range(1, config.max_steps + 1):
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
        total_flops += flops_per_step
        
        if step % 100 == 0:
            print(f"    ATOM Seed {seed} Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")
    
    elapsed = time.time() - start_time
    structs = count_structures(model)
    total_structures = structs['atoms'] + structs['aggregates'] * 10 + structs['abstractions'] * 100
    
    return {
        "seed": seed,
        "model": "atom",
        "params": n_params,
        "flops": total_flops,
        "time": elapsed,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_improvement": losses[0] - losses[-1],
        "atoms": structs['atoms'],
        "aggregates": structs['aggregates'],
        "abstractions": structs['abstractions'],
        "consolidations": structs['consolidations'],
        "total_structures": total_structures,
        "structures_per_flop": total_structures / max(total_flops, 1),
        "structures_per_param": total_structures / max(n_params, 1),
        "loss_per_flop": (losses[0] - losses[-1]) / max(total_flops, 1),
    }


def run_transformer(config: Config, seed: int) -> dict:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = torch.nn.Sequential(
        torch.nn.Embedding(config.vocab_size, config.d_model),
        TransformerEncoderLayer(d_model=config.d_model, nhead=4, dim_feedforward=128),
        torch.nn.Linear(config.d_model, config.vocab_size),
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    loader = create_data_loader(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start_time = time.time()
    total_flops = 0
    
    for step in range(1, config.max_steps + 1):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        emb = model[0](token_ids).unsqueeze(0)
        out = model[1](emb)
        out = out.squeeze(0)
        logits = model[2](out.reshape(-1, config.d_model))
        
        loss = criterion(logits, target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        total_flops += flops_per_step
        
        if step % 100 == 0:
            print(f"    TRANS Seed {seed} Step {step}: loss={loss.item():.4f}")
    
    elapsed = time.time() - start_time
    
    return {
        "seed": seed,
        "model": "transformer",
        "params": n_params,
        "flops": total_flops,
        "time": elapsed,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_improvement": losses[0] - losses[-1],
        "atoms": 0,
        "aggregates": 0,
        "abstractions": 0,
        "consolidations": 0,
        "total_structures": n_params,  # Proxy
        "structures_per_flop": n_params / max(total_flops, 1),
        "structures_per_param": 1.0,
        "loss_per_flop": (losses[0] - losses[-1]) / max(total_flops, 1),
    }


def main():
    print("="*70)
    print("ATOM AI — FINAL RIGOROUS BENCHMARK")
    print("="*70)
    print()
    print("Fixed budget | Same data | 3 seeds | 300 steps")
    print()
    
    config = Config(max_steps=300)
    all_results = []
    
    # ATOM
    print("--- ATOM Experiments ---")
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        result = run_atom(config, seed)
        all_results.append(result)
    
    # Transformer
    print("\n--- Transformer Baseline ---")
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        result = run_transformer(config, seed)
        all_results.append(result)
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    atom_results = [r for r in all_results if r['model'] == 'atom']
    trans_results = [r for r in all_results if r['model'] == 'transformer']
    
    atom_avg = {
        'final_loss': np.mean([r['final_loss'] for r in atom_results]),
        'loss_improvement': np.mean([r['loss_improvement'] for r in atom_results]),
        'total_structures': np.mean([r['total_structures'] for r in atom_results]),
        'structures_per_flop': np.mean([r['structures_per_flop'] for r in atom_results]),
        'structures_per_param': np.mean([r['structures_per_param'] for r in atom_results]),
        'params': np.mean([r['params'] for r in atom_results]),
        'flops': np.mean([r['flops'] for r in atom_results]),
    }
    
    trans_avg = {
        'final_loss': np.mean([r['final_loss'] for r in trans_results]),
        'loss_improvement': np.mean([r['loss_improvement'] for r in trans_results]),
        'total_structures': np.mean([r['total_structures'] for r in trans_results]),
        'structures_per_flop': np.mean([r['structures_per_flop'] for r in trans_results]),
        'structures_per_param': np.mean([r['structures_per_param'] for r in trans_results]),
        'params': np.mean([r['params'] for r in trans_results]),
        'flops': np.mean([r['flops'] for r in trans_results]),
    }
    
    print(f"\n{'Metric':<35} {'ATOM':<15} {'Transformer':<15} {'Ratio':<10}")
    print("-"*75)
    print(f"{'Final Loss':<35} {atom_avg['final_loss']:<15.4f} {trans_avg['final_loss']:<15.4f} {atom_avg['final_loss']/trans_avg['final_loss']:<10.2f}")
    print(f"{'Loss Improvement':<35} {atom_avg['loss_improvement']:<15.4f} {trans_avg['loss_improvement']:<15.4f} -")
    print(f"{'Parameters':<35} {atom_avg['params']:<15,.0f} {trans_avg['params']:<15,.0f} -")
    print(f"{'FLOPs':<35} {atom_avg['flops']:<15,.0f} {trans_avg['flops']:<15,.0f} -")
    print(f"{'Structures':<35} {atom_avg['total_structures']:<15,.0f} {trans_avg['total_structures']:<15,.0f} -")
    print(f"{'Structures/FLOP':<35} {atom_avg['structures_per_flop']:<15.2e} {trans_avg['structures_per_flop']:<15.2e} {atom_avg['structures_per_flop']/max(trans_avg['structures_per_flop'], 1e-10):<10.2f}")
    print(f"{'Structures/Param':<35} {atom_avg['structures_per_param']:<15.6f} {trans_avg['structures_per_param']:<15.6f} -")
    
    # Save
    results_dir = Path("./results/final_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "config": asdict(config),
        "atom_results": atom_results,
        "transformer_results": trans_results,
        "averages": {
            "atom": atom_avg,
            "transformer": trans_avg,
        }
    }
    
    results_file = results_dir / "final_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("="*70)
    
    # Conclusion
    print("\nCONCLUSION:")
    if atom_avg['structures_per_flop'] > trans_avg['structures_per_flop']:
        print(f"  ✅ ATOM creates {atom_avg['structures_per_flop']/max(trans_avg['structures_per_flop'], 1e-10):.1f}x more structures per FLOP")
    else:
        print(f"  ⚠️ Transformer creates more structures per FLOP")
    
    if atom_avg['final_loss'] < trans_avg['final_loss']:
        print(f"  ✅ ATOM achieves lower loss ({atom_avg['final_loss']:.4f} vs {trans_avg['final_loss']:.4f})")
    else:
        print(f"  ⚠️ Transformer achieves lower loss")
    
    print("="*70)


if __name__ == "__main__":
    main()
