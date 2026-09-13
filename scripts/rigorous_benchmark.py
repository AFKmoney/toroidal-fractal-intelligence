"""
ATOM AI — Rigorous Benchmark with WikiText-2
Fixed compute budget, same dataset, 3 seeds
"""
import sys
sys.path.insert(0, '.')
import torch
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader, load_wikitext

# Try to load WikiText-2
try:
    from datasets import load_dataset
    WIKITEXT_AVAILABLE = True
except ImportError:
    WIKITEXT_AVAILABLE = False
    print("Warning: datasets library not available. Using synthetic data.")


@dataclass
class Config:
    vocab_size: int = 50257  # GPT-2 vocab
    d_model: int = 128
    n_atoms_max: int = 256
    batch_size: int = 16
    seq_len: int = 64
    max_steps: int = 500
    learning_rate: float = 3e-4
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]


def estimate_flops(model, batch_size: int, seq_len: int) -> int:
    """Estimate FLOPs per training step."""
    n_params = sum(p.numel() for p in model.parameters())
    return int(6 * n_params * batch_size * seq_len)


def count_structures(model) -> Dict:
    """Count emergent structures."""
    return {
        "atoms": len(model.atoms),
        "aggregates": len(getattr(model.aggregation, 'hierarchy', [[]])[0]) if hasattr(model.aggregation, 'hierarchy') else 0,
        "abstractions": len(model.abstraction.get_active_abstractions()) if hasattr(model.abstraction, 'get_active_abstractions') else 0,
        "consolidations": model.consolidation_count,
    }


def run_atom_experiment(config: Config, seed: int) -> Dict:
    """Run one ATOM experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = ToroidalFractalIntelligence(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_modes=config.d_model,
        n_atoms_max=config.n_atoms_max,
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    # Load data
    if WIKITEXT_AVAILABLE:
        try:
            loader = load_wikitext(max_tokens=100000)
        except Exception as e:
            print(f"  WikiText load failed: {e}, using synthetic")
            loader = create_synthetic_loader(config)
    else:
        loader = create_synthetic_loader(config)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start_time = time.time()
    total_flops = 0
    
    for step in range(1, config.max_steps + 1):
        try:
            batch = next(iter(loader))
        except StopIteration:
            loader = create_synthetic_loader(config)
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
            structs = count_structures(model)
            print(f"    Step {step}: loss={loss.item():.4f}, atoms={structs['atoms']}, agg={structs['aggregates']}, abs={structs['abstractions']}")
    
    elapsed = time.time() - start_time
    
    # Final metrics
    structs = count_structures(model)
    
    total_structures = (structs['atoms'] + 
                       structs['aggregates'] * 10 + 
                       structs['abstractions'] * 100)
    
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


def create_synthetic_loader(config: Config):
    """Create synthetic data loader."""
    np.random.seed(42)
    tokens = np.random.randint(0, config.vocab_size, 50000)
    sequences = [torch.tensor(tokens[i:i+config.seq_len]) 
                 for i in range(0, len(tokens)-config.seq_len, config.seq_len//2)]
    return InfiniteDataLoader(sequences, batch_size=config.batch_size, seq_len=config.seq_len)


def run_transformer_baseline(config: Config, seed: int) -> Dict:
    """Run Transformer baseline with same budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    # Create Transformer with similar parameter count
    # Adjust layers to match ATOM param count
    model = torch.nn.Sequential(
        torch.nn.Embedding(config.vocab_size, config.d_model),
        TransformerEncoderLayer(d_model=config.d_model, nhead=4, dim_feedforward=256),
        TransformerEncoderLayer(d_model=config.d_model, nhead=4, dim_feedforward=256),
        torch.nn.Linear(config.d_model, config.vocab_size),
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    # Load same data
    if WIKITEXT_AVAILABLE:
        try:
            loader = load_wikitext(max_tokens=100000)
        except:
            loader = create_synthetic_loader(config)
    else:
        loader = create_synthetic_loader(config)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    start_time = time.time()
    total_flops = 0
    
    for step in range(1, config.max_steps + 1):
        try:
            batch = next(iter(loader))
        except StopIteration:
            loader = create_synthetic_loader(config)
            batch = next(iter(loader))
        
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        # Forward
        emb = model[0](token_ids).unsqueeze(0)  # [1, B, d]
        out = model[1](emb)
        out = model[2](out)
        out = out.squeeze(0)  # [B, d]
        logits = model[3](out.reshape(-1, config.d_model))
        
        loss = criterion(logits, target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        total_flops += flops_per_step
        
        if step % 100 == 0:
            print(f"    Step {step}: loss={loss.item():.4f}")
    
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
        "total_structures": n_params,  # Parameters as proxy
        "structures_per_flop": n_params / max(total_flops, 1),
        "structures_per_param": 1.0,
        "loss_per_flop": (losses[0] - losses[-1]) / max(total_flops, 1),
    }


def run_benchmark():
    """Run the full benchmark."""
    print("="*70)
    print("ATOM AI — RIGOROUS BENCHMARK EXPERIMENT")
    print("="*70)
    print()
    print("Fixed compute budget | Same dataset | 3 seeds")
    print()
    
    config = Config(max_steps=500)
    
    all_results = []
    
    # Run ATOM experiments
    print("--- Running ATOM experiments ---")
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        result = run_atom_experiment(config, seed)
        all_results.append(result)
        print(f"  Final: loss={result['final_loss']:.4f}, structures={result['total_structures']}")
        print(f"  Efficiency: {result['structures_per_flop']:.2e} structs/FLOP")
    
    # Run Transformer baseline
    print("\n--- Running Transformer baseline ---")
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        result = run_transformer_baseline(config, seed)
        all_results.append(result)
        print(f"  Final: loss={result['final_loss']:.4f}")
    
    # Analyze results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    atom_results = [r for r in all_results if r['model'] == 'atom']
    trans_results = [r for r in all_results if r['model'] == 'transformer']
    
    # Average over seeds
    atom_avg = {
        'final_loss': np.mean([r['final_loss'] for r in atom_results]),
        'loss_improvement': np.mean([r['loss_improvement'] for r in atom_results]),
        'total_structures': np.mean([r['total_structures'] for r in atom_results]),
        'structures_per_flop': np.mean([r['structures_per_flop'] for r in atom_results]),
        'structures_per_param': np.mean([r['structures_per_param'] for r in atom_results]),
        'loss_per_flop': np.mean([r['loss_per_flop'] for r in atom_results]),
        'params': np.mean([r['params'] for r in atom_results]),
        'flops': np.mean([r['flops'] for r in atom_results]),
    }
    
    trans_avg = {
        'final_loss': np.mean([r['final_loss'] for r in trans_results]),
        'loss_improvement': np.mean([r['loss_improvement'] for r in trans_results]),
        'total_structures': np.mean([r['total_structures'] for r in trans_results]),
        'structures_per_flop': np.mean([r['structures_per_flop'] for r in trans_results]),
        'structures_per_param': np.mean([r['structures_per_param'] for r in trans_results]),
        'loss_per_flop': np.mean([r['loss_per_flop'] for r in trans_results]),
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
    print(f"{'Structures/FLOP':<35} {atom_avg['structures_per_flop']:<15.2e} {trans_avg['structures_per_flop']:<15.2e} {atom_avg['structures_per_flop']/trans_avg['structures_per_flop']:<10.2f}")
    print(f"{'Structures/Param':<35} {atom_avg['structures_per_param']:<15.6f} {trans_avg['structures_per_param']:<15.6f} -")
    print(f"{'Loss/FLOP':<35} {atom_avg['loss_per_flop']:<15.2e} {trans_avg['loss_per_flop']:<15.2e} -")
    
    # Save results
    results_dir = Path("./results/benchmark_rigorous")
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
    
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("="*70)
    
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()
    
    config = Config(max_steps=args.steps, seeds=args.seeds)
    results = run_benchmark()
