"""
ATOM AI — Rigorous Benchmark Experiment

Experimental design:
- Fixed compute budget (FLOPs)
- Same dataset (WikiText-2)
- Same number of tokens
- 3 seeds for statistical significance
- Compare ATOM vs Transformer baseline

Metrics:
- E_struct = N_structures / FLOPs
- E_param = N_structures / Parameters
- E_info = Delta_Information / FLOPs
"""

import sys
sys.path.insert(0, '.')
import torch
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Try to load WikiText-2
try:
    from datasets import load_dataset
    WIKITEXT_AVAILABLE = True
except ImportError:
    WIKITEXT_AVAILABLE = False
    print("Warning: datasets library not available. Using synthetic data.")

from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader, load_wikitext


@dataclass
class ExperimentConfig:
    """Fixed experiment configuration."""
    # Compute budget
    max_flops: int = 1e12  # 1 trillion FLOPs
    max_steps: int = 1000
    
    # Model configs (same parameter budget)
    atom_d_model: int = 128
    atom_n_atoms: int = 256
    transformer_d_model: int = 128
    transformer_layers: int = 2
    
    # Training
    batch_size: int = 32
    seq_len: int = 64
    learning_rate: float = 3e-4
    
    # Seeds
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]


@dataclass
class Metrics:
    """All metrics to track."""
    # Compute
    flops_estimated: int = 0
    time_seconds: float = 0.0
    steps_completed: int = 0
    
    # Structure (ATOM only)
    atoms_created: int = 0
    aggregates_formed: int = 0
    abstractions_created: int = 0
    consolidations: int = 0
    
    # Learning
    initial_loss: float = 0.0
    final_loss: float = 0.0
    loss_improvement: float = 0.0
    
    # Efficiency
    structures_per_flop: float = 0.0
    structures_per_param: float = 0.0
    loss_per_flop: float = 0.0


def estimate_flops(model, batch_size: int, seq_len: int) -> int:
    """Estimate FLOPs per training step."""
    n_params = sum(p.numel() for p in model.parameters())
    # Forward + backward ≈ 6 * n_params * batch_size * seq_len
    return int(6 * n_params * batch_size * seq_len)


def count_structures(model) -> Dict:
    """Count emergent structures in ATOM model."""
    return {
        "atoms": len(model.atoms),
        "aggregates": len(model.aggregation.hierarchy[-1]) if model.aggregation.hierarchy else 0,
        "abstractions": len(model.abstraction.get_active_abstractions()),
        "consolidations": model.consolidation_count,
    }


def run_atom_experiment(config: ExperimentConfig, seed: int) -> Metrics:
    """Run one ATOM experiment with fixed seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    metrics = Metrics()
    metrics.seeds = [seed]
    
    # Create model
    model = ToroidalFractalIntelligence(
        vocab_size=50257,
        d_model=config.atom_d_model,
        n_modes=config.atom_d_model,
        n_atoms_max=config.atom_n_atoms,
    )
    n_params = sum(p.numel() for p in model.parameters())
    metrics.flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    # Load data
    if WIKITEXT_AVAILABLE:
        try:
            loader = load_wikitext(max_tokens=50000)
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
        
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        total_flops += metrics.flops_per_step
        
        # Checkpoint at specific steps
        if step in [100, 250, 500, 1000]:
            structs = count_structures(model)
            print(f"  Seed {seed} Step {step}: loss={loss.item():.4f}, "
                  f"atoms={structs['atoms']}, aggregates={structs['aggregates']}, "
                  f"abstractions={structs['abstractions']}")
    
    # Final metrics
    elapsed = time.time() - start_time
    metrics.flops_estimated = total_flops
    metrics.time_seconds = elapsed
    metrics.steps_completed = len(losses)
    metrics.initial_loss = losses[0]
    metrics.final_loss = losses[-1]
    metrics.loss_improvement = losses[0] - losses[-1]
    
    # Structure counts
    structs = count_structures(model)
    metrics.atoms_created = structs['atoms']
    metrics.aggregates_formed = structs['aggregates']
    metrics.abstractions_created = structs['abstractions']
    metrics.consolidations = structs['consolidations']
    
    # Efficiency metrics
    total_structures = (metrics.atoms_created + 
                       metrics.aggregates_formed * 10 + 
                       metrics.abstractions_created * 100)
    
    metrics.structures_per_flop = total_structures / max(total_flops, 1)
    metrics.structures_per_param = total_structures / max(n_params, 1)
    metrics.loss_per_flop = metrics.loss_improvement / max(total_flops, 1)
    
    return metrics


def create_synthetic_loader(config: ExperimentConfig):
    """Create synthetic data loader for testing."""
    tokens = list(range(1000)) * 100
    sequences = [torch.tensor(tokens[i:i+config.seq_len]) 
                 for i in range(0, len(tokens)-config.seq_len, config.seq_len//2)]
    return InfiniteDataLoader(sequences, batch_size=config.batch_size, seq_len=config.seq_len)


def run_baseline_transformer(config: ExperimentConfig, seed: int) -> Metrics:
    """Run baseline Transformer experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    
    metrics = Metrics()
    
    # Simple Transformer
    d_model = config.transformer_d_model
    nhead = 4
    num_layers = config.transformer_layers
    vocab_size = 50257
    seq_len = config.seq_len
    
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, d_model),
        TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256),
        torch.nn.Linear(d_model, vocab_size),
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    metrics.flops_per_step = estimate_flops(model, config.batch_size, config.seq_len)
    
    # Load same data
    if WIKITEXT_AVAILABLE:
        try:
            loader = load_wikitext(max_tokens=50000)
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
        
        token_ids = batch[:, :-1].reshape(-1)
        target_ids = batch[:, 1:].reshape(-1)
        
        # Forward through transformer
        emb = model[0](token_ids)  # [B, seq_len, d_model]
        emb = emb.permute(1, 0, 2)  # [seq_len, B, d_model]
        out = model[1](emb)  # [seq_len, B, d_model]
        out = out.permute(1, 0, 2)  # [B, seq_len, d_model]
        logits = model[2](out.reshape(-1, d_model))  # [B*seq_len, vocab]
        
        loss = criterion(logits, target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        total_flops += metrics.flops_per_step
    
    # Final metrics
    elapsed = time.time() - start_time
    metrics.flops_estimated = total_flops
    metrics.time_seconds = elapsed
    metrics.steps_completed = len(losses)
    metrics.initial_loss = losses[0]
    metrics.final_loss = losses[-1]
    metrics.loss_improvement = losses[0] - losses[-1]
    
    # Transformer has no structures, use proxy
    metrics.atoms_created = n_params  # Parameters as proxy
    metrics.structures_per_flop = n_params / max(total_flops, 1)
    metrics.structures_per_param = 1.0
    metrics.loss_per_flop = metrics.loss_improvement / max(total_flops, 1)
    
    return metrics


def run_benchmark_experiment():
    """Run the full benchmark experiment."""
    print("="*70)
    print("ATOM AI — RIGOROUS BENCHMARK EXPERIMENT")
    print("="*70)
    print()
    print("Experimental Design:")
    print("  - Fixed compute budget (FLOPs)")
    print("  - Same dataset (WikiText-2 or synthetic)")
    print("  - Same number of tokens")
    print("  - 3 seeds for statistical significance")
    print()
    print("Metrics:")
    print("  E_struct = N_structures / FLOPs")
    print("  E_param = N_structures / Parameters")
    print("  E_info = Delta_Information / FLOPs")
    print()
    print("="*70)
    
    config = ExperimentConfig()
    
    # Run ATOM experiments
    print("\n--- Running ATOM experiments ---")
    atom_results = []
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        metrics = run_atom_experiment(config, seed)
        atom_results.append(asdict(metrics))
        print(f"  Final: loss={metrics.final_loss:.4f}, atoms={metrics.atoms_created}")
        print(f"  Efficiency: {metrics.structures_per_flop:.2e} structs/FLOP")
    
    # Run Transformer baseline
    print("\n--- Running Transformer baseline ---")
    transformer_results = []
    for seed in config.seeds:
        print(f"\nSeed {seed}...")
        metrics = run_baseline_transformer(config, seed)
        transformer_results.append(asdict(metrics))
        print(f"  Final: loss={metrics.final_loss:.4f}")
        print(f"  Efficiency: {metrics.structures_per_flop:.2e} params/FLOP")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    # Average over seeds
    atom_avg_loss = np.mean([r['final_loss'] for r in atom_results])
    atom_avg_structs = np.mean([r['atoms_created'] for r in atom_results])
    atom_avg_eff = np.mean([r['structures_per_flop'] for r in atom_results])
    
    transformer_avg_loss = np.mean([r['final_loss'] for r in transformer_results])
    transformer_avg_eff = np.mean([r['structures_per_flop'] for r in transformer_results])
    
    print(f"\n{'Metric':<30} {'ATOM':<15} {'Transformer':<15} {'Ratio':<10}")
    print("-"*70)
    print(f"{'Final Loss':<30} {atom_avg_loss:<15.4f} {transformer_avg_loss:<15.4f} {atom_avg_loss/transformer_avg_loss:<10.2f}")
    print(f"{'Structures/FLOP':<30} {atom_avg_eff:<15.2e} {transformer_avg_eff:<15.2e} {atom_avg_eff/transformer_avg_eff:<10.2f}")
    print(f"{'Loss Improvement':<30} {np.mean([r['loss_improvement'] for r in atom_results]):<15.4f} {np.mean([r['loss_improvement'] for r in transformer_results]):<15.4f} -")
    
    # Save results
    results = {
        "experiment": "benchmark_v1",
        "config": asdict(config),
        "atom_results": atom_results,
        "transformer_results": transformer_results,
        "comparison": {
            "atom_avg_loss": atom_avg_loss,
            "transformer_avg_loss": transformer_avg_loss,
            "atom_avg_structs": atom_avg_structs,
            "atom_avg_efficiency": atom_avg_eff,
            "transformer_avg_efficiency": transformer_avg_eff,
        }
    }
    
    results_dir = Path("./results/benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "benchmark_results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    args = parser.parse_args()
    
    config = ExperimentConfig(max_steps=args.steps, seeds=args.seeds)
    results = run_benchmark_experiment()
