# Usage Guide

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

# Create model
model = ToroidalFractalIntelligence(
    vocab_size=50257,
    d_model=256,
    n_modes=512,
    n_atoms_max=2048,
)

# Load data
loader = InfiniteDataLoader.from_wikitext(max_tokens=100_000)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

for step in range(1000):
    batch = next(iter(loader))
    token_ids = batch[:, :-1].reshape(-1).long()
    target_ids = batch[:, 1:].reshape(-1).long()
    
    output = model(token_ids)
    loss = criterion(output['logits'], target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, atoms={len(model.atoms)}")

# Save
model.save("checkpoint.pt")

# Load and continue training
model2 = ToroidalFractalIntelligence(...)
model2.load("checkpoint.pt")
```

## Interactive Chat

```bash
python -m src.main --mode interactive --checkpoint checkpoints/final.pt
```

## Running Tests

```bash
python tests/test_fixes.py           # Quick sanity check
python tests/test_infinite_training.py # Prove infinite learning
python scripts/run_scaling_experiment.py # Scaling study
```

## Configuration

Key hyperparameters in `ToroidalFractalIntelligence.__init__`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Hidden dimension |
| `n_modes` | 256 | Spectral modes (state size) |
| `n_atoms_max` | 1024 | Max concurrent atoms |
| `dt` | 0.1 | RK4 time step |
| `k_neighbors` | 16 | Sparse interaction k |

## Metrics

Access during training:

```python
# Structural metrics
n_atoms = len(model.atoms)
n_aggregates = len(model.aggregation.hierarchy[-1])
n_abstractions = len(model.abstraction.active_ids)

# Shared parameters (Θ)
theta = model.dynamics.get_shared_params()

# Efficiency
efficiency = n_atoms / sum(p.numel() for p in model.parameters())
```
