# Toroidal Fractal Intelligence

An experimental AI architecture that generates intelligence from dynamically composed toroidal structures rather than static weight matrices.

## Vision

Current AI systems scale by increasing parameter count. This project explores whether the same (or greater) capacity can be achieved by:

1. Converting tokens into structured "atoms" with rich internal dynamics
2. Allowing these atoms to interact, aggregate, and abstract over time
3. Using a small shared set of rules to govern trillions of potential atom interactions

**Core hypothesis**: `capacity ≫ independent parameters` when structure is the primary carrier of information.

## Architecture

```
TOKEN STREAM
     │
     ▼
TOKEN → TOROIDAL MAP
     │
     ▼
TOROIDAL ATOMS / FIELD
     │
     ▼
FRACTAL SUPERPOSITION
     │
     ▼
     RK4 DYNAMICS
     │
     ▼
  INTERACTIONS
     │
     ▼
   AGGREGATION
     │
     ▼
   ABSTRACTION
     │
     ▼
  CONSOLIDATION
     │         │
     ▼         ▼
PERSISTENT   CURRENT
  STATE       STATE
     │         │
     └────┬────┘
          ▼
       OUTPUT
```

### The Toroidal Atom

```
A_i = (r_i, φ_i, ω_i, E_i, κ_i, M_i, τ_i, ρ_i)
```

| Property | Meaning |
|----------|---------|
| `r` | Position in latent space |
| `φ` | Phase |
| `ω` | Natural frequency |
| `E` | Energy / activation strength |
| `κ` | Coupling strength |
| `M` | Local memory vector |
| `τ` | Timescale |
| `ρ` | Hierarchical level (0–4) |

### Shared Parameters (Θ)

Instead of N×M weights, the system shares a small set of rules:

- `coupling_scale` — how strongly atoms influence each other
- `energy_decay` — how quickly unused atoms decay
- `phase_sync` — how aggressively phases align
- `attractor_centers` — learned stable points in state space

## Quick Start

```bash
# Install
pip install -e .

# Run quick test
python scripts/test_fixes.py

# Run scaling study
python scripts/run_scaling_experiment.py

# Start interactive chat
python -m src.main --mode interactive
```

## Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| `atoms_per_parameter` | atoms_created / trainable_params | > 0.01 |
| `loss_per_parameter` | loss / trainable_params | decreasing |
| `info_per_flop` | Δinformation / FLOPs | increasing |

## Current Status

**Prototype phase** — working implementation with proof-of-concept results.

- ✅ Full pipeline implemented (10 modules)
- ✅ Infinite training loop
- ✅ Structure emergence demonstrated
- ⏳ Scaling to prove hypothesis
- ⏳ Fair comparison vs Transformer baseline

## File Structure

```
src/
├── toroidal/          # Core architecture (10 modules)
│   ├── encoder.py     # Token → Atom conversion
│   ├── atom.py        # Atom data structures
│   ├── state.py       # Fractal superposition field
│   ├── dynamics.py    # RK4 integration
│   ├── interaction.py # Sparse k-NN interactions
│   ├── aggregation.py # Hierarchical grouping
│   ├── abstraction.py # Pattern extraction
│   ├── consolidation.py # Persistent memory
│   ├── production.py  # Output generation
│   └── model.py       # Complete model orchestration
├── agents/            # ThinkerAgent (internal reasoning)
├── io/                # Data loading (Wikitext, custom)
├── evaluation/        # Metrics & benchmarking
└── training/          # Training loop
docs/                  # Architecture docs, reports, logs
scripts/               # Experiment runners
tests/                 # Regression tests
results/               # Experiment outputs
```

## Research Questions

1. Does structural capacity scale better than parametric capacity?
2. Can shared rules govern trillions of atom interactions?
3. Does continuous learning without forgetting require this architecture?
4. What is the theoretical limit of `atoms / parameters`?

## Citation

```bibtex
@misc{toroidal_fractal_intelligence,
  title={Toroidal Fractal Intelligence: A Continuous Structured Learning Architecture},
  author={PHIL},
  year={2026},
  version={0.1.0}
}
```

## License

Proprietary — For research and evaluation purposes only.
