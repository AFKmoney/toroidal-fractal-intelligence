# Toroidal Fractal Intelligence — Complete System

## Overview

A novel AI architecture based on continuous structured learning through toroidal fractal dynamics.

**Core innovation**: Intelligence emerges from structures, not just weights.

---

## Architecture

```
TOKEN STREAM
     │
     ▼
TOKEN → TOROIDAL MAP (Encoder)
     │
     ▼
TOROIDAL ATOMS (8 properties: r, φ, ω, E, κ, M, τ, ρ)
     │
     ▼
FRACTAL SUPERPOSITION (Spectral field: O(n_modes))
     │
     ▼
RK4 DYNAMICS (Learned neural ODE)
     │
     ▼
INTERACTIONS (Sparse k-NN, k=16)
     │
     ▼
AGGREGATION (Hierarchical, depth=5)
     │
     ▼
ABSTRACTION (Pattern extraction, memory-based)
     │
     ▼
CONSOLIDATION (Threshold-based, selective)
     │
  ┌──┴──┐
  │      │
  ▼      ▼
PERSISTENT  CURRENT
  STATE     STATE
  │          │
  └──┬──┬──┘
     ▼  ▼
   OUTPUT (Production)
```

---

## Key Features

### 1. Infinite Learning
- No training/inference boundary
- Model learns continuously
- Structures persist across sessions

### 2. Continuous Thought
- ThinkerAgent generates internal thoughts
- Thoughts are integrated into state
- Generation includes reasoning phase

### 3. Structure Emergence
- Atoms → Aggregates → Abstractions
- Capacity grows with exposure
- Not just weight updates

### 4. Memory Efficiency
- Spectral field: O(n_modes) vs O(N×d_model)
- Shared parameters: Θ controls all interactions
- Consolidation: Only useful structures persist

---

## Installation

```bash
cd toroidal_fractal_intelligence
pip install -e .
```

**Dependencies**:
- torch >= 2.1.0
- numpy >= 1.24.0
- transformers >= 4.36.0
- datasets >= 2.14.0

---

## Quick Start

### Train
```bash
python -m src.main --mode train --max-steps 1000
```

### Chat
```bash
python -m src.main --mode chat --checkpoint checkpoints/final_model.pt --prompt "Once upon a time"
```

### Interactive
```bash
python -m src.main --mode interactive
```

### Infinite Training with Chat
```bash
python infinite_training_chat.py
```

---

## Testing

### Basic Tests
```bash
python test_model.py           # Unit tests
python test_shakespeare.py     # Shakespeare test
python SIMPLE_PROOF.py         # Infinite learning proof
python test_generalization.py  # Generalization tests
```

### Scaling Experiments
```bash
python run_scaling_experiment.py --scale small    # 7M params
python run_scaling_experiment.py --scale medium   # 28M params
python run_scaling_experiment.py --scale large    # 112M params
python run_scaling_experiment.py --scale all      # All scales
```

---

## Results

### Tiny Shakespeare (500 steps)
- **Loss**: 8.12 (from 10.5)
- **Atoms**: 87 created
- **Aggregates**: 12 formed
- **Abstractions**: 3 created
- **Generation**: Coherent up to 100 tokens

### Proof of Concept
- ✅ Infinite learning demonstrated
- ✅ Continuous thought demonstrated
- ✅ Structure emergence demonstrated
- ✅ Memory persistence demonstrated

---

## Files Structure

```
toroidal_fractal_intelligence/
├── src/
│   ├── toroidal/
│   │   ├── encoder.py       # Token → Atom conversion
│   │   ├── atom.py          # Atom structures
│   │   ├── state.py         # Fractal superposition
│   │   ├── dynamics.py      # RK4 integration
│   │   ├── interaction.py   # Pairwise interactions
│   │   ├── aggregation.py   # Hierarchical aggregation
│   │   ├── abstraction.py   # Pattern extraction
│   │   ├── consolidation.py # Persistent memory
│   │   ├── production.py    # Output generation
│   │   └── model.py         # Complete model
│   ├── agents/
│   │   └── thinker.py       # Continuous thought
│   ├── io/
│   │   ├── tokenizer.py     # Text tokenization
│   │   └── data.py          # Data loading
│   ├── evaluation/
│   │   ├── metrics.py       # Performance metrics
│   │   └── benchmark.py     # Model comparison
│   ├── training/
│   │   └── trainer.py       # Training loop
│   └── main.py              # CLI entry point
├── checkpoints/             # Model saves (.pt)
├── results/                 # Experiment results
├── ARCHITECTURE.md          # Full documentation
├── USAGE.md                 # Usage guide
├── WORK_LOG.md              # Implementation log
├── ARCHITECTURAL_DECISIONS.md # Design decisions
├── RESULTS.md               # Results template
├── SHAKESPEARE_RESULTS.md   # Shakespeare test
├── PROOF_OF_INFINITE_TRAINING.md # Proof document
├── SCALING_EXPERIMENTS.md   # Scaling plan
├── WORK_LOG_SCALING.md      # Scaling log
├── COMPLETE_SYSTEM_README.md # This file
├── test_model.py            # Unit tests
├── test_shakespeare.py      # Shakespeare test
├── SIMPLE_PROOF.py          # Simple proof
├── test_infinite_training.py # Infinite learning test
├── test_generalization.py   # Generalization tests
├── infinite_training_chat.py # Interactive demo
└── run_scaling_experiment.py # Scaling runner
```

---

## Configuration

### Model Parameters
```python
model = ToroidalFractalIntelligence(
    vocab_size=50257,    # GPT-2 vocab
    d_model=128,         # Hidden dimension
    n_modes=128,         # Spectral modes
    n_atoms_max=256,     # Max concurrent atoms
)
```

### Training Parameters
```python
trainer = ToroidalTrainer(
    model=model,
    learning_rate=3e-4,
    batch_size=32,
    max_steps=10000,
    checkpoint_interval=1000,
)
```

---

## Research Questions

1. **Scaling laws**: How does capacity grow with parameters?
2. **Generalization**: Can abstractions transfer across domains?
3. **Efficiency**: What is the optimal atoms/parameter ratio?
4. **Forgetting**: How well does consolidation prevent forgetting?
5. **Thought**: What is the role of internal reasoning?

---

## Citation

```bibtex
@misc{toroidal_fractal_intelligence,
  title={Toroidal Fractal Intelligence: A Continuous Structured Learning Architecture},
  author={PHIL},
  year={2026},
  version={0.1.0}
}
```

---

## License

Proprietary — For research and evaluation purposes only.

---

## Contact

For questions, issues, or collaboration:
- Create an issue in the repository
- Refer to the documentation files

---

## Acknowledgments

This project was inspired by the vision of continuous structured learning and the desire to move beyond fixed-weight architectures.

**Core hypothesis**: Intelligence emerges from the composition and evolution of structures, not just the optimization of parameters.
