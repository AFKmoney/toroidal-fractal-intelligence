# Toroidal Fractal Intelligence — Complete Documentation

## 1. Architecture Overview

### 1.1 Core Philosophy

ATOM (Toroidal Fractal Intelligence) is based on a fundamental shift:

**Traditional AI**: Parameters store knowledge, training optimizes weights
**ATOM**: Structures emerge from data, dynamics create intelligence

The key insight is that intelligence can emerge from the **organization of dynamic structures** rather than just the optimization of static parameters.

### 1.2 The Toroidal Atom

The primitive unit of ATOM is the **toroidal atom**:

```python
A_i = (r_i, φ_i, ω_i, E_i, κ_i, M_i, τ_i, ρ_i)
```

Where:
- `r`: Position in latent space
- `φ`: Phase
- `ω`: Natural frequency
- `E`: Energy/activation
- `κ`: Coupling strength
- `M`: Local memory
- `τ`: Time scale
- `ρ`: Hierarchical level

### 1.3 The Pipeline

```
Token → Encoder → Atom → Superposition → RK4 Dynamics → Interactions
    → Aggregation → Abstraction → Consolidation → Production → Output
```

## 2. Mathematical Foundations

### 2.1 Token to Atom Conversion

```python
T(x, c, S) → A
```

Where:
- `x`: Token
- `c`: Context
- `S`: Current state
- `A`: Toroidal atom or structural operation

The encoder can produce operations:
- CREATE: New atom
- MODIFY: Update existing atom
- MERGE: Combine atoms
- REINFORCE: Strengthen connections
- SPLIT: Divide atom
- ABSTRACT: Create abstraction
- CONSOLIDATE: Make persistent

### 2.2 Fractal Superposition

```python
S = Σ α_i A_i
```

The system represents N structures with computational cost that doesn't grow linearly with N.

### 2.3 RK4 Dynamics

```python
k1 = F(S_t)
k2 = F(S_t + dt/2 * k1)
k3 = F(S_t + dt/2 * k2)
k4 = F(S_t + dt * k3)
S_{t+dt} = S_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

## 3. Implementation Details

### 3.1 Core Modules

| Module | File | Purpose |
|--------|------|---------|
| ToroidalEncoder | encoder.py | Token → Atom conversion |
| ToroidalAtom | atom.py | Atom data structure |
| FractalSuperpositionState | state.py | Collective state representation |
| RK4DynamicsEngine | dynamics.py | Time integration |
| ToroidalInteraction | interaction.py | Pairwise interactions |
| AggregationEngine | aggregation.py | Hierarchical grouping |
| AbstractionEngine | abstraction.py | Pattern extraction |
| ConsolidationEngine | consolidation.py | Persistent memory |
| ProductionHead | production.py | Output generation |
| ToroidalFractalIntelligence | model.py | Complete model |

### 3.2 Parameters

Key hyperparameters:
- `d_model`: Dimension of state vectors
- `n_modes`: Number of spectral modes
- `n_atoms_max`: Maximum atoms in superposition
- `dt`: Time step for RK4
- `k_neighbors`: Number of neighbors for sparse interactions

### 3.3 Training

```python
# Training loop
for step in range(max_steps):
    token = next_token()
    atom = encoder(token, context, state)
    state = superposition(state, atom)
    state = rk4_dynamics(state)
    structures = aggregate(state)
    abstractions = abstract(structures)
    state = consolidate(state, abstractions)
    output = produce(state)
```

## 4. Experimental Results

### 4.1 Tiny Shakespeare

```
Parameters: 3,718
Steps: 500
Final loss: 8.12 (from 10.5)
Atoms created: 87
Aggregates: 12
Abstractions: 3
```

### 4.2 Scaling Study

```
Config    Params    Atoms    Loss    At/Param
small     692K      50       4.59    0.000072
medium    1.3M      30       4.61    0.000023
```

### 4.3 Benchmark

```
ATOM vs Transformer (100 steps):
- ATOM: 478K params, 100 atoms, loss=6.22
- Transformer: 41K params, 0 structures, loss=6.29
```

## 5. How to Use

### 5.1 Command Line

```bash
# Train
python -m src.main --mode train --max-steps 1000

# Chat
python -m src.main --mode chat --prompt "Once upon a time"

# Interactive
python -m src.main --mode interactive
```

### 5.2 Python API

```python
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import load_wikitext

# Create model
model = ToroidalFractalIntelligence(
    vocab_size=50257,
    d_model=256,
    n_modes=256,
    n_atoms_max=1024,
)

# Load data
loader = load_wikitext(max_tokens=100000)

# Train
for step in range(1000):
    batch = next(iter(loader))
    output = model(batch)
    loss = compute_loss(output, batch)
    loss.backward()
    optimizer.step()
```

### 5.3 Save/Load

```python
# Save
model.save("checkpoint.pt")

# Load
model.load("checkpoint.pt")
```

## 6. Research Questions

### 6.1 Primary Question

**Can structural capacity grow faster than parameter count?**

If true, this would demonstrate that:
- Intelligence can emerge from organization
- Compute can be decoupled from capacity
- Learning can be more efficient than standard paradigms

### 6.2 Sub-Questions

1. How does atoms/parameter ratio scale?
2. Can abstractions transfer across domains?
3. Does consolidation prevent forgetting?
4. How does structural efficiency compare to parameter efficiency?

## 7. Known Limitations

1. **Low efficiency**: Current atoms/parameter ratio is ~0.0001, target is >0.01
2. **Limited learning**: Loss improvement is minimal on small datasets
3. **No fair benchmark**: Parameter counts differ in comparisons
4. **Short training**: Most experiments use <500 steps

## 8. Future Work

### 8.1 Immediate

1. Run rigorous benchmark with fixed budgets
2. Test on WikiText-2 with 1000+ steps
3. Implement atom reuse (MODIFY instead of CREATE)
4. Add GPU support for larger models

### 8.2 Medium-term

1. Scale to d_model=512, n_atoms_max=10K
2. Test on multiple datasets
3. Compare with equivalent Transformers
4. Publish scaling laws paper

### 8.3 Long-term

1. Test on real-world applications
2. Explore human-level reasoning
3. Investigate general intelligence properties

## 9. References

- Original specification: See TOROIDAL_SPEC.md
- Architectural decisions: See ARCHITECTURAL_DECISIONS.md
- Mathematical foundations: See ARCHITECTURE.md

## 10. License

Proprietary — For research and evaluation purposes only.

---

**For questions and collaboration, contact the author.**
