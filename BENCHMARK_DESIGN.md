> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# ATOM AI — Benchmark Experiment Design

## Hypothesis to Test

**Primary**: "À budget de calcul fixe, ATOM produit plus de structure/information qu'un Transformer équivalent."

**Secondary**: "La capacité structurelle d'ATOM augmente plus vite que ses paramètres."

---

## Experimental Design

### Fixed Variables (Same for Both)
- Dataset: WikiText-2 (or synthetic equivalent)
- Number of tokens processed
- FLOPs budget (estimated)
- Learning rate
- Batch size
- Sequence length

### Variables to Measure
- Final loss
- Structures created (atoms, aggregates, abstractions)
- Efficiency metrics
- Training time

### Seeds
- 3 different random seeds for statistical significance
- Seeds: 42, 123, 456

---

## Metrics

### 1. Structural Efficiency
$$E_{struct} = \frac{N_{structures}}{FLOPs}$$

Where:
- $N_{structures} = N_{atoms} + 10 \times N_{aggregates} + 100 \times N_{abstractions}$
- Weights reflect estimated information content

### 2. Parameter Efficiency
$$E_{param} = \frac{N_{structures}}{Parameters}$$

### 3. Learning Efficiency
$$E_{info} = \frac{\Delta Loss}{FLOPs}$$

Where $\Delta Loss = Loss_{initial} - Loss_{final}$

---

## Implementation

### ATOM Model
```python
model = ToroidalFractalIntelligence(
    vocab_size=50257,
    d_model=128,
    n_modes=128,
    n_atoms_max=256,
)
```

### Transformer Baseline
```python
model = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, d_model),
    TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=256),
    torch.nn.Linear(d_model, vocab_size),
)
```

### Training Loop
```python
for step in range(1, max_steps + 1):
    batch = next(data_loader)
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    record_metrics()
```

---

## Expected Outcomes

### Best Case for ATOM
- Similar loss to Transformer
- 10-100x more structures per FLOP
- Demonstrates structural efficiency advantage

### Worst Case for ATOM
- Worse loss than Transformer
- Fewer structures per FLOP
- Architecture doesn't scale

### Most Likely
- Similar learning curve
- ATOM creates more structures
- Efficiency advantage depends on workload

---

## Files

- `benchmark_experiment.py` — Main experiment script
- `results/benchmark/` — Experiment results (JSON)
- `BENCHMARK_DESIGN.md` — This document

---

## Running the Experiment

```bash
# Quick test (100 steps)
python benchmark_experiment.py --steps 100

# Full experiment (1000 steps)
python benchmark_experiment.py --steps 1000

# Custom seeds
python benchmark_experiment.py --steps 1000 --seeds 1 2 3
```

---

## Interpretation Guidelines

### If ATOM wins on efficiency:
- Structural efficiency hypothesis supported
- Architecture buys capacity cheaply
- Publish scaling laws paper

### If Transformer wins on loss:
- ATOM learns less effectively
- May need architectural changes
- Still valuable as proof of concept

### If similar performance:
- Both architectures viable
- ATOM offers structure interpretability
- Different trade-offs

---

## Status

✅ Experiment design complete
✅ Code implemented
⏳ Ready to run (needs WikiText-2 or synthetic data)
