> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# ATOM AI Scaling Study — Structural Efficiency vs Compute Cost

## Research Question

**Can a small number of parameters generate and manipulate a much larger amount of structure without exploding compute cost?**

This is the core hypothesis of the Toroidal Fractal Intelligence (ATOM) architecture.

---

## Experimental Design

### Phase 1: Wikipedia Baseline

**Goal**: Measure structural efficiency on realistic data.

**Dataset**: Wikitext-2 (~100K tokens)

**Configuration**: Keep original architecture (d_model=128, n_atoms_max=256)

**Metrics**:
```
- atoms_created
- aggregates_formed
- abstractions_created
- FLOPs (estimated)
- RAM usage (peak)
- time_per_token
- loss
- structural_efficiency = useful_structures / FLOPs
- learning_efficiency = loss_improvement / FLOPs
```

### Phase 2: Progressive Scaling

**Goal**: Find the scaling limits of structural efficiency.

**Atom counts tested**:
```
256 → 512 → 1024 → 2048 → 4096
```

**d_model tested**:
```
128 → 256 → 512
```

**Controlled variables**:
- Same dataset (Wikitext-2)
- Same training steps (500)
- Same learning rate (3e-4)
- Same batch size (32)

### Phase 3: Compute Analysis

**Goal**: Understand where compute is spent.

**Measurements**:
1. Forward pass FLOPs
2. Backward pass FLOPs
3. Interaction FLOPs (k-NN)
4. Aggregation FLOPs
5. Abstraction FLOPs
6. Consolidation FLOPs

**Question**: Which component is the bottleneck?

---

## Key Metrics

### Structural Efficiency
```
SE = (atoms + aggregates*10 + abstractions*100) / FLOPs
```

**Interpretation**: How much "useful structure" per FLOP?

Higher SE = more efficient architecture.

### Learning Efficiency
```
LE = (initial_loss - final_loss) / FLOPs
```

**Interpretation**: How much learning per FLOP?

Higher LE = better learning efficiency.

### Memory Efficiency
```
ME = (atoms + aggregates*10 + abstractions*100) / RAM_GB
```

**Interpretation**: How much structure per GB of RAM?

Higher ME = better memory efficiency.

---

## Hypotheses

### H1: Structural Efficiency Increases with Scale
**Rationale**: More atoms allow more complex patterns, but FLOPs grow sublinearly due to sparse interactions.

**Prediction**:
- 256 atoms: SE = 1e-6 structures/FLOP
- 1024 atoms: SE = 2e-6 structures/FLOP
- 4096 atoms: SE = 4e-6 structures/FLOP

### H2: Learning Efficiency Plateaus
**Rationale**: Beyond a certain point, more structure doesn't help learning (diminishing returns).

**Prediction**:
- LE increases up to 1024 atoms
- LE plateaus or decreases beyond 1024 atoms

### H3: Memory Efficiency Decreases with Scale
**Rationale**: Memory grows linearly with atoms, but FLOPs grow sublinearly.

**Prediction**:
- ME decreases as atom count increases

### H4: Abstractions Become More Frequent
**Rationale**: More data → more patterns → more abstractions.

**Prediction**:
- 256 atoms: 2-5 abstractions
- 1024 atoms: 10-20 abstractions
- 4096 atoms: 30-50 abstractions

---

## Expected Outcomes

### Best Case
- Structural efficiency increases with scale
- Learning efficiency remains high
- Abstractions transfer across domains
- Architecture is viable for large-scale deployment

### Worst Case
- Structural efficiency decreases with scale
- FLOPs grow linearly with atoms (no benefit)
- No learning improvement beyond small scale
- Architecture doesn't scale

### Most Likely Case
- Structural efficiency increases up to ~1024 atoms
- Then plateaus or decreases
- Learning efficiency has optimal point at ~512 atoms
- Architecture is viable for "medium-scale" applications

---

## Files

- `study_structural_efficiency.py` — Main study runner
- `src/io/data.py` — Data loading (Wikitext)
- `results/scaling_study/` — Experiment results

---

## Running the Study

```bash
# Single experiment
python study_structural_efficiency.py --mode single --n-atoms 256 --d-model 128

# Scaling series
python study_structural_efficiency.py --mode series
```

---

## Next Steps

1. **Run Phase 1** (Wikitext baseline)
2. **Analyze bottlenecks** (which component spends most FLOPs?)
3. **Run Phase 2** (progressive scaling)
4. **Compare with baseline** (if we had a Transformer with same params)
5. **Document findings** in scaling laws paper

---

## Critical Question

**Does structure buy capacity more cheaply than parameters?**

If SE increases with scale, the answer is YES.
If SE decreases with scale, the architecture needs revision.

This study will answer that question.
