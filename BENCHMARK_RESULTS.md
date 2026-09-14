> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# Benchmark Results — Quick Test

## Experimental Setup

### Configuration
- Vocab size: 100
- d_model: 32
- Batch size: 4
- Sequence length: 16
- Steps: 50
- Seeds: 1 (quick test)

### Data
- Synthetic: random tokens from vocab
- Same data for both models

---

## Results

### ATOM Model
```
Parameters: 451,510
Final atoms: 50
Final loss: 4.5613
Training time: 5.95s
```

### Transformer Baseline
```
Parameters: 15,044
Structures: N/A (no emergent structures)
Final loss: 4.7866
Training time: 0.33s
```

---

## Comparison

| Metric | ATOM | Transformer | Ratio |
|--------|------|-------------|-------|
| Parameters | 451,510 | 15,044 | 30x more |
| Final Loss | 4.56 | 4.79 | 4.8% better |
| Training Time | 5.95s | 0.33s | 18x slower |
| Atoms Created | 50 | 0 | N/A |

---

## Key Observations

### 1. ATOM Creates Structures
- 50 atoms created in 50 steps
- 1 atom per token processed
- Linear growth with training

### 2. Loss Comparison
- ATOM: 4.56 (better)
- Transformer: 4.79 (worse)
- Difference: 4.8%

**Note**: Models have different parameter counts, so this is not a fair comparison.

### 3. Efficiency Analysis
```
ATOM:
- 451,510 params → 50 atoms
- Atoms/parameter: 0.00011

Transformer:
- 15,044 params → 0 structures
- No structural capacity
```

---

## Interpretation

### What This Shows
1. ✅ ATOM creates structures (atoms) from tokens
2. ✅ ATOM achieves comparable/better loss
3. ✅ Structure emergence is functional

### What This Doesn't Show
1. ❌ Fair comparison (different param counts)
2. ❌ Statistical significance (1 seed)
3. ❌ Scaling behavior
4. ❌ Long-term learning

---

## Next Steps

### Immediate
1. [ ] Run with same parameter count
2. [ ] Use 3+ seeds for significance
3. [ ] Increase steps to 500+
4. [ ] Use real dataset (WikiText-2)

### Analysis
1. [ ] Measure FLOPs accurately
2. [ ] Compare structures/FLOP
3. [ ] Test generalization
4. [ ] Run longer training

---

## Status

⚠️ **Quick test complete, full benchmark needed**

The prototype demonstrates:
- Structure creation works
- Loss is competitive
- System is stable

But the rigorous benchmark requires:
- Fixed parameter budget
- Fixed FLOPs budget
- Multiple seeds
- Real dataset
- 1000+ steps

---

## Files

- `very_quick_benchmark.py` — This test
- `benchmark_experiment.py` — Full benchmark (not run)
- `BENCHMARK_DESIGN.md` — Experimental design
