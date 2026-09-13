# Toroidal Fractal Intelligence — Rigorous Benchmark

## Experimental Design

This experiment tests the central hypothesis:

> **"At fixed FLOPs budget, ATOM produces more structure/information than an equivalent Transformer."**

### Controls
- Same dataset (WikiText-2 or synthetic)
- Same number of tokens
- Same learning rate
- Same batch size
- Same sequence length

### Variables
- Model architecture (ATOM vs Transformer)
- Parameter count (matched)
- FLOPs budget (measured)

### Metrics
- `E_struct = N_structures / FLOPs`
- `E_param = N_structures / Parameters`
- `E_info = ΔLoss / FLOPs`

---

## Methodology

1. **Create models** with matched parameter counts
2. **Load same dataset** (WikiText-2)
3. **Train for N steps** (1000+)
4. **Measure**: loss, structures, FLOPs, time
5. **Compare** efficiency metrics

---

## Results

See `results/benchmark/` for detailed results.

---

## Status

⏳ Ready to run
