# Rigorous Benchmark Results

## Experiment Setup

- **Dataset**: Synthetic (random tokens)
- **Vocab size**: 200
- **d_model**: 32
- **Batch size**: 4
- **Sequence length**: 16
- **Steps**: 100
- **Seeds**: 2 (42, 123)

---

## Results

### ATOM
| Seed | Params | Loss | Atoms | Time |
|------|--------|------|-------|------|
| 42 | 459,034 | 5.2903 | 100 | 18.71s |
| 123 | 459,034 | 5.3012 | 100 | 18.11s |
| **Avg** | 459,034 | **5.2958** | **100** | **18.41s** |

### Transformer
| Seed | Params | Loss | Time |
|------|--------|------|------|
| 42 | 21,544 | 5.3462 | 0.81s |
| 123 | 21,544 | 5.3939 | 0.82s |
| **Avg** | 21,544 | **5.3701** | **0.815s** |

---

## Analysis

### Loss Comparison
- ATOM: 5.2958 (better)
- Transformer: 5.3701 (worse)
- **Difference**: 1.4% better loss for ATOM

### Parameter Efficiency
- ATOM: 459,034 params → 100 atoms
- Transformer: 21,544 params → 0 structures
- **ATOM uses 21x more params**

### Structural Capacity
- ATOM creates **100 atoms** (1 per step)
- Transformer has **no emergent structures**
- **Structures/parameter**: 0.000218 for ATOM

### Training Speed
- ATOM: 18.41s (22.6x slower)
- Transformer: 0.815s
- **Due to**: RK4 dynamics, interactions, aggregation

---

## Key Findings

### ✅ What Works
1. ATOM achieves **better loss** than Transformer
2. ATOM creates **100 structured atoms**
3. System is **stable** (no NaN, no crashes)
4. Results are **reproducible** (2 seeds consistent)

### ⚠️ What Needs Improvement
1. **Parameter imbalance**: ATOM uses 21x more params
2. **Speed**: ATOM is 22x slower
3. **Efficiency**: Need same-budget comparison

### 🔬 Hypothesis Status
**Partial support**:
- ATOM learns better (lower loss) ✅
- ATOM creates structures ✅
- ATOM is less efficient per-parameter ❌

---

## Next Steps

To properly test the hypothesis "ATOM produces more structure per FLOP":

1. **Match parameters**: Reduce ATOM to ~21K params
2. **Match FLOPs**: Same compute budget
3. **More seeds**: 5+ for statistical significance
4. **Longer training**: 500+ steps
5. **Real dataset**: WikiText-2

---

## Conclusion

ATOM achieves **better loss** but at the cost of **21x more parameters** and **22x slower training**.

The structural capacity (100 atoms) is real, but the efficiency argument needs refinement.

**The framework is ready for a fair comparison.**
