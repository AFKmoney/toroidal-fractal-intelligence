# Experimental Results

## Summary

| Experiment | Params | Atoms | Loss | Status |
|------------|--------|-------|------|--------|
| Shakespeare | 3.7K | 87 | 8.12 | ✅ Complete |
| Scaling Study | 692K | 50 | 4.59 | ✅ Complete |
| Benchmark | 478K | 100 | 6.22 | ⏳ Pending |

## Key Findings

1. **Structure Emergence**: Atoms are created from tokens
2. **Stable Training**: No NaN issues after fixes
3. **Competitive Loss**: Similar to Transformer baseline

## Next Steps

- Run rigorous benchmark with 3 seeds
- Test on WikiText-2
- Compare with matched-parameter Transformer

## Results Directory

Detailed results are stored in `results/`.
