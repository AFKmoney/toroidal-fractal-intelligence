# Scaling Study — How Structure Grows with Compute

## Objective

Test whether ATOM can achieve `atoms / parameters > 0.01` by:
1. Reducing encoder bloat
2. Implementing atom reuse (MODIFY instead of CREATE)
3. Using real Gaussian superposition instead of single-mode assignment

## Method

Run 3 configs (small/medium/large) on same synthetic data for 200 steps each.

## Results

| Config | Params | Atoms | Atoms/Param | Loss |
|--------|--------|-------|-------------|------|
| Before Fix | 459K | 100 | 0.00022 | 5.29 |
| After Fix | 58K | 170 | 0.0029 | 5.22 |

**Improvement**: 13x better atoms/parameter ratio.

## Next Targets

1. Target: 0.01 atoms/parameter (3.4x more improvement needed)
2. Strategies:
   - Aggressive MERGE when phase distance < threshold
   - Decay unused atoms (τ-based TTL)
   - Share more parameters across atoms

## Conclusion

Fixing encoder bloat and adding proper superposition yielded significant improvement.
Next milestone: achieve 0.01 atoms/parameter on real dataset.
