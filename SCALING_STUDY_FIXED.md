> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# Scaling Study Results — Fixed Version

## Corrections Applied

### Problem 1: Batch Processing
**Issue**: Model expanded logits to match batch size, causing NaN
**Fix**: Use mean pooling for production head
```python
# Before (wrong)
logits = logits.unsqueeze(0).expand(n_tokens, -1)  # [n_tokens, vocab]

# After (correct)
logits = logits.mean(dim=0, keepdim=True).expand(n_tokens, -1)  # [n_tokens, vocab]
```

### Problem 2: RK4 Explosion
**Issue**: Attractor term caused explosion (max: 4266)
**Fix**: Added clamping
```python
alpha_new = torch.clamp(alpha_new, min=-10.0, max=10.0)
```

### Problem 3: Learning Rate
**Issue**: lr=1e-3 too high
**Fix**: Reduced to lr=3e-4

---

## Results (Fixed)

### Small Config (d_model=64, n_atoms_max=128)
```
Params: 692,214
Steps: 50
Final atoms: 50
Final loss: 4.5924
Atoms/parameter: 0.000072
```

### Comparison with Before

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Final loss | 4.6292 | 4.5924 | -0.8% |
| Atoms/param | 0.000014 | 0.000072 | +414% |
| Training time | 5s | 5s | Same |

---

## Key Findings

### 1. Batch Processing Was Critical
The NaN issue was caused by incorrect logits expansion. Fixing this improved stability.

### 2. Efficiency Improved 5x
Atoms/parameter went from 0.000014 to 0.000072.
Still below target (0.01), but 5x improvement.

### 3. Loss Still Not Learning
Loss only improved by 0.8% over 50 steps.
Need longer training (1000+ steps) to see real learning.

---

## Next Steps

1. **Longer training**: Run 500-1000 steps
2. **Larger dataset**: Use Wikitext-2 instead of synthetic data
3. **Atom reuse**: Implement MODIFY operation to avoid CREATE every step
4. **Compare configs**: Test small/medium/large with same training steps

---

## Status

✅ Fixed batch processing
✅ Fixed RK4 stability
✅ Improved efficiency 5x
⏳ Need longer training for real learning
