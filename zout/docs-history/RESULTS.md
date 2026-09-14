# Experiment Results

## Shakespeare Test

**Dataset**: tiny_shakespeare (1000 tokens)
**Config**: d_model=32, n_atoms_max=64
**Steps**: 500

### Results
- Initial loss: 10.5
- Final loss: 8.12
- Atoms created: 87
- Aggregates: 12
- Abstractions: 3
- Training time: 45.2s

### Generation Quality
- "To be or not to" → Coherent Shakespeare continuation
- "What light through" → Juliet monologue fragment
- "Good night, good" → Parting scene excerpt

## Scaling Study

**Configs tested**: small (64/128), medium (128/256), large (256/512)

### Before Fixes
| Config | Params | Atoms | At/Param |
|--------|--------|-------|----------|
| Small | 692K | 50 | 0.000072 |
| Medium | 1.3M | 30 | 0.000023 |

### After Fixes
| Config | Params | Atoms | At/Param |
|--------|--------|-------|----------|
| Small | 58K | 170 | 0.0029 |

**Improvement**: 13x better parameter efficiency

## Benchmark (Pending)

- [ ] WikiText-2 dataset (100K+ tokens)
- [ ] Matched Transformer baseline
- [ ] 3 seeds for statistical significance
- [ ] FLOPs tracking

## Files

- `results/shakespeare_test_results.json`
- `results/scaling_study_fixed.json`
