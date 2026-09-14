> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# Scaling Study Results — Toroidal Fractal Intelligence

## Experimental Setup

### Configuration Tested

| Config | d_model | n_modes | n_atoms_max | Parameters |
|--------|---------|---------|-------------|------------|
| Small | 64 | 64 | 128 | 692,214 |

### Dataset
- Custom text: "Sample text" × 20
- 123 sequences of length 32

### Training
- Optimizer: AdamW (lr=1e-3)
- Loss: CrossEntropy
- Steps: 10 (quick test)
- Batch size: 4

---

## Results

```
Step 1: loss=4.6024, atoms=1
Step 2: loss=4.6114, atoms=2
Step 3: loss=4.5836, atoms=3
Step 4: loss=4.6099, atoms=4
Step 5: loss=4.6135, atoms=5
Step 6: loss=4.6657, atoms=6
Step 7: loss=4.6431, atoms=7
Step 8: loss=4.5771, atoms=8
Step 9: loss=4.6467, atoms=9
Step 10: loss=4.6292, atoms=10
```

**Final Metrics**:
- Final atoms: 10
- Final loss: 4.6292
- Atoms per parameter: 0.000014

---

## Key Findings

### 1. Structure Emergence Confirmed
- Atoms created: 10 in 10 steps
- Each token creates one atom
- Linear growth: atoms = steps

### 2. Low Efficiency
- Atoms/parameter = 0.000014
- This is very low (target: >0.01)
- Model has 692K parameters but only creates 10 atoms

### 3. Loss Stabilization
- Loss stabilizes around 4.6
- This is near random (ln(100) = 4.605)
- Model is not learning effectively

---

## Issues Identified

### 1. Efficiency Problem
The model creates too few atoms relative to its parameter count.

**Possible causes**:
- Encoder is too complex for the task
- Atoms are not being retained (max limit?)
- Training is too short

### 2. Learning Problem
The loss is not decreasing significantly.

**Possible causes**:
- Learning rate too low
- Model capacity not being used
- Data too simple

### 3. Scaling Not Tested
Only small config was tested.

**Need to test**:
- Medium config (d_model=128)
- Large config (d_model=256)

---

## Next Steps

### Immediate
1. [ ] Run longer training (100+ steps)
2. [ ] Test medium and large configs
3. [ ] Tune learning rate

### Analysis
1. [ ] Measure FLOPs per step
2. [ ] Compare atoms/parameter across scales
3. [ ] Check if abstractions form

---

## Conclusion

The scaling study framework is **functional** but results show:
- ✅ Structure emergence works
- ⚠️ Efficiency is very low
- ⚠️ Learning is minimal

**Hypothesis**: The model needs more training data and longer training to show meaningful scaling.

**Next action**: Run full scaling study with 100 steps per config.
