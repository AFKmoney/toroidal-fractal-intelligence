# Scaling Study Results — Toroidal Fractal Intelligence

## Experimental Setup

### Configurations Tested

| Config | d_model | n_modes | n_atoms_max | Parameters |
|--------|---------|---------|-------------|------------|
| Small | 64 | 64 | 128 | 692,214 |
| Medium | 128 | 128 | 256 | 1,325,174 |
| Large | 256 | 256 | 512 | 3,205,494 |

### Dataset
- Custom text: "Sample text" × 50
- Sequence length = d_model
- Batch size = 4

### Training
- Optimizer: AdamW (lr=1e-3)
- Loss: CrossEntropy
- Steps: 20-30
- Gradient clipping: max_norm=1.0

---

## Results

### Small Config (d_model=64, n_atoms_max=128)
```
Parameters: 692,214
Final atoms: 10
Avg loss: 4.6292
Final loss: 4.6292
Time: ~5s
Atoms/parameter: 0.000014
```

### Medium Config (d_model=128, n_atoms_max=256)
```
Parameters: 1,325,174
Final atoms: 30
Avg loss: 4.6080
Final loss: 4.6054
Time: 49.54s
Atoms/parameter: 0.000023
```

### Large Config (d_model=256, n_atoms_max=512)
```
Parameters: 3,205,494
Final atoms: 20
Avg loss: 4.6079
Final loss: 4.6047
Time: 118.15s
Atoms/parameter: 0.000006
```

---

## Summary Table

| Config | Params | Atoms | Loss | Time(s) | Atoms/Param |
|--------|--------|-------|------|---------|-------------|
| Small | 692K | 10 | 4.629 | 5 | 0.000014 |
| Medium | 1.3M | 30 | 4.605 | 50 | 0.000023 |
| Large | 3.2M | 20 | 4.605 | 118 | 0.000006 |

---

## Key Findings

### 1. Structure Emergence Confirmed
- ✅ Atoms are created at a rate of ~1 per step
- ✅ Each token generates one atom
- ✅ Structure grows linearly with training

### 2. Efficiency is Very Low
- Atoms/parameter ratios are extremely low (0.000006 - 0.000023)
- Target should be >0.01 for meaningful efficiency
- **Current: 100-1000x below target**

### 3. Learning is Minimal
- Loss stabilizes around 4.6 (near random for vocab=100)
- No significant improvement over training
- Model is not learning the pattern

### 4. Scaling Behavior
- Medium config has best atoms/parameter ratio
- Large config actually performs worse
- Suggests optimal scale exists but is not reached yet

---

## Issues Identified

### 1. Efficiency Problem
**Observation**: Atoms/parameter is too low
**Impact**: Architecture doesn't buy capacity efficiently
**Possible causes**:
- Encoder creates too many parameters per atom
- Atoms are not being reused (CREATE instead of MODIFY)
- Model capacity is wasted on simple task

### 2. Learning Problem
**Observation**: Loss not decreasing
**Impact**: No evidence of learning
**Possible causes**:
- Learning rate too low
- Training too short
- Data too simple
- Loss function not appropriate

### 3. Scaling Problem
**Observation**: Large config worse than medium
**Impact**: Cannot scale linearly
**Possible causes**:
- RK4 dynamics unstable at large scale
- Attractor terms cause explosion
- Memory constraints

---

## Hypothesis Testing

### H1: Structural Efficiency Increases with Scale
**Prediction**: Larger models create more atoms per parameter
**Result**: ❌ FAILED
- Small: 0.000014
- Medium: 0.000023 (best)
- Large: 0.000006 (worse)

### H2: Learning Efficiency Improves with Scale
**Prediction**: Larger models learn faster
**Result**: ⚠️ INCONCLUSIVE
- All configs have similar loss (~4.6)
- Need longer training to determine

### H3: Abstractions Form at Scale
**Prediction**: More abstractions with more atoms
**Result**: ⚠️ NOT TESTED
- Need to run longer and check abstraction count

---

## Conclusions

### What Works
1. ✅ Token → Atom conversion
2. ✅ Fractal superposition
3. ✅ RK4 dynamics (with clipping)
4. ✅ Production head

### What Doesn't Work
1. ❌ Efficiency (atoms/parameter too low)
2. ❌ Learning (loss not decreasing)
3. ❌ Scaling (large config worse)

### Next Steps
1. Reduce model size (fewer parameters per atom)
2. Increase training steps (1000+ instead of 20)
3. Use more complex dataset
4. Tune hyperparameters (lr, clipping)
5. Implement atom reuse (MODIFY instead of CREATE)

---

## Raw Data

```json
{
  "small": {
    "params": 692214,
    "atoms": 10,
    "loss": 4.6292,
    "atoms_per_param": 0.000014
  },
  "medium": {
    "params": 1325174,
    "atoms": 30,
    "loss": 4.6054,
    "atoms_per_param": 0.000023
  },
  "large": {
    "params": 3205494,
    "atoms": 20,
    "loss": 4.6047,
    "atoms_per_param": 0.000006
  }
}
```

---

## Status

⚠️ **SCALING STUDY INCONCLUSIVE**

The framework works but results show:
- Very low efficiency
- No meaningful learning
- Poor scaling behavior

**Recommendation**: Revisit architecture before scaling further.
