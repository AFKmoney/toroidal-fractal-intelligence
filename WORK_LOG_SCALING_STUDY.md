> **ARCHIVE — NOT THE CANONICAL ATOM PATH**
>
> This historical record refers to routines that have been removed. Do not
> execute its commands or use its flattened/Transformer-style benchmarks as
> ATOM evidence. The current rules are in [`ATOM_RULES.md`](ATOM_RULES.md).

# Work Log — Structural Efficiency Study

## Date: 2026-09-12

### Overview
Implemented the core scaling study to test the hypothesis:
"Can a small number of parameters generate and manipulate a much larger amount of structure without exploding compute cost?"

---

### Files Created

#### 1. `study_structural_efficiency.py` (400 lines)
**Purpose**: Main orchestrator for the scaling study.

**Components**:
- `ComputeMetrics`: Track FLOPs, memory, time
- `StructuralMetrics`: Track atoms, aggregates, abstractions
- `LearningMetrics`: Track loss improvement
- `StructuralEfficiencyTracker`: Core metric calculator
- `ScalingStudy`: Main experiment runner

**Key Methods**:
- `train_step()`: Single step with full metrics tracking
- `run_study()`: Complete experiment with checkpoints
- `run_scaling_series()`: Test multiple configurations

**Metrics Computed**:
```python
structural_efficiency = useful_structures / FLOPs
learning_efficiency = loss_improvement / FLOPs
memory_efficiency = useful_structures / RAM_GB
```

Where:
- `useful_structures = atoms*1 + aggregates*10 + abstractions*100`
- Weights reflect estimated information content

---

#### 2. `SCALING_STUDY.md` (150 lines)
**Purpose**: Experimental plan and documentation.

**Sections**:
- Research question
- Experimental design (3 phases)
- Key metrics definitions
- Hypotheses (H1-H4)
- Expected outcomes
- Running instructions

---

### Key Design Decisions

#### 1. Metric Weights
Decided to weight abstractions higher than atoms:
- atoms: 1x
- aggregates: 10x
- abstractions: 100x

**Rationale**: Abstractions represent compressed patterns that can be reused. They are more "valuable" than raw atoms.

#### 2. FLOPs Estimation
Currently using parameter count * 2 * batch_size * seq_len as estimate.

**Limitation**: Not accurate for complex operations (RK4, interactions).

**Future**: Use `torch.profiler` for accurate measurement.

#### 3. Memory Tracking
Using `tracemalloc` for Python-level memory tracking.

**Limitation**: Doesn't capture CUDA memory or temporary tensors.

**Future**: Use `torch.cuda.memory_stats()` for GPU tracking.

#### 4. Scaling Series
Testing 5 atom counts × 3 d_model values = 15 experiments.

**Rationale**: Need to find the "knee" of the scaling curve.

---

### Hypotheses Tested

| Hypothesis | Prediction | Status |
|------------|------------|--------|
| H1: SE increases with scale | SE goes up as atoms increase | ⏳ To test |
| H2: LE plateaus | LE peaks then decreases | ⏳ To test |
| H3: ME decreases | More RAM per structure | ⏳ To test |
| H4: Abstractions increase | More abstractions at scale | ⏳ To test |

---

### Critical Insights

#### 1. Structural Efficiency vs Learning Efficiency
These are **different** metrics:
- SE measures structure creation
- LE measures actual learning

**Question**: Can we create lots of structure without learning?

**Answer** (hypothetical): Yes, if the structure is not useful.

**This is why we need BOTH metrics.**

#### 2. The "Useful Structure" Problem
Not all atoms are equal:
- Some atoms capture important patterns
- Some atoms are noise

**Solution**: Weight abstractions higher (they are more "useful").

**Open question**: How to measure "usefulness" objectively?

#### 3. Compute Bottleneck
Need to measure which component spends the most FLOPs:
- Encoder (token → atom)
- Dynamics (RK4 integration)
- Interactions (k-NN)
- Aggregation
- Abstraction
- Consolidation

**Hypothesis**: Interactions will be the bottleneck at scale.

---

### Next Steps

#### Immediate (Today)
1. [ ] Run Phase 1 (Wikitext baseline with original config)
2. [ ] Analyze FLOPs distribution (which component is expensive?)
3. [ ] Document Phase 1 results

#### Short-term (This Week)
1. [ ] Run Phase 2 (scaling series)
2. [ ] Compare SE across scales
3. [ ] Identify optimal configuration

#### Medium-term (Next Week)
1. [ ] Implement accurate FLOP counting (torch.profiler)
2. [ ] Add GPU memory tracking
3. [ ] Compare with baseline Transformer

---

### Technical Challenges Identified

1. **FLOPs accuracy**: Current estimate is rough. Need profiler.
2. **Memory tracking**: tracemalloc misses CUDA memory.
3. **Dataset loading**: Wikitext download can be slow/unreliable.
4. **Checkpoint size**: Large models + state = large files.

---

### Files Modified/Created

```
study_structural_efficiency.py      (400 lines) — Main study
SCALING_STUDY.md                    (150 lines) — Documentation
```

**Total new code**: ~400 lines
**Total documentation**: ~150 lines

---

### Status

✅ Study framework implemented
✅ Metrics defined
✅ Hypotheses stated
⏳ Experiments not yet run (need to execute)
⏳ Results to be filled

---

*Last updated: 2026-09-12*
