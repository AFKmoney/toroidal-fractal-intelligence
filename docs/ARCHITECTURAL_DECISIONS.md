# Architectural Decisions Record

## ADR-001: Spectral Field Representation

**Status**: Accepted
**Date**: 2026-09-12

### Context
Need to represent N atoms efficiently without O(N×d) memory growth.

### Decision
Use a fixed-size spectral field `α ∈ R^(n_modes × d_model)` where each atom contributes to modes via a Gaussian kernel centered at its frequency ω.

### Consequences
- Memory: O(n_modes × d_model) regardless of N
- Computation: O(N × n_modes) for projection, O(n_modes²) for interactions
- Tradeoff: n_modes must be chosen as a capacity knob

---

## ADR-002: Shared Parameters (Θ)

**Status**: Accepted
**Date**: 2026-09-12

### Context
Each atom having its own weight matrix would defeat the purpose of parameter efficiency.

### Decision
All atoms share the same dynamics function F(·; Θ) with ~10-50 shared parameters controlling coupling, decay, and attractor behavior.

### Consequences
- Parameter count stays constant as atoms grow
- Atoms differentiate via initial conditions (r, φ, ω) and interaction history
- Training focuses on Θ, not per-atom weights

---

## ADR-003: RK4 Integration

**Status**: Accepted
**Date**: 2026-09-12

### Context
Need stable continuous-time dynamics over the spectral field.

### Decision
Use 4th-order Runge-Kutta (RK4) for time integration with adaptive dt.

### Consequences
- Stable for stiff systems
- 4 function evaluations per step (expensive but necessary for stability)
- Can fall back to Euler for ablation studies

---

## ADR-004: Sparse k-NN Interactions

**Status**: Accepted
**Date**: 2026-09-12

### Context
Full O(N²) interaction graph is infeasible for large N.

### Decision
Use k-nearest-neighbors (k=16) based on phase+frequency distance for pairwise interactions.

### Consequences
- O(N × k) instead of O(N²)
- May miss long-range correlations
- Can switch to full attention if N stays small

---

## ADR-005: CREATE/MODIFY/MERGE Operations

**Status**: In Progress
**Date**: 2026-09-12

### Context
Naive CREATE for every token leads to O(N) growth and poor reuse.

### Decision
Encoder predicts operation type; MODIFY updates existing atom, MERGE combines two atoms.

### Consequences
- Requires similarity matching (cosine on φ, ω proximity)
- Adds complexity to encoder output interpretation
- Critical for achieving target atoms/parameter ratio

---

## ADR-006: Hierarchical Aggregation

**Status**: Accepted
**Date**: 2026-09-12

### Context
Flat atom lists don't capture compositional structure.

### Decision
Aggregate coherent atoms (high phase sync + spectral proximity) into higher-level structures up to depth 5.

### Consequences
- Enables abstraction layers
- Aggregates can themselves be aggregated (fractal property)
- Adds O(N log N) aggregation cost

---

## ADR-007: Selective Consolidation

**Status**: Accepted
**Date**: 2026-09-12

### Context
Not all structures are worth persisting.

### Decision
Only consolidate atoms/aggregates that meet stability thresholds (energy, repetition, predictive utility).

### Consequences
- Persistent memory is a small subset of total atoms
- Enables "forgetting" of transient structures
- Consolidation is itself a learnable process

---

## ADR-008: Production via Mean Pooling

**Status**: Fixed
**Date**: 2026-09-12

### Context
Original implementation had shape mismatches when producing logits for batched inputs.

### Decision
Use mean pooling over spectral modes to get a single [d_model] representation, then pass through decoder MLP.

### Consequences
- Simple but effective for batch production
- Loses per-mode detail but gains computational efficiency
- Fixed broadcasting bug in production.py

---

## ADR-009: Gaussian Mode Projection

**Status**: Fixed
**Date**: 2026-09-13

### Context
Original implementation assigned each atom to a single mode, losing spectral information.

### Decision
Spread each atom's contribution across all modes using a Gaussian kernel centered at ω.

### Consequences
- Better spectral representation
- More stable gradients
- Fixed broadcasting shapes in state.py

---

## ADR-010: Compact Encoder

**Status**: Fixed
**Date**: 2026-09-13

### Context
Original encoder used d_hidden=512 for d_model=32, creating 459K parameters for trivial task.

### Decision
Reduce d_hidden to match d_model, eliminating unnecessary parameters.

### Consequences
- Parameter count dropped from 459K to ~58K (8x reduction)
- Atoms/parameter improved from 0.00022 to ~0.003 (15x improvement)
- Still below target of 0.01, needs further optimization
