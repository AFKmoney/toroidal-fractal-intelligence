# Architectural Decisions Log

## ADR-001: Spectral Field vs Explicit Atom Storage

**Decision**: Use spectral field representation (fixed n_modes) instead of storing N explicit atoms.

**Context**: Storing N atoms explicitly requires O(N * d_model) memory. For N=1024, d_model=256, this is 256K floats per property × 8 properties = ~6.5M floats ≈ 26MB. As N grows to millions, this becomes infeasible.

**Alternatives Considered**:
1. Explicit atom list (O(N) memory)
2. Sparse tensor representation
3. Spectral field (O(n_modes) memory)

**Consequences**:
- ✅ Memory scales with n_modes, not N
- ✅ Natural frequency/phase separation
- ✅ Efficient RK4 integration on field
- ❌ Information loss in projection
- ❌ Requires careful mode initialization

**Status**: ✅ Accepted

---

## ADR-002: Shared Parameters Θ

**Decision**: Use a small set of shared parameters to control all atom interactions.

**Context**: The spec requires `capacity >> independent parameters`. If each atom had its own parameters, we'd lose this property.

**Alternatives Considered**:
1. Per-atom parameters (O(N) parameters)
2. Shared parameters (O(1) parameters)
3. Hybrid approach

**Consequences**:
- ✅ Enables billions of potential states with ~4 shared params
- ✅ Decouples capacity from training cost
- ✅ Follows spec principle: `F(A_i, A_j; Θ)`
- ❌ Less expressivity per interaction
- ❌ Requires careful tuning of Θ

**Status**: ✅ Accepted

---

## ADR-003: Sparse k-NN Interaction

**Decision**: Use k-NN attention (k=16) for pairwise interactions instead of full O(N^2) matrix.

**Context**: Full pairwise interaction requires O(N^2) computations. For N=1024, this is ~1M operations per step. For N=1M, this is ~1T operations—impossible.

**Alternatives Considered**:
1. Full pairwise matrix (O(N^2))
2. Sparse k-NN (O(N * k))
3. Field-only interactions (O(n_modes))

**Consequences**:
- ✅ Scales to large N
- ✅ Biological plausibility (local connectivity)
- ❌ May miss long-range interactions
- ✅ Recovered through field representation

**Status**: ✅ Accepted

---

## ADR-004: Hierarchical Aggregation Depth

**Decision**: Limit aggregation depth to 5 levels.

**Context**: Unlimited depth could lead to computational explosion. Depth 5 allows: atom → motif → structure → meta-structure → abstraction.

**Alternatives Considered**:
1. Unlimited depth (risky)
2. Fixed depth 3
3. Fixed depth 5
4. Dynamic depth based on coherence

**Consequences**:
- ✅ Controls computational complexity
- ✅ Matches spec: `A → G1 → G2 → G3 → ...`
- ❌ May miss deep abstractions
- ✅ Can increase if needed

**Status**: ✅ Accepted

---

## ADR-005: Dynamic Consolidation

**Decision**: Only consolidate structures that exceed stability threshold.

**Context**: Infinite learning requires bounded memory. If we consolidate everything, we lose flexibility. If we consolidate nothing, we lose persistence.

**Alternatives Considered**:
1. Consolidate all structures
2. Consolidate no structures
3. Threshold-based consolidation

**Consequences**:
- ✅ Bounded persistent memory
- ✅ Preserves flexibility for transient structures
- ✅ Enables true continuous learning
- ❌ Requires careful threshold tuning

**Status**: ✅ Accepted

---

## ADR-006: No Training/Inference Boundary

**Decision**: Design for infinite continuous training, not fixed training then inference.

**Context**: Spec explicitly states: "Il n'existe donc pas nécessairement de frontière stricte: TRAINING → FIN → INFERENCE".

**Alternatives Considered**:
1. Fixed training then inference
2. Fine-tuning mode
3. True continuous learning

**Consequences**:
- ✅ Matches spec vision
- ✅ No catastrophic forgetting (structures consolidate)
- ✅ Real-time adaptation
- ❌ Harder to evaluate (no fixed test set)
- ❌ Requires infinite data stream

**Status**: ✅ Accepted

---

## ADR-007: PyTorch Implementation

**Decision**: Use PyTorch for all tensor operations.

**Context**: Need differentiable operations for gradient-based learning. PyTorch is the standard for research.

**Alternatives Considered**:
1. NumPy only (no autograd)
2. TensorFlow
3. PyTorch

**Consequences**:
- ✅ Autograd for backprop
- ✅ GPU acceleration
- ✅ Ecosystem (transformers, datasets)
- ✅ Research standard
- ❌ Python dependency

**Status**: ✅ Accepted

---

## ADR-008: RK4 as Default Integrator

**Decision**: Use RK4 as default, but allow Euler/Heun for ablation.

**Context**: Spec says "RK4 ne doit pas être considéré comme une obligation absolue". But RK4 is the reference integrator.

**Alternatives Considered**:
1. Euler (simpler, less accurate)
2. RK4 (standard, good accuracy)
3. Adaptive step size

**Consequences**:
- ✅ Good accuracy for dynamics
- ✅ Stable for reasonable dt
- ❌ 4 function evaluations per step
- ✅ Can switch for comparison

**Status**: ✅ Accepted

---

## ADR-009: 8-Property Atom Model

**Decision**: Use the 8 properties from spec: (r, φ, ω, E, κ, M, τ, ρ).

**Context**: Spec defines these as minimum requirements. Can add/remove based on experiments.

**Alternatives Considered**:
1. Minimal 4 properties
2. 8 properties (spec)
3. Extended 12 properties

**Consequences**:
- ✅ Matches spec
- ✅ Each property has clear meaning
- ✅ Differentiable
- ❌ May be over-specified
- ✅ Can simplify based on experiments

**Status**: ✅ Accepted (with experimentation clause)

---

## ADR-010: Modular Architecture

**Decision**: Separate each component into its own module.

**Context**: Need maintainability and testability. Spec requires multiple mechanisms (interaction, aggregation, abstraction, consolidation).

**Alternatives Considered**:
1. Monolithic class
2. Modular classes
3. Functional approach

**Consequences**:
- ✅ Easy to test each component
- ✅ Easy to swap implementations
- ✅ Matches spec architecture diagram
- ❌ More files to manage
- ✅ Better for research iteration

**Status**: ✅ Accepted

---

## Summary

| ADR | Decision | Status |
|-----|----------|--------|
| 001 | Spectral field representation | Accepted |
| 002 | Shared parameters Θ | Accepted |
| 003 | Sparse k-NN interaction | Accepted |
| 004 | Max aggregation depth 5 | Accepted |
| 005 | Threshold-based consolidation | Accepted |
| 006 | No training/inference boundary | Accepted |
| 007 | PyTorch implementation | Accepted |
| 008 | RK4 default integrator | Accepted |
| 009 | 8-property atom model | Accepted |
| 010 | Modular architecture | Accepted |

All decisions support the core hypothesis: **capacity through structure composition, not parameter count**.
