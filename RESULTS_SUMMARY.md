# Toroidal Fractal Intelligence — Results Summary

## Scope

This project implements a prototype of the "Toroidal Fractal Intelligence" (ATOM) architecture.
The architecture aims to decouple structural capacity from parameter count by maintaining a
dynamically growing set of computational atoms over a shared set of learned rules.

## What we have

A working, trainable system with:

- Token → Toroidal Atom conversion with 8 primitive properties
- Fractal superposition over spectral modes
- RK4-driven dynamics with shared coupling rules
- Hierarchical aggregation / abstraction / consolidation
- Infinite continuous learning (no hard training/inference boundary)
- Internal `ThinkerAgent` for pre-generation reasoning
- Model save/load (`.pt`) that captures both weights and dynamic state

## What we have proven

- **Structure emergence**: atoms are created per token, aggregates/abstractions form as coherent groups emerge over time.
- **Infinite learning**: training can resume after chat/inference without state reset.
- **Stability**: after fixes in this round (NaN guards, broadcasting fixes, unified production head), the model trains without crash across small synthetic runs.
- **Architecture coherence**: the full pipeline from token to output is wired end-to-end.

## What remains to be proven

The original hypothesis — that a small parameter set can yield *much* larger effective capacity via structure composition — still needs quantitative evidence at scale. Specifically:

- **Scaling law**: does `structures / FLOPs` improve as we increase `n_atoms_max` and dataset size?
- **Fair comparison**: ATOM vs an equivalent-parameter Transformer on the same tokens for the same FLOPs.
- **Abstraction reuse**: do abstractions generalise across domains?
- **Infinite learning in practice**: does retention improve over time on a sustained stream?

Right now, measured `atoms / parameters` is still far below the target (`~0.00003` vs target `> 0.01`), which means either the encoding is too expensive, or most atoms are transient and don't contribute to persistent capacity. Fixing this is the immediate engineering target.

## How to read these results

- **Loss ~ 4.6 on tiny Shakespeare**: not meaningful. This is near random for a small vocab. The point was structural, not predictive.
- **Atoms/parameter 0.00003**: this is the key metric to move. It measures how many distinct computational primitives exist per trained weight. Right now, every new token tends to spawn a new atom instead of reusing or merging with existing ones.
- **Training time vs Transformer**: ATOM is currently slower per step because of the RK4 + interaction graph. If scaling holds, the *per-token information acquired* should be higher.

## Next experiments

1. **Fix atom reuse** — implement MODIFY/MERGE gates in the encoder so similar tokens merge into existing atoms rather than creating new ones.
2. **Real dataset** — run on Wikitext-2 with 1M+ tokens, measure `atoms / FLOPs` and `loss improvement / FLOPs`.
3. **Fair baseline** — build a Transformer with the same parameter count, same FLOPs, same data, compare loss and generalisation.
4. **Ablation** — remove aggregation/abstraction and measure the delta in `E_info`. If abstractions don't help, they're dead weight.

## Files

- `README.md` — quickstart and architecture overview
- `ARCHITECTURE.md` — mathematical foundations and module docs
- `BENCHMARK.md` — experimental design for the next benchmark run
- `docs/` — all previous documentation, reports, and logs
- `scripts/` — experiment runners and one-offs
- `tests/` — regression tests
- `src/` — source code
- `results/` — experiment outputs

## Citation

```bibtex
@misc{toroidal_fractal_intelligence,
  title={Toroidal Fractal Intelligence: A Continuous Structured Learning Architecture},
  author={PHIL},
  year={2026},
  version={0.1.0}
}
```
