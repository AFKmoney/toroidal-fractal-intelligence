# Benchmark Design

## Goal

Compare ATOM vs Transformer on equal footing:
- Same parameter count
- Same FLOPs budget
- Same dataset
- Same training time

## Metrics

1. **Loss**: Standard perplexity measure
2. **Structures/FLOP**: ATOM's unique advantage
3. **Info/FLOP**: Δlog_likelihood / FLOPs
4. **Generalization**: Performance on held-out domain

## Experimental Setup

```python
# Fixed budget
PARAMS = 100_000
FLOPS = 1e9
TOKENS = 100_000  # WikiText-2 subset

# ATOM config
ATOM = ToroidalFractalIntelligence(
    d_model=128,
    n_modes=256,
    n_atoms_max=1000,
)

# Transformer config (matched params)
TRANSFORMER = TransformerEncoder(
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=512,
)
```

## Expected Outcomes

| Scenario | Interpretation |
|----------|---------------|
| ATOM loss < Transformer | Structure helps learning |
| ATOM structures/FLOP >> 1 | Parameter efficiency working |
| ATOM generalizes better | Abstraction transfer working |

## Current Status

Prototype results show promise but need rigorous validation.
