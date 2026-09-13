"""Debug production NaN."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence

model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)

# Create some state
alpha = torch.randn(64, 64)
print(f"Alpha mean: {alpha.mean().item():.4f}, std: {alpha.std().item():.4f}")

# Test production
logits, conf = model.production.produce(alpha)
print(f"Logits shape: {logits.shape}")
print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
print(f"Logits min: {logits.min().item()}, max: {logits.max().item()}")

# Check decoder weights
print("\nDecoder weights:")
for name, param in model.production.decoder.named_parameters():
    print(f"  {name}: shape={param.shape}, has NaN={torch.isnan(param).any().item()}")

# Test decoder directly
state = alpha.mean(dim=0)
print(f"\nState shape: {state.shape}")
print(f"State has NaN: {torch.isnan(state).any().item()}")

out = model.production.decoder[0](state)
print(f"After linear 0: shape={out.shape}, has NaN={torch.isnan(out).any().item()}")

out = model.production.decoder[1](out)
print(f"After GELU: shape={out.shape}, has NaN={torch.isnan(out).any().item()}")

out = model.production.decoder[2](out)
print(f"After linear 1: shape={out.shape}, has NaN={torch.isnan(out).any().item()}")
print(f"Output min: {out.min().item()}, max: {out.max().item()}")
