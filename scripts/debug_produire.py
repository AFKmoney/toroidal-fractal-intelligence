"""Debug production shapes."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence

model = ToroidalFractalIntelligence(vocab_size=100, d_model=32, n_modes=32, n_atoms_max=64)
print("Production decoder:", model.production.decoder)
print("Production d_model:", model.production.d_model)
print("Production vocab_size:", model.production.vocab_size)

# Test produce directly
alpha = torch.randn(32, 32)  # [n_modes, d_model]
print("\nAlpha shape:", alpha.shape)

logits, conf = model.production.produce(alpha)
print("Logits shape:", logits.shape)
print("Confidence shape:", conf.shape)
