"""Debug production step by step."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence

model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)

# Create alpha
alpha = torch.randn(64, 64)
print(f"Alpha: mean={alpha.mean().item():.4f}, nan={torch.isnan(alpha).any().item()}")

# Test production
print("\nCalling produce...")
logits, conf = model.production.produce(alpha)
print(f"Logits: shape={logits.shape}, nan={torch.isnan(logits).any().item()}")

# Now test with model forward
print("\n=== MODEL FORWARD ===")
token_ids = torch.tensor([1, 2, 3])
print(f"Token IDs: {token_ids}")

# Manual forward
new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
print(f"Atom created: {new_atom is not None}")
print(f"Atom r: shape={new_atom.r.shape}, nan={torch.isnan(new_atom.r).any().item()}")

model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
print(f"State alpha: nan={torch.isnan(model.state.alpha).any().item()}")

alpha_new = model.dynamics.evolve(model.state.get_field(), input_token=new_atom.r)
print(f"Alpha new: nan={torch.isnan(alpha_new).any().item()}")

logits, conf = model.production.produce(alpha_new)
print(f"Logits: nan={torch.isnan(logits).any().item()}")
print(f"Logits value: {logits}")
