"""Debug model forward."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence

model = ToroidalFractalIntelligence(vocab_size=100, d_model=32, n_modes=32, n_atoms_max=64)

# Test encoder
token_ids = torch.tensor([1, 2, 3])
print("Token IDs:", token_ids.shape)

new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
if new_atom is not None:
    print("Atom created")
    print("r shape:", new_atom.r.shape)
    print("phi shape:", new_atom.phi.shape)
else:
    print("No atom created")

# Test state
alpha = model.state.get_field()
print("\nAlpha shape:", alpha.shape)

# Test dynamics
alpha_new = model.dynamics.evolve(alpha, input_token=new_atom.r if new_atom is not None else None)
print("Alpha new shape:", alpha_new.shape)

# Test production
logits, conf = model.production.produce(alpha_new)
print("Logits shape:", logits.shape)
print("SUCCESS!")
