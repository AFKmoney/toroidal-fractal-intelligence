"""Debug NaN - check initial values."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence

print("Creating model...")
model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)

print("\nChecking initial values...")
for name, param in model.named_parameters():
    if param.requires_grad:
        val = param.data
        if torch.isnan(val).any():
            print(f"  {name}: NaN in data!")
        if torch.isinf(val).any():
            print(f"  {name}: Inf in data!")

print("\nRunning forward pass...")
token_ids = torch.tensor([1, 2, 3])
with torch.no_grad():
    new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
    print(f"Atom r: {new_atom.r}")
    print(f"Atom phi: {new_atom.phi}")
    print(f"Atom omega: {new_atom.omega}")
    print(f"Atom E: {new_atom.E}")

print("\nAdding to state...")
model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
alpha = model.state.get_field()
print(f"Alpha shape: {alpha.shape}")
print(f"Alpha min: {alpha.min().item()}, max: {alpha.max().item()}")
print(f"Alpha has NaN: {torch.isnan(alpha).any().item()}")

print("\nEvolving dynamics...")
alpha_new = model.dynamics.evolve(alpha, input_token=new_atom.r)
print(f"Alpha new shape: {alpha_new.shape}")
print(f"Alpha new has NaN: {torch.isnan(alpha_new).any().item()}")
if torch.isnan(alpha_new).any():
    print(f"Alpha new min: {alpha_new.min().item()}, max: {alpha_new.max().item()}")

print("\nProduction...")
logits, conf = model.production.produce(alpha_new)
print(f"Logits shape: {logits.shape}")
print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
