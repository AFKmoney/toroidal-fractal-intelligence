"""Debug batch forward NaN."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)

tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)

batch = next(iter(loader))
token_ids = batch[:, :-1].reshape(-1)
print(f"Token IDs shape: {token_ids.shape}")

# Forward
print("\nForward pass...")
output = model(token_ids)
print(f"Output logits NaN: {torch.isnan(output['logits']).any().item()}")
print(f"Output logits shape: {output['logits'].shape}")

# Check each component
print("\n=== COMPONENT CHECK ===")

# Encoder
new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
print(f"Encoder: atom_r_nan={torch.isnan(new_atom.r).any().item()}")

# State
model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
print(f"State: alpha_nan={torch.isnan(model.state.alpha).any().item()}")

# Dynamics
alpha = model.state.get_field()
alpha_new = model.dynamics.evolve(alpha, input_token=new_atom.r)
print(f"Dynamics: alpha_new_nan={torch.isnan(alpha_new).any().item()}")

# Production
logits, conf = model.production.produce(alpha_new)
print(f"Production: logits_nan={torch.isnan(logits).any().item()}")
print(f"Production: logits_shape={logits.shape}")

# Check if issue is in the model.forward expansion
print("\n=== MODEL.FORWARD LOGITS ===")
# The model expands logits to match token_ids
n_tokens = token_ids.shape[0]
print(f"n_tokens: {n_tokens}")
print(f"logits before expand: {logits.shape}")

# This is what model.py does
logits_expanded = logits.unsqueeze(0).expand(n_tokens, -1) if logits.dim() == 2 else logits.unsqueeze(0).expand(n_tokens, -1)
print(f"logits after expand: {logits_expanded.shape}")
print(f"logits expanded NaN: {torch.isnan(logits_expanded).any().item()}")
