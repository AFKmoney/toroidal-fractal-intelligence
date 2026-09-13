"""Debug full forward NaN."""
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
target_ids = batch[:, 1:].reshape(-1)

print(f"Token IDs shape: {token_ids.shape}")
print(f"Token IDs sample: {token_ids[:10]}")

# Forward pass with detailed logging
print("\n=== FORWARD PASS ===")

# Step 1: Encoder
new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
print(f"1. Encoder: atom_r_nan={torch.isnan(new_atom.r).any().item()}")

# Step 2: Add to state
model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
print(f"2. State: alpha_nan={torch.isnan(model.state.alpha).any().item()}")

# Step 3: Dynamics
alpha = model.state.get_field()
alpha_new = model.dynamics.evolve(alpha, input_token=new_atom.r)
print(f"3. Dynamics: alpha_new_nan={torch.isnan(alpha_new).any().item()}")

# Step 4: Production
logits, conf = model.production.produce(alpha_new)
print(f"4. Production: logits_nan={torch.isnan(logits).any().item()}")
print(f"   logits shape: {logits.shape}")
print(f"   logits sample: {logits[0, :5]}")

# Check if issue is in the loop in model.py
print("\n=== CHECKING MODEL.FORWARD ===")
# Manually run what model.forward does
print(f"token_ids.dim(): {token_ids.dim()}")
if token_ids.dim() > 1:
    token_ids_flat = token_ids[:, -1]
    print(f"Using last token: shape={token_ids_flat.shape}")
else:
    token_ids_flat = token_ids
    print(f"Using all tokens: shape={token_ids_flat.shape}")

# The issue might be that we're calling encoder with batch of 124 tokens
# but it creates 124 atoms and adds them all to state
print(f"\nBefore forward: {len(model.atoms)} atoms")
output = model(token_ids)
print(f"After forward: {len(model.atoms)} atoms")
print(f"Output logits NaN: {torch.isnan(output['logits']).any().item()}")
