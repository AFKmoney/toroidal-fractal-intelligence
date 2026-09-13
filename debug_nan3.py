"""Debug NaN with debug prints."""
import sys
sys.path.insert(0, '.')
import torch
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

print("Creating model...")
model = ToroidalFractalIntelligence(vocab_size=100, d_model=64, n_modes=64, n_atoms_max=128)

print("Creating data...")
tokens = list(range(100)) * 50
sequences = [torch.tensor(tokens[i:i+32]) for i in range(0, len(tokens)-32, 16)]
loader = InfiniteDataLoader(sequences, batch_size=4, seq_len=32)

print("Forward pass...")
batch = next(iter(loader))
token_ids = batch[:, :-1].reshape(-1)
target_ids = batch[:, 1:].reshape(-1)
print(f"token_ids shape: {token_ids.shape}")
print(f"token_ids sample: {token_ids[:10]}")

# Run forward with debug
print("\nStep 1: Encoder")
new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
print(f"  Atom r shape: {new_atom.r.shape}")
print(f"  Atom r sample: {new_atom.r[0, :5]}")
print(f"  Atom r has NaN: {torch.isnan(new_atom.r).any().item()}")

print("\nStep 2: Add to state")
model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
print(f"  State alpha has NaN: {torch.isnan(model.state.alpha).any().item()}")

print("\nStep 3: Get field")
alpha = model.state.get_field()
print(f"  Alpha shape: {alpha.shape}")
print(f"  Alpha has NaN: {torch.isnan(alpha).any().item()}")

print("\nStep 4: Dynamics")
alpha_new = model.dynamics.evolve(alpha, input_token=new_atom.r if new_atom is not None else None)
print(f"  Alpha new has NaN: {torch.isnan(alpha_new).any().item()}")

print("\nStep 5: Production")
logits, conf = model.production.produce(alpha_new)
print(f"  Logits shape: {logits.shape}")
print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
if torch.isnan(logits).any():
    print(f"  Logits min: {logits.min()}, max: {logits.max()}")

print("\nStep 6: Expand logits")
n_tokens = token_ids.shape[0]
print(f"  Logits shape before expand: {logits.shape}")
if logits.dim() == 2:
    logits_expanded = logits.expand(n_tokens, -1)
else:
    logits_expanded = logits
print(f"  Logits expanded shape: {logits_expanded.shape}")
print(f"  Logits expanded has NaN: {torch.isnan(logits_expanded).any().item()}")
