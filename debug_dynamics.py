"""Debug dynamics NaN."""
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

# Forward
new_atom, op, conf = model.encoder(token_ids, None, model.atoms)
print(f"1. Encoder: {new_atom.r.shape}, nan={torch.isnan(new_atom.r).any().item()}")

model.state.add_atom_contribution(
    new_atom.r, new_atom.phi, new_atom.omega,
    new_atom.E, new_atom.kappa,
)
print(f"2. State: {model.state.alpha.shape}, nan={torch.isnan(model.state.alpha).any().item()}")

alpha = model.state.get_field()
print(f"3. Alpha: {alpha.shape}, nan={torch.isnan(alpha).any().item()}")

# Test dynamics
print(f"\n4. Testing dynamics...")
print(f"   alpha shape: {alpha.shape}")
print(f"   input_token shape: {new_atom.r.shape}")

# Call dynamics directly
d_alpha_dt = model.dynamics.dynamics(alpha, new_atom.r, None)
print(f"   d_alpha_dt shape: {d_alpha_dt.shape}")
print(f"   d_alpha_dt nan: {torch.isnan(d_alpha_dt).any().item()}")

# Check each term
base = model.dynamics.dynamics.dynamics_net(alpha.reshape(-1, 64))
base = base.reshape(alpha.shape)
print(f"   base nan: {torch.isnan(base).any().item()}")

phase_coupling = model.dynamics.dynamics.phase_sync * torch.sin(
    torch.sin(alpha - torch.roll(alpha, shifts=1, dims=0))
)
print(f"   phase_coupling nan: {torch.isnan(phase_coupling).any().item()}")

attractor = -torch.nn.functional.relu(
    torch.cdist(alpha, model.dynamics.dynamics.attractor_centers) ** 2
).mean(dim=-1, keepdim=True) * alpha
print(f"   attractor nan: {torch.isnan(attractor).any().item()}")

input_mean = new_atom.r.mean(dim=0)
drive = input_mean.unsqueeze(0).expand(64, -1) * model.dynamics.dynamics.coupling_scale
print(f"   drive nan: {torch.isnan(drive).any().item()}")

decay = (model.dynamics.dynamics.energy_decay - 1.0) * alpha
print(f"   decay nan: {torch.isnan(decay).any().item()}")

# Test RK4
print(f"\n5. Testing RK4...")
dt = model.dynamics.dt
k1 = d_alpha_dt
print(f"   k1 nan: {torch.isnan(k1).any().item()}")

k2 = model.dynamics.dynamics(alpha + dt/2 * k1, new_atom.r, None)
print(f"   k2 nan: {torch.isnan(k2).any().item()}")

k3 = model.dynamics.dynamics(alpha + dt/2 * k2, new_atom.r, None)
print(f"   k3 nan: {torch.isnan(k3).any().item()}")

k4 = model.dynamics.dynamics(alpha + dt * k3, new_atom.r, None)
print(f"   k4 nan: {torch.isnan(k4).any().item()}")

alpha_new = alpha + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
print(f"   alpha_new nan: {torch.isnan(alpha_new).any().item()}")
print(f"   alpha_new min: {alpha_new.min().item()}, max: {alpha_new.max().item()}")
