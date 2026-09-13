"""Debug NaN loss."""
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

print("Forward pass (no gradients)...")
batch = next(iter(loader))
token_ids = batch[:, :-1].reshape(-1)
target_ids = batch[:, 1:].reshape(-1)

with torch.no_grad():
    output = model(token_ids)
    logits = output['logits']
    print(f"Logits shape: {logits.shape}")
    print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
    print(f"Logits mean: {logits.mean().item():.4f}")
    
    loss = torch.nn.CrossEntropyLoss()(logits, target_ids)
    print(f"Loss: {loss.item()}")

print("\nBackward pass...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
output = model(token_ids)
logits = output['logits']
loss = torch.nn.CrossEntropyLoss()(logits, target_ids)
print(f"Loss before backward: {loss.item()}")

loss.backward()

# Check gradients
print("\nGradient check:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print(f"  {name}: NaN/Inf gradient!")
        elif grad_norm > 10:
            print(f"  {name}: Large gradient ({grad_norm:.2f})")

print("\nAfter backward, loss:", loss.item())
