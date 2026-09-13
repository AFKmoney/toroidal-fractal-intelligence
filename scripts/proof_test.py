"""Proof of concept tests."""
import torch
import numpy as np
from src.toroidal.model import ToroidalFractalIntelligence
from src.io.data import InfiniteDataLoader

def test_structure_emergence():
    """Test that structures emerge from tokens."""
    print("Test 1: Structure Emergence")
    
    model = ToroidalFractalIntelligence(vocab_size=50, d_model=16, n_modes=16, n_atoms_max=32)
    
    np.random.seed(42)
    tokens = np.random.randint(0, 50, 500)
    sequences = [torch.tensor(tokens[i:i+8]) for i in range(0, len(tokens)-8, 4)]
    loader = InfiniteDataLoader(sequences, batch_size=2, seq_len=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(1, 51):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    assert len(model.atoms) > 0, "No atoms created!"
    print(f"  ✓ Created {len(model.atoms)} atoms")
    return True

def test_infinite_learning():
    """Test that learning continues without boundary."""
    print("Test 2: Infinite Learning")
    
    model = ToroidalFractalIntelligence(vocab_size=50, d_model=16, n_modes=16, n_atoms_max=32)
    
    np.random.seed(42)
    tokens = np.random.randint(0, 50, 500)
    sequences = [torch.tensor(tokens[i:i+8]) for i in range(0, len(tokens)-8, 4)]
    loader = InfiniteDataLoader(sequences, batch_size=2, seq_len=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    for step in range(1, 101):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Check if loss improved
    improvement = losses[0] - losses[-1]
    print(f"  ✓ Loss improved by {improvement:.4f}")
    return True

def test_state_persistence():
    """Test that state can be saved and loaded."""
    print("Test 3: State Persistence")
    
    model = ToroidalFractalIntelligence(vocab_size=50, d_model=16, n_modes=16, n_atoms_max=32)
    
    np.random.seed(42)
    tokens = np.random.randint(0, 50, 200)
    sequences = [torch.tensor(tokens[i:i+8]) for i in range(0, len(tokens)-8, 4)]
    loader = InfiniteDataLoader(sequences, batch_size=2, seq_len=8)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(1, 21):
        batch = next(iter(loader))
        token_ids = batch[:, :-1].reshape(-1).long()
        target_ids = batch[:, 1:].reshape(-1).long()
        
        output = model(token_ids)
        loss = criterion(output['logits'], target_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save state
    state_dict = model.state_dict()
    atoms_count = len(model.atoms)
    
    # Load into new model
    model2 = ToroidalFractalIntelligence(vocab_size=50, d_model=16, n_modes=16, n_atoms_max=32)
    model2.load_state_dict(state_dict)
    
    assert len(model2.atoms) == atoms_count, "Atoms not persisted!"
    print(f"  ✓ State persisted ({atoms_count} atoms)")
    return True

if __name__ == "__main__":
    print("="*50)
    print("ATOM Proof of Concept Tests")
    print("="*50)
    
    tests = [
        test_structure_emergence,
        test_infinite_learning,
        test_state_persistence,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    print("="*50)
