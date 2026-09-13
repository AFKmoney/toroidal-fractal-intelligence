"""
Basic test for toroidal fractal intelligence model.
"""

import torch
from toroidal_fractal_intelligence import create_model
from toroidal_fractal_intelligence.io.tokenizer import ToroidalTokenizer


def test_model_creation():
    """Test that model can be created."""
    model = create_model(d_model=64, n_modes=64, n_atoms_max=128)
    assert model is not None
    print("✅ Model creation test passed")
    return model


def test_forward_pass(model):
    """Test forward pass."""
    token_ids = torch.randint(0, model.encoder.vocab_size, (8,))
    output = model(token_ids)

    assert "logits" in output
    assert "confidence" in output
    assert "aggregates" in output
    assert "abstractions" in output

    print(f"✅ Forward pass test passed")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Atoms: {len(model.atoms)}")
    print(f"   Aggregates: {len(output['aggregates'])}")
    print(f"   Abstractions: {len(output['abstractions'])}")


def test_training_step(model):
    """Test training step."""
    import torch.nn as nn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    token_ids = torch.randint(0, model.encoder.vocab_size, (8,))
    target_ids = torch.randint(0, model.encoder.vocab_size, (8,))

    metrics = model.train_step(token_ids, target_ids, optimizer)

    assert "loss" in metrics
    assert metrics["loss"] > 0
    assert metrics["n_atoms"] >= 0

    print(f"✅ Training step test passed")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Atoms: {metrics['n_atoms']}")


def test_save_load(model, tmp_path):
    """Test save and load."""
    path = tmp_path / "test_model.pt"
    model.save(str(path))

    # Create new model and load
    model2 = create_model(d_model=64, n_modes=64, n_atoms_max=128)
    model2.load(str(path))

    assert len(model2.atoms) == len(model.atoms)
    print(f"✅ Save/load test passed")


def test_chat(model, tokenizer):
    """Test text generation."""
    response = model.generate(
        tokenizer.encode("The future of AI", return_tensor=True),
        max_length=50
    )
    text = tokenizer.decode(response)

    assert len(text) > 0
    print(f"✅ Chat test passed")
    print(f"   Response: {text[:100]}...")


def main():
    print("=" * 60)
    print("TOROIDAL FRACTAL INTELLIGENCE — TEST SUITE")
    print("=" * 60)

    # Create model
    model = test_model_creation()

    # Test forward pass
    test_forward_pass(model)

    # Test training
    test_training_step(model)

    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_save_load(model, type('obj', (object,), {'__truediv__': lambda s, x: f'{tmpdir}/{x}'})())

    # Test chat
    tokenizer = ToroidalTokenizer("gpt2")
    test_chat(model, tokenizer)

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

    # Print model summary
    print("\nModel Summary:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Atoms: {len(model.atoms)}")
    print(f"  Modes: {model.state.n_modes}")
    print(f"  d_model: {model.encoder.d_model}")
    print(f"  Shared params: {model.get_shared_params_summary()}")


if __name__ == "__main__":
    main()
