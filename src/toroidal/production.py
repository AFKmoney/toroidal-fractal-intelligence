"""Production of outputs from current, persistent and abstract toroidal state."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductionHead(nn.Module):
    def __init__(self, d_model=256, vocab_size=32000):
        super().__init__()
        self.d_model, self.vocab_size = d_model, vocab_size
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, vocab_size))
        self.confidence_head = nn.Linear(d_model, 1)

    def produce(self, alpha, persistent_state=None, abstractions=None):
        state = alpha.mean(dim=0) if alpha.dim() > 1 else alpha
        if persistent_state is not None:
            state = state + (persistent_state.mean(dim=0) if persistent_state.dim() > 1 else persistent_state)
        if abstractions:
            reps = [a["pattern"].to(state.device, state.dtype).reshape(-1).mean(dim=0) for a in abstractions]
            if reps:
                # Keep abstraction influence bounded so transient state remains primary.
                state = state + 0.1 * torch.stack(reps).mean()
        logits = self.decoder(state).unsqueeze(0)
        confidence = torch.sigmoid(self.confidence_head(state)).reshape(1)
        return logits, confidence

    def sample(self, alpha, persistent_state=None, abstractions=None, temperature=1.0, top_k=50):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits, confidence = self.produce(alpha, persistent_state, abstractions)
        logits = logits / temperature
        vocab = logits.shape[-1]
        if top_k is not None and top_k > 0 and top_k < vocab:
            values, indices = torch.topk(logits, top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(-1, indices, values)
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs.squeeze(0), 1).item()
        return token_id, logits, confidence
