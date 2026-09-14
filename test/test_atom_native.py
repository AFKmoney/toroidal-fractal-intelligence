"""Tests for the atom-native input contract and adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.atom_native import AtomNativeModel
from src.io.atomizer import Atomizer


class AtomizerTests(unittest.TestCase):
    def test_round_trip_is_exact_for_utf8(self) -> None:
        text = "ATOM: déjà vu — 42\nfluide."
        atomizer = Atomizer(max_span_bytes=16)
        packets = atomizer.encode(text)
        reconstructed = b"".join(packet.payload for packet in packets).decode("utf-8")
        self.assertEqual(reconstructed, text)
        self.assertGreater(len(packets), 1)
        self.assertTrue(all(packet.features.shape == (320,) for packet in packets))

    def test_same_surface_span_can_depend_on_context(self) -> None:
        first = Atomizer(max_span_bytes=16).encode("a world ")
        second = Atomizer(max_span_bytes=16).encode("b world ")
        first_world = next(packet for packet in first if packet.payload == b"world ")
        second_world = next(packet for packet in second if packet.payload == b"world ")
        self.assertFalse(torch.allclose(first_world.features, second_world.features))

    def test_chunked_stream_is_equivalent_and_lossless(self) -> None:
        text = "alpha beta — gamma\n delta"
        one_shot = Atomizer(max_span_bytes=8).encode(text)
        chunked = list(Atomizer(max_span_bytes=8).stream(["alpha ", "beta ", "— ", "gamma\n", " delta"]))
        self.assertEqual(b"".join(packet.payload for packet in one_shot), text.encode("utf-8"))
        self.assertEqual(b"".join(packet.payload for packet in chunked), text.encode("utf-8"))

    def test_state_restore_preserves_next_packet_features(self) -> None:
        source = Atomizer(max_span_bytes=16)
        source.encode("prefix ")
        saved = source.state_dict()
        expected = source.packet_from_payload(b"next ")

        restored = Atomizer(max_span_bytes=16)
        restored.load_state_dict(saved)
        actual = restored.packet_from_payload(b"next ")
        torch.testing.assert_close(expected.features, actual.features)
        self.assertEqual(expected.start, actual.start)
        self.assertEqual(expected.end, actual.end)

    def test_pending_buffer_is_checkpointable(self) -> None:
        source = Atomizer(max_span_bytes=16)
        source.encode_bytes(b"partial", reset=True, flush=False)
        saved = source.state_dict()
        expected = source.encode_bytes(b" end ", reset=False, flush=True)

        restored = Atomizer(max_span_bytes=16)
        restored.load_state_dict(saved)
        actual = restored.encode_bytes(b" end ", reset=False, flush=True)
        self.assertEqual([packet.payload for packet in expected], [packet.payload for packet in actual])


class AtomNativeModelTests(unittest.TestCase):
    def test_forward_loss_backward_and_checkpoint_reload(self) -> None:
        atomizer = Atomizer(max_span_bytes=12)
        packets = atomizer.encode("alpha beta gamma delta epsilon.")
        self.assertGreaterEqual(len(packets), 3)

        model = AtomNativeModel(
            d_model=4,
            n_modes=4,
            n_atoms_max=32,
            max_payload_bytes=12,
            atomizer=Atomizer(max_span_bytes=12),
        )
        optimizer = torch.optim.AdamW(model.trainable_parameters, lr=1e-3)
        loss, info = model.transition_loss(packets[0], packets[1])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(info["surface"]["byte_logits"]).all())
        loss.backward()
        optimizer.step()
        self.assertGreater(len(model.core.atoms), 0)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atom_native.pt"
            model.save(path)
            reloaded = AtomNativeModel(
                d_model=4,
                n_modes=4,
                n_atoms_max=32,
                max_payload_bytes=12,
                atomizer=Atomizer(max_span_bytes=12),
            )
            reloaded.load(path)
            self.assertEqual(len(reloaded.core.atoms), len(model.core.atoms))
            self.assertEqual(reloaded.core.state.n_modes, 4)
            self.assertTrue(torch.isfinite(reloaded.core.state.alpha).all())


if __name__ == "__main__":
    unittest.main()
