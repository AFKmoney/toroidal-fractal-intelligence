"""
Tokenizer wrapper for toroidal fractal intelligence.

Uses HuggingFace transformers for standard tokenization.
"""

from __future__ import annotations

from transformers import AutoTokenizer


class ToroidalTokenizer:
    """
    Tokenizer wrapper for toroidal models.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        text: str,
        return_tensor: bool = True,
        device: str = "cpu",
    ) -> "torch.Tensor | list[int]":
        """
        Encode text to token IDs.

        Parameters
        ----------
        text : str
        return_tensor : bool
        device : str

        Returns
        -------
        token_ids : torch.Tensor or list[int]
        """
        import torch
        ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        if return_tensor:
            return ids.to(device)
        return ids.tolist()

    def decode(
        self,
        token_ids: "torch.Tensor | list[int]",
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Parameters
        ----------
        token_ids : torch.Tensor or list[int]
        skip_special_tokens : bool

        Returns
        -------
        text : str
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: list[str],
        padding: bool = True,
        return_tensor: bool = True,
        device: str = "cpu",
    ) -> "torch.Tensor | list[list[int]]":
        """
        Batch encode texts.

        Parameters
        ----------
        texts : list[str]
        padding : bool
        return_tensor : bool
        device : str

        Returns
        -------
        token_ids : torch.Tensor or list[list[int]]
        """
        import torch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if return_tensor:
            return encodings.input_ids.to(device)
        return encodings.input_ids.tolist()
