"""Custom pooling modules for SentenceTransformer protein models."""

from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.models.Module import Module


class LinearAttentionPooling(Module):
    """Pool token embeddings with a learned linear attention score.

    Args:
        word_embedding_dimension: Dimensionality of token embeddings.
    """

    config_keys = ["word_embedding_dimension"]

    def __init__(self, word_embedding_dimension: int) -> None:
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.attention = nn.Linear(word_embedding_dimension, 1, bias=False)
        nn.init.zeros_(self.attention.weight)

    def forward(
        self,
        features: dict[str, Tensor | Any],
        **kwargs,
    ) -> dict[str, Tensor | Any]:
        """Compute a softmax-weighted sum over non-padding tokens."""
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        scores = self.attention(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)

        features["sentence_embedding"] = pooled
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Return the sentence embedding dimension."""
        return self.word_embedding_dimension

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ) -> None:
        """Save configuration and learnable weights."""
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        """Load a saved pooling module."""
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = cls(**config)
        return cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )


class ContextualAttentionPooling(Module):
    """Pool token embeddings with a contextual two-layer MLP scorer.

    Args:
        word_embedding_dimension: Dimensionality of token embeddings.
        activation: Nonlinearity used between MLP layers.

    Raises:
        ValueError: If the requested activation is unsupported.
    """

    _ACTIVATIONS: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    config_keys = ["word_embedding_dimension", "activation"]

    def __init__(
        self,
        word_embedding_dimension: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.activation = activation
        activation_cls = self._ACTIVATIONS.get(activation)
        if activation_cls is None:
            raise ValueError(
                f"Unsupported activation '{activation}'. Expected one of "
                f"{sorted(self._ACTIVATIONS)}."
            )
        hidden_dimension = max(1, word_embedding_dimension // 2)
        self.attention_net = nn.Sequential(
            nn.Linear(word_embedding_dimension, hidden_dimension),
            activation_cls(),
            nn.Linear(hidden_dimension, 1, bias=False),
        )
        nn.init.zeros_(self.attention_net[-1].weight)

    def forward(
        self,
        features: dict[str, Tensor | Any],
        **kwargs,
    ) -> dict[str, Tensor | Any]:
        """Compute a contextual attention-weighted sum over tokens."""
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        scores = self.attention_net(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)

        features["sentence_embedding"] = pooled
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Return the sentence embedding dimension."""
        return self.word_embedding_dimension

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ) -> None:
        """Save configuration and learnable weights."""
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        """Load a saved pooling module."""
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = cls(**config)
        return cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )


class GeneralizedMeanPooling(Module):
    """Pool token embeddings with a learnable generalized mean.

    Args:
        word_embedding_dimension: Dimensionality of token embeddings.
        p: Initial generalized-mean exponent.
        eps: Numerical stability floor.
    """

    config_keys = ["word_embedding_dimension", "p", "eps"]

    def __init__(
        self,
        word_embedding_dimension: int,
        p: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(
        self,
        features: dict[str, Tensor | Any],
        **kwargs,
    ) -> dict[str, Tensor | Any]:
        """Compute masked GeM pooling over the token axis."""
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        p_value = self.p.clamp(min=1.0)

        # Apply signed generalized mean to avoid dropping negative embeddings
        x_p = token_embeddings.sign() * token_embeddings.abs().clamp(min=self.eps).pow(
            p_value
        )
        mean_p = (x_p * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = mean_p.sign() * mean_p.abs().clamp(min=self.eps).pow(1.0 / p_value)

        features["sentence_embedding"] = pooled
        return features

    def get_sentence_embedding_dimension(self) -> int:
        """Return the sentence embedding dimension."""
        return self.word_embedding_dimension

    def get_config_dict(self) -> dict[str, float | int]:
        """Return serializable configuration for this module."""
        return {
            "word_embedding_dimension": self.word_embedding_dimension,
            "p": float(self.p.detach().cpu().item()),
            "eps": self.eps,
        }

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ) -> None:
        """Save configuration and learnable weights."""
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        """Load a saved pooling module."""
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = cls(**config)
        return cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )
