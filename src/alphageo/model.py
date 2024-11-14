from __future__ import annotations
from typing import Any, Callable
import torch
from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Parameter,
    Dropout,
    Linear,
    Sequential,
    ReLU,
)

import math


class T5RelativeEmbeddings(Module):
    """
    Adapted from
        https://github.com/google-research/meliad/blob/main/transformer/position_t5.py
    which is probably itself adapted from
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py
    or from
        https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/t5/modeling_t5.py

    Config attributes:
        t5_num_buckets: Number of buckets to bucket distances between key and query positions into.
        t5_max_distance: Maximum distance before everything is lumped into the last distance bucket.
        num_heads: Number of heads in the attention layer. Each head will get a different relative position weighting.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore
        self.num_buckets: int = config["t5_num_buckets"]
        self.max_distance = config["t5_max_distance"]
        self.num_heads: int = config["num_heads"]
        self.relative_attention_bias = Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: Tensor):
        """
        Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to position.
        We use smaller buckets for small absolute relative_position and larger buckets for larger absolute relative_positions.
        All relative positions >= self.max_distance  map to the same bucket.
        All relative positions <= -self.max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on.

        Args:
            relative_position: An int Tensor.

        Returns:
            A Tensor with the same shape as relative_position, containing int values in the range [0, num_buckets).
            0...max_exact-1...num_buckets-1
        """

        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, self.num_buckets - 1),
        )

        relative_buckets = torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def forward(self, query_length: int, key_length: int):
        """
        Produce relative position embedding attention biases.

        Args:
            query_length: Number of queries.
            key_length: Number of keys.

        Returns:
            output: `(1, self.num_heads, query_length, key_length)` positional encodings for attention bias.
        """
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position
        )  # shape (query_length, key_length)
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values


class AGLayerNorm(Module):
    """
    pyTorch Module implementing LayerNorm as used by orginal AG, based on https://github.com/google-research/meliad/blob/main/transformer/nn_components.py

    Config attributes:
        embedding_dim: Embedding/internal size of model layers.
        layernom_epsilon: Value added to variance, for numerical stability (default: 1e-6).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore
        self.num_ele = config["embedding_dim"]
        self.epsilon = config.get("layernorm_epsilon", 1e-6)
        param: Parameter = Parameter(torch.ones(self.num_ele))  # type: ignore

        self.register_parameter(name="weight", param=param)

    def forward(self, xs: Tensor):
        """
        Apply layernorm to input tensor.
        Regardless of input dtype, normalizations are always calculated in FP32.
        Ordinarily, we would ensure this by excluding the AGLayerNorm Module from dtype conversion. Original AG converts the weight though, so we do to, to stay compatible.

        Args:
            xs: Float Tensor.

        Returns
            Normalized Tensor of same dtype and shape as xs.
        """
        xln = xs.to(torch.float32)
        var = torch.mean(torch.square(xln), dim=-1, keepdim=True)
        mul = torch.rsqrt(var + self.epsilon)
        ys = xs * mul
        ys = ys * self.weight
        return ys.to(xs.dtype)


class QKVLayer(Module):
    """
    Generate keys, values, and queries for attention.

    Adapted from https://github.com/google-research/meliad/blob/main/transformer/transformer_base.py

    Config attributes:
        embedding_dim: Embedding/internal size of model layers.
        num_heads: Number of heads in the attention layer. Each head will get a different relative position weighting.
        normalize_keys: Whether keys (and query) values are being normalized (default: True).
        pre_attn_dropout: Pre-attention dropout value (default: 0.0 -> no dropout).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore

        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        # each head gets 1/embedding_dim internal dimensions
        self.head_dim = self.embedding_dim // self.num_heads

        self.normalize_keys = config.get("normalize_keys", True)
        self.pre_attn_dropout = None
        if (dropout_rate := config.get("pre_attn_dropout", 0.0)) > 0.0:
            self.pre_attn_droput = Dropout(dropout_rate)

        self.queries_layer = Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.keys_layer = Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.values_layer = Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.pre_attn_layernorm = AGLayerNorm(config)

    def _normalize_kq(self, kq: Tensor):
        """
        Normalizes key/query values. Regardless of input dtype, normalization is carried out in FP32.

        Args:
            kq: Tensor of shape `(batch size, sequence length, self.num_heads, self.head_dim)`.

        Returns:
            Normalized Tensor of same shape and dtype as kq.
        """
        epsilon = torch.Tensor([1.0e-6]).to(kq.dtype).to(kq.device)
        kq_sum_sqr = torch.sum(torch.square(kq), dim=-1, keepdim=True)
        norm_kq = kq * torch.rsqrt(kq_sum_sqr.float() + epsilon).to(kq.dtype)
        return norm_kq

    def forward(self, xs: Tensor):
        """
        Calculate QKV values for input sequence embeddings.

        Args:
            xs: Float Tensor of shape `(batch size, sequence length, self.embedding_dim)`.

        Returns:
            `(queries, keys, values)` tuple of Float Tensors, each of shape (batch size, sequence_length, self.num_heads, self.head_dim).
        """
        batch_size, seq_len, _ = xs.shape

        xs = self.pre_attn_layernorm(xs)

        if self.pre_attn_dropout is not None:
            xs = self.pre_attn_dropout(xs)

        queries = self.queries_layer(xs)
        keys = self.keys_layer(xs)
        values = self.values_layer(xs)

        # each head gets its own qkv values
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.reshape(shape)
        keys = keys.reshape(shape)
        values = values.reshape(shape)

        if self.normalize_keys:
            queries = self._normalize_kq(queries)
            keys = self._normalize_kq(keys)

        return queries, keys, values


class MLP(Module):
    """
    Module implementing a simple Multi-layered Perceptron.
    Used as final output FFN of each DecoderLayer.

    Config attributes:
        embedding_dim: Embedding/internal size of model layers.
        mlp_num_layers: Dumber of layers in the final FFN.
        mlp_hidden_dim: Dimensionality of MLP hidden layers.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore

        # First layer input dim is equal to model embedding dim.
        cur_dim = config["embedding_dim"]

        modules: list[Module] = []
        # Create hidden layers, 1 less than total mlp_num_layers.
        for _ in range(0, config["mlp_num_layers"] - 1):
            modules.append(
                Sequential(
                    Linear(cur_dim, config["mlp_hidden_dim"], bias=False), ReLU()
                )
            )
            cur_dim = config["mlp_hidden_dim"]

        # Final output layer, mapping back to original model embedding dim.
        modules.append(Linear(cur_dim, config["embedding_dim"], bias=False))
        self.layers = ModuleList(modules)

    def forward(self, xs: Tensor):
        """
        Apply MLP to input.

        Args:
            xs: Float Tensor of shape `(batch size, sequence length, embedding_dim)`.

        Returns:
            Float Tensor of same shape as xs.
        """
        for layer in self.layers:
            xs = layer(xs)

        return xs


class DecoderLayer(Module):
    """
    Implements a single layer of a decoder transformer.
    Standard layer, designed to be close to AG's implementation at https://github.com/google-deepmind/alphageometry/blob/main/transformer_layer.py

    Config attributes:
        embedding_dim: Embedding/internal size of model layers.
        num_heads: Number of heads in the attention layer. Each head will get a different relative position weighting.
        attn_dropout: Dropout applied to attention matrix (default: 0.0); currently unused!
        post_attn_dropout: Dropout applied after attention calculations (default: 0.0); currently unused!
        pre_ffn_dropout: Dropout before layer FFN (default: 0.0); currently unused!
        post_ffn_dropout: Dropout after layer FFN (default: 0.0); currently unused!
        normalize_keys: Whether or not keys (and queries) will be normalized (default: True).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore

        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.embedding_dim // self.num_heads

        self.relative_positions = T5RelativeEmbeddings(config)

        self.attn_dropout = None
        if (dropout_rate := config.get("attn_dropout", 0.0)) > 0.0:
            self.attn_droput = Dropout(dropout_rate)

        self.post_attn_dropout = None
        if (dropout_rate := config.get("post_attn_dropout", 0.0)) > 0.0:
            self.post_attn_droput = Dropout(dropout_rate)

        self.pre_ffn_dropout = None
        if (dropout_rate := config.get("pre_ffn_dropout", 0.0)) > 0.0:
            self.pre_ffn_droput = Dropout(dropout_rate)

        self.post_ffn_dropout = None
        if (dropout_rate := config.get("post_ffn_dropout", 0.0)) > 0.0:
            self.post_ffn_droput = Dropout(dropout_rate)

        self.qkv = QKVLayer(config)

        # if keys and queries are normalized, attention weights have to be scaled by learned factors.
        self.normalize_keys = config.get("normalize_keys", True)
        if self.normalize_keys:
            self.register_parameter(
                name="attention_scale_factors",
                param=Parameter(  # type: ignore
                    torch.ones(
                        self.num_heads,
                    )
                ),
            )

        self.post_attn_mlp = Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.ffn = MLP(config)
        self.pre_ffn_layernorm = AGLayerNorm(config)

    def _get_causal_mask(self, num_qs: int, num_ks: int):
        """
        Helper function to generate a causal mask for causal Language Modeling. Ensures we only pay attention to the past.
        Essentially a recangular Boolean matrix, with diagonal and upper triangle set to False, and lower triangle set to True.

        Args:
            num_qs: Number of query tokens.
            num_ks: Number of key tokens. (Typically, num_qs==num_ks).
        """
        qidx = torch.arange(0, num_qs).reshape(num_qs, 1)
        kidx = torch.arange(0, num_ks).reshape(1, num_ks)
        mask = (kidx - qidx) < 0
        return mask

    def forward(self, xs: Tensor):
        """
        Apply the layer (query, key, value calculation, attention, ffn) to input sequence.

        Args:
            xs: Float Tensor of shape `(batch size, sequence length, embedding_dim)`.

        Returns:
            A float Tensor of same shape as xs.
        """
        batch_size, seq_length, _ = xs.shape

        queries, keys, values = self.qkv(
            xs
        )  # (batch_size, seq_len, num_heads, head_dim)
        rel_position_bias = self.relative_positions(
            seq_length, seq_length
        )  # (1, num_heads, queries_length=seq_length, key_length=seq_length)
        causal_mask = (
            self._get_causal_mask(seq_length, seq_length)
            .to(queries.device)
            .tile((self.num_heads, 1, 1))
        )

        attn = torch.einsum(
            "...qhd,...khd->...hqk", queries, keys
        )  # (batch_size, num_heads, seq_len, seq_len)
        attn = attn + rel_position_bias
        if self.normalize_keys:
            attn *= self.attention_scale_factors.reshape(1, self.num_heads, 1, 1)
        attn = torch.where(causal_mask, attn, -1_000_000.0)  # masking the future
        attn = attn.softmax(dim=-1) * causal_mask

        ys_hidden = torch.einsum(
            "...hqk,...khd->...qhd", attn, values
        )  # (batch_size, seq_len, num_heads, head_dim)
        ys_hidden = (
            ys_hidden.reshape(  # reshape to (batch_size, seq_len, embedding_dim)
                (batch_size, seq_length, self.num_heads * self.head_dim)
            )
        )
        ys_hidden = self.post_attn_mlp(ys_hidden) + xs  # residual

        ys = self.pre_ffn_layernorm(ys_hidden)
        ys = self.ffn(ys) + ys_hidden  # residual

        return ys


class Decoder(Module):
    """
    Transformer decoder stack.

    Config attributes:
        vocab_size: Input vocabulary size.
        embedding_dim: Embedding/internal size of model layers.
        num_layers: Number of DecoderLayers.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()  # type: ignore
        self.config = config
        self.embedding = Embedding(config["vocab_size"], config["embedding_dim"])
        self.layers = ModuleList(
            [DecoderLayer(config) for _ in range(config["num_layers"])]
        )
        self.final_layernorm = AGLayerNorm(config)
        self.dtype = self.final_layernorm.weight.dtype

    def _apply(self, fn: Callable[..., Any], recurse: bool = True):
        """
        Original AG runs inference in bfloat16, but certain operations (parts of AGLayerNorm) are forced to be in fp32.
        In particular, original AG *never* downcasts the input/outpout embedding, this is *always* in fp32.

        Changing pyTorch's _apply means we can ignore calls such as `model.bfloat16()`, `model.half()` etc., while still allowing `model.cuda()` etc..
        We won't be able to ignore calls to `model.to()` for dtypes this way... please just don't call it.
        E.g., to get the model onto the GPU in half precisions, don't call `model.to(device="cuda", dtype=torch.half)`, but do `model.cuda(); model.half()`.
        """
        if recurse:
            rep = repr(fn)
            is_dtype = any(
                [
                    typ in rep for typ in ["float", "double", "half", "int", "long"]
                ]  # skip dtype conversions for embedding
            )
            if not is_dtype:
                self.embedding._apply(fn)  # type: ignore
            self.layers._apply(fn, recurse=recurse)  # type: ignore
            self.final_layernorm._apply(fn, recurse=recurse)  # type: ignore

        super()._apply(fn, recurse=False)  # type: ignore
        self.dtype = self.final_layernorm.weight.dtype
        return self

    def forward(self, xs: Tensor):
        """
        Apply full decoder stack: Embed input tokens, send through DecoderLayers with attention, project back into token embedding space.

        Args:
            xs: LongTensor of shape `(batch_size, sequence_length)`.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length)`.
        """
        ys = self.embedding(xs)
        ys = ys.to(self.dtype)
        for layer in self.layers:
            ys = layer(ys)

        ys = self.final_layernorm(ys)
        logits = torch.nn.functional.linear(ys.float(), self.embedding.weight)
        logits /= math.sqrt(logits.shape[-1])
        return logits
