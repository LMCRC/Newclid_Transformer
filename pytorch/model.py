import torch
from torch import nn
import math


class T5RelativeEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_buckets = config["t5_num_buckets"]
        self.max_distance = config["t5_max_distance"]
        self.num_heads = config["num_heads"]
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position):
        relative_buckets = 0
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

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def forward(self, query_length, key_length):
        """Compute binned relative position bias"""
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


class AGLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_ele = config["embedding_dim"]
        self.epsilon = config.get("layernorm_epsilon", 1e-6)
        self.register_parameter(name='weight', param=nn.Parameter(torch.ones(self.num_ele, )))

    def forward(self, xs):
        xln = xs.to(torch.float32)
        var = torch.mean(torch.square(xln), dim=-1, keepdims=True)
        mul = torch.rsqrt(var + self.epsilon)
        ys = xs * mul
        ys = ys * self.weight
        return ys.to(xs.dtype)


class QKVLayer(nn.Module):
    """Generate keys, values, and queries for attention."""

    def __init__(self, config):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.embedding_dim // self.num_heads

        self.normalize_keys = config.get("normalize_keys", True)
        self.pre_attn_dropout = None
        if (dropout_rate := config.get("pre_attn_dropout", 0.0)) > 0.0:
            self.pre_attn_droput = nn.Dropout(dropout_rate)

        self.queries_layer = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        self.keys_layer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.values_layer = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )

        self.pre_attn_layernorm = AGLayerNorm(config)

    def _normalize_kq(self, kq):
        epsilon = 1.0e-6
        kq_sum_sqr = torch.sum(torch.square(kq), axis=-1, keepdims=True)
        norm_kq = kq * torch.rsqrt(kq_sum_sqr + epsilon)
        return norm_kq

    def forward(self, xs):
        batch_size, seq_len, _ = xs.shape

        xs = self.pre_attn_layernorm(xs)

        if self.pre_attn_dropout != None:
            xs = self.pre_attn_dropout(xs)

        queries = self.queries_layer(xs)
        keys = self.keys_layer(xs)
        values = self.values_layer(xs)

        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.reshape(shape)
        keys = keys.reshape(shape)
        values = values.reshape(shape)

        if self.normalize_keys:
            queries = self._normalize_kq(queries)
            keys = self._normalize_kq(keys)

        return queries, keys, values


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        cur_dim = config["embedding_dim"]

        modules = []
        for i in range(0, config["mlp_num_layers"] - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(cur_dim, config["mlp_hidden_dim"], bias=False), nn.ReLU()
                )
            )
            cur_dim = config["mlp_hidden_dim"]

        modules.append(nn.Linear(cur_dim, config["embedding_dim"], bias=False))
        self.layers = nn.ModuleList(modules)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)

        return xs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.embedding_dim // self.num_heads

        self.relative_positions = T5RelativeEmbeddings(config)

        self.gate_type = "residual"
        self.single_gate = False
        self.skip_ffn = False

        self.attn_dropout = None
        if (dropout_rate := config.get("attn_dropout", 0.0)) > 0.0:
            self.attn_droput = nn.Dropout(dropout_rate)

        self.post_attn_dropout = None
        if (dropout_rate := config.get("post_attn_dropout", 0.0)) > 0.0:
            self.post_attn_droput = nn.Dropout(dropout_rate)

        self.pre_ffn_dropout = None
        if (dropout_rate := config.get("pre_ffn_dropout", 0.0)) > 0.0:
            self.pre_ffn_droput = nn.Dropout(dropout_rate)

        self.post_ffn_dropout = None
        if (dropout_rate := config.get("post_ffn_dropout", 0.0)) > 0.0:
            self.post_ffn_droput = nn.Dropout(dropout_rate)

        self.qkv = QKVLayer(config)

        self.normalize_keys = config.get("normalize_keys", True)
        if self.normalize_keys:
            self.register_parameter(
                name="attention_scale_factors",
                param=nn.Parameter(
                    torch.ones(
                        self.num_heads,
                    )
                ),
            )

        self.post_attn_mlp = nn.Linear(
            self.embedding_dim, self.embedding_dim, bias=False
        )
        self.ffn = MLP(config)
        self.pre_ffn_layernorm = AGLayerNorm(config)

    def _get_causal_mask(self, num_qs, num_ks):
        qidx = torch.arange(0, num_qs).reshape(num_qs, 1)
        kidx = torch.arange(0, num_ks).reshape(1, num_ks)
        mask = (kidx - qidx) < 0
        return mask

    def forward(self, xs):
        batch_size, seq_length, _ = xs.shape

        queries, keys, values = self.qkv(xs)
        rel_position_bias = self.relative_positions(seq_length, seq_length)
        causal_mask = self._get_causal_mask(seq_length, seq_length).to(queries.device).tile((self.num_heads, 1, 1))

        attn = torch.einsum("...qhd,...khd->...hqk", queries, keys)
        attn = attn + rel_position_bias
        if self.normalize_keys:
            attn *= self.attention_scale_factors.reshape(1, self.num_heads, 1, 1)
        attn = torch.where(causal_mask, attn, -1_000_000.0)
        attn = attn.softmax(dim=-1)

        ys_hidden = torch.einsum("...hqk,...khd->...qhd", attn, values)
        ys_hidden = ys_hidden.reshape(
            (batch_size, seq_length, self.num_heads * self.head_dim)
        )
        ys_hidden = self.post_attn_mlp(ys_hidden) + xs

        ys = self.pre_ffn_layernorm(ys_hidden)
        ys = self.ffn(ys) + ys_hidden

        return ys


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["num_layers"])])
        self.final_layernorm = AGLayerNorm(config)

    def forward(self, xs):
        ys = self.embedding(xs)
        for layer in self.layers:
            ys = layer(ys)

        ys = self.final_layernorm(ys)
        logits = torch.nn.functional.linear(ys, self.embedding.weight)
        logits /= math.sqrt(logits.shape[-1])
        return logits
