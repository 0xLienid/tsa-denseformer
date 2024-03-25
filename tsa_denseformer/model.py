import math
import torch
from dataclasses import dataclass
from typing import Optional
from torch import nn
from .denseformers.modules import DWAModules
import torch.nn.functional as F


@dataclass
class ModelConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    hidden_dim: int = 4 * dim
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class PseudoEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        embedding_weights = torch.zeros(vocab_size, dim)
        for i in range(vocab_size):
            embedding_weights[i, 0] = i

        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AttentionHead(nn.Module):
    def __init__(self, head_size: int, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.dim, head_size, bias=False)
        self.query = nn.Linear(config.dim, head_size, bias=False)
        self.value = nn.Linear(config.dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len)))

    def forward(self, x, pos_emb):
        B, T, E = x.shape
        k = self.key(pos_emb)
        q = self.query(x)

        scores = q @ k.transpose(-2, -1) * E**-0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)

        v = self.value(x)
        out = scores @ v

        return out


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_size = config.dim // config.n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(self.head_size, config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.dim, config.dim)

    def forward(self, x, pos_emb):
        out = torch.cat([head(x, pos_emb=pos_emb)
                        for head in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim, config.dim)
        self.w3 = nn.Linear(config.dim, config.hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TSATransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.lc_weight = nn.Parameter(torch.randn(1))
        self.attention = CausalAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.norm2 = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, x_token, pos_emb):
        normalized_lc_weight = torch.sigmoid(self.lc_weight)
        x_attention = normalized_lc_weight * x + \
            (1 - normalized_lc_weight) * x_token
        x = x + self.attention(self.norm1(x_attention), pos_emb=pos_emb)
        x = x + self.feed_forward(self.norm2(x))
        return x


class TSADenseformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.dim)
        self.pseudo_embedding = PseudoEmbedding(config.vocab_size, config.dim)

        self.dwa_modules = DWAModules(config.n_layers, 1, 3)

        self.blocks = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.blocks.append(TSATransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = tokens.shape

        x = self.tok_embeddings(tokens)
        pos_emb = self.pos_embeddings(
            torch.arange(T, dtype=torch.long, device="cuda"))
        x_token_space = self.pseudo_embedding(tokens)

        self.dwa_modules.init_accumulators(x)

        for i, block in enumerate(self.blocks):
            x = block(x, x_token=x_token_space, pos_emb=pos_emb)
            x = self.dwa_modules(x, block_idx=i)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
