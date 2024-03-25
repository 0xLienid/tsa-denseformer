import math
import torch
from dataclasses import dataclass
from typing import Optional
from torch import nn
from .denseformers.modules import DWAModules
from ..utils import precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb, repeat_kv
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


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.wq = nn.Linear(config.dim, config.n_heads *
                            self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim,
                            config.dim, bias=False)

        self.flash = hasattr(torch.nn.functional,
                             "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full(
                (1, 1, config.max_seq_len, config.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

        def forward(
            self,
            x: torch.Tensor,
            pos_emb: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ):
            bsz, seqlen, _ = x.shape

            # QKV
            xq, posk, xv = self.wq(x), self.wk(pos_emb), self.wv(x)

            # RoPE relative positional embeddings
            xq, posk = apply_rotary_emb(xq, posk, freqs_cos, freqs_sin)

            # grouped multiquery attention: expand out keys and values
            posk = repeat_kv(posk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

            # make heads into a batch dimension
            xq = xq.transpose(1, 2)
            posk = posk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # Combine the causal mask with the padding attention mask
            if attn_mask is not None:
                # Convert the padding mask to the correct format: 0 for tokens to be masked, 1 for others
                # Then scale it to -inf for masked positions
                attn_mask = attn_mask.unsqueeze(1).repeat(1, seqlen, 1)
                attn_mask = (attn_mask == 0).to(x.dtype) * float("-inf")
                combined_mask = self.mask[:, :, :seqlen,
                                          :seqlen] + attn_mask.unsqueeze(1)
            else:
                combined_mask = self.mask[:, :, :seqlen, :seqlen]

            # attention
            if self.flash:
                output = F.scaled_dot_product_attention(
                    xq, posk, xv, attn_mask=combined_mask)
            else:
                scores = (xq @ posk.transpose(2, 3)) / math.sqrt(self.head_dim)
                assert hasattr(self, "mask")
                # TODO: add additional attention mask for padding
                scores = scores + combined_mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = scores @ xv

            # restore time as batch dimension and concat heads
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(output)


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
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.norm2 = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x,
        x_token,
        pos_emb,
        freqs_cos,
        freqs_sin,
        attn_mask: Optional[torch.Tensor] = None
    ):
        normalized_lc_weight = torch.sigmoid(self.lc_weight)
        x_attention = normalized_lc_weight * x + \
            (1 - normalized_lc_weight) * x_token
        x = x + self.attention(self.norm1(x_attention), pos_emb=pos_emb,
                               freqs_cos=freqs_cos, freqs_sin=freqs_sin, attn_mask=attn_mask)
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

        # tie the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.lm_head.weight

        # precompute the RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ):
        B, T = tokens.shape

        x = self.tok_embeddings(tokens)
        pos_emb = self.pos_embeddings(
            torch.arange(T, dtype=torch.long, device="cuda"))
        x_token_space = self.pseudo_embedding(tokens)

        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        self.dwa_modules.init_accumulators(x)

        for i, block in enumerate(self.blocks):
            x = block(x, x_token=x_token_space, pos_emb=pos_emb,
                      freqs_cos=freqs_cos, freqs_sin=freqs_sin, attn_mask=attn_mask)
            x = self.dwa_modules(x, block_idx=i)

        x = self.norm(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)

            targets = targets.view(-1)
            attn_mask = attn_mask.view(-1)
            targets[attn_mask == 0] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets, ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx
