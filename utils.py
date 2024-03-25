import torch
from torch.utils.data import DataLoader
from typing import Tuple


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenize_batch(examples, tokenizer, max_seq_len):
    input_ids = tokenizer.encode_batch(examples["text"])
    input_ids = [ids[:max_seq_len] for ids in input_ids]

    targets = []
    attn_masks = []
    for i, id in enumerate(input_ids):
        seq_len = len(id)

        if seq_len < max_seq_len:
            id = id + [50256] * (max_seq_len - seq_len)

        targets.append(input_ids[i][1:] + [50256])
        assert len(input_ids[i]) == len(targets[i])

        attn_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        attn_masks.append(attn_mask)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device="cpu")
    targets_tensor = torch.tensor(targets, dtype=torch.long, device="cpu")
    attn_masks_tensor = torch.tensor(
        attn_masks, dtype=torch.long, device="cpu")

    return input_ids_tensor, targets_tensor, attn_masks_tensor


def tokenize_dataset(dataset, tokenizer, max_seq_len, batch_size, num_batches):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batches = []
    batch_i = 0
    for batch in dataloader:
        input_ids, target_ids, attn_masks = tokenize_batch(
            batch, tokenizer, max_seq_len)
        batch_dict = {
            "inputs": input_ids,
            "targets": target_ids,
            "attn_masks": attn_masks
        }
        batches.append(batch_dict)

        batch_i += 1
        if batch_i >= num_batches:
            break

    return batches


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
