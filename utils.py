import torch
from torch.utils.data import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenize_batch(examples, tokenizer, max_seq_len):
    input_ids = tokenizer.encode_batch(examples["text"])
    input_ids = [ids[:max_seq_len] for ids in input_ids]
    input_ids = input_ids + [50256] * (max_seq_len - len(input_ids))
    targets = [ids[1:] + [50256] for ids in input_ids]

    return torch.Tensor(input_ids, dtype=torch.long), torch.Tensor(targets, dtype=torch.long)


def tokenize_dataset(dataset, tokenizer, max_seq_len, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batches = []

    for batch in dataloader:
        input_ids, target_ids = tokenize_batch(batch, tokenizer, max_seq_len)
        batch_dict = {
            "inputs": input_ids,
            "targets": target_ids
        }
        batches.append(batch_dict)

    return batches
