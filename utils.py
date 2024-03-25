import torch
from torch.utils.data import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenize_batch(examples, tokenizer, max_seq_len):
    input_ids = tokenizer.encode_batch(examples["text"])
    input_ids = [ids[:max_seq_len] for ids in input_ids]

    targets = []
    for i, id in enumerate(input_ids):
        if len(id) < max_seq_len:
            input_ids[i] = id + [50256] * (max_seq_len - len(id))

        targets.append(input_ids[i][1:] + [50256])
        assert len(input_ids[i]) == len(targets[i])

    return torch.tensor(input_ids, dtype=torch.long, device="cpu"), torch.tensor(targets, dtype=torch.long, device="cpu")


def tokenize_dataset(dataset, tokenizer, max_seq_len, batch_size, num_batches):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batches = []
    batch_i = 0
    for batch in dataloader:
        input_ids, target_ids = tokenize_batch(batch, tokenizer, max_seq_len)
        batch_dict = {
            "inputs": input_ids,
            "targets": target_ids
        }
        batches.append(batch_dict)

        batch_i += 1
        if batch_i >= num_batches:
            break

    return batches
