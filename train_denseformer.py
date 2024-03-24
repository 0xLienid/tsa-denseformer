import os
import datetime as dt
import torch
import tiktoken
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
from tsa_denseformer.model import ModelConfig, TSADenseformer
from utils import count_parameters

# Training Run Config
model_name = "tsa_denseformer"
run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
epochs = 1
accumulation_steps = 16
batch_size = 32
lr = 3e-4
min_lr = 3e-5
eval_steps = 250
training_steps = 10000
weight_decay = 0.01
lr_decay_steps = training_steps
dataset_name = "roneneldan/TinyStories"
wandb_project = "tsa-denseformer-experiments"
device = "cuda"
dtype = "bfloat16"

model_args = ModelConfig(
    dim=512,
    n_layers=16,
    n_heads=16,
    vocab_size=50000,
    hidden_dim=2048,
    norm_eps=1e-5,
    max_batch_size=16,
    max_seq_len=1024
)

# Setup
os.makedirs(f"runs/{model_name}/{run_id}", exist_ok=True)

# Create model
model = TSADenseformer(model_args)
total_params = count_parameters(model)
print(f"Total Parameters: {total_params}")
model.to(device)
print("compiling model...")
model = torch.compile(model)

# Create tokenizer
tokenizer = tiktoken.get_encoding("r50k_base")

# Load dataset
print("loading dataset...")
dataset = load_dataset(dataset_name)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length"))
train_dataloader = DataLoader(
    tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"], batch_size=batch_size, shuffle=False)


# Initialize wandb
wandb.init(project=wandb_project, name=run_id)

# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, min_lr)

# Training loop
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for batch in train_dataloader:
        inputs = batch["input_ids"].to(device)
        targets = inputs.clone()

        outputs, loss = model(inputs, targets=targets)

        loss = loss / accumulation_steps
        loss.backward()

        if (global_step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Step: {global_step + 1}, Loss: {loss.item()}")
            wandb.log({"loss": loss.item()})

        if (global_step + 1) % eval_steps == 0:
            model.eval()
            eval_loss = 0.0
            for eval_batch in eval_dataloader:
                eval_inputs = eval_batch["input_ids"].to(device)
                eval_targets = eval_inputs.clone()

                eval_outputs, eval_loss = model(
                    eval_inputs, targets=eval_targets)
                eval_loss += eval_loss.item()

            eval_loss /= len(eval_dataloader)
            print(f"Step: {global_step + 1}, Eval Loss: {eval_loss}")
            wandb.log({"eval_loss": eval_loss})
            model.train()

        torch.cuda.empty_cache()

        global_step += 1
        epoch_loss += loss.item()

        if global_step >= training_steps:
            break

    epoch_loss /= len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

    epoch += 1

# Save model
torch.save(model.state_dict(), f"runs/{model_name}/{run_id}/model.pt")
