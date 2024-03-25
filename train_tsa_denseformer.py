import os
import datetime as dt
import torch
import tiktoken
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from tsa_denseformer.model import ModelConfig, TSADenseformer
from utils import count_parameters, tokenize_dataset

load_dotenv()

# Training Run Config
model_name = "tsa_denseformer"
run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
epochs = 1
accumulation_steps = 16
batch_size = 8
lr = 3e-4
min_lr = 3e-5
eval_steps = 250
training_steps = 10000
weight_decay = 0.01
lr_decay_steps = training_steps // accumulation_steps
dataset_name = "roneneldan/TinyStories"
wandb_project = "tsa-denseformer-experiments"
device = "cuda"
dtype = "bfloat16"

model_args = ModelConfig(
    dim=512,
    n_layers=32,
    n_heads=32,
    vocab_size=50257,
    hidden_dim=2048,
    norm_eps=1e-5,
    max_batch_size=8,
    max_seq_len=512
)

# Setup
os.makedirs(f"runs/{model_name}/{run_id}", exist_ok=True)

# Create model
model = TSADenseformer(model_args)
total_params = count_parameters(model)
print(f"Total Parameters: {total_params}")
model.to(device)

# Create tokenizer
tokenizer = tiktoken.get_encoding("r50k_base")

# Load dataset
print("loading dataset...")
dataset = load_dataset(dataset_name)
train_batches = tokenize_dataset(
    dataset["train"], tokenizer, model_args.max_seq_len, batch_size, 10000)
eval_batches = tokenize_dataset(
    dataset["validation"], tokenizer, model_args.max_seq_len, batch_size, 50)

# Initialize wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(project=wandb_project, name=model_name + "--" + run_id)
print("wandb run initialized")

# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, min_lr)
print("optimizer and scheduler initialized")

# Training loop
global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch in train_batches:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        attn_masks = batch["attn_masks"].to(device)

        _, loss = model(inputs, targets=targets, attn_mask=attn_masks)

        del inputs, targets, attn_masks

        loss = loss / accumulation_steps
        loss.backward()
        epoch_loss += loss.item()

        if (global_step + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            print(
                f"Step: {global_step + 1}, Loss: {loss.item() * accumulation_steps}")
            try:
                run.log({"loss": loss.item()})
            except:
                print(f"Failed to push {loss.item()} to wandb")

        del loss

        if (global_step + 1) % eval_steps == 0:
            model.eval()
            eval_run_loss = 0.0
            for i, eval_batch in enumerate(eval_batches):
                eval_inputs = eval_batch["inputs"].to(device)
                eval_targets = eval_batch["targets"].to(device)
                eval_attn_masks = eval_batch["attn_masks"].to(device)

                _, eval_loss = model(
                    eval_inputs, targets=eval_targets, attn_mask=eval_attn_masks)
                eval_loss += eval_loss.item()
                del eval_targets, eval_attn_masks, eval_loss

                if i == 0:
                    gen_outputs = model.generate(
                        eval_inputs[0][:10].unsqueeze(0), max_new_tokens=20)
                    print(
                        f"Generated: {tokenizer.decode(gen_outputs[0].tolist())}")
                    del eval_inputs, gen_outputs

            eval_run_loss /= len(eval_batches)
            print(f"Step: {global_step + 1}, Eval Loss: {eval_run_loss}")
            try:
                run.log({"eval_loss": eval_run_loss})
            except:
                print(f"Failed to push {eval_run_loss} to wandb")
            model.train()

        torch.cuda.empty_cache()

        global_step += 1

        if global_step >= training_steps:
            break

    epoch_loss /= len(train_batches)
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

    epoch += 1

# Save model
torch.save(model.state_dict(), f"runs/{model_name}/{run_id}/model.pt")
