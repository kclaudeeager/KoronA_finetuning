import os
import torch
os.environ["DISABLE_SLIDING_WINDOW_ATTENTION"] = "1"
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import your existing functions and classes
from KornA import KronABres
import logging



class ModelWithKronABres(nn.Module):
    def __init__(self, base_model, hidden_dim, a1=48, a2=32, scale=1.0, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.kronabres = KronABres(hidden_dim, a1, a2, scale, dropout)
        self.input_proj = nn.Linear(base_model.config.vocab_size, hidden_dim) if base_model.config.vocab_size != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, base_model.config.vocab_size) if hidden_dim != base_model.config.vocab_size else nn.Identity()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(-1, vocab_size)
        projected_logits = self.input_proj(flat_logits)
        kronabres_input = projected_logits.view(batch_size, seq_len, -1)
        adapted_logits = self.kronabres(kronabres_input)
        final_logits = self.output_proj(adapted_logits.view(-1, adapted_logits.size(-1)))
        final_logits = final_logits.view(batch_size, seq_len, vocab_size)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": final_logits} if return_dict else (loss, final_logits)



def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Rank {rank}] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()
# Custom data collator
def custom_data_collator(features):
    batch = {}
    
    # Initialize with first example's keys
    for key in features[0].keys():
        # Ensure all elements are tensors
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
    
    return batch

def tokenize_data(dataset, tokenizer):
    system_msg = (
        "You are a mathematical reasoning assistant. Follow these guidelines to solve problems:\n"
        "1. **Accuracy**: Perform all mathematical operations correctly.\n"
        "2. **Clarity**: Understand the problem statement and explain your reasoning step by step.\n"
        "3. **Concepts**: Apply mathematical concepts appropriately.\n"
        "4. **Units**: Ensure correct unit conversions and usage.\n"
        "5. **Logic**: Use logical reasoning to justify each step.\n"
        "6. **Final Answer**: Provide a clear and relevant final answer, formatted as \\boxed{answer}.\n"
        "Solve each problem systematically and ensure your explanation is easy to follow."
    )

    def tokenize_function(examples):
        # Prepend the system message to the input text
        examples["input_text"] = [
            f"{system_msg}\n\n{input_text}" for input_text in examples["input_text"]
        ]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples["target_text"],
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding token id in labels with -100 to ignore in loss calculation
        for i, label in enumerate(model_inputs["labels"]):
            model_inputs["labels"][i] = [
                -100 if token == tokenizer.pad_token_id else token for token in label
            ]
        
        return model_inputs
    
    # Apply batched tokenization and remove original columns
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text"]  # Ensure string fields are removed
    )
    
    # Convert to PyTorch tensors format
    tokenized_dataset.set_format(type="torch")
    
    return tokenized_dataset



def load_and_preprocess_dataset(dataset_name, sample_size=None):
    """
    Load and preprocess datasets for fine-tuning.

    Args:
        dataset_name (str): Name of the dataset (e.g., "gsm8k", "meta-math/MetaMathQA").
        sample_size (int, optional): Number of samples to load for training (only applicable for certain datasets).

    Returns:
        Tuple[Dataset, Dataset]: Preprocessed training and validation datasets.
    """
    def preprocess_data(example):
        # Format the input as a prompt-response pair
        if dataset_name == "gsm8k":
            query = f"Question: {example['question']}\nAnswer:"
            response = example["answer"]
        elif dataset_name == "meta-math/MetaMathQA":
            query = example["query"]
            response = example["response"]
        else:
            query = f"Question: {example['input_text']}\nAnswer:"
            response = example["target_text"]
        
        return {"input_text": query, "target_text": response}

    # Handle meta-math/MetaMathQA dataset
    if dataset_name == "meta-math/MetaMathQA":
        if sample_size is None:
            sample_size = 50000  # Default sample size for training
        
        # Load train split with specified sample size
        train_dataset = load_dataset(
            dataset_name, 
            split=f"train[:{sample_size}]", 
            trust_remote_code=True
        )
        
        # Load validation split
        valid_dataset = load_dataset(
            dataset_name, 
            split=f"train[{sample_size}:{sample_size+1000}]", 
            trust_remote_code=True
        )
        
        # Preprocess both datasets
        train_dataset = train_dataset.map(preprocess_data)
        valid_dataset = valid_dataset.map(preprocess_data)
    
    # Handle GSM8K or other datasets
    else:
        # Load full dataset
        dataset = load_dataset(dataset_name, "main", trust_remote_code=True)
        
        # For GSM8K, use train split and create validation split
        train_dataset = dataset["train"].map(preprocess_data)
        valid_dataset = dataset["test"].map(preprocess_data)
    
    return train_dataset, valid_dataset


def distributed_fine_tune(rank, world_size, model_name, output_path):
    setup_logging(rank)
    setup(rank, world_size)
    logging.info(f"Starting process on rank {rank}")
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in base_model.parameters():
            param.requires_grad = False

        hidden_size = base_model.config.hidden_size
        a1, a2 = 48, 32
        assert a1 * a2 == hidden_size, f"a1 * a2 must equal hidden_size: {a1} * {a2} != {hidden_size}"

        model = ModelWithKronABres(base_model=base_model, hidden_dim=hidden_size, a1=a1, a2=a2, scale=1.0, dropout=0.1)
        model = model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])

        meta_train_data, meta_valid_data = load_and_preprocess_dataset("meta-math/MetaMathQA", sample_size=20000 // world_size)
        gsm_train_data, gsm_valid_data = load_and_preprocess_dataset("gsm8k")

        meta_train_tokenized = tokenize_data(meta_train_data, tokenizer)
        gsm_train_tokenized = tokenize_data(gsm_train_data, tokenizer)

        meta_train_sampler = DistributedSampler(meta_train_tokenized, num_replicas=world_size, rank=rank, shuffle=True)
        gsm_train_sampler = DistributedSampler(gsm_train_tokenized, num_replicas=world_size, rank=rank, shuffle=True)

        batch_size_per_gpu = 4
        meta_train_dataloader = DataLoader(meta_train_tokenized, batch_size=batch_size_per_gpu, sampler=meta_train_sampler, collate_fn=custom_data_collator, pin_memory=True)
        gsm_train_dataloader = DataLoader(gsm_train_tokenized, batch_size=batch_size_per_gpu, sampler=gsm_train_sampler, collate_fn=custom_data_collator, pin_memory=True)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=0.01)
        num_training_steps = len(meta_train_dataloader) * 3 + len(gsm_train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
        scaler = torch.amp.GradScaler('cuda') 

        def train_epoch(dataloader, num_epochs):
            for epoch in range(num_epochs):
                dataloader.sampler.set_epoch(epoch)
                model.train()
                total_loss = 0
                for step, batch in enumerate(dataloader):
                    batch = {k: v.to(rank) for k, v in batch.items()}
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                        loss = outputs["loss"]
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                dist.all_reduce(torch.tensor(avg_loss, device=rank), op=dist.ReduceOp.SUM)
                if rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        train_epoch(meta_train_dataloader, num_epochs=3)
        train_epoch(gsm_train_dataloader, num_epochs=1)

        if rank == 0:
            os.makedirs(output_path, exist_ok=True)
            model.module.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"Model saved to {output_path}")

    except Exception as e:
        print(f"Error on rank {rank}: {e}")
    finally:
        cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_path = "krona_finetuned_qwen_distributed"
    mp.spawn(distributed_fine_tune, args=(world_size, model_name, output_path), nprocs=world_size)


if __name__ == '__main__':
    main()