import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# Assuming KronA is implemented in a separate file
from KornA import KronABres


class ModelWithKronABres(nn.Module):
    def __init__(self, base_model, hidden_dim, a1=48, a2=32, scale=1.0, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.kronabres = KronABres(hidden_dim, a1, a2, scale, dropout)
        
        # Add projection layers if needed
        self.input_proj = nn.Linear(base_model.config.vocab_size, hidden_dim) if base_model.config.vocab_size != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, base_model.config.vocab_size) if hidden_dim != base_model.config.vocab_size else nn.Identity()
        
    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape logits for KronABres
        flat_logits = logits.view(-1, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
        
        # Project to KronABres input dimension
        projected_logits = self.input_proj(flat_logits)  # Shape: [batch_size * seq_len, hidden_dim]
        
        # Reshape to [batch_size, seq_len, hidden_dim]
        kronabres_input = projected_logits.view(batch_size, seq_len, -1)
        
        # Apply KronABres
        adapted_logits = self.kronabres(kronabres_input)
        
        # Reshape back to original logits dimensions
        final_logits = self.output_proj(adapted_logits.view(-1, adapted_logits.size(-1)))
        final_logits = final_logits.view(batch_size, seq_len, vocab_size)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            return (loss, final_logits) + outputs[2:]
        
        # Return without cross_attentions
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=final_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    # Add generation method for inference
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    # Add save_pretrained method
    def save_pretrained(self, path):
        # Save the base model
        self.base_model.save_pretrained(path)
        # Save KronABres parameters separately
        kronabres_state = self.kronabres.state_dict()
        torch.save(kronabres_state, f"{path}/kronabres_adapter.pt")

    @classmethod
    def from_pretrained(cls, path, base_model_name, hidden_dim, a1=48, a2=32, scale=1.0, dropout=0.1):
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(path)
        # Initialize the model
        model = cls(base_model, hidden_dim, a1, a2, scale, dropout)
        # Load KronABres parameters if available
        kronabres_path = f"{path}/kronabres_adapter.pt"
        if os.path.exists(kronabres_path):
            kronabres_state = torch.load(kronabres_path, map_location=torch.device('cpu'))
            model.kronabres.load_state_dict(kronabres_state)
        else:
            print(f"Warning: {kronabres_path} not found. KronABres adapter will be randomly initialized.")
        return model


# Custom data collator
def custom_data_collator(features):
    batch = {}
    
    # Initialize with first example's keys
    for key in features[0].keys():
        # Ensure all elements are tensors
        if isinstance(features[0][key], torch.Tensor):
            batch[key] = torch.stack([f[key] for f in features])
    
    return batch


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


# Fine-tune the model with progress bar and better error handling
def fine_tune_model(model, train_dataloader, optimizer, num_epochs=3, lr_scheduler=None):
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Create progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            try:
                # Move all tensors to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        # Print epoch results
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")


# Main script
import os  # Import os to check for existing checkpoints

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define model and checkpoint paths
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_path = "krona_finetuned_qwen"
    
    # Check if the model checkpoint already exists
    if os.path.exists(output_path):
        print(f"Checkpoint found at {output_path}. Reloading the model...")
        # Reload the model from the checkpoint
        model = ModelWithKronABres.from_pretrained(
            output_path,
            base_model_name=model_name,
            hidden_dim=1536,  # Replace with the correct hidden_dim
            a1=48,
            a2=32,
            scale=1.0,
            dropout=0.1
        )
        
        # Attempt to load the tokenizer from the checkpoint
        try:
            tokenizer = AutoTokenizer.from_pretrained(output_path)
        except OSError:
            print("Tokenizer not found in the checkpoint. Loading tokenizer from the base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print("No checkpoint found. Initializing model and tokenizer from scratch...")
        # Load model and tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure tokenizer has proper padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze base model parameters
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Ensure a1 * a2 matches hidden_size
        hidden_size = base_model.config.hidden_size
        a1 = 48
        a2 = 32
        
        assert a1 * a2 == hidden_size, f"a1 * a2 must equal hidden_size: {a1} * {a2} != {hidden_size}"
        
        # Initialize the model
        model = ModelWithKronABres(
            base_model=base_model,
            hidden_dim=hidden_size,
            a1=a1,
            a2=a2,
            scale=1.0,
            dropout=0.1
        )
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model.to(device)
    
    # Load and process datasets
    meta_train_data, meta_valid_data = load_and_preprocess_dataset("meta-math/MetaMathQA", sample_size=20000)
    gsm_train_data, gsm_valid_data = load_and_preprocess_dataset("gsm8k")
    
    # Tokenize datasets
    meta_train_tokenized = tokenize_data(meta_train_data, tokenizer)
    gsm_train_tokenized = tokenize_data(gsm_train_data, tokenizer)
    
    # Create dataloaders
    meta_train_dataloader = DataLoader(
        meta_train_tokenized,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_data_collator
    )
    
    gsm_train_dataloader = DataLoader(
        gsm_train_tokenized,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_data_collator
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5,
        weight_decay=0.01
    )
    
    # Set up learning rate scheduler
    num_training_steps = len(meta_train_dataloader) * 3 + len(gsm_train_dataloader)  # 3 epochs for meta-math, 1 for gsm8k
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )
    
    # Fine-tune on meta-math for 3 epochs
    print("Fine-tuning on meta-math...")
    fine_tune_model(model, meta_train_dataloader, optimizer, num_epochs=3, lr_scheduler=lr_scheduler)
    
    # Fine-tune on GSM8K for 1 epoch
    print("Fine-tuning on GSM8K...")
    fine_tune_model(model, gsm_train_dataloader, optimizer, num_epochs=1, lr_scheduler=lr_scheduler)
    
    # Save the fine-tuned model
    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Model saved successfully!")
    
    # Reload the model to verify
    print("Reloading the model for verification...")
    reloaded_model = ModelWithKronABres.from_pretrained(
        output_path,
        base_model_name=model_name,
        hidden_dim=hidden_size,
        a1=a1,
        a2=a2,
        scale=1.0,
        dropout=0.1
    )
    reloaded_model.to(device)
    print("Model reloaded successfully!")