from T5Integretion import KronABT5, KronAForT5
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric

def train_krona_on_glue(task="cola", model_type="krona", epochs=20, batch_size=100, 
                         learning_rate=1e-3, scale=1.0, warmup_steps=500, use_residual=False,
                         num_biases=1):
    """
    Train a KronA variant on a GLUE task
    
    Args:
        task: GLUE task name (cola, rte, mrpc, sst2, stsb, mnli, qnli, qqp)
        model_type: Type of KronA to use (krona, kronab, kronabres)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        scale: Scale factor for KronA modules
        warmup_steps: Number of warmup steps for learning rate scheduler
        use_residual: Whether to use learnable residual (only for kronabres)
        num_biases: Number of bias vectors (1 or 2)
    """
    # Load dataset
    dataset = load_dataset("glue", task)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    # Task-specific preprocessing
    task_to_prefix = {
        "cola": "cola sentence: ",
        "sst2": "sst2 sentence: ",
        "mrpc": "mrpc sentence1: ",
        "qqp": "qqp question1: ",
        "stsb": "stsb sentence1: ",
        "mnli": "mnli hypothesis: ",
        "qnli": "qnli question: ",
        "rte": "rte sentence1: "
    }
    
    def preprocess_function(examples):
        prefix = task_to_prefix[task]
        
        # Handle different task formats
        texts = []
        if task in ["mrpc", "qqp", "stsb"]:
            for i in range(len(examples["sentence1"])):
                texts.append(prefix + examples["sentence1"][i] + " sentence2: " + examples["sentence2"][i])
        elif task in ["mnli", "qnli", "rte"]:
            for i in range(len(examples["premise"])):
                texts.append(prefix + examples["premise"][i] + " premise: " + examples["hypothesis"][i])
        else:  # cola, sst2
            for sent in examples["sentence"]:
                texts.append(prefix + sent)
        
        # Tokenize inputs
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
        
        # Tokenize targets
        if task == "stsb":
            # Regression task
            targets = [str(float(label)) for label in examples["label"]]
        else:
            # Classification task
            targets = [str(label) for label in examples["label"]]
        
        target_encodings = tokenizer(targets, padding="max_length", truncation=True, max_length=8)
        inputs["labels"] = target_encodings["input_ids"]
        
        return inputs
    
    # Preprocess datasets
    train_dataset = dataset["train"].map(preprocess_function, batched=True)
    eval_dataset = dataset["validation"].map(preprocess_function, batched=True)
    
    # Convert to PyTorch Dataset format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Initialize model
    if model_type == "krona":
        model = KronAForT5(scale=scale)
    else:  # "kronab" or "kronabres"
        model = KronABT5(scale=scale, use_residual=(model_type=="kronabres"), num_biases=num_biases)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Load metric
    if task == "stsb":
        metric = load_metric("glue", "stsb")
        is_regression = True
    else:
        metric = load_metric("glue", task)
        is_regression = False
    
    # Training loop
    best_score = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg. Training Loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        references = []
        
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=8
                )
            
            # Decode predictions and references
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            if is_regression:
                # Convert string predictions to floats for regression task
                pred_values = [float(pred) for pred in pred_texts]
                ref_values = [float(ref) for ref in ref_texts]
                predictions.extend(pred_values)
                references.extend(ref_values)
            else:
                # Convert string labels to integers for classification
                pred_values = [int(pred) if pred.isdigit() else 0 for pred in pred_texts]
                ref_values = [int(ref) if ref.isdigit() else 0 for ref in ref_texts]
                predictions.extend(pred_values)
                references.extend(ref_values)
        
        # Compute metrics
        if task == "cola":
            # Matthew's correlation
            result = metric.compute(predictions=predictions, references=references)
            score = result["matthews_correlation"]
            metric_name = "Matthews Corr"
        elif task == "stsb":
            # Pearson and Spearman correlation
            result = metric.compute(predictions=predictions, references=references)
            score = (result["pearson"] + result["spearmanr"]) / 2
            metric_name = "Avg Correlation"
        elif task in ["mrpc", "qqp"]:
            # F1 score
            result = metric.compute(predictions=predictions, references=references)
            score = result["f1"]
            metric_name = "F1"
        else:
            # Accuracy
            result = metric.compute(predictions=predictions, references=references)
            score = result["accuracy"]
            metric_name = "Accuracy"
        
        print(f"Epoch {epoch+1}/{epochs} - {metric_name}: {score:.4f}")
        
        # Save best model
        if score > best_score:
            best_score = score
            if model_type == "krona":
                # Merge weights for inference if using KronA
                model.merge_weights()
            torch.save(model.state_dict(), f"krona_{model_type}_{task}_best.pt")
    
    print(f"Best {metric_name}: {best_score:.4f}")
    return best_score