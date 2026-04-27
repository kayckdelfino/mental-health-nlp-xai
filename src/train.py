import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import TextDataset

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def train_epoch(
    model: "PreTrainedModel",
    data_loader: DataLoader[TextDataset],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, float]:
    """
    Runs one training epoch for the model.

    Args:
        model (PreTrainedModel): PyTorch/Transformers model.
        data_loader (DataLoader[TextDataset]): DataLoader with training batches.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device (cpu or cuda).
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler.
    Returns:
        Tuple[float, float]: (accuracy, average loss)
    """
    model.train()  # Enable training mode (activates dropout, batchnorm, etc.)

    losses = []
    correct_predictions = 0
    for batch in data_loader:
        # Move batch data to the selected device (GPU or CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass: compute model outputs and loss
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        # Get predicted class for each sample in the batch
        _, preds = torch.max(logits, dim=1)
        # Count correct predictions for accuracy calculation
        correct_predictions += torch.sum(preds == labels)
        # Store loss for averaging later
        losses.append(loss.item())

        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()
        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step()
        # Reset gradients before next batch
        optimizer.zero_grad()

    # Compute accuracy and average loss for the epoch
    total = len(data_loader.dataset)  # type: ignore[attr-defined]
    acc = float(correct_predictions) / total if total > 0 else 0.0
    return acc, sum(losses) / len(losses) if losses else 0.0


def eval_model(
    model: "PreTrainedModel", data_loader: DataLoader[TextDataset], device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluates the model on the validation or test set.

    Args:
        model (PreTrainedModel): PyTorch/Transformers model.
        data_loader (DataLoader[TextDataset]): Validation/test DataLoader.
        device (torch.device): Device (cpu or cuda).
    Returns:
        Tuple[float, float, List[int], List[int]]: (accuracy, average loss, true labels, predicted labels)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    losses = []
    correct_predictions = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in data_loader:
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass: compute outputs and loss
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            # Get predicted class for each sample
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            # Store true and predicted labels for metrics
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Compute accuracy and average loss
    total = len(data_loader.dataset)  # type: ignore[attr-defined]
    acc = float(correct_predictions) / total if total > 0 else 0.0
    return (
        acc,
        sum(losses) / len(losses) if losses else 0.0,
        all_labels,
        all_preds,
    )


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test splits from CSV files in a directory.

    Args:
        data_dir (str): Path to the directory containing train.csv, validation.csv, test.csv.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df) as pandas DataFrames.
    """
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, val, test


def prepare_tokenizer_and_labels(
    model_name: str, train_df: pd.DataFrame, label_col: str = "label"
) -> Tuple["PreTrainedTokenizerBase", Dict[str, int], Dict[int, str]]:
    """
    Creates the appropriate tokenizer for the model and generates label2id/id2label mappings.

    Args:
        model_name (str): Model name ('bert-base-uncased', 'mental-roberta-base', etc).
        train_df (pd.DataFrame): Training DataFrame to extract labels.
        label_col (str): Name of the label column.
    Returns:
        Tuple[PreTrainedTokenizerBase, Dict[str, int], Dict[int, str]]: (tokenizer, label2id, id2label)
    """
    labels = sorted(train_df[label_col].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    if "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer, label2id, id2label


def prepare_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: "PreTrainedTokenizerBase",
    label2id: Dict[str, int],
    batch_size: int = 16,
    max_length: int = 128,
    text_col: str = "clean_text",
    label_col: str = "label",
) -> Tuple[DataLoader[TextDataset], DataLoader[TextDataset], DataLoader[TextDataset]]:
    """
    Creates PyTorch DataLoaders for training, validation, and test sets from DataFrames and tokenizer.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        tokenizer (PreTrainedTokenizerBase): Model tokenizer.
        label2id (Dict[str, int]): Label to index mapping.
        batch_size (int): Batch size.
        max_length (int): Maximum sequence length.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column.
    Returns:
        Tuple[DataLoader[TextDataset], DataLoader[TextDataset], DataLoader[TextDataset]]: (train_loader, val_loader, test_loader)
    """
    train_dataset = TextDataset(
        texts=train_df[text_col].tolist(),
        labels=[label2id[label] for label in train_df[label_col]],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = TextDataset(
        texts=val_df[text_col].tolist(),
        labels=[label2id[label] for label in val_df[label_col]],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = TextDataset(
        texts=test_df[text_col].tolist(),
        labels=[label2id[label] for label in test_df[label_col]],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader: DataLoader[TextDataset] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader[TextDataset] = DataLoader(val_dataset, batch_size=batch_size)
    test_loader: DataLoader[TextDataset] = DataLoader(
        test_dataset, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def save_model_and_tokenizer(
    model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase", save_dir: str
) -> None:
    """
    Saves the model and tokenizer in Hugging Face format, with subfolders model/ and tokenizer/.

    Args:
        model (PreTrainedModel): Trained PyTorch model.
        tokenizer (PreTrainedTokenizerBase): Model tokenizer.
        save_dir (str): Directory to save model and tokenizer.

    Returns:
        None
    """
    model_dir = os.path.join(save_dir, "model")
    tokenizer_dir = os.path.join(save_dir, "tokenizer")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)


def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
    device: Optional[torch.device] = None,
    return_history: bool = False,
) -> Tuple[
    "PreTrainedModel",
    "PreTrainedTokenizerBase",
    DataLoader[TextDataset],
    DataLoader[TextDataset],
    DataLoader[TextDataset],
    Dict[str, int],
    Dict[int, str],
    Optional[Dict[str, List[float]]],
]:
    """
    Complete fine-tuning pipeline for a Transformer model on a text classification dataset.
    Includes data loading, tokenization, DataLoader creation, training, validation, and saving the best model.

    Args:
        model_name (str): Model name (e.g., 'bert-base-uncased', 'mental-roberta-base').
        data_dir (str): Directory containing train.csv, validation.csv, and test.csv files.
        output_dir (str): Directory to save the model and tokenizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        max_length (int): Maximum sequence length for tokenization.
        seed (int): Random seed for reproducibility.
        device (torch.device, optional): Device for training (cuda/cpu).
        return_history (bool, optional): If True, returns training/validation loss and accuracy history.
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase, DataLoader[TextDataset], DataLoader[TextDataset], DataLoader[TextDataset], Dict[str, int], Dict[int, str], Optional[Dict[str, List[float]]]:
            (trained model, tokenizer, train_loader, val_loader, test_loader, label2id, id2label, [history])

    Example:
        >>> model, tokenizer, train_loader, val_loader, test_loader, label2id, id2label, history = train_model(
        ...     model_name='bert-base-uncased', data_dir='...', output_dir='...', return_history=True)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load train, validation, and test splits from CSV files
    train_df, val_df, test_df = load_data(data_dir)

    # Prepare tokenizer and label mappings (label2id, id2label)
    tokenizer, label2id, id2label = prepare_tokenizer_and_labels(model_name, train_df)

    # Create DataLoaders for each split
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer, label2id, batch_size, max_length
    )
    num_labels = len(label2id)

    # Instantiate the correct model architecture (BERT or RoBERTa)
    if "roberta" in model_name:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    # Move model to the selected device
    model.to(device)  # type: ignore[assignment]

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs

    # Linear warmup for first 10% of steps, then linear decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Track the best model (highest validation accuracy)
    best_val_acc = 0.0
    best_model_state = None

    # Store training/validation metrics for analysis
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    # Main training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train for one epoch
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler
        )

        # Evaluate on validation set
        val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
        print(
            f"Train acc: {train_acc:.4f} | loss: {train_loss:.4f} | Val acc: {val_acc:.4f} | loss: {val_loss:.4f}"
        )

        # Store metrics for this epoch
        history["train_acc"].append(float(train_acc))
        history["train_loss"].append(float(train_loss))
        history["val_acc"].append(float(val_acc))
        history["val_loss"].append(float(val_loss))

        # Save model state if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model and tokenizer in HuggingFace format for reproducibility
    output_dir_hf = os.path.join(output_dir, f"{model_name.replace('/', '_')}_hf")
    save_model_and_tokenizer(model, tokenizer, output_dir_hf)

    return (
        model,
        tokenizer,
        train_loader,
        val_loader,
        test_loader,
        label2id,
        id2label,
        history if return_history else None,
    )
