from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from dataset import TextDataset

if TYPE_CHECKING:
    import pandas as pd
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def evaluate_model_on_test(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    test_df: "pd.DataFrame",
    batch_size: int = 16,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate a text classification model on a test DataFrame.

    Args:
        model (PreTrainedModel): Loaded PyTorch/Transformers model (fine-tuned or pre-trained).
        tokenizer (PreTrainedTokenizerBase): Corresponding tokenizer for the model.
        test_df (pd.DataFrame): DataFrame containing test data (must have 'clean_text' and 'label' columns).
        batch_size (int, optional): Batch size for evaluation (default: 16).
        max_length (int, optional): Maximum token sequence length (default: 128).
        device (torch.device, optional): Device for execution (CPU or GPU). If None, selects automatically.

    Returns:
        Dict[str, Any]: Dictionary with evaluation metrics:
            - 'accuracy' (float): overall accuracy.
            - 'precision' (float): weighted precision.
            - 'recall' (float): weighted recall.
            - 'f1' (float): weighted F1-score.
            - 'classification_report' (str): detailed per-class report.
            - 'confusion_matrix' (np.ndarray): confusion matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    texts = test_df["clean_text"].tolist()
    labels = test_df["label"].tolist()

    # Create label mappings for consistent evaluation
    unique_labels = sorted(list(set(labels)))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Convert string labels to integer indices
    y_true = [label2id[lbl] for lbl in labels]

    # Create a dataset and DataLoader for the test set
    dataset = TextDataset(texts, y_true, tokenizer, max_length=max_length)
    loader: DataLoader[TextDataset] = DataLoader(dataset, batch_size=batch_size)

    y_pred = []
    with torch.no_grad():  # Disable gradients for efficiency
        for batch in loader:
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass: get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predicted class for each sample
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]
    )
    cm = confusion_matrix(y_true, y_pred)

    # Return all metrics in a dictionary
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }
