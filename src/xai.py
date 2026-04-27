from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lime.lime_text import LimeTextExplainer

if TYPE_CHECKING:
    from lime.explanation import Explanation
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def lime_explain_instance(
    text: str,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    class_names: List[str],
    max_length: int = 128,
    num_features: int = 10,
    num_samples: int = 1000,
    device: Optional[torch.device] = None,
) -> "Explanation":
    """
    Generates a LIME explanation for a text instance using a HuggingFace model.

    Args:
        text (str): Input text to be explained.
        model (PreTrainedModel): Loaded HuggingFace model.
        tokenizer (PreTrainedTokenizerBase): Corresponding tokenizer for the model.
        class_names (List[str]): List of class names.
        max_length (int, optional): Maximum sequence length for tokenization. Default: 128.
        num_features (int, optional): Number of most important words to display. Default: 10.
        num_samples (int, optional): Number of LIME samples. Default: 1000.
        device (torch.device, optional): Device to run the model. Default: None.

    Returns:
        Explanation: LIME explanation object.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set model to evaluation mode

    def predict_proba(texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for a list of input texts.

        This function is used by LIME to generate perturbed samples and get model predictions.
        It must return a probability distribution over classes for each input.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) with class probabilities.
        """
        if not texts:
            return np.zeros((0, len(class_names)))

        # Tokenize and prepare inputs for the model
        inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move inputs to the selected device (GPU or CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get probabilities from the model using softmax
        with torch.no_grad():
            outputs = model(**inputs)

            # Some models return logits as attribute, others as first tuple element
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    # Create a LIME explainer for text classification
    explainer = LimeTextExplainer(class_names=class_names)

    # Generate explanation for the provided text instance
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
    return explanation


def get_lime_top_words(
    explanation: "Explanation", label_index: int = 0, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns a list of the most influential words according to LIME, with their weights.

    Args:
        explanation (Explanation): LIME explanation object.
        label_index (int, optional): Class index. Default: 0.
        top_n (int, optional): Number of words to return. Default: 10.

    Returns:
        List[Tuple[str, float]]: List of tuples (word, weight).
    """
    labels = explanation.available_labels()
    if not labels:
        return []

    if label_index >= len(labels):
        label_index = 0

    label = labels[label_index]
    return explanation.as_list(label=label)[:top_n]


def plot_lime_explanation(
    explanation: "Explanation",
    label_index: int = 0,
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Plots the LIME explanation for a text instance.

    Args:
        explanation (Explanation): LIME explanation object.
        label_index (int, optional): Index of the class for which the explanation will be plotted. Default: 0.
        figsize (Tuple[int, int], optional): Figure size. Default: (8, 4).

    Returns:
        None
    """
    labels = explanation.available_labels()

    if not labels:
        print("No label available for LIME plot.")
        return

    if label_index >= len(labels):
        label_index = 0

    label_to_plot = labels[label_index]
    fig = explanation.as_pyplot_figure(label=label_to_plot)
    fig.set_size_inches(*figsize)
    plt.tight_layout()
    plt.show()


def get_attention_weights(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Extracts attention weights from a HuggingFace model for an input text.

    Args:
        model (PreTrainedModel): Loaded HuggingFace model.
        tokenizer (PreTrainedTokenizerBase): Corresponding tokenizer for the model.
        text (str): Input text.
        max_length (int, optional): Maximum sequence length for tokenization. Default: 128.
        device (torch.device, optional): Device to run the model. Default: None.

    Returns:
        Tuple[List[str], np.ndarray]:
            - tokens (List[str]): List of tokens from the input text.
            - attn_weights (np.ndarray): Attention weights array with shape (layers, heads, seq_len, seq_len).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set model to evaluation mode

    # Tokenize the input text for the model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    # Move inputs to the selected device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    # Get attention weights from the model
    with torch.no_grad():
        try:
            # Some models require output_attentions=True to return attention weights
            outputs = model(**inputs, output_attentions=True)

        except TypeError:
            # Fallback for models that always return attentions
            outputs = model(**inputs)

        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            raise ValueError(
                "Model did not return attentions. Ensure the model supports output_attentions=True."
            )

    # Stack attentions across layers (shape: layers, batch, heads, seq_len, seq_len)
    attention_tensor = torch.stack(attentions)

    # Remove batch dimension if present (for single input)
    if attention_tensor.shape[1] == 1:
        attention_tensor = attention_tensor.squeeze(1)

    # Convert to numpy for further processing/visualization
    attn_weights = attention_tensor.cpu().numpy()

    # Convert input IDs back to tokens for interpretability
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())  # type: ignore[attr-defined]
    return tokens, attn_weights


def get_top_attention_tokens(
    tokens: List[str], attn_matrix: np.ndarray, layer: int = -1, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns the tokens that received the most attention in the specified layer, with numeric values.

    Args:
        tokens (list): List of tokens.
        attn_matrix (np.ndarray): Attention matrix (layers, heads, seq_len, seq_len).
        layer (int, optional): Layer index. Default: -1 (last).
        top_n (int, optional): Number of tokens to return. Default: 10.

    Returns:
        List[Tuple[str, float]]: List of tuples (token, received attention value).
    """
    if attn_matrix.ndim != 4:
        # Attention matrix must have 4 dimensions: (layers, heads, seq_len, seq_len)
        raise ValueError(
            "attn_matrix must have 4 dimensions: (layers, heads, seq_len, seq_len)"
        )

    # Support negative indexing for layer (e.g., -1 for last layer)
    if layer < 0:
        layer = attn_matrix.shape[0] + layer

    if layer >= attn_matrix.shape[0]:
        layer = attn_matrix.shape[0] - 1

    # Average attention across all heads for the selected layer
    mean_attn = np.mean(attn_matrix[layer], axis=0)

    # Sum attention received by each token
    attention_received = mean_attn.sum(axis=0)

    # Get indices of top tokens by attention received
    top_indices = np.argsort(attention_received)[::-1][:top_n]

    # Return list of (token, attention value) tuples
    return [(tokens[i], float(attention_received[i])) for i in top_indices]


def plot_mean_attention_heatmap(
    tokens: List[str],
    attn_matrix: np.ndarray,
    layer: int = -1,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plots a heatmap of the MEAN attention weights across all heads for a specific layer.

    Args:
        tokens (List[str]): List of tokens.
        attn_matrix (np.ndarray): Attention weights array with shape (layers, heads, seq_len, seq_len).
        layer (int, optional): Index of the layer to visualize. Default: -1 (last).
        figsize (Tuple[int, int], optional): Figure size.
    """
    if attn_matrix.ndim != 4:
        # Attention matrix must have 4 dimensions: (layers, heads, seq_len, seq_len)
        raise ValueError(
            "attn_matrix must have 4 dimensions: (layers, heads, seq_len, seq_len)"
        )

    # Support negative indexing for layer
    if layer < 0:
        layer = attn_matrix.shape[0] + layer

    if layer >= attn_matrix.shape[0]:
        layer = attn_matrix.shape[0] - 1

    # Average attention across all heads for the selected layer
    mean_attn = np.mean(attn_matrix[layer], axis=0)

    # Plot heatmap of mean attention weights
    plt.figure(figsize=figsize)
    sns.heatmap(mean_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Mean Attention Heatmap - Layer {layer}", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
