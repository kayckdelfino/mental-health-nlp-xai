from typing import TYPE_CHECKING, Dict, List, cast

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset):
    """
    Custom Dataset for text classification tasks using Transformers.
    Converts raw texts and labels into tokenized tensors for use with PyTorch DataLoader.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> dataset = TextDataset(['text1', 'text2'], [0, 1], tokenizer)
        >>> sample = dataset[0]
        >>> print(sample['input_ids'].shape)
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 128,
    ) -> None:
        """
        Args:
            texts (List[str]): List of input texts.
            labels (List[int]): List of integer labels.
            tokenizer (PreTrainedTokenizerBase): Tokenizer instance.
            max_length (int): Maximum sequence length for tokenization.
        """
        # Store references to data and tokenizer
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns:
            int: Number of examples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single tokenized example for the given index.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys:
                - input_ids (torch.Tensor): Token IDs for the text.
                - attention_mask (torch.Tensor): Attention mask for the text.
                - labels (torch.Tensor): Integer label for the example.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize the text using the provided tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add special tokens required by the model
            max_length=self.max_length,  # Truncate/pad to max_length
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_attention_mask=True,  # Return attention mask for padding
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Tokenizer returns Tensor at runtime; cast fixes typing
        input_ids = cast(torch.Tensor, encoding["input_ids"])
        attention_mask = cast(torch.Tensor, encoding["attention_mask"])

        # Return a dictionary compatible with PyTorch DataLoader
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
