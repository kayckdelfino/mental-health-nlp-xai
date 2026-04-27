import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def clean_text(text: str | None) -> str:
    """
    Cleans the input text by removing URLs, mentions, hashtags, and extra spaces.

    Args:
        text (str | None): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()  # Lowercase for normalization

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove user mentions (e.g., @username)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # Remove hashtags (e.g., #topic)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataframe(
    df: "pd.DataFrame",
    text_column: str,
    min_length: int = 4,
) -> "pd.DataFrame":
    """
    Cleans and filters a DataFrame of texts and labels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.
        min_length (int, optional): Minimum number of words. Default is 4.

    Returns:
        pd.DataFrame: Cleaned DataFrame with text columns.
    """
    # Remove rows with missing text
    df = df.dropna(subset=[text_column]).copy()
    # Clean the text column and store in 'clean_text'
    df["clean_text"] = df[text_column].astype(str).apply(clean_text)

    # Compute text length (number of words)
    df["text_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    # Filter out short texts below min_length
    df = df[df["text_length"] >= min_length].copy()

    # Remove duplicate texts
    df = df.drop_duplicates(subset=["clean_text"])

    return df.reset_index(drop=True)
