# Data Instructions

[← Back to main README](../README.md)

This directory contains instructions for obtaining and organizing the datasets used in this project.

Datasets are **not included** in this repository.

---

## Datasets

The experiments rely on the following public datasets from Kaggle:

---

### 1. Suicide and Depression Detection

**Task**: Binary classification
**Classes**: `suicide`, `not_suicide`
**Source**: [kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

---

### 2. Sentiment Analysis for Mental Health

**Task**: Multiclass classification
**Classes**: `suicide`, `anxiety`, `bipolar`, `depression`, `normal`, `stress`, `personality_disorder`
**Source**: [kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

---

### 3. Sentimental Analysis for Tweets

**Task**: Binary classification
**Classes**: `depression`, `not_depression`
**Source**: [kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets)

---

### 4. Mental Health Corpus

**Task**: Binary classification
**Classes**: `poisonous`, `not_poisonous`
**Source**: [kaggle.com/datasets/reihanenamdari/mental-health-corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus)

---

## Expected Directory Structure

After downloading and extracting the datasets, organize them as follows:

```
data/
├── raw/
│   ├── suicide-and-depression-detection/
│   |   └── raw.csv
│   ├── sentiment-analysis-for-mental-health/
│   |   └── raw.csv
│   ├── sentimental-analysis-for-tweets/
│   |   └── raw.csv
│   └── mental-health-corpus/
│       └── raw.csv
└── processed/
```

- `raw/`: original dataset files (must be renamed to `raw.csv` to match the expected pipeline input)
- `processed/`: generated automatically during preprocessing ([`notebooks/02-preprocessing.ipynb`](../notebooks/02-preprocessing.ipynb))

After running the preprocessing notebook, `processed/` will have the following structure:

```
data/
└── processed/
    ├── suicide-and-depression-detection/
    │   ├── clean.csv
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    ├── sentiment-analysis-for-mental-health/
    │   ├── clean.csv
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    ├── sentimental-analysis-for-tweets/
    │   ├── clean.csv
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    └── mental-health-corpus/
        ├── clean.csv
        ├── train.csv
        ├── validation.csv
        └── test.csv
```

Each split file contains at least two columns: `clean_text` (preprocessed text) and `label` (target class).

> **Note**: The class labels shown above reflect the standardized values used throughout the pipeline (post-preprocessing). Raw dataset files may contain different representations (e.g., `non-suicide` instead of `not_suicide`, or integer values `0`/`1` instead of string labels). The preprocessing notebook handles all normalization automatically.

---

## Formatting Requirements

To ensure compatibility with the pipeline:

* Files must be in **CSV format**
* Files should use UTF-8 encoding to avoid parsing issues
* Each dataset must contain:
  - a text column (input)
  - a label column (target)
* Column names may differ across datasets — preprocessing scripts handle normalization
* The preprocessing pipeline internally maps dataset-specific column names to a unified format

---

## Notes

* Some preprocessing steps are dataset-specific
* If dataset structure or schema changes, adjustments in preprocessing may be required
* Deduplication and filtering are handled programmatically in the pipeline

---

## Usage

Once datasets are placed in `data/raw/`, proceed with:

1. [`notebooks/01-data-exploration.ipynb`](../notebooks/01-data-exploration.ipynb) — initial inspection and distribution analysis
2. [`notebooks/02-preprocessing.ipynb`](../notebooks/02-preprocessing.ipynb) — cleaning, balancing, and splitting

---

## License and Usage

All datasets belong to their respective authors and are subject to their original licenses and terms of use.

This project uses them **strictly for research and educational purposes**.
