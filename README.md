![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-blue)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)

# Transformer Models and Explainability for Detecting Psychological Distress in Social Media Text

This repository presents a structured experimental pipeline for analyzing linguistic cues associated with psychological distress in social media text using Transformer-based models and Explainable AI (XAI) techniques.

The project combines supervised fine-tuning with interpretability methods to support transparent and reproducible analysis in a sensitive domain.

This work was originally developed as part of an undergraduate thesis (B.Sc. in Computer Engineering) and is made available as an open, reproducible research repository.

---

## Table of Contents

- [Transformer Models and Explainability for Detecting Psychological Distress in Social Media Text](#transformer-models-and-explainability-for-detecting-psychological-distress-in-social-media-text)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Datasets](#datasets)
  - [Methodology](#methodology)
    - [Preprocessing](#preprocessing)
    - [Data Splitting](#data-splitting)
    - [Data Balancing](#data-balancing)
  - [Models](#models)
  - [Training Configuration](#training-configuration)
  - [Evaluation](#evaluation)
  - [Explainability (XAI)](#explainability-xai)
    - [LIME (Local Explanations)](#lime-local-explanations)
    - [Attention Inspection](#attention-inspection)
  - [Results Summary](#results-summary)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
  - [Hugging Face Access (Required)](#hugging-face-access-required)
  - [Usage](#usage)
  - [Reproducibility](#reproducibility)
  - [Future Work](#future-work)
  - [Limitations](#limitations)
  - [Ethical Considerations](#ethical-considerations)
  - [Citation](#citation)
  - [License](#license)
  - [Disclaimer](#disclaimer)

---

## Overview

* **Models**: BERT-base (`google-bert/bert-base-uncased`), MentalRoBERTa (`mental/mental-roberta-base`)
* **Tasks**: Binary and multiclass text classification
* **Datasets**: 4 public Kaggle datasets (mental health domain)
* **Techniques**: Supervised fine-tuning, LIME (local explanations), attention-based inspection
* **Scope**: Each dataset is treated as an independent classification task; no cross-dataset training; models fine-tuned from pretrained checkpoints with fixed hyperparameters for comparability

The repository is organized as a step-by-step experimental pipeline:

```
data exploration → preprocessing → model training → evaluation → explainability analysis
```

---

## Datasets

This project uses the following public datasets from Kaggle:

* Suicide and Depression Detection
* Sentiment Analysis for Mental Health
* Sentimental Analysis for Tweets
* Mental Health Corpus

Datasets are **not included** in this repository.

See [`data/README.md`](./data/README.md) for:

* download links
* folder structure
* formatting instructions

---

## Methodology

### Preprocessing

* Lowercasing
* Removal of URLs, mentions, hashtags
* Deduplication (**performed before splitting**)
* Filtering of short texts (dataset-dependent)

### Data Splitting

* Stratified split: **80% train / 10% validation / 10% test**
* Fixed random seed: **42**

### Data Balancing

* Undersampling applied in 3 datasets (before split)
* One dataset intentionally kept imbalanced (multiclass scenario)

---

## Models

* **BERT-base** (general-domain)
* **MentalRoBERTa** (domain-adapted for mental health)

Both models are fine-tuned independently on each dataset.

---

## Training Configuration

| Parameter     | Value              |
| ------------- | ------------------ |
| Batch size    | 16                 |
| Epochs        | 3                  |
| Learning rate | 2e-5               |
| Max length    | 256                |
| Optimizer     | AdamW              |
| Scheduler     | Linear with warmup |
| Random seed   | 42                 |

All experiments use **fixed hyperparameters** for comparability.

---

## Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
  * Weighted average (multiclass scenario)

Additional diagnostics:

* Per-class analysis (multiclass)
* Confusion matrices (available in notebooks)

---

## Explainability (XAI)

Two complementary approaches are used:

### LIME (Local Explanations)

* Identifies influential words per prediction
* Model-agnostic

### Attention Inspection

* Mean attention weights from the **first Transformer layer**
* Top-N tokens analyzed

Attention is treated as an **inspection mechanism**, not as a causal explanation.

---

## Results Summary

All models were evaluated on held-out test sets. Fine-tuning consistently yielded substantial gains over pre-trained baselines across all datasets.

| Dataset                              | Model         | Accuracy | F1 (weighted) |
|--------------------------------------|---------------|----------|---------------|
| Suicide and Depression Detection     | BERT-base     | 0.9809   | 0.9809        |
| Suicide and Depression Detection     | MentalRoBERTa | 0.9918   | 0.9918        |
| Sentiment Analysis for Mental Health | BERT-base     | 0.8383   | 0.8378        |
| Sentiment Analysis for Mental Health | MentalRoBERTa | 0.8457   | 0.8460        |
| Sentimental Analysis for Tweets      | BERT-base     | 0.9955   | 0.9955        |
| Sentimental Analysis for Tweets      | MentalRoBERTa | 0.9887   | 0.9887        |
| Mental Health Corpus                 | BERT-base     | 0.9666   | 0.9666        |
| Mental Health Corpus                 | MentalRoBERTa | 0.9651   | 0.9651        |

> Full per-class metrics, confusion matrices, and pre-trained vs. fine-tuned comparisons are available in [`notebooks/04-evaluation.ipynb`](./notebooks/04-evaluation.ipynb).

---

## Repository Structure

```
.
├── src/                # Core pipeline modules (preprocessing, training, evaluation, XAI)
├── notebooks/          # Step-by-step experimental workflow (01 → 05)
├── data/               # Dataset instructions and processed splits (no raw data included)
├── models/             # Saved model artifacts — not versioned, generated locally at runtime
├── requirements.txt
├── .gitignore
└── LICENSE
```

> The `models/` directory is not included in the repository. It is created automatically during training (notebook `03`). The `data/processed/` directory is created automatically during preprocessing (notebook `02`).

---

## Installation

Requirements: **Python 3.10+**

```bash
pip install -r requirements.txt
```

> **Note**: [`requirements.txt`](./requirements.txt) includes a GPU-enabled PyTorch build. If you encounter compatibility issues, install the appropriate version for your system from the [PyTorch official installer](https://pytorch.org/get-started/locally/).
> Training without a GPU is possible but significantly slower (hours per experiment vs. minutes).

To reproduce the full pipeline:

1. Install dependencies (see above)
2. Authenticate with Hugging Face (see [Hugging Face Access](#hugging-face-access-required))
3. Download datasets and place them under `data/raw/` (see [`data/README.md`](./data/README.md))
4. Run notebooks in order: `01 → 02 → 03 → 04 → 05` (see [`notebooks/README.md`](./notebooks/README.md))

---

## Hugging Face Access (Required)

To use **MentalRoBERTa**, authentication with Hugging Face is required, as the model is a gated resource. You may also need to request access via the model page before downloading it programmatically.

```bash
hf auth login
```

More info: [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub)

---

## Usage

The experimental pipeline is driven through notebooks. See [`notebooks/README.md`](./notebooks/README.md) for per-notebook descriptions.

The `src/` modules can also be used programmatically:

```python
import sys
sys.path.append("src/")

from preprocessing import preprocess_dataframe
from train import train_model
from evaluate import evaluate_model_on_test

# Preprocess a raw DataFrame
df_clean = preprocess_dataframe(df_raw, text_column="text", min_length=4)

# Fine-tune a model
model, tokenizer, *_, history = train_model(
    model_name="google-bert/bert-base-uncased",
    data_dir="data/processed/suicide-and-depression-detection",
    output_dir="models/",
    return_history=True,
)

# Evaluate on test set
metrics = evaluate_model_on_test(model, tokenizer, test_df)
print(metrics["classification_report"])
```

See [`src/README.md`](./src/README.md) for module documentation.

---

## Reproducibility

* Fixed seed (`42`) across NumPy, PyTorch, and CUDA
* Data splitting uses fixed `random_state=42` throughout
* Final outputs included in notebooks for transparency

Re-running the pipeline should reproduce consistent results. Minor numerical variations may occur across different hardware configurations, CUDA versions, or operating systems, as full platform-level determinism cannot be guaranteed.

---

## Future Work

Potential directions to extend this work include:

* Exploring alternative data balancing and augmentation strategies
* Incorporating classical baselines (e.g., Logistic Regression, SVM) for stronger comparative analysis
* Evaluating cross-dataset and cross-lingual generalization
* Expanding explainability analysis with additional XAI methods (e.g., SHAP)
* Conducting interdisciplinary validation of XAI findings with domain experts

---

## Limitations

* Social media data ≠ clinical diagnosis
* Undersampling alters real-world distributions
* Multiclass imbalance impacts minority classes
* Sequence length (256) may truncate long context
* XAI methods are **interpretative, not causal**

---

## Ethical Considerations

This project:

* Does **not** perform clinical diagnosis
* Uses only **public datasets**
* Should **not** be used for individual-level decisions

Intended for:

* research
* educational purposes
* methodological analysis

---

## Citation

If you use this repository in your research, please reference it directly:

> Delfino, K. (2025). *Transformer Models and Explainability for Detecting Psychological Distress in Social Media Text*. GitHub repository. [github.com/kayckdelfino/mental-health-nlp-xai](https://github.com/kayckdelfino/mental-health-nlp-xai)

A formal citation entry (BibTeX, APA) will be added here upon publication.

---

## License

MIT License — see [`LICENSE`](./LICENSE)

---

## Disclaimer

This project is intended strictly for research and educational purposes.

It must not be used for:

* clinical diagnosis
* psychological evaluation
* decision-making about individuals
