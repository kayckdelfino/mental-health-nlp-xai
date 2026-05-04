# Notebooks

[← Back to main README](../README.md)

This directory contains the step-by-step experimental pipeline.

Each notebook corresponds to a stage of the workflow and is designed to be readable, reproducible, and self-contained.

---

## Pipeline

Run notebooks in the following order:

1. [`01-data-exploration.ipynb`](01-data-exploration.ipynb)
   - Initial inspection of all 4 datasets (shape, types, missing values)
   - Class distribution and text length analysis
   - Word frequency and word cloud visualizations
   - Comparative analysis across datasets

2. [`02-preprocessing.ipynb`](02-preprocessing.ipynb)
   - Text cleaning via [`src/preprocessing.py`](../src/preprocessing.py) (lowercase, URL/mention/hashtag removal, deduplication)
   - Dataset-specific column normalization and label standardization
   - Class balancing via undersampling (3 datasets) and intentional imbalance (1 multiclass dataset)
   - Stratified 80/10/10 train/validation/test split
   - Exports processed splits to `data/processed/`

3. [`03-model-training.ipynb`](03-model-training.ipynb)
   - Fine-tuning of BERT-base and MentalRoBERTa on each processed dataset (8 experiments total)
   - Training and validation loss/accuracy curves per epoch
   - Best-model checkpointing; models saved to `models/`

4. [`04-evaluation.ipynb`](04-evaluation.ipynb)
   - Comparison of fine-tuned vs. pre-trained performance
   - Metrics: accuracy, precision, recall, F1-score (weighted)
   - Per-class analysis and confusion matrices for all model vs. dataset combinations
   - Summary table of all results

5. [`05-xai-analysis.ipynb`](05-xai-analysis.ipynb)
   - LIME explanations: influential words per prediction
   - Attention inspection: top-attended tokens (first Transformer layer)
   - Comparative XAI analysis across models and datasets

---

## Execution Notes

* Sequential execution is recommended to reproduce the full pipeline
* Some notebooks may take significant time depending on hardware
* Training (notebook `03`) requires a CUDA-compatible GPU for practical execution; CPU execution is possible but very slow
* Outputs from final runs are preserved in notebooks for transparency
* Re-running the pipeline should produce consistent training results; LIME explanations are stochastic and may vary across runs

---

## Setup

See [Installation](../README.md#installation) in the main README for full setup instructions, including GPU/CPU configuration and Hugging Face authentication.

Ensure datasets are placed in `data/raw/` before running notebook `01`. See [`data/README.md`](../data/README.md).
