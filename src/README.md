# Source Code Overview

[← Back to main README](../README.md)

This directory contains the core implementation of the experimental pipeline.

The code is modularized to support preprocessing, training, evaluation, and explainability.

---

## Structure

```
src/
├── __init__.py        # Package marker (empty; enables use as a package)
├── dataset.py         # Dataset loading and handling
├── preprocessing.py   # Text cleaning and preprocessing pipeline
├── train.py           # Model training and fine-tuning
├── evaluate.py        # Evaluation metrics and analysis
└── xai.py             # Explainability methods (LIME, attention)
```

---

## Design Principles

* Modular and reusable components
* Clear separation of concerns
* Reproducibility through fixed seeds
* Compatibility with notebook-based workflows

---

## Usage

The `src/` modules are primarily used within the notebooks and can also be imported programmatically:

```python
import sys
sys.path.append("src/")

from preprocessing import preprocess_dataframe
from train import train_model
```

> **Note**: All modules use flat imports internally, so `src/` must be added to `sys.path` before importing. The notebooks handle this automatically at startup.

---

## Reproducibility

* Random seeds are fixed across:
  * NumPy
  * PyTorch
  * data splitting

* Deterministic behavior is enforced where possible

> **`max_length` note**: The default value in module function signatures is `128`, but all experiments in the notebooks use `MAX_LENGTH = 256` (as documented in the [Training Configuration](../README.md#training-configuration) table). Always pass `max_length=256` explicitly when reproducing the reported results.

---

## Notes

* This is a research-oriented codebase, not a production system
* Model loading relies on the Hugging Face Transformers library
* Some components are simplified for clarity and experimentation
* Type hints are used to improve readability and maintainability
* XAI implementations are intended for **interpretation**, not causal inference
* No training scripts are exposed as CLI tools; execution is notebook-driven

---

## Extending the Code

You can extend this module by:

* Adding new models or architectures
* Implementing additional evaluation metrics
* Integrating alternative XAI techniques (e.g., SHAP)

Keep consistency with existing structure and typing.
