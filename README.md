# Kaggle Competition Template

A structured, production-ready template repository for Kaggle data science competitions.

## Directory Structure

```
├── data/
│   ├── raw/          # Raw dataset files (git-ignored)
│   ├── processed/    # Processed features and datasets (git-ignored)
│   └── external/     # Third-party or external datasets (git-ignored)
├── notebooks/        # Jupyter notebooks for EDA and experimentation
│   ├── 01_eda.ipynb
│   └── 02_baseline.ipynb
├── src/              # Reusable Python modules
│   ├── __init__.py
│   ├── data.py       # Data loading, cleaning, and preprocessing
│   ├── features.py   # Feature engineering pipelines
│   ├── models.py     # Model definitions and training loops
│   └── utils.py      # Evaluation metrics, seed setting, logger
├── models/           # Saved model artifacts, weights, and checkpoints (git-ignored)
├── submissions/      # Generated submission CSVs and logs (git-ignored)
├── requirements.txt  # Python package dependencies
├── environment.yml   # Conda environment configuration
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Or using Conda
conda env create -f environment.yml
conda activate kaggle-env
```

### 2. Kaggle API Setup

Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).

Download competition data:
```bash
kaggle competitions download -c <competition-name> -p data/raw/
unzip data/raw/<competition-name>.zip -d data/raw/
```

### 3. Workflow & Best Practices

1. **Exploration**: Use `notebooks/01_eda.ipynb` for initial data exploration.
2. **Modularize Code**: Extract reusable feature engineering and data processing logic into `src/features.py` and `src/data.py`.
3. **Validation**: Ensure local cross-validation strategy matches the competition evaluation metric (`src/utils.py`).
4. **Submissions**: Output final submission files to `submissions/submission_<experiment_id>.csv`.
