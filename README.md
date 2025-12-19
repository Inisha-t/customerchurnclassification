# MAST6100 Classification Project (GitHub-ready)

This repo provides a reproducible pipeline for a classification task using:
- GLM (Logistic Regression)
- Random Forest
- SVM (RBF)
- Deep Learning (Keras MLP)

## 1) Dataset
Recommended dataset: **Telco Customer Churn** (Kaggle).

Download the CSV from Kaggle and place it here:
`data/raw/Telco-Customer-Churn.csv`

Common filename from Kaggle is:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`
If so, either rename it to `Telco-Customer-Churn.csv` or update `DATA_PATH` in `src/config.py`.

## 2) Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Run the pipeline
### A) Train classical ML models
```bash
python src/train_classical.py
```

### B) Train the deep learning model
```bash
python src/train_deep.py
```

### C) Evaluate saved models on the test set
```bash
python src/evaluate.py
```

Outputs:
- Saved models in `models/`
- Metrics summary in `reports/metrics.json`
- ROC curves in `reports/figures/`

## 4) Reproducibility
- Fixed random seeds in training scripts
- Train/test split is consistent across runs

## 5) What to submit
- Your report (DOCX/PDF)
- Slides (PPTX/PDF)
- This repo (ZIP or GitHub link)
