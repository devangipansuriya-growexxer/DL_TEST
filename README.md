# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:** Devangi Pansuriya  
**Student ID:** 1121  
**Submission date:** 27/03/2026  


# Deep Learning Readmission Predictor

## Problem
Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

## My model

### Architecture:
I implemented a 3-layer Multilayer Perceptron (MLP) with ReLU activations and Dropout (0.3). Given the relatively small dataset size (3,800 records), an extremely deep network would likely overfit. The dropout layers act as regularization. I used Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) for final binary classification.

### Key preprocessing decisions:
Missing values were removed using dropna(), and categorical features were converted using one-hot encoding. All features were then scaled using StandardScaler to stabilize training for the neural network.

### How I handled class imbalance:
Class imbalance (91% vs 9%) was handled using weighted loss (BCEWithLogitsLoss with pos_weight=10) to force the model to pay more attention to the minority class and improve recall.




## Model Description
The model is a simple 3-layer Multilayer Perceptron (MLP) built with PyTorch, trained on tabular data to classify binary readmission outcomes within 30 days. It uses ReLU activations and Dropout layers to prevent overfitting.


## How to Run
Initialize a Python environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Train the model:


Run the inference script:
```bash
python src/predict.py --input data/test.csv
```
This script will load `data/test.csv`, apply preprocessing, run the dummy model forward pass, and output `predictions.csv` containing patient IDs, prediction probabilities


## Repository structure


```text
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb
├── src/
│   └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md
```

---

## Model Performance (Validation Set)

| Metric | Value |
|--------|------|
| AUROC | 0.5796 |
| F1 (minority class) | 0.5616 |
| Precision (minority class) | 0.7321 |
| Recall (minority class) | 0.4514 |
| Decision threshold used | 0.55 |

---


## Limitations and Honest Assessment

The current model is limited by simple preprocessing choices like dropping missing values and one-hot encoding, which may remove useful signal and create sparse features. The neural network is relatively small and trained on a highly imbalanced dataset, leading to weak discrimination performance (low precision and modest AUROC). Although class weighting improves recall, the model still over-predicts positives, which could cause high false alarms in real-world use. In production, it may also fail under dataset shift across hospitals or patient populations due to lack of external validation. With more time, I would improve missing data handling, try stronger models like gradient boosting, and focus on better calibration and imbalance techniques.