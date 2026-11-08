# Homework 6 — Tree-Based Methods, XGBoost and SMOTE

DSCI-552 (Machine Learning for Data Science) — Homework 6.

## Structure

```
.
├── README.md
├── requirements.txt
├── Homework6.pdf
├── Homework 6 Data/
│   ├── aps_failure_training_set.csv
│   ├── aps_failure_test_set.csv
│   └── aps_failure_description.txt
├── Shelar_Kartik_HW6.ipynb
└── Shelar_Kartik_HW6.pdf
```

## Contents

- `Homework 6 Data/` — extracted contents of the course's `Homework 6 Data.zip`
  (the zip itself is not in the repository). The notebook reads the files from
  this folder.
  - APS Failure at Scania Trucks (UCI): 60,000 training rows (1,000 positive)
    and 16,000 test rows, 170 numeric features with missing values ("na").
    The first 20 lines of each CSV are a licence header.
- `Shelar_Kartik_HW6.ipynb` — full analysis:
  - Data preparation: median imputation, coefficient of variation, correlation
    matrix, scatter plots and box plots by class for the 13 features with the
    highest CV, and the class imbalance.
  - Random forest without and with class weights: confusion matrices, ROC,
    AUC, misclassification rates and out-of-bag error.
  - XGBoost with an L1 penalty (`reg_alpha`) chosen by 5-fold stratified
    cross-validation, without compensation and with SMOTE applied inside the
    cross-validation folds.
  - ISLR exercises 6.6.3, 6.6.5, 8.4.5 and 9.7.3.
- `Shelar_Kartik_HW6.pdf` — the notebook printed to PDF (19 pages).
- `Homework6.pdf` — assignment.

## Setup

```bash
pip install -r requirements.txt
```

Then open `Shelar_Kartik_HW6.ipynb` in Jupyter, keeping `Homework 6 Data/` in
the same folder. A full run takes about 4–5 minutes, mostly the two random
forests and the cross-validation searches for XGBoost (with and without SMOTE).
