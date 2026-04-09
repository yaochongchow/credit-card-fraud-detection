# Credit Card Fraud Detection

Machine learning models for detecting fraudulent credit card transactions on a highly imbalanced dataset. Four classifiers are benchmarked across multiple class-balancing strategies, with GPU acceleration via cuML and XGBoost.

## Models

| Notebook | Model |
|---|---|
| `LogisticRegression.ipynb` | Logistic Regression (cuML) |
| `RandomForrest.ipynb` | Random Forest (cuML) |
| `SVM.ipynb` | Support Vector Machine (cuML) |
| `XGBOOST.ipynb` | XGBoost (GPU) |

## Dataset

- **File:** `creditcard.csv`
- **Features:** 30 input features + binary `Class` target (0 = legitimate, 1 = fraud)
- **Split:** 80/20 train-test with stratification (`random_state=42`)
- **Challenge:** Highly imbalanced — fraud is the minority class

## Balancing Strategies

Each model is trained and evaluated under four conditions:

1. **None** — no resampling
2. **Class Weight** — cost-sensitive learning (with random undersampling for LR)
3. **SMOTE** — synthetic minority oversampling
4. **SMOTE + Class Weight** — combined approach

## Results Summary

### ROC-AUC

| Model | None | Class Weight | SMOTE | SMOTE + CW |
|---|---|---|---|---|
| Logistic Regression | 0.9840 | 0.9745 | 0.9772 | 0.9772 |
| Random Forest | 0.9805 | 0.9840 | 0.9805 | 0.9759 |
| SVM | 0.9725 | 0.9745 | 0.9774 | 0.9675 |
| XGBoost | 0.9776 | 0.9776 | **0.9845** | **0.9845** |

### PR-AUC (Precision-Recall, more meaningful for imbalanced data)

| Model | None | Class Weight | SMOTE | SMOTE + CW |
|---|---|---|---|---|
| Logistic Regression | 0.7152 | 0.4837 | 0.8004 | 0.8004 |
| Random Forest | 0.8827 | **0.8867** | 0.8808 | 0.8845 |
| SVM | 0.3904 | 0.7380 | 0.7351 | 0.6973 |
| XGBoost | 0.8777 | 0.8777 | 0.8786 | 0.8786 |

**Best overall:** XGBoost + SMOTE (ROC-AUC: 0.9845), Random Forest + Class Weight (PR-AUC: 0.8867)

## Key Findings

- **Tree-based models** (RF, XGBoost) consistently outperform LR and SVM across both metrics
- **SMOTE** is most effective for XGBoost and SVM; Random Forest responds better to class weighting
- **Entropy split criterion** outperformed Gini in all Random Forest runs
- **Decision threshold of 0.3** (vs. default 0.5) was tested across all models to prioritize fraud recall
- All models evaluated with **StratifiedKFold cross-validation** (3–5 folds)

## Requirements

- Python 3.x
- `scikit-learn`, `imbalanced-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`
- GPU: `cuML` (RAPIDS) for accelerated Logistic Regression, Random Forest, and SVM
