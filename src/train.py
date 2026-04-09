"""Training, cross-validation, and threshold tuning."""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score


def tune(
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
):
    """
    Randomised hyperparameter search with stratified CV.

    Returns the best estimator (already fitted on full X_train / y_train).
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_splitter,
        refit=True,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Find the decision threshold that maximises F1 on a validation set.

    NOTE: X_val / y_val must come from a held-out split — never the test set.
    """
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])
