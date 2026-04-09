"""Class-imbalance handling strategies."""

import numpy as np
from imblearn.over_sampling import SMOTE


STRATEGIES = ["none", "class_weight", "smote", "smote_cw"]


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Oversample the minority class with SMOTE."""
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def get_class_weight(strategy: str) -> dict | None:
    """Return the class_weight argument for sklearn estimators."""
    if strategy in ("class_weight", "smote_cw"):
        return "balanced"
    return None


def prepare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = 42,
) -> tuple:
    """
    Apply the requested imbalance strategy to the training data.

    Parameters
    ----------
    strategy : one of 'none', 'class_weight', 'smote', 'smote_cw'

    Returns
    -------
    X_res, y_res : resampled arrays (unchanged for 'none' / 'class_weight')
    sample_weight : None (class_weight is passed directly to estimators)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    if strategy in ("smote", "smote_cw"):
        X_res, y_res = apply_smote(X_train, y_train, random_state)
    else:
        X_res, y_res = X_train.copy(), y_train.copy()

    return X_res, y_res
