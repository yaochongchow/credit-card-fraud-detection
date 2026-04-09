"""Model definitions and hyperparameter search spaces."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_model(name: str, class_weight=None, random_state: int = 42):
    """
    Instantiate a classifier by name.

    Parameters
    ----------
    name         : 'lr' | 'rf' | 'svm' | 'xgb'
    class_weight : passed to models that support it ('balanced' or None)
    """
    models = {
        "lr": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm": SVC(
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "xgb": XGBClassifier(
            tree_method="hist",
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=random_state,
        ),
    }
    if name not in models:
        raise ValueError(f"name must be one of {list(models)}")
    return models[name]


def get_param_grid(name: str) -> dict:
    """Return the randomised search parameter grid for a given model."""
    grids = {
        "lr": {
            "C": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
            "fit_intercept": [True, False],
            "tol": [1e-4, 1e-3, 1e-2],
            "max_iter": [500, 1000, 2000],
        },
        "rf": {
            "n_estimators": [200, 400, 600, 800, 1000],
            "max_depth": [8, 12, 16, 20, 24],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
        },
        "svm": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "tol": [1e-3],
        },
        "xgb": {
            "n_estimators": [300, 500, 800, 1200],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.02, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10, 20],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "gamma": [0.0, 0.1, 0.5, 1.0],
        },
    }
    return grids[name]
