"""Data loading and preprocessing for the credit card fraud dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load the raw creditcard.csv dataset."""
    df = pd.read_csv(path)
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Prepare features and labels for modelling.

    Steps:
    - Log-transform Amount to reduce right skew
    - Drop the raw Amount and Time columns after transformation
    - Standardise all features with StandardScaler (fit on train only)
    - Stratified 80/20 train-test split

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    df = df.copy()

    # Log-transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
