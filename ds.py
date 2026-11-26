"""
ds.py

Homework 3: Classification with Logistic Regression

You will implement a complete binary classification workflow, suitable for
datasets like `covid-survey.csv` that contain mixed numeric and categorical
features.

Complete all questions by filling in the functions below.

- Do NOT rename any existing functions.
- Do NOT change function signatures.
- You may add small helper functions if needed, but do not change the
  public API defined here.
- This file will be graded using pytest and pylint.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Part 1: Data Loading & Basic Cleaning
# ============================================================

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to a CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Notes
    -----
    - Use `pd.read_csv`.
    - Do not modify the CSV contents here; just read the file.
    """
    raise NotImplementedError("Q1: Implement load_csv().")


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "mean",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    This is especially relevant for survey-style data like `covid-survey.csv`,
    where respondents may skip questions.

    Suggested behavior:
    - For numeric columns:
        * If `numeric_strategy == "mean"`, fill NaNs with the column mean.
        * If `numeric_strategy == "median"`, fill NaNs with the column median.
    - For non-numeric columns:
        * If `categorical_strategy == "mode"`, fill NaNs with the most
          frequent value (mode).

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    numeric_strategy : {"mean", "median"}, default "mean"
        Strategy to fill missing values in numeric columns.
    categorical_strategy : {"mode"}, default "mode"
        Strategy to fill missing values in non-numeric columns.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values handled.

    Notes
    -----
    - You should NOT modify `df` in-place; return a cleaned copy.
    - If a column is entirely missing (all NaN), you may leave it as-is
      or drop it in your own workflow (but do not drop inside this function).
    """
    raise NotImplementedError("Q2: Implement handle_missing_values().")


# ============================================================
# Part 2: Feature/Target Preparation & Splitting
# ============================================================
def create_categorical_column(
    df: pd.DataFrame,
    source_col: str,
    new_col: str,
    bins: Iterable[float],
    labels: Iterable[str] | None = None,
    right: bool = False,
) -> pd.DataFrame:
    """
    Create a categorical column from a numeric column using binning.

    Example (common for COVID survey / clinical data):
        - Turn numeric age into age_group categories, then one-hot encode.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    source_col : str
        Name of the numeric column to bin (e.g., "age").
    new_col : str
        Name of the new categorical column to create (e.g., "age_group").
    bins : Iterable[float]
        Bin edges to pass to pd.cut.
        Example: [0, 18, 30, 45, 60, 200]
    labels : Iterable[str] or None, default None
        Labels for the bins. If None, interval labels are used.
    right : bool, default False
        Whether bins include the rightmost edge (pd.cut right=...).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the added categorical column.

    Notes
    -----
    - Do NOT modify df in-place; return a copy.
    - Uses pd.cut to create categories.
    - This function should raise a KeyError if source_col is missing.
    """
    df_out = df.copy()

    if source_col not in df_out.columns:
        raise KeyError(f"Column not found: {source_col}")

    df_out[new_col] = pd.cut(
        df_out[source_col],
        bins=list(bins),
        labels=list(labels) if labels is not None else None,
        right=right,
        include_lowest=True,
    )

    return df_out


def select_features_and_target(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into feature matrix X and target vector y.

    This is where you define which columns of the survey (e.g., symptoms,
    demographics, behavior) are used to predict the target (e.g., a binary
    COVID outcome).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    feature_cols : Iterable[str]
        Column names to use as features.
    target_col : str
        Name of the target (label) column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing only the columns in `feature_cols`.
    y : pd.Series
        Target vector containing the values from `target_col`.

    Notes
    -----
    - Do not modify `df` in-place.
    - Preserve the original index.
    - You may assume the target is binary (0/1 or equivalent).
    """
    raise NotImplementedError("Q3: Implement select_features_and_target().")


def train_test_split_df(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Name of the target column (used for stratification, if desired).
    test_size : float, default 0.2
        Fraction of rows to put in the test set (between 0 and 1).
    random_state : int, default 42
        Random seed for reproducible splits.

    Returns
    -------
    train_df : pd.DataFrame
        Training subset of the data.
    test_df : pd.DataFrame
        Test subset of the data.

    Notes
    -----
    - You may use `sklearn.model_selection.train_test_split` with
      `stratify=df[target_col]` to preserve class balance.
    - Do not shuffle with a different random seed inside the function.
    """
    raise NotImplementedError("Q4: Implement train_test_split_df().")


# ============================================================
# Part 3: Preprocessing for Modeling
# ============================================================

def standardize_numeric_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize numeric columns in train and test sets using training statistics.

    For each column in `columns`:
        - Compute mean and standard deviation using ONLY `train_df`.
        - Transform both train and test columns:
            z = (x - mean) / std

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data.
    columns : Iterable[str]
        Names of numeric columns to standardize.

    Returns
    -------
    train_std : pd.DataFrame
        Copy of train_df with standardized columns.
    test_std : pd.DataFrame
        Copy of test_df with standardized columns.

    Notes
    -----
    - Do NOT modify the original train_df/test_df in-place.
    - If a column has zero variance in the training set, you may set
      the standardized values to 0 for that column.
    - This function is intended for continuous/numeric features such as
      age, symptom scores, or counts from a survey.
    """
    raise NotImplementedError("Q5: Implement standardize_numeric_columns().")


def encode_categorical_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Iterable[str],
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode the same categorical columns in train and test sets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data.
    columns : Iterable[str]
        Names of categorical columns to encode.
    drop_first : bool, default False
        If True, drop the first dummy column for each categorical variable.

    Returns
    -------
    train_encoded : pd.DataFrame
        Training data with one-hot encoded columns.
    test_encoded : pd.DataFrame
        Test data with one-hot encoded columns.

    Notes
    -----
    - Use `pd.get_dummies` on the training data first, then align
      the test data to the same columns (adding any missing columns with 0).
    - Do NOT modify the original train_df/test_df in-place.
    - This function is appropriate for coding survey responses such as
      gender, mask usage, job type, political affiliation, etc.
    """
    raise NotImplementedError("Q6: Implement encode_categorical_columns().")


# ============================================================
# Part 4: Model Training
# ============================================================

def train_logistic_regression(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> object:
    """
    Train a binary logistic regression classifier.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels (binary: 0/1).
    C : float, default 1.0
        Inverse of regularization strength (as in scikit-learn).
    max_iter : int, default 1000
        Maximum number of iterations for the solver.

    Returns
    -------
    model : object
        Trained model object (e.g., a scikit-learn LogisticRegression instance).

    Notes
    -----
    - You may use `sklearn.linear_model.LogisticRegression`.
    - The function should only train the model and return it.
    - Do not compute metrics here; that belongs in evaluation functions.
    """
    raise NotImplementedError("Q7: Implement train_logistic_regression().")


# ============================================================
# Part 5: Prediction & Evaluation
# ============================================================

def predict_probabilities(
    model: object,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Predict class probabilities for the positive class (label 1).

    Parameters
    ----------
    model : object
        Trained classifier with a `predict_proba`-like interface.
    X : pd.DataFrame or np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of probabilities for the positive class.

    Notes
    -----
    - If using scikit-learn, you can call `model.predict_proba(X)[:, 1]`.
    """
    raise NotImplementedError("Q8: Implement predict_probabilities().")


def predict_labels(
    proba: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert predicted probabilities into binary class labels.

    Parameters
    ----------
    proba : np.ndarray
        1D array of probabilities for the positive class.
    threshold : float, default 0.5
        Classification threshold. Values >= threshold are predicted as 1,
        values < threshold as 0.

    Returns
    -------
    np.ndarray
        1D array of predicted class labels (0 or 1).

    Notes
    -----
    - This function should NOT depend on any specific model library.
    - It should work on any NumPy array of probabilities.
    """
    raise NotImplementedError("Q9: Implement predict_labels().")


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Compute basic classification metrics.

    Suggested metrics:
    - 'accuracy'
    - 'precision'
    - 'recall'
    - 'f1'

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name -> value.

    Notes
    -----
    - You may use scikit-learn's metric functions (e.g.,
      `accuracy_score`, `precision_score`, etc.), or implement them
      yourself using basic Python / NumPy.
    - For precision/recall/F1, you should handle the case where there
      are no positive predictions (e.g., `zero_division=0` if you use
      scikit-learn).
    """
    raise NotImplementedError("Q10: Implement classification_metrics().")


def confusion_matrix_counts(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, int]:
    """
    Compute confusion matrix counts for a binary classifier.

    Specifically, return:
    - 'tn': true negatives
    - 'fp': false positives
    - 'fn': false negatives
    - 'tp': true positives

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_pred : array-like
        Predicted binary labels (0/1).

    Returns
    -------
    dict[str, int]
        A dictionary with keys 'tn', 'fp', 'fn', 'tp'.

    Notes
    -----
    - You may use `sklearn.metrics.confusion_matrix` or compute counts directly.
    - In a COVID survey context, interpreting false positives vs false negatives
      is especially important; you will do that in your notebook, not here.
    """
    raise NotImplementedError("Q11: Implement confusion_matrix_counts().")
