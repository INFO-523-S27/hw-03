"""
test_ds.py

Pytest-based tests for HW-03: Classification with Logistic Regression.

These tests assume the following functions are defined in ds.py:

- load_csv
- handle_missing_values
- select_features_and_target
- train_test_split_df
- create_categorical_column   <-- NEW
- standardize_numeric_columns
- encode_categorical_columns
- train_logistic_regression
- predict_probabilities
- predict_labels
- classification_metrics
- confusion_matrix_counts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ds


# ============================================================
# Helpers
# ============================================================

def _make_simple_classification_df() -> pd.DataFrame:
    """Create a tiny synthetic binary classification dataset."""
    data = {
        "x1": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "x2": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "y":  [0,   0,   0,   1,   1,   1],
    }
    return pd.DataFrame(data)


def _find_covid_dataset_path() -> Path | None:
    """
    Locate the COVID dataset if present in common course/repo locations.
    Skips integration tests if not found.
    """
    candidates = [
        Path("data") / "covid-19-symptoms-dataset.csv",
        Path("covid-19-symptoms-dataset.csv"),
        Path("/mnt/data") / "covid-19-symptoms-dataset.csv",  # dev/CI convenience
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ============================================================
# Tests for Part 1: Data Loading & Basic Cleaning
# ============================================================

def test_load_csv(tmp_path):
    """load_csv should correctly read a small CSV into a DataFrame."""
    csv_path = tmp_path / "tiny.csv"
    df_expected = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_expected.to_csv(csv_path, index=False)

    df_loaded = ds.load_csv(str(csv_path))

    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == df_expected.shape
    pd.testing.assert_frame_equal(df_loaded, df_expected)


def test_handle_missing_values_mean_mode():
    """handle_missing_values should fill numeric NaNs and categorical NaNs."""
    df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0], "cat": ["a", None, "a", "b"]})
    df_before = df.copy(deep=True)

    cleaned = ds.handle_missing_values(df)

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_before)

    # All missing values should be filled
    assert cleaned.isna().sum().sum() == 0

    # Numeric with mean strategy
    num_mean = df_before["num"].mean()
    assert cleaned.loc[2, "num"] == pytest.approx(num_mean)

    # Categorical with mode strategy
    cat_mode = df_before["cat"].mode(dropna=True).iloc[0]
    assert cleaned.loc[1, "cat"] == cat_mode


def test_handle_missing_values_median():
    """handle_missing_values should support median strategy for numeric."""
    df = pd.DataFrame({"num": [1.0, np.nan, 7.0], "cat": ["x", "y", None]})
    df_before = df.copy(deep=True)

    cleaned = ds.handle_missing_values(df, numeric_strategy="median")

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_before)

    assert cleaned.isna().sum().sum() == 0
    median_val = df_before["num"].median()
    assert cleaned.loc[1, "num"] == pytest.approx(median_val)


# ============================================================
# Tests for Part 2: Feature/Target Preparation & Splitting
# ============================================================

def test_select_features_and_target_basic():
    """select_features_and_target should return X (features) and y (target)."""
    df = pd.DataFrame({"x1": [0, 1, 2], "x2": [10, 20, 30], "y": [0, 1, 0]})
    df_before = df.copy(deep=True)

    X, y = ds.select_features_and_target(df, feature_cols=["x1", "x2"], target_col="y")

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_before)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ["x1", "x2"]
    assert y.name == "y"
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_train_test_split_df_shapes_and_indices():
    """train_test_split_df should split data into disjoint train and test sets."""
    df = _make_simple_classification_df()
    train_df, test_df = ds.train_test_split_df(df, target_col="y", test_size=0.33, random_state=0)

    assert len(train_df) + len(test_df) == len(df)
    assert len(train_df) > 0
    assert len(test_df) > 0

    assert set(train_df.index).isdisjoint(set(test_df.index))

    combined_idx = sorted(list(train_df.index) + list(test_df.index))
    assert combined_idx == sorted(df.index.tolist())


# ============================================================
# NEW: Test for creating a categorical column
# ============================================================

def test_create_categorical_column_age_groups():
    """create_categorical_column should add a binned categorical column without mutating input."""
    df = pd.DataFrame({"age": [5, 17, 18, 29, 30, 65]})
    df_before = df.copy(deep=True)

    out = ds.create_categorical_column(
        df,
        source_col="age",
        new_col="age_group",
        bins=[0, 18, 30, 200],
        labels=["child", "young_adult", "adult"],
        right=False,
    )

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_before)

    assert "age_group" in out.columns
    assert out["age_group"].isna().sum() == 0

    # right=False => [0,18), [18,30), [30,200)
    assert out.loc[0, "age_group"] == "child"         # 5
    assert out.loc[2, "age_group"] == "young_adult"   # 18
    assert out.loc[5, "age_group"] == "adult"         # 65


# ============================================================
# Tests for Part 3: Preprocessing
# ============================================================

def test_standardize_numeric_columns_mean_zero_std_one():
    """standardize_numeric_columns should make training columns mean~0 and std~1."""
    train_df = pd.DataFrame(
        {"num1": [1.0, 2.0, 3.0, 4.0], "num2": [10.0, 20.0, 30.0, 40.0], "other": ["a", "b", "a", "b"]}
    )
    test_df = pd.DataFrame({"num1": [2.0, 5.0], "num2": [15.0, 50.0], "other": ["a", "b"]})

    train_before = train_df.copy(deep=True)
    test_before = test_df.copy(deep=True)

    cols = ["num1", "num2"]
    train_std, test_std = ds.standardize_numeric_columns(train_df, test_df, cols)

    # Original unchanged (correctly checked)
    pd.testing.assert_frame_equal(train_df, train_before)
    pd.testing.assert_frame_equal(test_df, test_before)

    # Means approx 0 and std approx 1 on training set
    for col in cols:
        col_mean = float(train_std[col].mean())
        col_std = float(train_std[col].std(ddof=0))  # population std
        assert col_mean == pytest.approx(0.0, abs=1e-6)
        assert col_std == pytest.approx(1.0, rel=1e-3)

    assert np.issubdtype(train_std["num1"].dtype, np.floating)
    assert np.issubdtype(test_std["num1"].dtype, np.floating)

    assert "other" in train_std.columns
    assert list(train_std["other"]) == list(train_before["other"])


def test_standardize_numeric_columns_constant_column():
    """If a column is constant in training, standardized values can be all zeros."""
    train_df = pd.DataFrame({"num1": [5.0, 5.0, 5.0], "num2": [1.0, 2.0, 3.0]})
    test_df = pd.DataFrame({"num1": [5.0, 5.0], "num2": [4.0, 5.0]})

    train_std, test_std = ds.standardize_numeric_columns(train_df, test_df, ["num1", "num2"])

    assert np.allclose(train_std["num1"].values, 0.0)
    assert np.allclose(test_std["num1"].values, 0.0)


def test_encode_categorical_columns_train_test_alignment():
    """encode_categorical_columns should align train/test dummies."""
    train_df = pd.DataFrame({"color": ["red", "blue", "green", "red"], "value": [1, 2, 3, 4]})
    test_df = pd.DataFrame({"color": ["blue", "red"], "value": [10, 20]})

    train_before = train_df.copy(deep=True)
    test_before = test_df.copy(deep=True)

    train_enc, test_enc = ds.encode_categorical_columns(train_df, test_df, columns=["color"], drop_first=False)

    # Original unchanged
    pd.testing.assert_frame_equal(train_df, train_before)
    pd.testing.assert_frame_equal(test_df, test_before)

    assert "color" not in train_enc.columns
    assert "color" not in test_enc.columns

    dummy_cols = [c for c in train_enc.columns if c.startswith("color_")]
    assert dummy_cols, "No dummy columns created for 'color'"

    for col in dummy_cols:
        assert col in test_enc.columns

    assert "value" in train_enc.columns
    assert "value" in test_enc.columns
    assert list(train_enc["value"]) == list(train_before["value"])


def test_encode_categorical_columns_empty_list_noop():
    """encode_categorical_columns should handle columns=[] gracefully (no changes)."""
    train_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    test_df = pd.DataFrame({"a": [5], "b": [6]})

    train_enc, test_enc = ds.encode_categorical_columns(train_df, test_df, columns=[], drop_first=False)

    pd.testing.assert_frame_equal(train_enc, train_df)
    pd.testing.assert_frame_equal(test_enc, test_df)


# ============================================================
# Tests for Part 4: Model Training, Prediction & Evaluation
# ============================================================

def test_train_logistic_regression_and_predict():
    """train_logistic_regression should return a usable model."""
    df = _make_simple_classification_df()
    X = df[["x1", "x2"]]
    y = df["y"]

    X_train, X_test = X.iloc[:4], X.iloc[4:]
    y_train = y.iloc[:4]

    model = ds.train_logistic_regression(X_train, y_train)

    assert hasattr(model, "predict_proba")

    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    train_acc = (train_pred == y_train.values).mean()
    assert train_acc >= 0.75


def test_predict_probabilities_shape_and_range():
    """predict_probabilities should return 1D probabilities in [0, 1]."""
    df = _make_simple_classification_df()
    X = df[["x1", "x2"]]
    y = df["y"]

    X_train, X_test = X.iloc[:4], X.iloc[4:]
    y_train = y.iloc[:4]

    model = ds.train_logistic_regression(X_train, y_train)
    proba = ds.predict_probabilities(model, X_test)

    assert isinstance(proba, np.ndarray)
    assert proba.ndim == 1
    assert proba.shape[0] == len(X_test)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_labels_thresholding():
    """predict_labels should threshold probabilities into 0/1 labels."""
    proba = np.array([0.1, 0.4, 0.5, 0.9])

    labels_default = ds.predict_labels(proba)
    assert np.array_equal(labels_default, np.array([0, 0, 1, 1]))

    labels_high = ds.predict_labels(proba, threshold=0.8)
    assert np.array_equal(labels_high, np.array([0, 0, 0, 1]))


def test_classification_metrics_basic():
    """classification_metrics should compute accuracy, precision, recall, f1."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = ds.classification_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    assert metrics["accuracy"] == pytest.approx(0.75, rel=1e-6)
    assert metrics["precision"] == pytest.approx(2 / 3, rel=1e-6)
    assert metrics["recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["f1"] == pytest.approx(0.8, rel=1e-6)


def test_confusion_matrix_counts_basic():
    """confusion_matrix_counts should return tn, fp, fn, tp."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    cm = ds.confusion_matrix_counts(y_true, y_pred)

    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert cm["tn"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 0
    assert cm["tp"] == 2
