# HW-03: Binary Classification with Logistic Regression

## Objectives

The learning objectives of this assignment are to:

1. Implement a complete **binary classification pipeline** in Python.
2. Perform **data cleaning and preprocessing** for real-world survey data.
3. Apply **feature scaling and categorical encoding** correctly.
4. Train and evaluate a **logistic regression classifier**.
5. Compute and interpret **classification metrics**, including:
   - Accuracy
   - Precision
   - Recall
   - F1-score
6. Analyze **confusion matrix outcomes** and threshold effects.
7. Reflect on the **ethical consequences of model errors**.
8. Write **testable, modular Python code** that passes automated unit tests.
9. Follow professional **software quality standards** using automated linting.

---

## Setup Your Environment

Before starting, set up a Python environment. You will need the following tools and libraries:

- [Python (version 3.9 or higher)](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pytest](https://docs.pytest.org/)
- [pylint](https://pylint.pycqa.org/)

---

### Installing Dependencies

Install the required dependencies with:

```bash
pip install -r requirements.txt
````

---

## Check Out the Starter Code

When you accepted the assignment, GitHub created a clone of the assignment
template for you at:

```
https://github.com/INFO-511-S25/classification-hw-03-<your-username>
```

It also set up a separate `feedback` branch and a
[feedback pull request](https://docs.github.com/en/education/manage-coursework-with-github-classroom/teach-with-github-classroom/leave-feedback-with-pull-requests)
that will allow the instructional team to give you feedback on your work.

---

### Codespaces (Recommended)

**Codespaces are enabled.**

To launch a Codespace:

1. Click the green **Code** button
2. Select the **Codespaces** tab
3. Click **Create codespace on main**

This provides a fully configured environment with all dependencies installed.

---

### Local Machine (Alternative)

If you prefer working locally, clone the repository:

```bash
git clone https://github.com/INFO-511-S25/classification-hw-03-<your-username>.git
cd classification-hw-03-<your-username>
```

You should do all your work in the default branch, `main`.

---

## File Descriptions

### ✅ `ds.py` (This is the only file you will modify)

This file contains all required functions for the HW-03 binary classification workflow.

### **Part 1: Data Loading & Cleaning**

* `load_csv(path)`
* `handle_missing_values(df, numeric_strategy="mean", categorical_strategy="mode")`

### **Part 2: Feature/Target Selection & Splitting**

* `select_features_and_target(df, feature_cols, target_col)`
* `create_categorical_column(df, source_col, new_col, bins, labels, right)`
* `train_test_split_df(df, target_col, test_size=0.2, random_state=42)`

### **Part 3: Preprocessing for Modeling**

* `standardize_numeric_columns(train_df, test_df, columns)`
* `encode_categorical_columns(train_df, test_df, columns, drop_first=False)`

### **Part 4: Model Training**

* `train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000)`

### **Part 5: Prediction & Evaluation**

* `predict_probabilities(model, X)`
* `predict_labels(proba, threshold=0.5)`
* `classification_metrics(y_true, y_pred)`
* `confusion_matrix_counts(y_true, y_pred)`

Each function includes a **clear docstring specifying expected behavior**.

---

### ✅ `test_ds.py` (DO NOT MODIFY)

This file contains the **automated pytest tests** used for grading.

* You **must not modify** this file.
* Passing this file locally means your autograder will pass on GitHub.

---

### ✅ `hw-03-classification.ipynb` (Analysis Notebook)

This is where you will:

* Load the COVID syptoms dataset
* Clean and preprocess the data
* Train your classification model
* Evaluate results
* Perform threshold analysis
* Reflect on ethical implications

⚠️ This notebook **must run top-to-bottom without errors.**

---

### ✅ `data/`

Contains:

* `covid-19-symptoms-dataset.csv` (classification dataset)

⚠️ **Do NOT modify the dataset.**

---

### ✅ `requirements.txt`

Lists all required Python dependencies for HW-03.

---

### ✅ `.gitignore`

Prevents virtual environments, cache files, OS junk, and generated data from being committed.

---

## Instructions

1. Open `ds.py` and implement **every function**.
2. Follow the behavior described in each function’s **docstring**.
3. **Do not rename** any functions.
4. **Do not modify** any files other than `ds.py`.
5. Once `ds.py` passes all tests, complete the **classification notebook**.

---

## Running Tests

Run the full test suite with:

```bash
pytest
```

* Initially, many tests will fail.
* Once your implementation is correct, **all tests will pass**.

Example output when all tests pass:

```plaintext
============================= test session starts ==============================
collected 11 items

test_ds.py ...........                                      [100%]

============================== 11 passed in 0.12s ===============================
```

---

## Code Quality Checks (Required)

Run pylint to verify code style:

```bash
pylint ds.py
```

A perfect score looks like:

```plaintext
--------------------------------------------------------------------
Your code has been rated at 10.00/10
```

---

## Submission Instructions

1. Commit your completed `ds.py` file:

```bash
git add ds.py
git commit -m "Completed HW-03 Classification Pipeline"
git push
```

2. Make sure your notebook is saved and committed:

```bash
git add hw-03-classification.ipynb
git commit -m "Completed HW-03 Classification Analysis"
git push
```

3. **Do NOT:**

* Modify `test_ds.py`
* Modify `requirements.txt`
* Modify files in `data/`
* Merge the `feedback` pull request

4. Verify that your GitHub Actions checks are ✅ green.

---

## Grading

### ✅ Points Allocation

* **70 points** — All automated correctness (`pytest`) tests pass
* **10 points** — All automated style (`pylint`) checks pass
* **20 points** — Notebook analysis & interpretation:

  * Correct use of preprocessing
  * Appropriate model training
  * Correct interpretation of metrics
  * Threshold analysis
  * Ethical reflection

---

### ❌ Deductions

You will lose **5 points each** for:

* Modifying any file other than `ds.py`
* Renaming required functions
* Deleting or modifying docstrings
* Merging the `feedback` pull request
* Hardcoding values to pass specific test cases
* Modifying the dataset

---

✅ **This assignment prepares you directly for real-world machine learning workflows.**
Take the time to write clean, correct, and professional classification code.

Good luck—and model responsibly! 🤖📊
