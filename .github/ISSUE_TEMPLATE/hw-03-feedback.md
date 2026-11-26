name: HW-03 Feedback
description: Grading rubric for Homework 03 — Binary Classification
title: "HW-03 Feedback"
labels: ["grading", "hw-03"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        ## HW-03: Binary Classification Feedback  
        **Student Repository:** _auto-filled by GitHub Classroom_  
        **Autograder Status:** ✅ / ❌  
        **Notebook Reviewed:** ✅ / ❌  

        This rubric evaluates:
        - Correct implementation of the classification pipeline in `ds.py`
        - Proper use of preprocessing, modeling, and evaluation
        - Conceptual understanding via the notebook
        - Ethical reasoning about model errors

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 1: Data Loading & Cleaning (10 points)

  - type: checkboxes
    id: part1
    attributes:
      options:
        - label: "load_csv correctly loads CSV data"
        - label: "handle_missing_values fills numeric NaNs correctly"
        - label: "handle_missing_values fills categorical NaNs correctly"
        - label: "Original DataFrame not modified in-place"
        - label: "No hardcoded values used"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 2: Feature/Target Selection & Splitting (10 points)

  - type: checkboxes
    id: part2
    attributes:
      options:
        - label: "select_features_and_target returns correct X and y"
        - label: "train_test_split_df returns valid train/test sets"
        - label: "Train/test indices are disjoint"
        - label: "Stratification preserves class balance (if used)"
        - label: "Random state handled correctly"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 3: Preprocessing (20 points)

  - type: checkboxes
    id: part3
    attributes:
      options:
        - label: "standardize_numeric_columns uses training-only statistics"
        - label: "Standardized columns have mean ≈ 0 and std ≈ 1"
        - label: "Constant-value columns handled correctly"
        - label: "encode_categorical_columns correctly one-hot encodes"
        - label: "Train/test dummy alignment preserved"
        - label: "drop_first functionality works correctly"
        - label: "Original DataFrames not modified in-place"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 4: Model Training (10 points)

  - type: checkboxes
    id: part4
    attributes:
      options:
        - label: "train_logistic_regression returns a trained model"
        - label: "Model exposes predict_proba"
        - label: "Training accuracy is reasonable"
        - label: "Hyperparameters handled correctly"
        - label: "No metrics computed inside training function"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 5: Prediction & Thresholding (10 points)

  - type: checkboxes
    id: part5
    attributes:
      options:
        - label: "predict_probabilities returns valid shape"
        - label: "Probabilities are within [0, 1]"
        - label: "predict_labels correctly applies threshold"
        - label: "Custom thresholds work correctly"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 6: Evaluation Metrics & Confusion Matrix (10 points)

  - type: checkboxes
    id: part6
    attributes:
      options:
        - label: "classification_metrics returns all required metrics"
        - label: "Accuracy computed correctly"
        - label: "Precision computed correctly"
        - label: "Recall computed correctly"
        - label: "F1-score computed correctly"
        - label: "Zero-division handled safely"
        - label: "confusion_matrix_counts returns correct tn/fp/fn/tp"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 7: Notebook Analysis & Interpretation (20 points)

  - type: checkboxes
    id: part7
    attributes:
      options:
        - label: "Used ds.py functions (not raw sklearn shortcuts)"
        - label: "Clearly described features vs target"
        - label: "Explained class imbalance"
        - label: "Interpreted evaluation metrics correctly"
        - label: "Compared multiple thresholds"
        - label: "Explained threshold effects on precision/recall"
        - label: "Correctly interpreted confusion matrix"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 8: Ethical Reflection (10 points)

  - type: checkboxes
    id: part8
    attributes:
      options:
        - label: "Clearly explained false negatives"
        - label: "Clearly explained false positives"
        - label: "Identified which error is more harmful"
        - label: "Connected to real-world public health consequences"
        - label: "Response shows critical reasoning"

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Part 9: Software Quality & Professionalism (10 points)

  - type: checkboxes
    id: part9
    attributes:
      options:
        - label: "All pytest tests passed"
        - label: "pylint score ≥ 9.0"
        - label: "Clean, readable implementations"
        - label: "No unnecessary code duplication"
        - label: "Sensibly named variables and functions"
        - label: "No test-specific hardcoding detected"

  - type: markdown
    attributes:
      value: |
        ---
        ## ❌ Policy Violations (Auto Deductions – 5 pts each)

  - type: checkboxes
    id: deductions
    attributes:
      options:
        - label: "Modified files other than ds.py"
        - label: "Renamed required functions"
        - label: "Deleted or altered docstrings"
        - label: "Merged feedback pull request"
        - label: "Hardcoded values to pass tests"

  - type: markdown
    attributes:
      value: |
        ---
        ## 📝 Final Comments

  - type: textarea
    id: comments
    attributes:
      label: Instructor / TA Comments
      description: Provide specific feedback, strengths, and areas for improvement.
      placeholder: |
        - Strengths:
        - Areas for Improvement:
        - Modeling Observations:
        - Ethical Reasoning Comments:
    validations:
      required: false

  - type: markdown
    attributes:
      value: |
        ---
        ## ✅ Final Score (Out of 100)

  - type: input
    id: final_score
    attributes:
      label: Final Score
      description: Enter the final numeric score.
      placeholder: "e.g., 87"
    validations:
      required: true
