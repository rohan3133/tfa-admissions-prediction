# TFA Admissions Prediction (R)

Predicting whether an applicant completes the admissions process, comparing five classification approaches on ~75,000 anonymized applicant records from a TFA (Teach For America) admissions dataset.

## Overview

| | |
|---|---|
| **Goal** | Predict `Completed Admissions Process` (1 = completed, 0 = did not) |
| **Data** | 74,839 applicant records, 39 fields, application years 2015 and 2016 |
| **Class balance** | 80.7% completed / 19.3% not completed (imbalanced) |
| **Language** | R |

**Models**

- K-Nearest Neighbors (tuned over k)
- Naive Bayes (baseline and tuned with 10-fold cross-validation)
- C5.0 decision tree (baseline and tuned with 10-fold cross-validation)
- Artificial neural network (`neuralnet`, single hidden layer)
- Support Vector Machine: RBF kernel tuned with repeated cross-validation and ROC-based selection, plus linear-kernel, kernel-comparison and SMOTE experiments

**Techniques:** 80/20 train-test splits, cross-validated hyperparameter tuning (`caret`), class-imbalance handling with SMOTE, evaluation with confusion matrices (accuracy, sensitivity, specificity) and ROC curves.

## Data

`data/tfa_data.csv` contains one row per applicant. Fields include cumulative GPA, major/minor (raw and cleaned), a STEM flag, undergraduate school (masked IDs) and selectivity, essay length / unique-word / topic / sentiment features, application dates, region preference, and funnel steps (event attended, met, invited, responded). The records are anonymized: there are no names or contact details, and universities appear as masked IDs.

## Repository structure

```
tfa-admissions-prediction/
├── data/
│   └── tfa_data.csv
└── scripts/
    ├── complete_analysis.R          # KNN, Naive Bayes, C5.0, ANN and SVM in one script
    ├── knn_nb_decision_tree.R       # earlier KNN / Naive Bayes / decision-tree pipeline
    ├── decision_tree_c50.R          # standalone C5.0 with cross-validation and tuning
    ├── verify_decision_tree.R       # sanity checks on the decision-tree logic
    └── svm_experiments/
        ├── svm_caret_rbf.R          # RBF SVM via caret
        ├── svm_rbf_baseline.R       # standalone RBF SVM
        ├── svm_rbf_optimized.R      # RBF SVM with tuning plots
        ├── svm_linear.R             # linear SVM (memory-optimized sampling)
        ├── svm_kernel_comparison.R  # linear (vanilladot) vs RBF (rbfdot)
        ├── svm_smote.R              # SVM with SMOTE to address class imbalance
        └── svm_quick_pipeline.R     # fast SVM pipeline (QUICK_SVM_RUN=1 for a small subset)
```

## How to run

1. Install R packages: `caret`, `class`, `e1071`, `C50`, `neuralnet`, `fastDummies`, `kernlab`, `smotefamily`, `pROC`, `doParallel`, `plyr`, `dplyr`, `ggplot2`.
2. Set the working directory to the repository root, since scripts read `data/tfa_data.csv`.
3. Run a script, for example:

```r
source("scripts/complete_analysis.R")
```

The SVM scripts are computationally heavy; `svm_quick_pipeline.R` has a quick mode: set the environment variable `QUICK_SVM_RUN=1` to train on a small subset.

## Author

Rohan Satishkumar, MS Marketing Analytics, Illinois Institute of Technology
