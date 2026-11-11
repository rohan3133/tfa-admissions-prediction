# ============================================================================
# TFA ADMISSIONS ANALYSIS - SVM MODEL ONLY
# ============================================================================
# Target: Completed.Admissions.Process (0/1)
# ============================================================================

# Load necessary libraries
library(caret)
library(kernlab)    # For SVM
library(e1071)      # Dependency for caret

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================

cat("\nLoading data...\n")
# Load dataset
if (!file.exists("data/tfa_data.csv")) {
  stop("tfa_data.csv not found in the working directory.")
}
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

# Remove rows with any NA values
tfa_df <- na.omit(tfa_df)

# Safely remove 'personid' and 'appyear' if they exist
if ("personid" %in% colnames(tfa_df)) {
  tfa_df$personid <- NULL
}
if ("appyear" %in% colnames(tfa_df)) {
  tfa_df$appyear <- NULL
}

# Target variable setup
target_col <- "Completed.Admissions.Process"
if (!(target_col %in% colnames(tfa_df))) {
  stop("Column 'Completed.Admissions.Process' not found in tfa_df.")
}

# Prepare SVM dataset
svm.df <- tfa_df

# Scale numeric columns
numeric_columns_svm <- c(
  'Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length',
  'Essays.Unique.Words', 'Essays.Sentiment',
  'Sign.up.Date', 'Started.Date', 'Submitted.Date'
)

numeric_exists_svm <- numeric_columns_svm[numeric_columns_svm %in% colnames(svm.df)]

# Convert date columns to numeric if they exist
date_cols <- c('Sign.up.Date', 'Started.Date', 'Submitted.Date')
for (col in date_cols) {
  if (col %in% colnames(svm.df)) {
    svm.df[[col]] <- as.numeric(as.Date(svm.df[[col]]))
  }
}

if (length(numeric_exists_svm) > 0) {
  svm.df[, numeric_exists_svm] <- scale(svm.df[, numeric_exists_svm])
}

# Ensure target is factor with proper levels for caret (Class0, Class1)
# This is crucial for twoClassSummary and ROC calculation
svm.df[[target_col]] <- factor(
  as.character(svm.df[[target_col]]),
  levels = c("0", "1"),
  labels = c("Class0", "Class1")
)

# ============================================================================
# SECTION 2: TRAIN/TEST SPLIT
# ============================================================================

set.seed(123)
svm.index <- createDataPartition(svm.df[[target_col]], p = 0.8, list = FALSE)

svm.train <- svm.df[svm.index, ]
svm.test  <- svm.df[-svm.index, ]

cat("\nData Split Summary:\n")
cat("Training Set:", nrow(svm.train), "rows\n")
cat("Test Set:    ", nrow(svm.test), "rows\n")

# ============================================================================
# SECTION 3: SVM MODEL TRAINING (TUNED)
# ============================================================================

cat("\n============================================\n")
cat("TRAINING SVM MODEL (RBF KERNEL)\n")
cat("============================================\n")

# Define tuning grid
# Note: Full grid search takes a long time. 
# We use a reduced grid here for reasonable execution time.
svm_tune_grid <- expand.grid(
  C = c(1, 10, 100),
  sigma = c(0.01, 0.05, 0.1)
)

# Set up cross-validation
svm_train_control <- trainControl(
  method = "cv",           # Simple CV is faster than repeatedcv
  number = 5,              # 5-fold CV
  classProbs = TRUE,       # Needed for ROC
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

cat("Starting grid search with 5-fold CV...\n")

set.seed(123)
svm_model <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train,
  method = "svmRadial",
  preProcess = c("center", "scale"),
  trControl = svm_train_control,
  tuneGrid = svm_tune_grid,
  metric = "ROC"
)

# Display Best Model
cat("\nBest Hyperparameters:\n")
print(svm_model$bestTune)

# ============================================================================
# SECTION 4: EVALUATION
# ============================================================================

cat("\n============================================\n")
cat("EVALUATION ON TEST SET\n")
cat("============================================\n")

# Predictions
svm_pred_class <- predict(svm_model, newdata = svm.test)
svm_pred_prob  <- predict(svm_model, newdata = svm.test, type = "prob")

# Confusion Matrix
svm_conf_matrix <- confusionMatrix(svm_pred_class, svm.test[[target_col]])
print(svm_conf_matrix)

# Extract and Print Key Metrics
accuracy <- svm_conf_matrix$overall["Accuracy"]
sensitivity <- svm_conf_matrix$byClass["Sensitivity"]
specificity <- svm_conf_matrix$byClass["Specificity"]

cat("\n--------------------------------------------\n")
cat("FINAL SVM PERFORMANCE METRICS\n")
cat("--------------------------------------------\n")
cat(sprintf("Accuracy:    %.4f\n", accuracy))
cat(sprintf("Sensitivity: %.4f\n", sensitivity))
cat(sprintf("Specificity: %.4f\n", specificity))
cat("--------------------------------------------\n")
