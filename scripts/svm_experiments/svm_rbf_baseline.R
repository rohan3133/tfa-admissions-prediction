
# --------------------------------------------------
# SVM MODEL (Radial Basis Function) - Standalone Script
# --------------------------------------------------

# Load necessary libraries
library(caret)

# 1. Load Data
# --------------------------------------------------
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

# 2. Basic Cleaning (Matching pred.r)
# --------------------------------------------------

# Remove rows with any NA values
tfa_df <- na.omit(tfa_df)

# Safely remove 'personid' and 'appyear' if they exist
if ("personid" %in% colnames(tfa_df)) {
  tfa_df$personid <- NULL
}
if ("appyear" %in% colnames(tfa_df)) {
  tfa_df$appyear <- NULL
}

# Target and basic factor conversion
target_col <- "Completed.Admissions.Process"

# Make sure target exists and is factor
if (!(target_col %in% colnames(tfa_df))) {
  stop("Column 'Completed.Admissions.Process' not found in tfa_df.")
}
tfa_df[[target_col]] <- as.factor(tfa_df[[target_col]])

# Optional: if these columns exist, convert to factor (matches pred.r style)
if ("School.Selectivity" %in% colnames(tfa_df)) {
  tfa_df$School.Selectivity <- as.factor(tfa_df$School.Selectivity)
}
if ("Is.Math..Sci..or.Eng.Major.Minor" %in% colnames(tfa_df)) {
  tfa_df$Is.Math..Sci..or.Eng.Major.Minor <- as.factor(tfa_df$Is.Math..Sci..or.Eng.Major.Minor)
}
if ("Attended.Event" %in% colnames(tfa_df)) {
  tfa_df$Attended.Event <- as.factor(tfa_df$Attended.Event)
}

# 3. Data Preparation (Exact replica of KNN prep in pred.r)
# --------------------------------------------------

# Copy to a working object for SVM
svm_df <- tfa_df

# Identify numeric-like columns (continuous) automatically
numeric_cols_logical <- sapply(svm_df, is.numeric)
numeric_columns      <- names(svm_df)[numeric_cols_logical]
numeric_columns      <- setdiff(numeric_columns, target_col)  # exclude target if numeric

# Everything else (non-target, non-numeric) is treated as categorical
other_cols <- setdiff(colnames(svm_df), c(numeric_columns, target_col))

cat("\nNumeric columns (to be scaled):\n")
print(numeric_columns)
cat("\nOther predictor columns (treated as categorical for encoding):\n")
print(other_cols)

# Scale numeric columns (if any)
if (length(numeric_columns) > 0) {
  svm_df[, numeric_columns] <- scale(svm_df[, numeric_columns, drop = FALSE])
}

# Encode all non-numeric predictors as numeric codes via as.factor -> as.numeric
for (col in other_cols) {
  svm_df[[col]] <- as.numeric(as.factor(svm_df[[col]]))
}

# Remove any remaining rows with NA/NaN/Inf across predictors + target
all_cols_for_cc <- c(setdiff(colnames(svm_df), target_col), target_col)
svm_df <- svm_df[complete.cases(svm_df[, all_cols_for_cc]), ]

# Final predictor set = all columns except target
svm_predictor_columns <- setdiff(colnames(svm_df), target_col)

# 4. Train/Test Split
# --------------------------------------------------
set.seed(123)

# Use createDataPartition for split (consistent with caret usage)
svm_index <- createDataPartition(svm_df[[target_col]], p = 0.8, list = FALSE)

svm.train <- svm_df[svm_index, ]
svm.test  <- svm_df[-svm_index, ]

# 5. SVM Model Training
# --------------------------------------------------

# Recode target for caret's twoClassSummary
# Convert "0"/"1" into "Class0"/"Class1"
svm.train[[target_col]] <- factor(
  ifelse(svm.train[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

svm.test[[target_col]] <- factor(
  ifelse(svm.test[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

# Define trainControl
svm_ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

cat("\n### Training SVM Model... ###\n")

# Train SVM model
svm_model <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train,
  method = "svmRadial",
  metric = "ROC",
  trControl = svm_ctrl,
  tuneLength = 10
)

# 6. Evaluation
# --------------------------------------------------

# Print model object
cat("\n### SVM – Final Tuned Model (RBF Kernel) ###\n")
print(svm_model)

# Predict on svm.test
svm_pred_class <- predict(svm_model, newdata = svm.test)

# Confusion Matrix
cat("\n### Confusion Matrix – SVM ###\n")
cm_svm <- confusionMatrix(svm_pred_class, svm.test[[target_col]])
print(cm_svm)

# Print Accuracy, Sensitivity, Specificity
svm_acc  <- cm_svm$overall["Accuracy"]
svm_sens <- cm_svm$byClass["Sensitivity"]
svm_spec <- cm_svm$byClass["Specificity"]

cat("\nSVM Accuracy:   ", svm_acc, "\n")
cat("SVM Sensitivity:", svm_sens, "\n")
cat("SVM Specificity:", svm_spec, "\n")

# Create and print svm_results dataframe
svm_results <- data.frame(
  Model = "SVM_Radial",
  Accuracy = svm_acc,
  Sensitivity = svm_sens,
  Specificity = svm_spec
)

print(svm_results)
