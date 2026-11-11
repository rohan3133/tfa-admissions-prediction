
# --------------------------------------------------
# SVM MODEL (Linear Kernel) - Memory Optimized
# --------------------------------------------------
# Linear SVM with memory optimization via data sampling
# - 3-fold CV for speed
# - 50% data sampling to avoid memory limits
# - Linear kernel (no complex hyperparameter tuning needed)
# - Parallel processing enabled
# - Should complete in 15-30 minutes
# - Avoids the 16GB memory limit error

# Load necessary libraries
library(caret)
library(pROC) # Required for ROC curve
library(doParallel) # For parallel processing

# Setup parallel processing
cl <- makePSOCKcluster(detectCores() - 1) # Use all cores except 1
registerDoParallel(cl)
cat("\n### Parallel processing enabled with", detectCores() - 1, "cores ###\n")

# 1. Load Data
# --------------------------------------------------
# Use tfa_data.csv as found in the directory
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

# 3. Data Preparation
# --------------------------------------------------

# Copy to a working object for SVM
svm_df <- tfa_df

# Identify numeric-like columns (continuous) automatically
numeric_cols_logical <- sapply(svm_df, is.numeric)
numeric_columns      <- names(svm_df)[numeric_cols_logical]
numeric_columns      <- setdiff(numeric_columns, target_col)  # exclude target if numeric

# Everything else (non-target, non-numeric) is treated as categorical
other_cols <- setdiff(colnames(svm_df), c(numeric_columns, target_col))

cat("\nNumeric columns (will be pre-processed by caret):\n")
print(numeric_columns)
cat("\nOther predictor columns (treated as categorical for encoding):\n")
print(other_cols)

# Encode all non-numeric predictors as numeric codes via as.factor -> as.numeric
for (col in other_cols) {
  svm_df[[col]] <- as.numeric(as.factor(svm_df[[col]]))
}

# Remove any remaining rows with NA/NaN/Inf across predictors + target
all_cols_for_cc <- c(setdiff(colnames(svm_df), target_col), target_col)
svm_df <- svm_df[complete.cases(svm_df[, all_cols_for_cc]), ]

# Final predictor set = all columns except target
svm_predictor_columns <- setdiff(colnames(svm_df), target_col)

# 4. Train/Test Split with MEMORY OPTIMIZATION
# --------------------------------------------------
set.seed(123)

# Use createDataPartition for split (consistent with caret usage)
svm_index <- createDataPartition(svm_df[[target_col]], p = 0.8, list = FALSE)

svm.train <- svm_df[svm_index, ]
svm.test  <- svm_df[-svm_index, ]

# MEMORY OPTIMIZATION: Sample 50% of training data to avoid 16GB limit
set.seed(456)
sample_index <- createDataPartition(svm.train[[target_col]], p = 0.5, list = FALSE)
svm.train <- svm.train[sample_index, ]

cat("\nOriginal training set would be too large for memory.\n")
cat("Using 50% sample to avoid 16GB memory limit.\n")
cat("Training set size (sampled):", nrow(svm.train), "\n")
cat("Test set size:", nrow(svm.test), "\n")

# 5. SVM Model Training (LINEAR KERNEL)
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
  number = 3,  # 3-fold CV for speed
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  allowParallel = TRUE,  # Enable parallel processing
  savePredictions = TRUE # Needed for ROC plot
)

cat("\n### Training Linear SVM Model (Memory Optimized)... ###\n")
cat("Configuration: 3-fold CV, Linear Kernel, 50% data sample, parallel processing\n")
cat("Estimated time: 15-30 minutes\n\n")

# Record start time
start_time <- Sys.time()

# Train SVM model with LINEAR kernel
# Linear kernel only tunes the cost parameter (C)
# Much faster than RBF which tunes both C and sigma
svm_model <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train,
  method = "svmLinear",  # LINEAR KERNEL
  metric = "ROC",
  trControl = svm_ctrl,
  tuneLength = 5,  # Test 5 different cost values
  preProcess = c("center", "scale")
)

# Record end time
end_time <- Sys.time()
training_duration <- difftime(end_time, start_time, units = "mins")
cat("\n### Training completed in", round(training_duration, 2), "minutes ###\n")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# 6. Evaluation & Plots
# --------------------------------------------------

# Print model object
cat("\n### SVM – Final Tuned Model (Linear Kernel) ###\n")
print(svm_model)

# Plot 1: Tuning Parameter Performance
pdf("svm_linear_tuning_plot.pdf")
plot(svm_model, main = "Linear SVM Tuning: ROC vs Cost (C)")
dev.off()
cat("\n[Saved] svm_linear_tuning_plot.pdf\n")

# Predict on svm.test (full test set)
svm_pred_class <- predict(svm_model, newdata = svm.test)
svm_pred_prob  <- predict(svm_model, newdata = svm.test, type = "prob")

# Confusion Matrix
cat("\n### Confusion Matrix – Linear SVM ###\n")
cm_svm <- confusionMatrix(svm_pred_class, svm.test[[target_col]])
print(cm_svm)

# Print Accuracy, Sensitivity, Specificity
svm_acc  <- cm_svm$overall["Accuracy"]
svm_sens <- cm_svm$byClass["Sensitivity"]
svm_spec <- cm_svm$byClass["Specificity"]

cat("\nLinear SVM Accuracy:   ", svm_acc, "\n")
cat("Linear SVM Sensitivity:", svm_sens, "\n")
cat("Linear SVM Specificity:", svm_spec, "\n")

# Plot 2: ROC Curve
roc_obj <- roc(svm.test[[target_col]], svm_pred_prob$Class1)
auc_val <- auc(roc_obj)

pdf("svm_linear_roc_curve.pdf")
plot(roc_obj, main = paste("Linear SVM ROC Curve (AUC =", round(auc_val, 3), ")"), col = "blue", lwd = 2)
dev.off()
cat("\n[Saved] svm_linear_roc_curve.pdf\n")

# Plot 3: Variable Importance
var_imp <- varImp(svm_model, scale = FALSE)
pdf("svm_linear_var_importance.pdf")
plot(var_imp, main = "Linear SVM Variable Importance")
dev.off()
cat("\n[Saved] svm_linear_var_importance.pdf\n")

# Create and print svm_results dataframe
svm_results <- data.frame(
  Model = "SVM_Linear_MemOpt",
  Accuracy = svm_acc,
  Sensitivity = svm_sens,
  Specificity = svm_spec,
  AUC = as.numeric(auc_val),
  Training_Time_Minutes = as.numeric(training_duration),
  Training_Sample_Size = nrow(svm.train),
  Test_Size = nrow(svm.test)
)

cat("\n### FINAL RESULTS ###\n")
print(svm_results)

# Provide guidance on next steps
cat("\n### NEXT STEPS ###\n")
if (svm_acc >= 0.70) {
    cat("[SUCCESS] Linear SVM achieved good accuracy (>= 70%)!\n")
    cat("You can now run svm_model_fast.r (RBF kernel) to see if RBF improves performance.\n")
    cat("RBF will also use 50% sampling to avoid memory issues.\n")
} else {
    cat("[INFO] Linear SVM accuracy is below 70%.\n")
    cat("RBF kernel (svm_model_fast.r with sampling) will likely perform better.\n")
    cat("Linear SVM works best when data is linearly separable.\n")
}

cat("\n### Script completed successfully! ###\n")
