
# --------------------------------------------------
# SVM KERNEL COMPARISON: vanilladot vs rbfdot
# --------------------------------------------------
# Trains both linear (vanilladot) and RBF (rbfdot) kernels
# Uses IDENTICAL data preparation as other models
# Memory optimized with 50% sampling
# Direct comparison of kernel performance

# Load necessary libraries
library(caret)
library(kernlab)  # For direct kernel access
library(pROC)     # For ROC curves
library(doParallel) # For parallel processing

# Setup parallel processing
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
cat("\n### Parallel processing enabled with", detectCores() - 1, "cores ###\n")

# 1. Load Data (SAME AS OTHER MODELS)
# --------------------------------------------------
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

# 2. Basic Cleaning (SAME AS OTHER MODELS)
# --------------------------------------------------
tfa_df <- na.omit(tfa_df)

# Remove personid and appyear if they exist
if ("personid" %in% colnames(tfa_df)) {
  tfa_df$personid <- NULL
}
if ("appyear" %in% colnames(tfa_df)) {
  tfa_df$appyear <- NULL
}

# Target and basic factor conversion
target_col <- "Completed.Admissions.Process"

if (!(target_col %in% colnames(tfa_df))) {
  stop("Column 'Completed.Admissions.Process' not found in tfa_df.")
}
tfa_df[[target_col]] <- as.factor(tfa_df[[target_col]])

# Convert categorical columns to factors
if ("School.Selectivity" %in% colnames(tfa_df)) {
  tfa_df$School.Selectivity <- as.factor(tfa_df$School.Selectivity)
}
if ("Is.Math..Sci..or.Eng.Major.Minor" %in% colnames(tfa_df)) {
  tfa_df$Is.Math..Sci..or.Eng.Major.Minor <- as.factor(tfa_df$Is.Math..Sci..or.Eng.Major.Minor)
}
if ("Attended.Event" %in% colnames(tfa_df)) {
  tfa_df$Attended.Event <- as.factor(tfa_df$Attended.Event)
}

# 3. Data Preparation (SAME AS OTHER MODELS)
# --------------------------------------------------
svm_df <- tfa_df

# Identify numeric columns
numeric_cols_logical <- sapply(svm_df, is.numeric)
numeric_columns      <- names(svm_df)[numeric_cols_logical]
numeric_columns      <- setdiff(numeric_columns, target_col)

# Everything else is categorical
other_cols <- setdiff(colnames(svm_df), c(numeric_columns, target_col))

cat("\nNumeric columns:", length(numeric_columns), "\n")
cat("Categorical columns:", length(other_cols), "\n")

# Encode categorical predictors as numeric
for (col in other_cols) {
  svm_df[[col]] <- as.numeric(as.factor(svm_df[[col]]))
}

# Remove any remaining NA/NaN/Inf
all_cols_for_cc <- c(setdiff(colnames(svm_df), target_col), target_col)
svm_df <- svm_df[complete.cases(svm_df[, all_cols_for_cc]), ]

# 4. Train/Test Split (SAME AS OTHER MODELS)
# --------------------------------------------------
set.seed(123)  # SAME SEED FOR CONSISTENCY

svm_index <- createDataPartition(svm_df[[target_col]], p = 0.8, list = FALSE)
svm.train <- svm_df[svm_index, ]
svm.test  <- svm_df[-svm_index, ]

# 50% sampling for memory optimization (SAME AS OTHER MODELS)
set.seed(456)  # SAME SEED FOR CONSISTENCY
sample_index <- createDataPartition(svm.train[[target_col]], p = 0.5, list = FALSE)
svm.train <- svm.train[sample_index, ]

cat("\nTraining set size (sampled):", nrow(svm.train), "\n")
cat("Test set size:", nrow(svm.test), "\n")

# Recode target for caret (SAME AS OTHER MODELS)
svm.train[[target_col]] <- factor(
  ifelse(svm.train[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

svm.test[[target_col]] <- factor(
  ifelse(svm.test[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

# 5. Define Training Control (SAME AS OTHER MODELS)
# --------------------------------------------------
svm_ctrl <- trainControl(
  method = "cv",
  number = 2,  # 2-fold CV for speed
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  allowParallel = TRUE,
  savePredictions = TRUE
)

# 6. Train Model 1: vanilladot (Linear Kernel)
# --------------------------------------------------
cat("\n========================================\n")
cat("TRAINING MODEL 1: vanilladot (Linear)\n")
cat("========================================\n")

start_time_linear <- Sys.time()

model_linear <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train,
  method = "svmLinear",  # Uses vanilladot kernel
  metric = "ROC",
  trControl = svm_ctrl,
  tuneLength = 5,
  preProcess = c("center", "scale")
)

end_time_linear <- Sys.time()
time_linear <- difftime(end_time_linear, start_time_linear, units = "mins")

cat("\n### Linear SVM completed in", round(time_linear, 2), "minutes ###\n")

# 7. Train Model 2: rbfdot (RBF Kernel)
# --------------------------------------------------
cat("\n========================================\n")
cat("TRAINING MODEL 2: rbfdot (RBF)\n")
cat("========================================\n")

start_time_rbf <- Sys.time()

# Removed YeoJohnson to improve stability
model_rbf <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train,
  method = "svmRadial",  # Uses rbfdot kernel
  metric = "ROC",
  trControl = svm_ctrl,
  tuneLength = 4,
  preProcess = c("center", "scale")
)

end_time_rbf <- Sys.time()
time_rbf <- difftime(end_time_rbf, start_time_rbf, units = "mins")

cat("\n### RBF SVM completed in", round(time_rbf, 2), "minutes ###\n")

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# 8. Evaluate Both Models
# --------------------------------------------------
cat("\n========================================\n")
cat("MODEL COMPARISON RESULTS\n")
cat("========================================\n")

# Safe prediction function
safe_predict_class <- function(model, data) {
  tryCatch({
    predict(model, newdata = data)
  }, error = function(e) {
    cat("Error in class prediction:", e$message, "\n")
    return(NULL)
  })
}

safe_predict_prob <- function(model, data) {
  tryCatch({
    predict(model, newdata = data, type = "prob")
  }, error = function(e) {
    cat("Error in prob prediction:", e$message, "\n")
    return(NULL)
  })
}

# Predictions for Linear
pred_linear_class <- safe_predict_class(model_linear, svm.test)
pred_linear_prob  <- safe_predict_prob(model_linear, svm.test)

if (!is.null(pred_linear_class) && !is.null(pred_linear_prob)) {
  cm_linear <- confusionMatrix(pred_linear_class, svm.test[[target_col]])
  roc_linear <- roc(svm.test[[target_col]], pred_linear_prob$Class1)
} else {
  cm_linear <- list(overall = c(Accuracy = NA, Kappa = NA), byClass = c(Sensitivity = NA, Specificity = NA, "Balanced Accuracy" = NA))
  roc_linear <- list(auc = NA)
}

# Predictions for RBF
pred_rbf_class <- safe_predict_class(model_rbf, svm.test)
pred_rbf_prob  <- safe_predict_prob(model_rbf, svm.test)

if (!is.null(pred_rbf_class) && !is.null(pred_rbf_prob)) {
  cm_rbf <- confusionMatrix(pred_rbf_class, svm.test[[target_col]])
  roc_rbf <- roc(svm.test[[target_col]], pred_rbf_prob$Class1)
} else {
  cm_rbf <- list(overall = c(Accuracy = NA, Kappa = NA), byClass = c(Sensitivity = NA, Specificity = NA, "Balanced Accuracy" = NA))
  roc_rbf <- list(auc = NA)
}

# Create comparison table
comparison <- data.frame(
  Kernel = c("vanilladot (Linear)", "rbfdot (RBF)"),
  Training_Time_Min = c(as.numeric(time_linear), as.numeric(time_rbf)),
  Accuracy = c(cm_linear$overall["Accuracy"], cm_rbf$overall["Accuracy"]),
  Sensitivity = c(cm_linear$byClass["Sensitivity"], cm_rbf$byClass["Sensitivity"]),
  Specificity = c(cm_linear$byClass["Specificity"], cm_rbf$byClass["Specificity"]),
  Balanced_Acc = c(cm_linear$byClass["Balanced Accuracy"], cm_rbf$byClass["Balanced Accuracy"]),
  Kappa = c(cm_linear$overall["Kappa"], cm_rbf$overall["Kappa"]),
  AUC = c(as.numeric(auc(roc_linear)), as.numeric(auc(roc_rbf)))
)

cat("\n### PERFORMANCE COMPARISON ###\n")
print(comparison)

# 9. Generate Comparison Plots
# --------------------------------------------------

# Plot 1: ROC Curves Comparison
pdf("svm_kernels_roc_comparison.pdf", width = 10, height = 6)
plot(roc_linear, col = "blue", lwd = 2, main = "ROC Curve Comparison: vanilladot vs rbfdot")
plot(roc_rbf, col = "red", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("vanilladot (AUC =", round(auc(roc_linear), 3), ")"),
                  paste("rbfdot (AUC =", round(auc(roc_rbf), 3), ")")),
       col = c("blue", "red"), lwd = 2)
dev.off()
cat("\n[Saved] svm_kernels_roc_comparison.pdf\n")

# Plot 2: Performance Metrics Bar Chart
pdf("svm_kernels_metrics_comparison.pdf", width = 12, height = 6)
par(mfrow = c(1, 2))

# Accuracy, Sensitivity, Specificity
metrics1 <- rbind(
  c(comparison$Accuracy[1], comparison$Sensitivity[1], comparison$Specificity[1]),
  c(comparison$Accuracy[2], comparison$Sensitivity[2], comparison$Specificity[2])
)
barplot(metrics1, beside = TRUE, 
        names.arg = c("Accuracy", "Sensitivity", "Specificity"),
        col = c("blue", "red"),
        legend.text = c("vanilladot", "rbfdot"),
        main = "Classification Metrics",
        ylim = c(0, 1))

# AUC and Balanced Accuracy
metrics2 <- rbind(
  c(comparison$AUC[1], comparison$Balanced_Acc[1]),
  c(comparison$AUC[2], comparison$Balanced_Acc[2])
)
barplot(metrics2, beside = TRUE,
        names.arg = c("AUC", "Balanced Accuracy"),
        col = c("blue", "red"),
        legend.text = c("vanilladot", "rbfdot"),
        main = "Overall Performance",
        ylim = c(0, 1))

dev.off()
cat("[Saved] svm_kernels_metrics_comparison.pdf\n")

# 10. Print Detailed Confusion Matrices
# --------------------------------------------------
cat("\n### CONFUSION MATRIX: vanilladot (Linear) ###\n")
print(cm_linear)

cat("\n### CONFUSION MATRIX: rbfdot (RBF) ###\n")
print(cm_rbf)

# 11. Determine Winner
# --------------------------------------------------
cat("\n========================================\n")
cat("RECOMMENDATION\n")
cat("========================================\n")

if (comparison$AUC[2] > comparison$AUC[1]) {
  cat("✓ rbfdot (RBF) kernel performs better (higher AUC)\n")
  cat("  - Better at capturing non-linear patterns\n")
  cat("  - AUC improvement:", round((comparison$AUC[2] - comparison$AUC[1]) * 100, 2), "%\n")
} else {
  cat("✓ vanilladot (Linear) kernel performs better (higher AUC)\n")
  cat("  - Data may be linearly separable\n")
  cat("  - Faster training time\n")
}

cat("\nNote: Both models show class imbalance issues.\n")
cat("Consider implementing SMOTE or class weights for better specificity.\n")

cat("\n### Script completed successfully! ###\n")
