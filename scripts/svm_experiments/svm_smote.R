
# --------------------------------------------------
# SVM with SMOTE (Synthetic Minority Over-sampling)
# --------------------------------------------------
# Addresses class imbalance by creating synthetic samples of Class0
# Uses RBF kernel (best performing from previous tests)

# Load necessary libraries
library(caret)
library(pROC)
library(doParallel)
library(smotefamily)  # For SMOTE

# Setup parallel processing
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)
cat("\n### Parallel processing enabled with", detectCores() - 1, "cores ###\n")

# 1. Load Data
# --------------------------------------------------
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

# 2. Basic Cleaning
# --------------------------------------------------
tfa_df <- na.omit(tfa_df)

if ("personid" %in% colnames(tfa_df)) {
  tfa_df$personid <- NULL
}
if ("appyear" %in% colnames(tfa_df)) {
  tfa_df$appyear <- NULL
}

target_col <- "Completed.Admissions.Process"

if (!(target_col %in% colnames(tfa_df))) {
  stop("Column 'Completed.Admissions.Process' not found in tfa_df.")
}
tfa_df[[target_col]] <- as.factor(tfa_df[[target_col]])

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
svm_df <- tfa_df

numeric_cols_logical <- sapply(svm_df, is.numeric)
numeric_columns      <- names(svm_df)[numeric_cols_logical]
numeric_columns      <- setdiff(numeric_columns, target_col)

other_cols <- setdiff(colnames(svm_df), c(numeric_columns, target_col))

cat("\nNumeric columns:", length(numeric_columns), "\n")
cat("\nCategorical columns:", length(other_cols), "\n")

# Encode categorical predictors as numeric
for (col in other_cols) {
  svm_df[[col]] <- as.numeric(as.factor(svm_df[[col]]))
}

# Remove any remaining NA/NaN/Inf
all_cols_for_cc <- c(setdiff(colnames(svm_df), target_col), target_col)
svm_df <- svm_df[complete.cases(svm_df[, all_cols_for_cc]), ]

# 4. Train/Test Split
# --------------------------------------------------
set.seed(123)

svm_index <- createDataPartition(svm_df[[target_col]], p = 0.8, list = FALSE)
svm.train <- svm_df[svm_index, ]
svm.test  <- svm_df[-svm_index, ]

cat("\nOriginal training set size:", nrow(svm.train), "\n")
cat("Test set size:", nrow(svm.test), "\n")

# Check class distribution BEFORE SMOTE
cat("\nClass distribution BEFORE SMOTE:\n")
print(table(svm.train[[target_col]]))

# 5. Apply SMOTE to Balance Training Data
# --------------------------------------------------
cat("\n### Applying SMOTE to balance training data... ###\n")

# Prepare data for smotefamily (needs numeric target)
train_for_smote <- svm.train
train_for_smote[[target_col]] <- as.numeric(as.character(train_for_smote[[target_col]]))

# Apply SMOTE
# K = 5 (number of nearest neighbors)
# dup_size = 0 means auto-calculate to balance classes
smote_result <- SMOTE(train_for_smote[, -which(names(train_for_smote) == target_col)],
                      train_for_smote[[target_col]],
                      K = 5,
                      dup_size = 0)

# Extract balanced data
svm.train.balanced <- smote_result$data
colnames(svm.train.balanced)[ncol(svm.train.balanced)] <- target_col
svm.train.balanced[[target_col]] <- as.factor(svm.train.balanced[[target_col]])

cat("\nBalanced training set size:", nrow(svm.train.balanced), "\n")
cat("Class distribution AFTER SMOTE:\n")
print(table(svm.train.balanced[[target_col]]))

# 6. Recode Target for Caret
# --------------------------------------------------
svm.train.balanced[[target_col]] <- factor(
  ifelse(svm.train.balanced[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

svm.test[[target_col]] <- factor(
  ifelse(svm.test[[target_col]] == "1", "Class1", "Class0"),
  levels = c("Class1", "Class0")
)

# 7. Train SVM with RBF Kernel
# --------------------------------------------------
svm_ctrl <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE,
  allowParallel = TRUE,
  savePredictions = TRUE
)

cat("\n### Training SVM with SMOTE-balanced data... ###\n")
cat("Configuration: 3-fold CV, RBF kernel, SMOTE-balanced training data\n\n")

start_time <- Sys.time()

svm_model <- train(
  as.formula(paste(target_col, "~ .")),
  data = svm.train.balanced,
  method = "svmRadial",
  metric = "ROC",
  trControl = svm_ctrl,
  tuneLength = 4,
  preProcess = c("center", "scale")
)

end_time <- Sys.time()
training_duration <- difftime(end_time, start_time, units = "mins")

cat("\n### Training completed in", round(training_duration, 2), "minutes ###\n")

stopCluster(cl)
registerDoSEQ()

# 8. Evaluation
# --------------------------------------------------
cat("\n### SVM – Final Tuned Model (RBF Kernel with SMOTE) ###\n")
print(svm_model)

# Predictions
svm_pred_class <- predict(svm_model, newdata = svm.test)
svm_pred_prob  <- predict(svm_model, newdata = svm.test, type = "prob")

# Confusion Matrix
cat("\n### Confusion Matrix – SVM with SMOTE ###\n")
cm_svm <- confusionMatrix(svm_pred_class, svm.test[[target_col]])
print(cm_svm)

# Metrics
svm_acc  <- cm_svm$overall["Accuracy"]
svm_sens <- cm_svm$byClass["Sensitivity"]
svm_spec <- cm_svm$byClass["Specificity"]
svm_bal_acc <- cm_svm$byClass["Balanced Accuracy"]

cat("\nSVM Accuracy:          ", svm_acc, "\n")
cat("SVM Sensitivity:       ", svm_sens, "\n")
cat("SVM Specificity:       ", svm_spec, "\n")
cat("SVM Balanced Accuracy: ", svm_bal_acc, "\n")

# ROC Curve
roc_obj <- roc(svm.test[[target_col]], svm_pred_prob$Class1)
auc_val <- auc(roc_obj)

pdf("svm_smote_roc_curve.pdf")
plot(roc_obj, main = paste("SVM with SMOTE - ROC Curve (AUC =", round(auc_val, 3), ")"), 
     col = "blue", lwd = 2)
dev.off()
cat("\n[Saved] svm_smote_roc_curve.pdf\n")

# Results Summary
svm_results <- data.frame(
  Model = "SVM_RBF_SMOTE",
  Accuracy = svm_acc,
  Sensitivity = svm_sens,
  Specificity = svm_spec,
  Balanced_Accuracy = svm_bal_acc,
  AUC = as.numeric(auc_val),
  Training_Time_Minutes = as.numeric(training_duration)
)

cat("\n### FINAL RESULTS ###\n")
print(svm_results)

cat("\n### Comparison with Previous Best (No SMOTE) ###\n")
cat("Previous Best - Specificity: 2.4%, Balanced Accuracy: 50.7%\n")
cat("With SMOTE    - Specificity:", round(svm_spec * 100, 1), "%, Balanced Accuracy:", 
    round(svm_bal_acc * 100, 1), "%\n")

if (svm_spec > 0.10) {
  cat("\n[SUCCESS] SMOTE significantly improved specificity!\n")
} else {
  cat("\n[INFO] Specificity still low. May need further tuning or different approach.\n")
}

cat("\n### Script completed successfully! ###\n")
