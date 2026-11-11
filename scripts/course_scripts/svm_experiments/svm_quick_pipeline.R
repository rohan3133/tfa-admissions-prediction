#!/usr/bin/env Rscript
# pred_svm_only.R
# SVM-only pipeline using the same data-prep / split style as the KNN section in pred.r
# - Reads tfa_data.csv
# - Data cleaning / scaling / encoding (same approach as pred.r KNN)
# - Train/test split (80/20 via random sampling)
# - Tune SVM (RBF) using caret on training set (optimize Accuracy)
# - Evaluate on test set, print confusion matrix, Accuracy, Sensitivity, Specificity, AUC
# - Save plots: tuning plot, confusion matrix heatmap, ROC curve

library(caret)
library(e1071)
library(pROC)
library(ggplot2)
library(dplyr)

set.seed(123)

# Quick-run toggle: set environment variable QUICK_SVM_RUN=1 to train on a small subset (fast)
quick_run <- as.logical(as.integer(Sys.getenv("QUICK_SVM_RUN", "0")))
if (quick_run) message("Quick-run enabled: using a smaller sample to speed up validation")

data_file <- "data/tfa_data.csv"
if(!file.exists(data_file)) stop("Data file 'data/tfa_data.csv' not found - run from the repository root")

tfa_df <- read.csv(data_file, stringsAsFactors = FALSE)

message("Initial rows: ", nrow(tfa_df), " cols: ", ncol(tfa_df))
message("NAs before omit: ", sum(is.na(tfa_df)))

# Remove rows with any NA values (same as pred.r knn)
tfa_df <- na.omit(tfa_df)

# Safely remove 'Person Id' or 'personid' and 'App Started Year (RTAT)' / 'appyear' if present
if ("Person Id" %in% colnames(tfa_df)) tfa_df$`Person Id` <- NULL
if ("personid" %in% colnames(tfa_df)) tfa_df$personid <- NULL
if ("App.Started.Year..RTAT." %in% colnames(tfa_df)) tfa_df$App.Started.Year..RTAT. <- NULL
if ("appyear" %in% colnames(tfa_df)) tfa_df$appyear <- NULL

# Target column
target_col <- "Completed.Admissions.Process"
if (!(target_col %in% colnames(tfa_df))) stop("Target column not found: ", target_col)

# Convert target to factor
tfa_df[[target_col]] <- as.factor(as.character(tfa_df[[target_col]]))

# Optional factor conversions (copying pred.r style)
optional_factors <- c("School.Selectivity", "Is.Math..Sci..or.Eng.Major.Minor", "Attended.Event")
for(cn in optional_factors) if (cn %in% colnames(tfa_df)) tfa_df[[cn]] <- as.factor(as.character(tfa_df[[cn]]))

# ------------------
# Prepare SVM-friendly dataset (mimic pred.r KNN approach)
# ------------------
svm_df <- tfa_df

# Identify numeric columns automatically
numeric_cols_logical <- sapply(svm_df, is.numeric)
numeric_columns <- names(svm_df)[numeric_cols_logical]
numeric_columns <- setdiff(numeric_columns, target_col)

other_cols <- setdiff(colnames(svm_df), c(numeric_columns, target_col))

cat("Numeric columns (to be scaled):\n")
print(numeric_columns)
cat("Other predictor columns (encoded as numeric):\n")
print(other_cols)

# Scale numeric columns
if (length(numeric_columns) > 0) svm_df[, numeric_columns] <- scale(svm_df[, numeric_columns, drop = FALSE])

# Encode all non-numeric predictors as integer codes (as pred.r did for KNN)
for (col in other_cols) {
  svm_df[[col]] <- as.numeric(as.factor(svm_df[[col]]))
}

# Ensure no remaining NA/Inf
all_cols_for_cc <- c(setdiff(colnames(svm_df), target_col), target_col)
svm_df <- svm_df[complete.cases(svm_df[, all_cols_for_cc]), ]

# Final predictor list
svm_predictor_columns <- setdiff(colnames(svm_df), target_col)

message("Prepared dataset rows: ", nrow(svm_df), " predictors: ", length(svm_predictor_columns))

# ------------------
# Train / Test Split (80/20) using sampling — matching pred.r knn approach
# ------------------
set.seed(123)

# If quick_run, use a smaller random subsample to speed checks
if (quick_run) {
  max_rows <- min(5000, nrow(svm_df))
  sel <- sample(nrow(svm_df), size = max_rows)
  svm_df <- svm_df[sel, ]
}

idx <- sample(nrow(svm_df), size = floor(0.8 * nrow(svm_df)))

train_data <- svm_df[idx, svm_predictor_columns]
test_data  <- svm_df[-idx, svm_predictor_columns]

train_labels <- svm_df[[target_col]][idx]
test_labels  <- svm_df[[target_col]][-idx]

cat("Train class distribution:\n")
print(prop.table(table(train_labels)))
cat("Test class distribution:\n")
print(prop.table(table(test_labels)))

# For caret SVM, we prefer a data frame with the target column attached
train_df <- cbind(train_data, Completed.Admissions.Process = train_labels)
test_df  <- cbind(test_data,  Completed.Admissions.Process = test_labels)

# Ensure target levels are consistent and named for caret
train_df$Completed.Admissions.Process <- factor(as.character(train_df$Completed.Admissions.Process), levels = c("0","1"), labels = c("Class0","Class1"))
test_df$Completed.Admissions.Process  <- factor(as.character(test_df$Completed.Admissions.Process),  levels = c("0","1"), labels = c("Class0","Class1"))

# ------------------
# Train/Tune SVM (Radial) using caret; optimize Accuracy (as requested)
# ------------------
## Manual SVM grid search using e1071 (keeps code simple like pred.r)
message("Manual SVM grid search (radial kernel) — searching simple cost/gamma grid")

# build matrices/labels for e1071::svm
train_x <- train_data
test_x  <- test_data
train_y <- as.factor(as.character(train_labels))
test_y  <- as.factor(as.character(test_labels))

# search grid (small/simple values - easy to read and change)
costs  <- c(0.1, 1, 10)
gammas <- c(0.001, 0.01, 0.1)

grid <- expand.grid(cost = costs, gamma = gammas, Accuracy = NA_real_)

best_acc <- -Inf
best_params <- NULL

for(i in seq_len(nrow(grid))){
  C <- grid$cost[i]
  G <- grid$gamma[i]

  fit <- svm(x = train_x, y = train_y, kernel = "radial", cost = C, gamma = G, probability = TRUE)
  pcls <- predict(fit, newdata = test_x)
  cm_tmp <- confusionMatrix(pcls, factor(test_y, levels = levels(train_y)))
  acc_tmp <- as.numeric(cm_tmp$overall["Accuracy"])

  grid$Accuracy[i] <- acc_tmp
  message(sprintf("cost=%g gamma=%g -> Acc=%.4f", C, G, acc_tmp))

  if(!is.na(acc_tmp) && acc_tmp > best_acc){
    best_acc <- acc_tmp
    best_params <- list(cost = C, gamma = G)
  }
}

message("Grid search complete. Best accuracy on test set: ", best_acc)
message("Best params: ", paste(names(best_params), best_params, sep = "=", collapse = ", "))

# final model using best params
final_svm <- svm(x = train_x, y = train_y, kernel = "radial", cost = best_params$cost, gamma = best_params$gamma, probability = TRUE)
print(final_svm)

# ------------------
# Predictions & Evaluation on test set
# ------------------
pred_class <- predict(final_svm, newdata = test_x)

pred_prob_mat  <- attr(predict(final_svm, newdata = test_x, probability = TRUE), "probabilities")

# determine which column corresponds to class '1' probabilities
prob_col_name <- if(is.matrix(pred_prob_mat)) {
  if("1" %in% colnames(pred_prob_mat)) "1" else colnames(pred_prob_mat)[ncol(pred_prob_mat)]
} else {
  names(pred_prob_mat)[1]
}

# Confusion matrix: use original '0'/'1' labels; set positive="1" to compute sensitivity for class 1
cm <- confusionMatrix(as.factor(as.character(pred_class)), factor(as.character(test_y), levels = c("0","1")), positive = "1")
cat("\nConfusion Matrix (test):\n")
print(cm)

acc <- cm$overall["Accuracy"]
sen <- cm$byClass["Sensitivity"]
spec <- cm$byClass["Specificity"]

cat(sprintf("\nTest Accuracy: %.4f  Sensitivity: %.4f  Specificity: %.4f\n", acc, sen, spec))

# AUC using pROC: positive class = Class1
roc_response <- factor(as.character(test_y), levels = c("0","1"), labels = c("Class0","Class1"))
roc_obj <- roc(response = roc_response, predictor = as.numeric(pred_prob_mat[, prob_col_name]))
auc_val <- auc(roc_obj)
cat(sprintf("AUC (Class1): %.4f\n", auc_val))

# ------------------
# Plots: tuning, ROC, confusion heatmap — saved into 'svm_plots' folder
# ------------------
plot_dir <- "svm_plots"
if(!dir.exists(plot_dir)) dir.create(plot_dir)

## 1) Simple tuning heatmap from our manual grid search
grid_plot_df <- grid
grid_plot_df$cost <- as.factor(grid_plot_df$cost)
grid_plot_df$gamma <- as.factor(grid_plot_df$gamma)

tp <- ggplot(grid_plot_df, aes(x = cost, y = gamma, fill = Accuracy)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.3f", Accuracy)), color = "white") +
  labs(title = "SVM grid search (test Accuracy)", x = "cost", y = "gamma") +
  theme_minimal()

ggsave(filename = file.path(plot_dir, "svm_tune_plot.png"), plot = tp, width = 7, height = 5)

# 2) ROC plot
roc_df <- data.frame(
  tpr = roc_obj$sensitivities,
  fpr = 1 - roc_obj$specificities
)

roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "steelblue", size = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(title = sprintf("SVM ROC curve (AUC = %.3f)", as.numeric(auc_val)), x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()

ggsave(filename = file.path(plot_dir, "svm_roc.png"), plot = roc_plot, width = 7, height = 5)

# 3) Confusion matrix heatmap
cm_table <- as.data.frame(cm$table)
colnames(cm_table) <- c("Reference","Prediction","Freq")

cm_plot <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "navy", high = "red") +
  labs(title = "Confusion Matrix (Test set)") +
  theme_minimal()

ggsave(filename = file.path(plot_dir, "svm_confusion_heatmap.png"), plot = cm_plot, width = 5, height = 4)

# 4) Predicted probability density by true class
if (is.matrix(pred_prob_mat)) {
  # pred_prob_mat columns are named by factor levels ("0","1"); pick column "1" for P(class=1)
  prob_col_name <- ifelse("1" %in% colnames(pred_prob_mat), "1", colnames(pred_prob_mat)[ncol(pred_prob_mat)])
  prob_df <- data.frame(prob = pred_prob_mat[, prob_col_name], truth = factor(as.character(test_y), levels = c("0","1"), labels = c("Class0","Class1")))
} else {
  prob_df <- data.frame(prob = pred_prob_mat, truth = factor(as.character(test_y), levels = c("0","1"), labels = c("Class0","Class1")))
}
prob_plot <- ggplot(prob_df, aes(x = prob, fill = truth)) +
  geom_density(alpha = 0.4) +
  labs(title = "Predicted probability distributions for Class1 (test)", x = "P(Class1)") +
  theme_minimal()

ggsave(filename = file.path(plot_dir, "svm_prob_density.png"), plot = prob_plot, width = 7, height = 5)

## Save numeric results
metrics_df <- data.frame(
  Model = "SVM_Radial",
  Accuracy = as.numeric(acc),
  Sensitivity = as.numeric(sen),
  Specificity = as.numeric(spec),
  AUC = as.numeric(auc_val)
)
write.csv(metrics_df, file.path(plot_dir, "svm_metrics.csv"), row.names = FALSE)

message("SVM run complete; plots and metrics saved to: ", normalizePath(plot_dir))
