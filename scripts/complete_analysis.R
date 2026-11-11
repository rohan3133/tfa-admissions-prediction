# ============================================================================
# TFA ADMISSIONS ANALYSIS - COMPLETE SCRIPT WITH SVM
# ============================================================================
# Models: KNN, Naive Bayes, Decision Tree (C5.0), ANN, and SVM
# Target: Completed.Admissions.Process (0/1)
# ============================================================================

# Load necessary libraries
library(caret)
library(class)
library(e1071)      # For Naive Bayes
library(C50)        # For Decision Tree
library(neuralnet)  # For ANN
library(fastDummies)
library(kernlab)    # For SVM

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL PREPARATION
# ============================================================================

getwd()
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

summary(tfa_df)
head(tfa_df)
str(tfa_df)
sum(is.na(tfa_df))

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

# Optional: if these columns exist, convert to factor
if ("School.Selectivity" %in% colnames(tfa_df)) {
  tfa_df$School.Selectivity <- as.factor(tfa_df$School.Selectivity)
}
if ("Is.Math..Sci..or.Eng.Major.Minor" %in% colnames(tfa_df)) {
  tfa_df$Is.Math..Sci..or.Eng.Major.Minor <- as.factor(tfa_df$Is.Math..Sci..or.Eng.Major.Minor)
}
if ("Attended.Event" %in% colnames(tfa_df)) {
  tfa_df$Attended.Event <- as.factor(tfa_df$Attended.Event)
}

# ============================================================================
# SECTION 2: KNN MODEL
# ============================================================================

cat("\n============================================\n")
cat("KNN MODEL\n")
cat("============================================\n")

# Build a clean KNN dataset: all predictors numeric
knn_df <- tfa_df

# Identify numeric-like columns (continuous) automatically
numeric_cols_logical <- sapply(knn_df, is.numeric)
numeric_columns      <- names(knn_df)[numeric_cols_logical]
numeric_columns      <- setdiff(numeric_columns, target_col)  # exclude target if numeric

# Everything else (non-target, non-numeric) is treated as categorical
other_cols <- setdiff(colnames(knn_df), c(numeric_columns, target_col))

cat("\nNumeric columns (to be scaled):\n")
print(numeric_columns)
cat("\nOther predictor columns (treated as categorical for KNN encoding):\n")
print(other_cols)

# Scale numeric columns (if any)
if (length(numeric_columns) > 0) {
  knn_df[, numeric_columns] <- scale(knn_df[, numeric_columns, drop = FALSE])
}

# Encode all non-numeric predictors as numeric codes via as.factor -> as.numeric
for (col in other_cols) {
  knn_df[[col]] <- as.numeric(as.factor(knn_df[[col]]))
}

# Remove any remaining rows with NA/NaN/Inf across predictors + target
all_cols_for_cc <- c(setdiff(colnames(knn_df), target_col), target_col)
knn_df <- knn_df[complete.cases(knn_df[, all_cols_for_cc]), ]

# Final predictor set = all columns except target
knn_predictor_columns <- setdiff(colnames(knn_df), target_col)

# Quick check
str(knn_df[, c(knn_predictor_columns, target_col)])

# Train/Test split for KNN
set.seed(123)
idx <- sample(nrow(knn_df), size = 0.8 * nrow(knn_df))

train_data <- knn_df[idx,  knn_predictor_columns]
test_data  <- knn_df[-idx, knn_predictor_columns]

train_labels <- knn_df[[target_col]][idx]
test_labels  <- knn_df[[target_col]][-idx]

prop.table(table(train_labels))
prop.table(table(test_labels))

# KNN: try multiple k values, pick best, show confusion matrix
ks <- c(1, 3, 5, 7, 9, 11)
knn_accuracy <- numeric(length(ks))

for (i in seq_along(ks)) {
  k_val <- ks[i]
  
  knn_pred <- knn(
    train = train_data,
    test  = test_data,
    cl    = train_labels,
    k     = k_val
  )
  
  cm <- confusionMatrix(knn_pred, test_labels)
  knn_accuracy[i] <- cm$overall["Accuracy"]
  
  cat("\n-------------------------------------\n")
  cat("k =", k_val, "\n")
  print(cm)
}

# Best k based on accuracy
best_k <- ks[which.max(knn_accuracy)]
cat("\nBest k:", best_k, "\n")

# Final KNN model with best_k
knn_pred_best <- knn(
  train = train_data,
  test  = test_data,
  cl    = train_labels,
  k     = best_k
)

conf_matrix_best_k <- confusionMatrix(knn_pred_best, test_labels)
conf_matrix_best_k

# Store KNN best accuracy for comparison
knn_accuracy_best <- conf_matrix_best_k$overall["Accuracy"]

# ============================================================================
# SECTION 3: NAIVE BAYES MODEL
# ============================================================================

cat("\n============================================\n")
cat("NAIVE BAYES MODEL\n")
cat("============================================\n")

# View the first few rows of the completedadm column
print(head(tfa_df$Completed.Admissions.Process))

# Check the number of unique values in the completedadm column
print(length(unique(tfa_df$Completed.Admissions.Process)))

# Scale numeric variables
numeric_columns <- c('Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length', 
                     'Essays.Unique.Words', 'Essays.Sentiment', 'Sign.up.Date', 'Started.Date', 'Submitted.Date')
tfa_df[, numeric_columns] <- scale(tfa_df[, numeric_columns])

# Summary of scaled numeric columns
summary(tfa_df[numeric_columns])

# Prepare NB dataset
nb.df <- tfa_df
str(nb.df)

numeric_columns <- c(
  'Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length',
  'Essays.Unique.Words', 'Essays.Sentiment',
  'Sign.up.Date', 'Started.Date', 'Submitted.Date'
)

numeric_exists <- numeric_columns[numeric_columns %in% colnames(nb.df)]
nb.df[, numeric_exists] <- scale(nb.df[, numeric_exists])

set.seed(123)
idx.nb <- createDataPartition(nb.df$Completed.Admissions.Process, p = 0.8, list = FALSE)
train.df <- nb.df[idx.nb, ]
test.df  <- nb.df[-idx.nb, ]

# Basic model
nb.basic <- naiveBayes(Completed.Admissions.Process ~ ., data = train.df)
nb_basic_pred_class <- predict(nb.basic, newdata = test.df)
nb_basic_pred_prob  <- predict(nb.basic, newdata = test.df, type = "raw")
nbpred.df <- data.frame(actual = test.df$Completed.Admissions.Process, predicted = nb_basic_pred_class, nb_basic_pred_prob)
test_conf_matrix <- confusionMatrix(predict(nb.basic, newdata = test.df), test.df$Completed.Admissions.Process)
print(test_conf_matrix)

# Apply train control
tc <- trainControl(p = 0.8, method = "cv", verboseIter = TRUE, number = 10)
train.df$Completed.Admissions.Process <- factor(
  as.character(train.df$Completed.Admissions.Process),
  levels = c("0","1"),
  labels = c("Class0","Class1")
)
test.df$Completed.Admissions.Process <- factor(
  as.character(test.df$Completed.Admissions.Process),
  levels = c("0","1"),
  labels = c("Class0","Class1")
)
model <- train(train.df[, colnames(train.df)[colnames(train.df) != "Completed.Admissions.Process"]], 
               train.df$Completed.Admissions.Process, method = "nb", trControl = tc)
print("Cross-Validated Model Summary:")
print(model)
confusionMatrix(predict(model, newdata = test.df), test.df$Completed.Admissions.Process)

# Tunegrid setting
modelLookup('nb')
nbtune_grid <- expand.grid(usekernel = c(TRUE, FALSE), fL = c(0, 0.5, 1), adjust = c(0.5, 1, 1.5))
model_tuned <- train(train.df[, colnames(train.df)[colnames(train.df) != "Completed.Admissions.Process"]], 
                     train.df$Completed.Admissions.Process, method = "nb", trControl = tc, tuneGrid = nbtune_grid)
print(model_tuned)
nb_tuned_cm <- confusionMatrix(predict(model_tuned, newdata = test.df), test.df$Completed.Admissions.Process)
print(nb_tuned_cm)

# Store NB tuned accuracy for comparison
nb_accuracy_best <- nb_tuned_cm$overall["Accuracy"]

# ============================================================================
# SECTION 4: DECISION TREE MODEL (C5.0)
# ============================================================================

cat("\n============================================\n")
cat("DECISION TREE MODEL (C5.0)\n")
cat("============================================\n")

# Use the Naive Bayes dataset as base (scaled, cleaned)
dt.df <- nb.df
str(dt.df)

# Ensure target is factor
dt.df$Completed.Admissions.Process <- factor(dt.df$Completed.Admissions.Process)

# Train/Test split
set.seed(2001)
dt.index <- createDataPartition(dt.df$Completed.Admissions.Process, 
                                p = 0.8, list = FALSE)

dt.train <- dt.df[dt.index, ]
dt.test  <- dt.df[-dt.index, ]

# Class proportions
proportions(table(dt.df$Completed.Admissions.Process))
proportions(table(dt.train$Completed.Admissions.Process))
proportions(table(dt.test$Completed.Admissions.Process))

# Check tuning parameters for C5.0
modelLookup("C5.0")

# Basic Decision Tree Model
dt.basic <- train(
  Completed.Admissions.Process ~ .,
  data = dt.train,
  method = "C5.0",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

dt.basic
pred.basic <- predict(dt.basic, newdata = dt.test)

cat("\n### Confusion Matrix – Basic DT ###\n")
confusionMatrix(pred.basic, dt.test$Completed.Admissions.Process)

# Tuned Decision Tree with Explicit Tuning Grid
dt.grid <- expand.grid(
  model  = c("tree", "rules"),
  winnow = c(TRUE, FALSE),
  trials = seq(1, 10, 1)
)

dt.tuned <- train(
  Completed.Admissions.Process ~ .,
  data = dt.train,
  method = "C5.0",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = dt.grid
)

dt.tuned

# Evaluate tuned model
pred.tuned <- predict(dt.tuned, newdata = dt.test)

cat("\n### Confusion Matrix – Tuned DT ###\n")
dt_tuned_cm <- confusionMatrix(pred.tuned, dt.test$Completed.Admissions.Process)
print(dt_tuned_cm)

# Store DT tuned accuracy for comparison
dt_accuracy_best <- dt_tuned_cm$overall["Accuracy"]

# Example Predictions (using actual rows from test set)
example1 <- dt.test[1, ]
predict(dt.tuned, newdata = example1)

example2 <- dt.test[2, ]
predict(dt.tuned, newdata = example2)

# ============================================================================
# SECTION 5: ARTIFICIAL NEURAL NETWORK (ANN)
# ============================================================================

cat("\n============================================\n")
cat("ARTIFICIAL NEURAL NETWORK (ANN)\n")
cat("============================================\n")

# Load dataset fresh for ANN
data <- read.csv("tfa_dataset.csv", stringsAsFactors = FALSE)

# Rename columns for ANN
names(data)[names(data) == "Completed Admissions Process"] <- "completedadm"
names(data)[names(data) == "Is Math, Sci, or Eng Major Minor"] <- "stem"
names(data)[names(data) == "School Selectivity"] <- "schoolsel"
names(data)[names(data) == "Attended Event"] <- "attendedevent"

# Prepare data for ANN
data$completedadm <- as.numeric(data$completedadm)  # target 0/1
data$stem <- as.numeric(data$stem)                 # already 0/1

# Convert categorical columns to dummy variables
categorical_cols <- c("schoolsel", "attendedevent")
existing_cat <- intersect(categorical_cols, names(data))
if(length(existing_cat) > 0){
  data <- fastDummies::dummy_cols(data, select_columns = existing_cat, remove_first_dummy = TRUE)
  data <- data[, !names(data) %in% existing_cat]  # remove original columns
}

# Keep only numeric columns and remove NAs
data_numeric <- na.omit(data[, sapply(data, is.numeric)])

# Split data into train and test sets
set.seed(123)
train_idx <- createDataPartition(data_numeric$completedadm, p = 0.8, list = FALSE)
train <- data_numeric[train_idx, ]
test  <- data_numeric[-train_idx, ]

x_train <- train[, setdiff(names(train), "completedadm")]
y_train <- train$completedadm
x_test  <- test[, setdiff(names(test), "completedadm")]
y_test  <- test$completedadm

# Scale features
x_train <- scale(x_train)
x_test  <- scale(x_test)

# Combine features and target for neuralnet
train_nn <- cbind(x_train, completedadm = y_train)

# Build ANN model
set.seed(123)
ann_model <- neuralnet(
  completedadm ~ ., 
  data = train_nn,
  hidden = 5,       # single hidden layer with 5 neurons
  linear.output = FALSE
)

# Predict using ANN
pred <- compute(ann_model, x_test)
pred_class <- ifelse(pred$net.result[,1] > 0.5, 1, 0)

# Evaluate ANN performance
ann_conf_matrix <- confusionMatrix(as.factor(pred_class), as.factor(y_test))
print(ann_conf_matrix)

# Store ANN accuracy for comparison
ann_accuracy <- ann_conf_matrix$overall["Accuracy"]

# ============================================================================
# SECTION 6: SUPPORT VECTOR MACHINE (SVM) - WELL-TUNED
# ============================================================================

cat("\n============================================\n")
cat("SUPPORT VECTOR MACHINE (SVM)\n")
cat("============================================\n")

# SVM is a max-margin classifier that finds the optimal hyperplane to separate classes.
# We'll use the RBF (Radial Basis Function) kernel for non-linear decision boundaries.
# Key hyperparameters:
#   - C (cost): Controls the trade-off between smooth decision boundary and classifying training points correctly.
#            Higher C = less tolerance for misclassification (may overfit).
#   - sigma: Controls the "reach" of each training example in RBF kernel.
#            Higher sigma = more influence from distant points (smoother boundary).

# Prepare SVM dataset (following the same pattern as nb.df and dt.df)
svm.df <- tfa_df

# Ensure numeric columns are scaled
numeric_columns_svm <- c(
  'Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length',
  'Essays.Unique.Words', 'Essays.Sentiment',
  'Sign.up.Date', 'Started.Date', 'Submitted.Date'
)

numeric_exists_svm <- numeric_columns_svm[numeric_columns_svm %in% colnames(svm.df)]
svm.df[, numeric_exists_svm] <- scale(svm.df[, numeric_exists_svm])

# Ensure target is factor with proper levels for caret (Class0, Class1 for twoClassSummary)
svm.df$Completed.Admissions.Process <- factor(
  as.character(svm.df$Completed.Admissions.Process),
  levels = c("0", "1"),
  labels = c("Class0", "Class1")
)

str(svm.df)

# Train/Test split for SVM
set.seed(123)
svm.index <- createDataPartition(svm.df$Completed.Admissions.Process, p = 0.8, list = FALSE)

svm.train <- svm.df[svm.index, ]
svm.test  <- svm.df[-svm.index, ]

# Check class proportions
cat("\n### Class Proportions ###\n")
cat("Full dataset:\n")
print(prop.table(table(svm.df$Completed.Admissions.Process)))
cat("\nTraining set:\n")
print(prop.table(table(svm.train$Completed.Admissions.Process)))
cat("\nTest set:\n")
print(prop.table(table(svm.test$Completed.Admissions.Process)))

# Define comprehensive tuning grid for SVM with RBF kernel
# We'll search over a wide range of C and sigma values to find optimal hyperparameters
svm_tune_grid <- expand.grid(
  C = c(0.1, 1, 5, 10, 25, 50, 100),
  sigma = c(0.001, 0.005, 0.01, 0.05, 0.1)
)

cat("\n### SVM Tuning Grid ###\n")
print(svm_tune_grid)
cat("\nTotal combinations to test:", nrow(svm_tune_grid), "\n")

# Set up cross-validation with repeated CV for robust performance estimation
# We use twoClassSummary to get ROC, Sensitivity, and Specificity metrics
svm_train_control <- trainControl(
  method = "repeatedcv",
  number = 5,              # 5-fold cross-validation
  repeats = 3,             # Repeat 3 times for stability
  classProbs = TRUE,       # Needed for ROC calculation
  summaryFunction = twoClassSummary,
  verboseIter = TRUE       # Show progress
)

# Train SVM model with comprehensive hyperparameter tuning
cat("\n### Training SVM with RBF Kernel and Cross-Validation ###\n")
cat("This may take a few minutes...\n\n")

set.seed(123)
svm_model <- train(
  Completed.Admissions.Process ~ .,
  data = svm.train,
  method = "svmRadial",           # RBF kernel SVM
  preProcess = c("center", "scale"),  # Additional scaling within caret
  trControl = svm_train_control,
  tuneGrid = svm_tune_grid,
  metric = "ROC"                  # Optimize for ROC AUC
)

# Display SVM model results
cat("\n### SVM Model Summary ###\n")
print(svm_model)

cat("\n### Best Hyperparameters ###\n")
print(svm_model$bestTune)

# Plot model performance across hyperparameters
cat("\n### Plotting SVM Performance ###\n")
plot(svm_model, main = "SVM Performance Across Hyperparameters")

# Make predictions on test set
svm_pred_class <- predict(svm_model, newdata = svm.test)
svm_pred_prob  <- predict(svm_model, newdata = svm.test, type = "prob")

# Evaluate SVM performance
cat("\n### SVM Confusion Matrix and Statistics ###\n")
svm_conf_matrix <- confusionMatrix(svm_pred_class, svm.test$Completed.Admissions.Process)
print(svm_conf_matrix)

# Store SVM accuracy for comparison
svm_accuracy <- svm_conf_matrix$overall["Accuracy"]

# Extract additional metrics
cat("\n### SVM Performance Metrics Summary ###\n")
cat("Accuracy:    ", round(svm_conf_matrix$overall["Accuracy"], 4), "\n")
cat("Kappa:       ", round(svm_conf_matrix$overall["Kappa"], 4), "\n")
cat("Sensitivity: ", round(svm_conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Specificity: ", round(svm_conf_matrix$byClass["Specificity"], 4), "\n")
cat("Pos Pred Value: ", round(svm_conf_matrix$byClass["Pos Pred Value"], 4), "\n")
cat("Neg Pred Value: ", round(svm_conf_matrix$byClass["Neg Pred Value"], 4), "\n")

# ============================================================================
# SECTION 7: MODEL COMPARISON
# ============================================================================

cat("\n============================================\n")
cat("MODEL COMPARISON SUMMARY\n")
cat("============================================\n")

# Create comparison table
model_results <- data.frame(
  Model = c("KNN", "Naive Bayes (tuned)", "Decision Tree (tuned)", "ANN", "SVM (tuned)"),
  Accuracy = c(
    round(knn_accuracy_best, 4),
    round(nb_accuracy_best, 4),
    round(dt_accuracy_best, 4),
    round(ann_accuracy, 4),
    round(svm_accuracy, 4)
  )
)

# Sort by accuracy (descending)
model_results <- model_results[order(-model_results$Accuracy), ]

cat("\n### Model Performance Comparison ###\n")
print(model_results)

# Identify best model
best_model_name <- model_results$Model[1]
best_model_acc  <- model_results$Accuracy[1]

cat("\n### Best Performing Model ###\n")
cat("Model:", best_model_name, "\n")
cat("Accuracy:", best_model_acc, "\n")

cat("\n============================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================\n")
