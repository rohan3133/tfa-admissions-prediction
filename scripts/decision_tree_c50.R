# Decision Tree Model for TFA Dataset
# Uses C5.0 Algorithm with Cross-Validation and Tuning

# 1. Load Libraries
library(caret)
library(C50)
library(plyr) # for revalue if needed, though caret handles most

# 2. Load Data
# Ensure we use the correct filename
if (file.exists("data/tfa_data.csv")) {
  tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)
} else {
  stop("Error: tfa_data.csv not found.")
}

# 3. Data Cleaning & Preprocessing

# Remove rows with NAs immediately to ensure clean data for modeling
tfa_df <- na.omit(tfa_df)

# Convert Date columns to numeric (days since epoch)
# This is crucial as models cannot handle raw date strings
date_cols <- c("Sign.up.Date", "Started.Date", "Submitted.Date", "Application.Deadline")
for (col in date_cols) {
  if (col %in% colnames(tfa_df)) {
    # Convert to Date object then to numeric
    tfa_df[[col]] <- as.numeric(as.Date(tfa_df[[col]]))
  }
}

# Identify Target Column
target_col <- "Completed.Admissions.Process"
if (!target_col %in% colnames(tfa_df)) {
  stop("Target column 'Completed.Admissions.Process' not found.")
}

# Convert Target to Factor (valid R variable names for levels are safer for caret)
tfa_df[[target_col]] <- factor(tfa_df[[target_col]], levels = c(0, 1), labels = c("No", "Yes"))

# Identify Predictors
# Exclude ID-like columns or columns that might leak information if necessary
# Based on previous files, we'll keep most but remove IDs if they exist
exclude_cols <- c("personid", "appyear", "Person.Id") 
predictors <- setdiff(colnames(tfa_df), c(target_col, exclude_cols))

# Convert Character columns to Factors
# Decision trees can handle factors, but caret often prefers dummy vars or numeric. 
# C5.0 handles factors well natively.
for (col in predictors) {
  if (is.character(tfa_df[[col]])) {
    tfa_df[[col]] <- as.factor(tfa_df[[col]])
  }
}

# 4. Split Data (80% Train, 20% Test)
set.seed(123)
trainIndex <- createDataPartition(tfa_df[[target_col]], p = 0.8, list = FALSE, times = 1)
train_data <- tfa_df[trainIndex, ]
test_data  <- tfa_df[-trainIndex, ]

# 5. Train Decision Tree Model (C5.0)
# We use 10-fold Cross Validation to ensure robustness
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE, # Needed for ROC/AUC if we wanted that
  summaryFunction = defaultSummary,
  sampling = "up" # Handle class imbalance by oversampling the minority class
)

# Tuning Grid for C5.0
# trials: Number of boosting iterations (1 = single tree, >1 = boosting)
# model: 'tree' or 'rules'
# winnow: Feature selection (TRUE/FALSE)
grid <- expand.grid(
  trials = c(1, 10),
  model = c("tree"),
  winnow = c(FALSE)
)

cat("Training C5.0 Decision Tree Model...\n")
dt_model <- train(
  x = train_data[, predictors],
  y = train_data[[target_col]],
  method = "C5.0",
  trControl = ctrl,
  tuneGrid = grid
)

# 6. Evaluate Model
cat("\nBest Model Parameters:\n")
print(dt_model$bestTune)

cat("\nModel Summary:\n")
print(dt_model)

# Predictions on Test Set
predictions <- predict(dt_model, newdata = test_data)
probs <- predict(dt_model, newdata = test_data, type = "prob")

# Confusion Matrix
cm <- confusionMatrix(predictions, test_data[[target_col]], mode = "everything")

cat("\n------------------------------------------------\n")
cat("Confusion Matrix and Statistics (Test Set)\n")
cat("------------------------------------------------\n")
print(cm)

# Save the model for later use
saveRDS(dt_model, "decision_tree_model.rds")
cat("\nModel saved to 'decision_tree_model.rds'\n")
