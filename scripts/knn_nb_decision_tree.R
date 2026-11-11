# Load data

getwd()
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)

summary(tfa_df)
head(tfa_df)
str(tfa_df)
sum(is.na(tfa_df))

# Remove rows with any NA values
tfa_df <- na.omit(tfa_df)

# Convert date columns to numeric
date_cols <- c("Sign.up.Date", "Started.Date", "Submitted.Date", "Application.Deadline")
for (col in date_cols) {
  if (col %in% colnames(tfa_df)) {
    tfa_df[[col]] <- as.numeric(as.Date(tfa_df[[col]]))
  }
}

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

# Optional: if these columns exist, convert to factor (matches your earlier style)
if ("School.Selectivity" %in% colnames(tfa_df)) {
  tfa_df$School.Selectivity <- as.factor(tfa_df$School.Selectivity)
}
if ("Is.Math..Sci..or.Eng.Major.Minor" %in% colnames(tfa_df)) {
  tfa_df$Is.Math..Sci..or.Eng.Major.Minor <- as.factor(tfa_df$Is.Math..Sci..or.Eng.Major.Minor)
}
if ("Attended.Event" %in% colnames(tfa_df)) {
  tfa_df$Attended.Event <- as.factor(tfa_df$Attended.Event)
}

# --------------------------------------------------
# Build a clean KNN dataset: all predictors numeric
# --------------------------------------------------

# Copy to a working object for KNN
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


# --------------------------------------------------
# Train/Test split for KNN
# --------------------------------------------------

library(caret)
library(class)

set.seed(123)

idx <- sample(nrow(knn_df), size = 0.8 * nrow(knn_df))

train_data <- knn_df[idx,  knn_predictor_columns]
test_data  <- knn_df[-idx, knn_predictor_columns]

train_labels <- knn_df[[target_col]][idx]
test_labels  <- knn_df[[target_col]][-idx]

prop.table(table(train_labels))
prop.table(table(test_labels))


# --------------------------------------------------
# KNN: try multiple k values, pick best, show confusion matrix
# --------------------------------------------------

# [SKIPPED KNN SECTION FOR SPEED]
# ks <- c(1, 3, 5, 7, 9, 11)
# ... (commented out)



# View the first few rows of the completedadm column
print(head(tfa_df$Completed.Admissions.Process))

# Check the number of unique values in the completedadm column
print(length(unique(tfa_df$Completed.Admissions.Process)))

# Scale numeric variables
numeric_columns <- c('Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length', 'Essays.Unique.Words', 'Essays.Sentiment', 'Sign.up.Date', 'Started.Date', 'Submitted.Date')
tfa_df[, numeric_columns] <- scale(tfa_df[, numeric_columns])

# Summary of scaled numeric columns
summary(tfa_df[numeric_columns])
###knn models
nb.df<-tfa_df
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
###basic model
# [SKIPPED NAIVE BAYES SECTION FOR SPEED]
# nb.basic <- naiveBayes(Completed.Admissions.Process ~ ., data = train.df)
# ...


# DECISION TREE MODEL (C5.0)

library(C50)
library(caret)

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
confusionMatrix(pred.tuned, dt.test$Completed.Admissions.Process)

# Example Predictions (using actual rows from test set)

example1 <- dt.test[1, ]
predict(dt.tuned, newdata = example1)

example2 <- dt_test[2, ]
predict(dt.tuned, newdata = example2)






