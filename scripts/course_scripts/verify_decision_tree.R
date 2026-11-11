# Verify Decision Tree Logic
library(caret)
library(C50)
library(e1071)

# 1. Load Data
if (!file.exists("data/tfa_data.csv")) {
  stop("tfa_data.csv not found")
}
tfa_df <- read.csv("data/tfa_data.csv", stringsAsFactors = FALSE)
tfa_df <- na.omit(tfa_df)

if ("personid" %in% colnames(tfa_df)) tfa_df$personid <- NULL
if ("appyear" %in% colnames(tfa_df)) tfa_df$appyear <- NULL

target_col <- "Completed.Admissions.Process"
tfa_df[[target_col]] <- as.factor(tfa_df[[target_col]])

# 2. Prep Data (mimicking NB section)
nb.df <- tfa_df
numeric_columns <- c(
  'Cumulative.GPA', 'Essay.1.Length', 'Essay.2.Length', 'Essay.3.Length',
  'Essays.Unique.Words', 'Essays.Sentiment',
  'Sign.up.Date', 'Started.Date', 'Submitted.Date'
)

# Fix for dates (found during SVM debug)
date_cols <- c('Sign.up.Date', 'Started.Date', 'Submitted.Date')
for (col in date_cols) {
  if (col %in% colnames(nb.df)) {
    nb.df[[col]] <- as.numeric(as.Date(nb.df[[col]]))
  }
}

numeric_exists <- numeric_columns[numeric_columns %in% colnames(nb.df)]
if(length(numeric_exists) > 0) {
    nb.df[, numeric_exists] <- scale(nb.df[, numeric_exists])
}

# 3. Decision Tree Section
dt.df <- nb.df
dt.df$Completed.Admissions.Process <- factor(dt.df$Completed.Admissions.Process)

set.seed(2001)
# SUBSET FOR VERIFICATION
subset_idx <- sample(nrow(dt.df), 2000)
dt.df <- dt.df[subset_idx, ]
dt.index <- createDataPartition(dt.df$Completed.Admissions.Process, p = 0.8, list = FALSE)
dt.train <- dt.df[dt.index, ]
dt.test  <- dt.df[-dt.index, ]

cat("\nTraining Basic DT...\n")
dt.basic <- train(
  Completed.Admissions.Process ~ .,
  data = dt.train,
  method = "C5.0",
  trControl = trainControl(method = "cv", number = 5), # Reduced to 5 for speed
  tuneLength = 3 # Reduced for speed
)
print(dt.basic)

cat("\nPredicting...\n")
pred.basic <- predict(dt.basic, newdata = dt.test)
cm <- confusionMatrix(pred.basic, dt.test$Completed.Admissions.Process)
print(cm)

cat("\nDecision Tree Verification Complete.\n")
