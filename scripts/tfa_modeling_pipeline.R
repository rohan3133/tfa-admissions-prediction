# =============================================================================
# Teach For America (TFA) admissions: predicting who completes the process
# -----------------------------------------------------------------------------
# Goal  : Predict whether an applicant completes the admissions process, so that
#         recruiters can identify applicants at risk of dropping out early.
# Data  : data/tfa_data.csv  (74,839 applicants, 2015-2016 cohorts, 39 columns)
# Run   : from the repository root ->  Rscript scripts/tfa_modeling_pipeline.R
# Output: figures/*.png and results/*.csv
#
# Design decisions (see README for the reasoning):
#   * "Confirmed TFA Offer" is NOT used as a predictor. An offer can only be
#     confirmed after the process is completed (every confirmed applicant in the
#     data has Completed = 1), so it would leak the answer.
#   * Raw date columns are turned into timing features (days between steps).
#   * Evaluation uses AUC (with a 95% CI), balanced accuracy and top-decile lift,
#     because 80.7% of applicants complete and accuracy alone is uninformative.
# =============================================================================

suppressPackageStartupMessages({
  library(caret)    # stratified split, tuning, confusion matrices
  library(e1071)    # Naive Bayes
  library(class)    # k-nearest neighbours
  library(rpart)    # decision tree
  library(nnet)     # single-hidden-layer neural network
  library(ranger)   # fast random forest
  library(pROC)     # ROC / AUC
  library(ggplot2)
})

dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# Optional quick smoke test: TFA_SAMPLE=8000 Rscript scripts/tfa_modeling_pipeline.R
SAMPLE <- as.integer(Sys.getenv("TFA_SAMPLE", "0"))
POS <- "Completed"; NEG <- "NotCompleted"

# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------
raw <- read.csv("data/tfa_data.csv", check.names = FALSE, stringsAsFactors = FALSE)
if (SAMPLE > 0) { set.seed(1); raw <- raw[sample(nrow(raw), SAMPLE), ] }
cat("Applicants:", nrow(raw), " | missing values:", sum(is.na(raw)), "\n")

y <- factor(ifelse(raw[["Completed Admissions Process"]] == 1, POS, NEG),
            levels = c(NEG, POS))
cat("Completion rate:", round(mean(y == POS), 4), "\n")
cat("Confirmed offer -> completed:",
    mean(y[raw[["Confirmed TFA Offer"]] == 1] == POS), "(leakage check, must be 1)\n")

# -----------------------------------------------------------------------------
# 2. Split first, then build features (so imputation uses training data only)
# -----------------------------------------------------------------------------
set.seed(123)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)[, 1]
is_train  <- seq_len(nrow(raw)) %in% train_idx

as_days <- function(x) as.numeric(as.Date(x))
gpa_raw <- raw[["Cumulative GPA"]]
gpa_med <- median(gpa_raw[is_train & gpa_raw > 0])   # GPA of exactly 0 = not reported

feat <- data.frame(
  gpa                    = ifelse(gpa_raw == 0, gpa_med, gpa_raw),
  gpa_not_reported       = as.integer(gpa_raw == 0),
  stem_major_minor       = raw[["Is Math, Sci, or Eng Major Minor"]],
  essay1_length          = raw[["Essay 1 Length"]],
  essay2_length          = raw[["Essay 2 Length"]],
  essay3_length          = raw[["Essay 3 Length"]],
  essays_unique_words    = raw[["Essays Unique Words"]],
  essays_sentiment       = raw[["Essays Sentiment"]],
  region_pref_level      = raw[["Region Preference Level"]],
  attended_event         = raw[["Attended Event"]],
  invited                = raw[["Invited"]],
  met                    = raw[["Met"]],
  responded              = raw[["Responded"]],
  # timing features engineered from the date columns
  days_signup_to_start   = as_days(raw[["Started Date"]])  - as_days(raw[["Sign-up Date"]]),
  days_start_to_submit   = as_days(raw[["Submitted Date"]]) - as_days(raw[["Started Date"]]),
  days_submit_before_deadline = as_days(raw[["Application Deadline"]]) - as_days(raw[["Submitted Date"]]),
  stringsAsFactors = FALSE
)
for (k in 0:9) feat[[paste0("essay_topic_", k)]] <- raw[[paste0("Essays Topic ", k)]]
feat$app_year           <- factor(raw[["App Started Year (RTAT)"]])
feat$school_selectivity <- factor(raw[["School Selectivity"]])
feat$major1             <- factor(raw[["Major 1 - Cleaned"]])
feat$major2             <- factor(raw[["Major 2 - Cleaned"]])
feat$minor              <- factor(raw[["Minor - Cleaned"]])
feat$university         <- factor(raw[["Undergraduate University - Cleaned"]])
cat("Predictors:", ncol(feat), "\n")

x_train_df <- feat[train_idx, ]; x_test_df <- feat[-train_idx, ]
y_train <- y[train_idx];         y_test <- y[-train_idx]
base_rate <- mean(y_train == POS)
cat("Train:", nrow(x_train_df), " Test:", nrow(x_test_df),
    " | majority-class baseline accuracy on test:", round(mean(y_test == POS), 4), "\n")

# One-hot design matrix, scaled with TRAINING means / sds (for LR, KNN, NN)
mm_all   <- model.matrix(~ ., data = feat)[, -1, drop = FALSE]
mu       <- colMeans(mm_all[train_idx, , drop = FALSE])
sdv      <- apply(mm_all[train_idx, , drop = FALSE], 2, sd); sdv[sdv == 0] <- 1
X        <- sweep(sweep(mm_all, 2, mu, "-"), 2, sdv, "/")
X_train  <- X[train_idx, , drop = FALSE]; X_test <- X[-train_idx, , drop = FALSE]
cat("Design matrix columns:", ncol(X), "\n")

# -----------------------------------------------------------------------------
# 3. Evaluation helper
#    AUC (DeLong CI), accuracy at 0.5, balanced accuracy / recall at the
#    base-rate threshold, and top-decile lift for flagging non-completers.
# -----------------------------------------------------------------------------
evaluate <- function(p, model, truth = y_test, base = base_rate) {
  r   <- roc(truth, p, levels = c(NEG, POS), direction = "<", quiet = TRUE)
  ci  <- as.numeric(ci.auc(r))
  hard05 <- factor(ifelse(p >= 0.5, POS, NEG), levels = c(NEG, POS))
  hardb  <- factor(ifelse(p >= base, POS, NEG), levels = c(NEG, POS))
  cm  <- confusionMatrix(hardb, truth, positive = NEG)
  n10 <- ceiling(0.10 * length(p))
  low <- order(p)[seq_len(n10)]                       # 10% least likely to complete
  data.frame(
    Model = model,
    AUC = ci[2], AUC_CI_low = ci[1], AUC_CI_high = ci[3],
    Accuracy_at_0.5 = mean(hard05 == truth),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    Recall_NotCompleted = unname(cm$byClass["Sensitivity"]),
    Precision_NotCompleted = unname(cm$byClass["Pos Pred Value"]),
    NotCompleted_Rate_Bottom_Decile = mean(truth[low] == NEG),
    Lift_Bottom_Decile = mean(truth[low] == NEG) / mean(truth == NEG),
    row.names = NULL
  )
}

probs <- list()   # P(Completed) on the test set for each model
secs  <- function(t0) round(as.numeric(difftime(Sys.time(), t0, units = "secs")))

# -----------------------------------------------------------------------------
# 4. Models
# -----------------------------------------------------------------------------

# 4a. Logistic regression (interpretable baseline)
t0 <- Sys.time()
lr <- suppressWarnings(glm(I(y_train == POS) ~ ., data = as.data.frame(X_train), family = binomial))
probs[["Logistic Regression"]] <- suppressWarnings(predict(lr, as.data.frame(X_test), type = "response"))
cat("Logistic regression done in", secs(t0), "s\n")

# 4b. Naive Bayes
t0 <- Sys.time()
nb <- naiveBayes(x = x_train_df, y = y_train)
probs[["Naive Bayes"]] <- predict(nb, x_test_df, type = "raw")[, POS]
cat("Naive Bayes done in", secs(t0), "s\n")

# 4c. Decision tree (rpart, complexity parameter tuned by 5-fold CV on AUC)
t0 <- Sys.time()
set.seed(123)
dt <- train(x = x_train_df, y = y_train, method = "rpart", metric = "ROC",
            trControl = trainControl(method = "cv", number = 5, classProbs = TRUE,
                                     summaryFunction = twoClassSummary),
            tuneGrid = data.frame(cp = c(0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)))
probs[["Decision Tree"]] <- predict(dt, x_test_df, type = "prob")[, POS]
cat("Decision tree: best cp =", dt$bestTune$cp, "| done in", secs(t0), "s\n")

# 4d. k-nearest neighbours. Compact feature set (no high-cardinality dummies),
#     k chosen on a validation hold-out taken from the training data.
t0 <- Sys.time()
compact <- !grepl("^(major1|major2|minor|university)", colnames(X))
set.seed(123)
val   <- sample(nrow(X_train), min(6000, floor(0.15 * nrow(X_train))))
knn_p <- function(train_x, train_y, test_x, k) {
  fit <- knn(train_x, test_x, cl = train_y, k = k, prob = TRUE)
  pr  <- attr(fit, "prob")
  ifelse(fit == POS, pr, 1 - pr)
}
# class::knn stops when k plus tied neighbours exceeds 1,000, so the grid stays below that
ks   <- c(75, 150, 300, 450, 600)
kauc <- sapply(ks, function(k) tryCatch({
  p <- knn_p(X_train[-val, compact], y_train[-val], X_train[val, compact], k)
  as.numeric(auc(roc(y_train[val], p, levels = c(NEG, POS), direction = "<", quiet = TRUE)))
}, error = function(e) NA_real_))
cat("KNN validation AUC by k:", paste(ks, round(kauc, 4), sep = "=", collapse = " "), "\n")
best_k <- NA; p_knn <- NULL
for (k in ks[order(-kauc, na.last = NA)]) {          # best k first; fall back if it errors
  p_knn <- tryCatch(knn_p(X_train[, compact], y_train, X_test[, compact], k), error = function(e) NULL)
  if (!is.null(p_knn)) { best_k <- k; break }
}
stopifnot(!is.null(p_knn))
probs[["KNN"]] <- p_knn
cat("KNN: best k =", best_k, "| done in", secs(t0), "s\n")

# 4e. Neural network (one hidden layer), size / weight decay tuned on a hold-out
t0 <- Sys.time()
y01 <- as.numeric(y_train == POS)
grid <- expand.grid(size = c(6, 12), decay = c(0.001, 0.003, 0.01))
nn_auc <- apply(grid, 1, function(g) {
  set.seed(123)
  m <- nnet(X_train[-val, ], y01[-val], size = g["size"], decay = g["decay"],
            maxit = 150, MaxNWts = 1e6, entropy = TRUE, trace = FALSE)
  p <- as.numeric(predict(m, X_train[val, ]))
  as.numeric(auc(roc(y_train[val], p, levels = c(NEG, POS), direction = "<", quiet = TRUE)))
})
bg <- grid[which.max(nn_auc), ]
cat("NN validation AUC:", paste0("size", grid$size, "/decay", grid$decay, "=", round(nn_auc, 4),
                                  collapse = " "), "\n")
set.seed(123)
nn <- nnet(X_train, y01, size = bg$size, decay = bg$decay, maxit = 150, MaxNWts = 1e6,
           entropy = TRUE, trace = FALSE)
probs[["Neural Network"]] <- as.numeric(predict(nn, X_test))
cat("NN: size =", bg$size, "decay =", bg$decay, "| done in", secs(t0), "s\n")

# 4f. Random forest (ranger) with permutation importance
t0 <- Sys.time()
set.seed(123)
rf <- ranger(y = y_train, x = x_train_df, num.trees = 500, probability = TRUE,
             importance = "permutation", min.node.size = 50, num.threads = 2,
             respect.unordered.factors = "order")
probs[["Random Forest"]] <- predict(rf, x_test_df)$predictions[, POS]
cat("Random forest done in", secs(t0), "s\n")

# 4g. Cohort-shift check: train on the 2015 cohort, test on the 2016 cohort.
#     A random split can overstate performance if cohorts differ, so this asks
#     whether a model built on one year still works on the next.
yr  <- raw[["App Started Year (RTAT)"]]
cols <- setdiff(names(feat), "app_year")
f15 <- feat[yr == 2015, cols]; f16 <- feat[yr == 2016, cols]
med15 <- median(gpa_raw[yr == 2015 & gpa_raw > 0])           # impute with 2015 data only
f15$gpa[f15$gpa_not_reported == 1] <- med15; f16$gpa[f16$gpa_not_reported == 1] <- med15
set.seed(123)
rf_t <- ranger(y = y[yr == 2015], x = f15, num.trees = 500, probability = TRUE,
               min.node.size = 50, num.threads = 2, respect.unordered.factors = "order")
p_t  <- predict(rf_t, f16)$predictions[, POS]
temporal <- evaluate(p_t, "Random Forest, train 2015 -> test 2016", truth = y[yr == 2016],
                     base = mean(y[yr == 2015] == POS))
temporal$Train_n <- sum(yr == 2015); temporal$Test_n <- sum(yr == 2016)
write.csv(temporal, "results/temporal_validation.csv", row.names = FALSE)
cat("\n=== Cohort shift: train 2015 -> test 2016 ===\n"); print(temporal, digits = 3)

# -----------------------------------------------------------------------------
# 5. Results table
# -----------------------------------------------------------------------------
res <- do.call(rbind, lapply(names(probs), function(m) evaluate(probs[[m]], m)))
res <- res[order(-res$AUC), ]
baseline <- data.frame(
  Majority_class_accuracy = mean(y_test == POS),
  Base_non_completion_rate = mean(y_test == NEG),
  Test_n = length(y_test), Train_n = length(y_train)
)
write.csv(res, "results/model_comparison.csv", row.names = FALSE)
write.csv(baseline, "results/baseline.csv", row.names = FALSE)
cat("\n=== Test-set results (sorted by AUC) ===\n"); print(res, digits = 3)
cat("\nMajority-class baseline accuracy:", round(baseline$Majority_class_accuracy, 4), "\n")

imp <- sort(rf$variable.importance, decreasing = TRUE)
imp_df <- data.frame(Feature = names(imp), Importance = unname(imp))
write.csv(imp_df, "results/random_forest_importance.csv", row.names = FALSE)
cat("\nTop 10 features (permutation importance):\n"); print(head(imp_df, 10), digits = 3)

# -----------------------------------------------------------------------------
# 6. Figures
# -----------------------------------------------------------------------------
theme_set(theme_minimal(base_size = 12) +
            theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank()))
pal <- c("#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030", "#718096")

# 6a. ROC curves
roc_df <- do.call(rbind, lapply(names(probs), function(m) {
  r <- roc(y_test, probs[[m]], levels = c(NEG, POS), direction = "<", quiet = TRUE)
  data.frame(Model = sprintf("%s (AUC %.3f)", m, as.numeric(auc(r))),
             FPR = 1 - r$specificities, TPR = r$sensitivities)
}))
p_roc <- ggplot(roc_df, aes(FPR, TPR, colour = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) + scale_colour_manual(values = pal) +
  coord_equal() +
  labs(title = "Predicting admissions completion: ROC curves (held-out test set)",
       x = "False positive rate", y = "True positive rate (recall)", colour = NULL)
ggsave("figures/roc_curves.png", p_roc, width = 8, height = 5.5, dpi = 150, bg = "white")

# 6b. Permutation importance (top 15)
top <- head(imp_df, 15); top$Feature <- factor(top$Feature, levels = rev(top$Feature))
p_imp <- ggplot(top, aes(Importance, Feature)) + geom_col(fill = pal[1]) +
  labs(title = "What predicts completion?",
       subtitle = "Random forest permutation importance (top 15 features)",
       x = "Increase in prediction error when the feature is shuffled", y = NULL)
ggsave("figures/feature_importance.png", p_imp, width = 8, height = 5.5, dpi = 150, bg = "white")

# 6c. Decile lift for the best model by AUC
best <- res$Model[1]
dec  <- data.frame(p = probs[[best]], notc = as.numeric(y_test == NEG))
dec$decile <- cut(rank(dec$p, ties.method = "first"), breaks = quantile(rank(dec$p, ties.method = "first"),
                  probs = seq(0, 1, 0.1)), include.lowest = TRUE, labels = 1:10)
dd <- aggregate(notc ~ decile, dec, mean)
p_dec <- ggplot(dd, aes(decile, notc)) + geom_col(fill = pal[1]) +
  geom_hline(yintercept = mean(dec$notc), linetype = "dashed", colour = pal[5]) +
  scale_y_continuous(labels = scales::percent) +
  annotate("text", x = 10, y = mean(dec$notc), label = "overall rate", vjust = -0.6, hjust = 1, colour = pal[5], size = 3.5) +
  labs(title = sprintf("Non-completion rate by predicted-risk decile (%s)", best),
       x = "Predicted-risk decile (1 = least likely to complete, 10 = most likely)",
       y = "Applicants who did NOT complete")
ggsave("figures/risk_deciles.png", p_dec, width = 8, height = 5, dpi = 150, bg = "white")
write.csv(dd, "results/risk_deciles.csv", row.names = FALSE)

# 6d. Simple, interpretable view of the strongest single drivers (levels kept in natural order)
drv <- function(var, lab, df) {
  g <- aggregate(list(completed = as.numeric(y == POS)), list(level = df[[var]]), mean)
  n <- aggregate(list(n = rep(1, length(y))), list(level = df[[var]]), sum)
  g <- merge(g, n); g$driver <- lab; g$level <- as.character(g$level); g
}
sel_levels <- c("least_selective", "less_selective", "selective", "more_selective", "most_selective", "unknown")
gpa_lv <- c("< 2.5", "2.5-3.0", "3.0-3.5", "3.5+")
voc_lv <- c("Q1 (fewest)", "Q2", "Q3", "Q4 (most)")
ddl_lv <- c("After deadline", "On deadline", "1-7 days before", "8-30 days before", "30+ days before")
evt_lv <- c("No event", "Attended event")
eda <- data.frame(
  gpa_band = cut(feat$gpa, c(-Inf, 2.5, 3.0, 3.5, Inf), labels = gpa_lv),
  vocab = cut(feat$essays_unique_words,
              quantile(feat$essays_unique_words, c(0, .25, .5, .75, 1)), include.lowest = TRUE,
              labels = voc_lv),
  deadline = cut(feat$days_submit_before_deadline, c(-Inf, -1, 0, 7, 30, Inf), labels = ddl_lv),
  selectivity = factor(feat$school_selectivity, levels = sel_levels),
  event = factor(ifelse(feat$attended_event == 1, "Attended event", "No event"), levels = evt_lv),
  stringsAsFactors = FALSE)
panel_order <- c("Cumulative GPA", "Unique words across essays (quartile)", "Submission timing",
                 "School selectivity", "Recruitment event")
eda_long <- rbind(drv("gpa_band", panel_order[1], eda), drv("vocab", panel_order[2], eda),
                  drv("deadline", panel_order[3], eda), drv("selectivity", panel_order[4], eda),
                  drv("event", panel_order[5], eda))
eda_long <- eda_long[eda_long$n >= 200, ]
eda_long$driver <- factor(eda_long$driver, levels = panel_order)
eda_long$level  <- factor(eda_long$level, levels = rev(unique(c(gpa_lv, voc_lv, ddl_lv, sel_levels, evt_lv))))
write.csv(eda_long, "results/completion_rate_by_driver.csv", row.names = FALSE)
p_eda <- ggplot(eda_long, aes(completed, level)) + geom_col(fill = pal[1]) +
  geom_vline(xintercept = mean(y == POS), linetype = "dashed", colour = pal[5]) +
  facet_wrap(~driver, scales = "free_y", ncol = 2) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "Completion rate by applicant characteristic",
       subtitle = "Dashed line = overall completion rate", x = "Completion rate", y = NULL)
ggsave("figures/completion_rate_by_driver.png", p_eda, width = 9, height = 8.5, dpi = 150, bg = "white")

cat("\nDone. See figures/ and results/.\n")
