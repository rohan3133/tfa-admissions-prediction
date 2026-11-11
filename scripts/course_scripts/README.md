# Original coursework scripts

These are the scripts written during the course while exploring KNN, Naive Bayes, C5.0 decision trees, a neural network (`neuralnet`) and several SVM variants (linear, RBF, kernel comparison, SMOTE). They are kept for transparency and for the modeling ideas they explore.

**For the verified, reproducible analysis use [`../tfa_modeling_pipeline.R`](../tfa_modeling_pipeline.R).** Its results are the ones reported in the repository README.

Known issues in the scripts in this folder, found during the portfolio review:

- **Outcome leakage.** They keep `Confirmed TFA Offer` as a predictor. Every applicant with a confirmed offer has `Completed Admissions Process = 1`, so the column leaks the answer and inflates accuracy.
- **Identifier left in.** The code tries to drop `personid` and `appyear`, but `read.csv()` renames the columns to `Person.Id` and `App.Started.Year..RTAT.`, so the drops never match and the ID stays in as a feature.
- **Dates treated as numbers.** `Sign-up Date`, `Started Date` and `Submitted Date` are text dates; calling `scale()` on them fails in `complete_analysis.R`, and the other scripts convert them to arbitrary factor codes.
- **`complete_analysis.R`, ANN section** reads a file called `tfa_dataset.csv` that is not part of the repository.
- They depend on packages (`C50`, `neuralnet`, `smotefamily`) that were not available when the pipeline was re-run, so they have **not** been re-run end to end.
- RBF-kernel SVM tuning is expensive on ~60,000 training rows. Some scripts tune on a 50% subsample and `svm_quick_pipeline.R` has a quick-run option, but expect long run times.
