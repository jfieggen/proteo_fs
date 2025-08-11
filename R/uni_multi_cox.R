###############################################################################
# cax_lasso_v2.R
# Complete script for proteomic feature selection via Cox LASSO with bootstrapping,
# including preprocessing, univariate ranking (by coefficient size),
# simple Cox on top features, and bootstrapped LASSO for average hazard ratios.
###############################################################################

# ----- 1) LOAD LIBRARIES -----
library(survival)      # For Cox models
library(dplyr)         # Data wrangling
library(tidyr)         # Pivoting if needed

# ----- 2) USER PATHS & SETTINGS -----
features_path <- "/well/clifton/users/ncu080/proteo_fs/data/train_features.csv"
outcomes_path <- "/well/clifton/users/ncu080/proteo_fs/data/train_outcome_cox.csv"
out_dir      <- "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/R_cox_lasso"

# ----- 3) READ DATA -----
cat("Reading features and outcomes...\n")
features <- read.csv(features_path)
outcomes <- read.csv(outcomes_path)
cat("Initial number of features:", ncol(features), "\n")
cat("Outcome event counts:\n")
print(table(outcomes$outcome))

# ----- 4) PREPROCESSING OF FEATURES -----
cat("\n--- Preprocessing features ---\n")
# A) Remove near-zero variance features
nzv_threshold <- 1e-10
nzv <- apply(features, 2, function(col) var(col, na.rm = TRUE)) < nzv_threshold
if(any(nzv)) {
  cat("Removing", sum(nzv), "near-zero variance features.\n")
  features <- features[, !nzv, drop = FALSE]
}
cat("Features remaining after NZV removal:", ncol(features), "\n")

# B) Remove highly correlated features (R^2 > 0.9)
cor_matrix <- cor(features, use = "pairwise.complete.obs")
n_col <- ncol(cor_matrix)
to_remove <- c()
missing_counts <- colSums(is.na(features))  # for tie-break

for(i in seq_len(n_col - 1)) {
  for(j in (i+1):n_col) {
    if(!is.na(cor_matrix[i, j]) && (cor_matrix[i, j]^2 > 0.9)) {
      col_to_remove <- ifelse(missing_counts[i] >= missing_counts[j], i, j)
      to_remove <- c(to_remove, col_to_remove)
    }
  }
}
to_remove <- unique(to_remove)
if(length(to_remove) > 0) {
  cat("Removing", length(to_remove), "highly correlated features.\n")
  features <- features[, -to_remove, drop = FALSE]
}
cat("Features remaining after correlation filter:", ncol(features), "\n")

# C) Impute missing values with column medians
features <- as.data.frame(
  apply(features, 2, function(col) {
    col[is.na(col)] <- median(col, na.rm = TRUE)
    col
  })
)
cat("Missing value imputation complete.\n")

# Optionally standardize/scale features:
features <- scale(features)

# ----- 5) MERGE OUTCOMES WITH FEATURES -----
data_all <- cbind(outcomes, features)
cat("Merged data dimensions:", dim(data_all), "\n")

#write csv
write.csv(data_all, "/well/clifton/users/ncu080/proteo_fs/data/cox_processed_data.csv")

# Create survival object (assumes columns "time" and "outcome" exist)
y_surv <- Surv(data_all$time, data_all$outcome)

# ----- 6) ADD NOISE FEATURES -----
cat("\n--- Adding noise features ---\n")
# Generate 10 random variables using the range of the first proteomic feature.
ref_min <- min(features[[1]], na.rm = TRUE)
ref_max <- max(features[[1]], na.rm = TRUE)
rand_vars <- replicate(10, runif(nrow(data_all), min = ref_min, max = ref_max))
rand_vars <- as.data.frame(rand_vars)
colnames(rand_vars) <- paste0("rand_var_", 1:10)
# Bind noise features to data_all
data_all <- cbind(data_all, rand_vars)
# Define real feature names and update total features list.
real_feature_names <- colnames(features)
all_feature_names <- c(real_feature_names, colnames(rand_vars))
cat("Total features including noise:", length(all_feature_names), "\n")

# ----- 7) UNIVARIATE COX RANKING (SORTED BY COEFFICIENT SIZE) -----
cat("\n--- Running univariate Cox models (ranking by absolute coefficient) ---\n")
univ_results <- data.frame(
  feature = real_feature_names,
  coefficient = NA_real_,
  abs_coefficient = NA_real_,
  p_value = NA_real_,
  stringsAsFactors = FALSE
)

for(i in seq_along(real_feature_names)) {
  fn <- real_feature_names[i]
  fit <- coxph(y_surv ~ data_all[[fn]], data = data_all)
  s <- summary(fit)
  univ_results$coefficient[i] <- s$coefficients[,"coef"][1]
  univ_results$abs_coefficient[i] <- abs(univ_results$coefficient[i])
  univ_results$p_value[i] <- s$coefficients[,"Pr(>|z|)"][1]
}
# Sort descending by absolute coefficient
univ_results <- univ_results %>% arrange(desc(abs_coefficient))

# Write full univariate results (all feature effects) to CSV
all_features_file <- file.path(out_dir, "univ_ranking_sorted_by_coef.csv")
write.csv(univ_results, all_features_file, row.names = FALSE)
cat("Full univariate ranking saved to:", all_features_file, "\n")

# Compute Bonferroni corrected p-values
n_tests <- nrow(univ_results)
univ_results$p_bonferroni <- pmin(univ_results$p_value * n_tests, 1)

# Write out only those features with corrected p-value < 0.05 to a separate CSV
sig_univ_results <- univ_results %>% filter(p_bonferroni < 0.05)
sig_features_file <- file.path(out_dir, "univ_ranking_significant_by_bonferroni.csv")
write.csv(sig_univ_results, sig_features_file, row.names = FALSE)
cat("Significant features (Bonferroni corrected p < 0.05) saved to:", sig_features_file, "\n")

# ----- 8) SIMPLE MULTIVARIABLE COX MODEL WITH TOP 50 FEATURES -----
cat("\n--- Fitting multivariable Cox with top 50 features ---\n")
top_50 <- univ_results$feature[1:min(50, nrow(univ_results))]
X_top50 <- data_all[, top_50, drop = FALSE]
fit_top50 <- coxph(Surv(time, outcome) ~ ., data = cbind(outcomes, X_top50))
coxtxtfile <- file.path(out_dir, "cox_top50_summary.txt")
sink(coxtxtfile)
cat("=== Multivariable Cox Model (Top 50 Features) ===\n")
print(summary(fit_top50))
sink()
cat("Simple Cox summary saved to:", coxtxtfile, "\n")

cat("\n--- All steps completed successfully. ---\n")
