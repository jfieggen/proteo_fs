###############################################################################
# cox_lasso_feature_selection_modified_5fold.R
# 
# Changes from previous version:
# - Only 1 round of 5-fold CV.
# - Removed ROSE resampling.
# - Use a lambda grid of only 5 values.
# - Timing messages (Sys.time()) added to measure overhead of each major step.
# - Omit the second bootstrapped LASSO run ("all features") and only run with
#   the filtered features.
# - Replaced caret’s train() call with cv.glmnet for improved performance.
###############################################################################

library(survival)
library(glmnet)
library(dplyr)
library(tidyr)
library(caret)
# library(ROSE)  # Not used now
library(parallel)
library(purrr)
library(tibble)

# ----- TIMING FUNCTION WRAPPER -----
log_time <- function(step_name) {
  cat("\n[", step_name, "] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
}

# ----- 1) READ PREPROCESSED DATA -----
log_time("Start: Reading data")
data_all <- read.csv("/well/clifton/users/ncu080/proteo_fs/data/cox_processed_data.csv")
cat("Read processed data with dimensions:", dim(data_all), "\n")

# Assume outcome columns are named "time" and "outcome".
# All other columns are assumed to be features.
protein_list <- setdiff(colnames(data_all), c("time", "outcome"))
cat("Number of real features:", length(protein_list), "\n")

# Create a factor version of the outcome for classification
data_all$outcome_fac <- as.factor(data_all$outcome)

# ----- Median Imputation -----
log_time("Median Imputation")
cat("--- Imputing missing values using column medians ---\n")
data_all[protein_list] <- lapply(data_all[protein_list], function(col) {
  col[is.na(col)] <- median(col, na.rm = TRUE)
  return(col)
})
cat("Median imputation complete.\n")

# ----- 2) ADD NOISE FEATURES -----
log_time("Add noise features")
cat("\n--- Adding noise features ---\n")
ref_min <- min(data_all[[protein_list[1]]], na.rm = TRUE)
ref_max <- max(data_all[[protein_list[1]]], na.rm = TRUE)
rand_vars <- replicate(10, runif(nrow(data_all), min = ref_min, max = ref_max))
rand_vars <- as.data.frame(rand_vars)
colnames(rand_vars) <- paste0("rand_var_", 1:10)
data_all <- cbind(data_all, rand_vars)
cat("Added 10 noise features.\n")

# Update feature lists
real_feature_names <- protein_list
all_feature_names <- c(real_feature_names, colnames(rand_vars))
cat("Total features including noise:", length(all_feature_names), "\n")

# ----- 3) UNIVARIATE ASSOCIATIONS -----
log_time("Univariate Cox")
cat("\n--- Univariate Cox associations ---\n")
# 3a. Cox
cox_univ <- lapply(real_feature_names, function(feat) {
  fit <- coxph(Surv(time, outcome) ~ data_all[[feat]], data = data_all)
  s <- summary(fit)
  coef_val <- s$coefficients[,"coef"][1]
  p_val <- s$coefficients[,"Pr(>|z|)"][1]
  data.frame(feature = feat, cox_coef = coef_val, cox_p = p_val)
})
cox_univ_df <- do.call(rbind, cox_univ)
n_tests <- nrow(cox_univ_df)
cox_univ_df$p_bonferroni <- pmin(cox_univ_df$cox_p * n_tests, 1)
# Sort by corrected p-value
cox_univ_df <- cox_univ_df %>% arrange(p_bonferroni)
write.csv(cox_univ_df, file = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/cox_univariate.csv", row.names = FALSE)
cat("Univariate Cox associations saved.\n")

log_time("Univariate Logistic")
cat("\n--- Univariate Logistic associations ---\n")
# 3b. Logistic
logistic_univ <- lapply(real_feature_names, function(feat) {
  fit <- glm(as.factor(outcome) ~ data_all[[feat]], data = data_all, family = "binomial")
  s <- summary(fit)
  coef_val <- s$coefficients[2, "Estimate"]
  p_val <- s$coefficients[2, "Pr(>|z|)"]
  data.frame(feature = feat, log_coef = coef_val, log_p = p_val)
})
logistic_univ_df <- do.call(rbind, logistic_univ)
n_tests_log <- nrow(logistic_univ_df)
logistic_univ_df$p_bonferroni <- pmin(logistic_univ_df$log_p * n_tests_log, 1)
logistic_univ_df <- logistic_univ_df %>% arrange(p_bonferroni)
write.csv(logistic_univ_df, file = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/logistic_univariate.csv", row.names = FALSE)
cat("Univariate Logistic associations saved.\n")

# ----- 4) DEFINE FILTERED FEATURES -----
log_time("Filtering features")
filtered_features <- cox_univ_df %>% filter(p_bonferroni < 0.05) %>% pull(feature)
cat("Number of features passing Cox univariate significance (p_bonferroni < 0.05):", length(filtered_features), "\n")

# ----- 5) SET BOOTSTRAP & LASSO PARAMETERS -----
# We will run logistic LASSO (family="binomial") using cv.glmnet.
n_boot <- 12   # number of bootstrap iterations
subsample_frac <- 0.50

# A smaller lambda grid with 5 values.
# This is just an example set spanning 0.001 to ~0.316.
lambda_grid <- 10^(-seq(3, 0.5, length.out = 5))

# ----- 6) BOOTSTRAPPED LASSO FUNCTION (Using cv.glmnet) -----
run_bootstrap_lasso <- function(features_to_use) {
  log_time("Start bootstrapped LASSO function")
  # Use the full dataset as training data.
  u.train <- data_all
  # Define predictors: selected features + noise features.
  cols_use <- c(features_to_use, colnames(rand_vars))
  
  # Draw bootstrap subsamples (n_boot iterations, 50% subsample, with replacement)
  jj <- lapply(1:n_boot, function(x) sample(nrow(u.train), round(nrow(u.train) * subsample_frac), replace = TRUE))
  
  # Run logistic LASSO on each bootstrap using mclapply (parallel).
  las.morb <- mclapply(seq_along(jj), function(i) {
    cat("Bootstrap iteration:", i, "at", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
    tryCatch({
      # Prepare training data as a matrix (required by glmnet)
      x_train <- as.matrix(u.train[jj[[i]], cols_use])
      y_train <- u.train[jj[[i]], "outcome_fac"]
      # cv.glmnet performs its own 5-fold cross-validation
      cv_fit <- cv.glmnet(x_train, y_train,
                          family = "binomial",
                          alpha = 1,
                          lambda = lambda_grid,
                          nfolds = 5)
      # Extract coefficients at the optimal lambda (lambda.min)
      cf <- as.matrix(coef(cv_fit, s = "lambda.min"))
      gc()
      return(cf)
    }, error = function(e) {
      cat("Error in bootstrap iteration", i, ":", e$message, "\n")
      # Return a matrix of zeros with the expected dimensions:
      n_rows <- ncol(u.train[, cols_use, drop = FALSE]) + 1  # +1 for the intercept
      return(matrix(0, nrow = n_rows, ncol = 1))
    })
  }, mc.cores = 12, mc.allow.recursive = TRUE)
  
  # Remove any NULL entries (if any)
  las.morb <- Filter(Negate(is.null), las.morb)
  if (length(las.morb) == 0) {
    stop("All bootstrap iterations failed.")
  }
  
  # Save bootstrapped LASSO model objects.
  out_file <- "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/bs_lasso_filtered.RData"
  save(las.morb, file = out_file)
  cat("Bootstrapped LASSO models saved to:", out_file, "\n")
  
  # Generate feature selection ranking:
  p.select <- matrix(NA, nrow = nrow(las.morb[[1]]), ncol = length(las.morb))
  for(i in seq_along(las.morb)) {
    tmp <- tryCatch({
      las.morb[[i]][,1]
    }, error = function(e) rep(0, nrow(las.morb[[1]])))
    p.select[, i] <- tmp
  }
  
  colnames(p.select) <- paste0("boot_", seq_len(ncol(p.select)))
  row.names(p.select) <- rownames(las.morb[[1]])
  
  # Convert to tibble, filter out the intercept, compute aggregate importance
  p.select_df <- as_tibble(p.select, rownames = "feature") %>%
    filter(feature != "(Intercept)") %>%
    mutate(select = abs(rowSums(across(where(is.numeric))))) %>%
    arrange(desc(select))
  
  p.select_df <- as.data.frame(p.select_df)
  p.select_df$select.perc <- p.select_df$select / max(p.select_df$select)
  
  # Save ranking.
  rank_file <- "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/feature_selection_ranking_filtered.csv"
  write.csv(p.select_df, rank_file, row.names = FALSE)
  cat("Feature selection ranking saved to:", rank_file, "\n")
  
  log_time("End bootstrapped LASSO function")
  return(p.select_df)
}

# ----- 7) RUN BOOTSTRAPPED LASSO ON FILTERED FEATURES (plus noise) ONLY -----
log_time("Run: Bootstrapped LASSO (filtered)")
ranking_filtered <- run_bootstrap_lasso(filtered_features)

cat("\n--- All steps completed successfully. ---\n")
log_time("End of Script")
