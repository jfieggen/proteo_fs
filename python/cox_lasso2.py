#!/usr/bin/env python

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from lifelines import CoxPHFitter

# --- File paths ---
TRAIN_FEATURES_PATH = "/well/clifton/users/ncu080/proteo_fs/data/train_features.csv"
TRAIN_OUTCOME_PATH = "/well/clifton/users/ncu080/proteo_fs/data/train_outcome_cox.csv"

MODEL_SAVE_DIR = "/well/clifton/users/ncu080/proteo_fs/outputs/models/cox_lasso"
COEF_SAVE_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/all_cox_lasso.csv"
SUMMARY_SAVE_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/cox_lasso_summary.csv"
GENERAL_COX_COEF_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/general_cox_top20.csv"

# Create directories if they don't exist.
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COEF_SAVE_PATH), exist_ok=True)

# --- Helper functions for feature filtering ---

def remove_correlated_features(X, corr_threshold=0.9):
    """
    Identify and remove one feature from each pair of highly correlated features.
    The feature with more missing values is dropped.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    
    correlated_pairs = np.where(upper > corr_threshold)
    for row_idx, col_idx in zip(*correlated_pairs):
        feat1 = upper.index[row_idx]
        feat2 = upper.columns[col_idx]
        if feat1 in to_drop or feat2 in to_drop:
            continue
        missing1 = X[feat1].isnull().sum()
        missing2 = X[feat2].isnull().sum()
        if missing1 >= missing2:
            to_drop.add(feat1)
        else:
            to_drop.add(feat2)
    
    return list(to_drop)

def remove_low_variance_features(X, var_threshold=1e-10):
    """
    Identify features with near-zero variance.
    """
    low_variance = X.var() < var_threshold
    return list(X.columns[low_variance])

# --- Define scoring function for grid search ---
def cox_concordance_scorer(estimator, X, y):
    """
    Computes the concordance index (C-index) for the Cox model.
    If the estimated coefficients are all (or nearly) zero, return a very poor score.
    Note: We use the negative of the risk score because a higher risk implies lower survival time.
    """
    # Access the underlying Cox model from the pipeline.
    cox_model = estimator.named_steps["cox"]
    # Penalize the all-zero solution
    if np.all(np.abs(cox_model.coef_) < 1e-6):
        return -999  # A very poor score to discourage selection of an all-zero model.
    
    risk_scores = -estimator.predict(X)
    c_index = concordance_index_censored(y["outcome"], y["time"], risk_scores)[0]
    return c_index

# --- Function to pick the best alpha purely by highest CV score ---
def pick_best_alpha(cv_results, candidate_alphas):
    """
    From grid search CV results, select the alpha with the highest mean test concordance index.
    """
    mean_scores = cv_results["mean_test_score"]
    best_index = np.argmax(mean_scores)
    return candidate_alphas[best_index]

# --- Bootstrap training function for one iteration ---
def Bootstrap_train_cox(X, y, seed, performance_CUTOFF, candidate_alphas, imputation_strategy="mean"):
    """
    For a given bootstrap seed:
      - Sample the data with replacement.
      - Build a pipeline (imputation -> scaling -> LASSO Cox model).
      - Run grid search (CV=5) over candidate alphas, using the concordance index as performance metric.
      - Select the alpha that yields the best mean CV score.
      - Retrain on the full bootstrap sample, save the model, and return the coefficients.
    """
    print(f"Bootstrap seed {seed} starting...")
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X.iloc[indices].reset_index(drop=True)
    y_boot = y[indices]  # y is a structured numpy array
    
    # Build pipeline: impute -> scale -> Cox LASSO
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
        ("scaler", StandardScaler()),
        ("cox", CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=20000))
    ])
    
    # Grid search over candidate alphas.
    param_grid = {"cox__alphas": [np.array([alpha]) for alpha in candidate_alphas]}
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring=cox_concordance_scorer, n_jobs=1, verbose=1)
    gs.fit(X_boot, y_boot)
    
    best_alpha = pick_best_alpha(gs.cv_results_, list(candidate_alphas))
    
    # Retrain pipeline with selected best_alpha.
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation_strategy)),
        ("scaler", StandardScaler()),
        ("cox", CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=20000, alphas=np.array([best_alpha])))
    ])
    final_pipeline.fit(X_boot, y_boot)
    
    # Save model.
    model_filename = os.path.join(MODEL_SAVE_DIR, f"cox_lasso_seed{seed}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(final_pipeline, f)
    
    # Extract coefficients.
    cox_model = final_pipeline.named_steps["cox"]
    coefs = cox_model.coef_  # shape: (n_features,)
    feature_names = X.columns.tolist()
    coef_dict = {feat: coef for feat, coef in zip(feature_names, coefs)}
    
    result = {"seed": seed, "best_alpha": best_alpha}
    result.update(coef_dict)
    
    print(f"Bootstrap seed {seed} finished with best alpha = {best_alpha}")
    return result

# --- Main training function ---
def Train_cox_lasso_bootstrap(n_bootstraps=500, performance_CUTOFF=0.95, imputation_strategy="mean", n_jobs=10):
    # Load training features and outcomes.
    X = pd.read_csv(TRAIN_FEATURES_PATH)
    df_y = pd.read_csv(TRAIN_OUTCOME_PATH)
    
    # Ensure outcome data contains required columns.
    if "time" not in df_y.columns or "outcome" not in df_y.columns:
        raise ValueError("Outcome file must have columns 'time' and 'outcome'.")
    
    # --- Feature Filtering ---
    # 1. Remove near-constant (low variance) features.
    low_var_features = remove_low_variance_features(X, var_threshold=1e-10)
    if low_var_features:
        print(f"Removing near-constant features: {low_var_features}")
        X = X.drop(columns=low_var_features)
    else:
        print("No near-constant features detected.")
    
    # 2. Remove one feature from each highly correlated pair.
    correlated_features = remove_correlated_features(X, corr_threshold=0.9)
    if correlated_features:
        print(f"Removing highly correlated features (corr > 0.9): {correlated_features}")
        X = X.drop(columns=correlated_features)
    else:
        print("No highly correlated features detected.")
    
    # 3. Warning: more features than samples.
    n_samples, n_features = X.shape
    if n_features > n_samples:
        print(f"Warning: Number of features ({n_features}) exceeds number of samples ({n_samples}).")
    
    # 4. Check for features with >50% missing data.
    missing_fraction = X.isnull().mean()
    high_missing = missing_fraction[missing_fraction > 0.5].index.tolist()
    if high_missing:
        print(f"Warning: The following features have >50% missing data: {high_missing}")
    
    # --- Create structured array for survival outcome ---
    y = Surv.from_dataframe(event="outcome", time="time", data=df_y)
    
    # --- 1) Baseline (Almost Unpenalized) Cox Model for Sanity Check ---
    # Use lifelines' CoxPHFitter (similar to R's coxph) with a tiny ridge penalization.
    print("Training a baseline (almost unpenalized) Cox model using lifelines for sanity check...")
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    # Create a DataFrame for lifelines that includes survival columns.
    df_for_lifelines = X_scaled.copy()
    df_for_lifelines["time"] = df_y["time"].values
    df_for_lifelines["event"] = df_y["outcome"].values

    # Use a tiny ridge penalizer (penalizer=1e-5) to improve numerical stability.
    cph = CoxPHFitter(penalizer=1e-5, max_steps=500, tol=1e-7)
    cph.fit(df_for_lifelines, duration_col="time", event_col="event", show_progress=True)
    
    # Extract coefficients and compute hazard ratios.
    coef_series = cph.params_.copy()
    hr_series = np.exp(coef_series)
    sorted_by_abs = coef_series.abs().sort_values(ascending=False)
    top20_features = sorted_by_abs.index[:20]
    df_general_cox_top20 = pd.DataFrame({
        "feature": top20_features,
        "coefficient": coef_series.loc[top20_features].values,
        "hazard_ratio": hr_series.loc[top20_features].values
    })
    df_general_cox_top20.to_csv(GENERAL_COX_COEF_PATH, index=False)
    print(f"Saved top 20 general Cox model coefficients to: {GENERAL_COX_COEF_PATH}\n")
    
    # --- 2) Bootstrapped LASSO Cox Model ---
    # Adjust candidate_alphas range if needed.
    candidate_alphas = np.logspace(-5, 0, 100)
    seeds = list(range(n_bootstraps))
    
    print("Starting bootstrapped LASSO Cox model training...")
    pool = mp.Pool(processes=n_jobs)
    args = [(X, y, seed, performance_CUTOFF, candidate_alphas, imputation_strategy) for seed in seeds]
    results = pool.starmap(Bootstrap_train_cox, args)
    pool.close()
    pool.join()
    
    # Save all bootstrap coefficients.
    df_coefs = pd.DataFrame(results)
    df_coefs.to_csv(COEF_SAVE_PATH, index=False)
    print(f"Saved bootstrap coefficients to: {COEF_SAVE_PATH}")
    
    # Summarize top 20 features across bootstraps.
    feature_cols = [col for col in df_coefs.columns if col not in ["seed", "best_alpha"]]
    summary_list = []
    for feat in feature_cols:
        coef_values = df_coefs[feat].values
        if np.all(coef_values == 0):
            continue
        hazard_ratios = np.exp(coef_values)
        median_hr = np.median(hazard_ratios)
        lower_hr = np.percentile(hazard_ratios, 2.5)
        upper_hr = np.percentile(hazard_ratios, 97.5)
        freq = np.mean(coef_values != 0)
        summary_list.append({
            "feature": feat,
            "median_hazard_ratio": median_hr,
            "ci_lower": lower_hr,
            "ci_upper": upper_hr,
            "selection_frequency": freq
        })
    df_summary = pd.DataFrame(summary_list)
    df_summary["effect_size"] = np.abs(df_summary["median_hazard_ratio"] - 1)
    df_summary = df_summary.sort_values(by=["selection_frequency", "effect_size"], ascending=False)
    top20 = df_summary.head(20)
    top20.to_csv(SUMMARY_SAVE_PATH, index=False)
    print(f"Saved bootstrapped LASSO summary to: {SUMMARY_SAVE_PATH}")
    
    return df_coefs, top20

if __name__ == "__main__":
    # Parameters (adjust as needed).
    n_bootstraps = 2  # For quick testing; increase for full analysis.
    performance_CUTOFF = 0.80
    imputation_strategy = "mean"  # Options: "mean", "median", "most_frequent", etc.
    n_jobs = 10  # Number of parallel processes.
    
    df_coefs, top20 = Train_cox_lasso_bootstrap(
        n_bootstraps=n_bootstraps,
        performance_CUTOFF=performance_CUTOFF,
        imputation_strategy=imputation_strategy,
        n_jobs=n_jobs
    )
