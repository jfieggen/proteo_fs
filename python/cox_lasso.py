#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import pickle

# --- File paths ---
TRAIN_FEATURES_PATH = "/well/clifton/users/ncu080/proteo_fs/data/train_features.csv"
TRAIN_OUTCOME_PATH = "/well/clifton/users/ncu080/proteo_fs/data/train_outcome_cox.csv"

MODEL_SAVE_DIR = "/well/clifton/users/ncu080/proteo_fs/outputs/models/cox_lasso"
COEF_SAVE_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/all_cox_lasso.csv"
SUMMARY_SAVE_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/cox_lasso_summary.csv"

# Optional: path to save the top 20 coefficients from the general Cox model
GENERAL_COX_COEF_PATH = "/well/clifton/users/ncu080/proteo_fs/outputs/top_features/general_cox_top20.csv"

# Create directories if they don't exist.
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COEF_SAVE_PATH), exist_ok=True)

# --- Define scoring function ---
def cox_concordance_scorer(estimator, X, y):
    """
    Computes the concordance index (C-index) for the Cox model.
    Note: We use the negative of the risk score because a higher risk implies lower survival time.
    """
    risk_scores = -estimator.predict(X)
    c_index = concordance_index_censored(y['outcome'], y['time'], risk_scores)[0]
    return c_index

# --- Function to pick the best alpha purely by highest CV score ---
def pick_best_alpha(cv_results, candidate_alphas):
    """
    From grid search CV results, select the alpha with the highest mean test concordance index.
    """
    mean_scores = cv_results['mean_test_score']
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
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', StandardScaler()),
        ('cox', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=20000))
    ])
    
    # Grid search over candidate alphas.
    param_grid = {'cox__alphas': [np.array([alpha]) for alpha in candidate_alphas]}
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring=cox_concordance_scorer, n_jobs=1, verbose=1)
    gs.fit(X_boot, y_boot)
    
    best_alpha = pick_best_alpha(gs.cv_results_, list(candidate_alphas))
    
    # Retrain pipeline with selected best_alpha.
    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', StandardScaler()),
        ('cox', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=20000, alphas=np.array([best_alpha])))
    ])
    final_pipeline.fit(X_boot, y_boot)
    
    # Save model.
    model_filename = os.path.join(MODEL_SAVE_DIR, f"cox_lasso_seed{seed}.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(final_pipeline, f)
    
    # Extract coefficients.
    cox_model = final_pipeline.named_steps['cox']
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
    if 'time' not in df_y.columns or 'outcome' not in df_y.columns:
        raise ValueError("Outcome file must have columns 'time' and 'outcome'.")
    
    # Create structured array for survival outcome.
    y = Surv.from_dataframe(event="outcome", time="time", data=df_y)

    # -------------------------------------------------------------------
    # 1) Train an unpenalized Cox model with imputation + scaling
    # -------------------------------------------------------------------
    print("Training a general (unpenalized) Cox model for sanity check...")
    # Create a pipeline for imputation and scaling
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', StandardScaler())
    ])
    # Fit-transform X
    X_processed = preprocessing_pipeline.fit_transform(X)
    
    # Fit unpenalized Cox model on the processed data
    cox_ph = CoxPHSurvivalAnalysis()
    cox_ph.fit(X_processed, y)
    
    # Extract coefficients (cox_ph.coef_) and keep track of feature names
    feature_names = X.columns
    coefs_series = pd.Series(cox_ph.coef_, index=feature_names)
    
    # Sort by absolute magnitude and pick the top 20
    sorted_by_abs = coefs_series.abs().sort_values(ascending=False)
    top20_indices = sorted_by_abs.index[:20]
    top20_coefs = coefs_series.loc[top20_indices]
    
    # Compute hazard ratios for interpretability
    top20_hrs = np.exp(top20_coefs)
    
    # Create a DataFrame with the results
    df_general_cox_top20 = pd.DataFrame({
        "feature": top20_coefs.index,
        "coefficient": top20_coefs.values,
        "hazard_ratio": top20_hrs.values
    })
    
    # Save to CSV
    df_general_cox_top20.to_csv(GENERAL_COX_COEF_PATH, index=False)
    print(f"Saved top 20 general Cox model coefficients to: {GENERAL_COX_COEF_PATH}\n")
    # -------------------------------------------------------------------

    # Now proceed with the LASSO bootstrap approach
    candidate_alphas = np.logspace(-2.5, 0, 100)
    seeds = list(range(n_bootstraps))
    
    # Parallelize bootstrap iterations.
    pool = mp.Pool(processes=n_jobs)
    args = [(X, y, seed, performance_CUTOFF, candidate_alphas, imputation_strategy) for seed in seeds]
    results = pool.starmap(Bootstrap_train_cox, args)
    pool.close()
    pool.join()
    
    # Save all bootstrap coefficients.
    df_coefs = pd.DataFrame(results)
    df_coefs.to_csv(COEF_SAVE_PATH, index=False)
    
    # Summarize top 20 features across bootstraps.
    feature_cols = [col for col in df_coefs.columns if col not in ['seed', 'best_alpha']]
    summary_list = []
    for feat in feature_cols:
        coef_values = df_coefs[feat].values
        # Skip features never selected.
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
    df_summary['effect_size'] = np.abs(df_summary['median_hazard_ratio'] - 1)
    df_summary = df_summary.sort_values(by=['selection_frequency', 'effect_size'], ascending=False)
    top20 = df_summary.head(20)
    top20.to_csv(SUMMARY_SAVE_PATH, index=False)
    
    return df_coefs, top20

if __name__ == '__main__':
    # Parameters (adjust as needed).
    n_bootstraps = 2  # Number of bootstrap iterations, set to 2 for a quick test
    performance_CUTOFF = 0.80
    imputation_strategy = "mean"  # Can be "median" or "most_frequent", etc.
    n_jobs = 10  # Number of parallel processes.
    
    df_coefs, top20 = Train_cox_lasso_bootstrap(
        n_bootstraps=n_bootstraps,
        performance_CUTOFF=performance_CUTOFF,
        imputation_strategy=imputation_strategy,
        n_jobs=n_jobs
    )
