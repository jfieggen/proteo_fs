#!/usr/bin/env python
"""
xgboost_feature_selector.py

This script:
  - Imports the processed training and test datasets (and labels) via the load_data() function from process_data.py,
  - Performs a grid search over an XGBoost model (using the survival:cox objective) with five-fold cross-validation
    and early stopping,
  - Trains a final model on the full training set using the best hyperparameters,
  - Evaluates the final model on both training and test data using:
       * Concordance index (c-index)
       * ROC AUC
       * AUPRC (Average Precision)
  - Saves the final model to disk.
"""

import numpy as np
import xgboost as xgb
from itertools import product
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, average_precision_score

# Import processed data using the new load_data() function from process_data.py.
from process_data import load_data

# Load the processed data.
dtrain, y_train_xgb, dtest, y_test_xgb = load_data()

# Set a random seed for reproducibility
seed = 42

# Define the grid of hyperparameters to search over.
# Note: "n_estimators" in the grid is used as the maximum number of boosting rounds.
param_grid = {
    "learning_rate": [0.1],
    "n_estimators": [50, 100],
    "max_depth": [3, 4],
    "min_child_weight": [4, 6],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],
    "reg_lambda": [2, 4],
    "reg_alpha": [0, 1]
}

# Set base parameters that are common to all model configurations.
base_params = {
    "objective": "survival:cox",      # Use Cox loss for survival analysis
    "eval_metric": "cox-nloglik",      # Negative log partial likelihood
    "tree_method": "gpu_hist",         # Use GPU-optimized histogram algorithm
    "predictor": "gpu_predictor",      # Use GPU for prediction as well
    "seed": seed,
    "verbosity": 1
}

# Prepare for five-fold cross validation.
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

best_score = float("inf")
best_params = None
best_num_rounds = None

# Loop over all parameter combinations in the grid.
grid_keys = list(param_grid.keys())
grid_values = list(param_grid.values())

print("Starting grid search over hyperparameters ...")
for combination in product(*grid_values):
    comb_dict = dict(zip(grid_keys, combination))
    # Merge grid parameters with the base parameters.
    params = {**base_params, **comb_dict}
    
    # For each combination, run 5-fold CV.
    fold_scores = []
    fold_best_rounds = []
    
    for train_idx, val_idx in kf.split(dtrain):
        # Split the data (using .iloc since dtrain is a DataFrame).
        X_train, X_val = dtrain.iloc[train_idx], dtrain.iloc[val_idx]
        y_train, y_val = y_train_xgb.iloc[train_idx], y_train_xgb.iloc[val_idx]
        
        # Create DMatrix objects for XGBoost.
        dtrain_fold = xgb.DMatrix(X_train, label=y_train)
        dval_fold = xgb.DMatrix(X_val, label=y_val)
        
        watchlist = [(dval_fold, 'eval')]
        
        # Use the “n_estimators” value as the maximum boosting rounds.
        n_rounds = params["n_estimators"]
        # Remove "n_estimators" before training as xgb.train does not accept it.
        params_train = params.copy()
        params_train.pop("n_estimators", None)
        
        # Train with early stopping (patience = 10 rounds).
        model = xgb.train(
            params_train,
            dtrain_fold,
            num_boost_round=n_rounds,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        fold_scores.append(model.best_score)
        fold_best_rounds.append(model.best_iteration)
    
    avg_score = np.mean(fold_scores)
    avg_rounds = int(np.mean(fold_best_rounds))
    
    print(f"Params: {comb_dict} | Avg CV Score (cox-nloglik): {avg_score:.4f} | Avg best rounds: {avg_rounds}")
    
    if avg_score < best_score:
        best_score = avg_score
        best_params = params.copy()
        best_num_rounds = avg_rounds

print("\nBest hyperparameter combination found:")
print(best_params)
print(f"Best CV cox-nloglik score: {best_score:.4f}")
print(f"Best number of rounds (from CV): {best_num_rounds}")

# Train the final model on the full training data using the best parameters.
dtrain_full = xgb.DMatrix(dtrain, label=y_train_xgb)
final_params = best_params.copy()
# Remove "n_estimators" (it was used only for setting the maximum boosting rounds).
n_rounds_final = final_params.pop("n_estimators", best_num_rounds)

print("\nTraining final model on the full training data ...")
final_model = xgb.train(final_params, dtrain_full, num_boost_round=n_rounds_final)

###############################################################################
# Evaluate the final model on the training data.
###############################################################################
train_preds = final_model.predict(dtrain_full)
train_time = y_train_xgb.abs()         # Survival times (absolute value)
train_event = (y_train_xgb > 0).astype(int)  # Event indicator: 1 if event, 0 if censored

# For c-index, use -train_preds (so that higher risk correlates with event).
train_c_index = concordance_index(train_time, -train_preds, train_event)
print(f"\nFinal Model Concordance Index on training data: {train_c_index:.4f}")

# For classification metrics, use risk scores defined as -train_preds.
train_risk = -train_preds
train_auc = roc_auc_score(train_event, train_risk)
train_auprc = average_precision_score(train_event, train_risk)
print(f"Final Model ROC AUC on training data: {train_auc:.4f}")
print(f"Final Model AUPRC on training data: {train_auprc:.4f}")

###############################################################################
# Evaluate the final model on the test data.
###############################################################################
dtest_matrix = xgb.DMatrix(dtest, label=y_test_xgb)
test_preds = final_model.predict(dtest_matrix)
test_time = y_test_xgb.abs()
test_event = (y_test_xgb > 0).astype(int)

test_c_index = concordance_index(test_time, -test_preds, test_event)
print(f"\nFinal Model Concordance Index on test data: {test_c_index:.4f}")

test_risk = -test_preds
test_auc = roc_auc_score(test_event, test_risk)
test_auprc = average_precision_score(test_event, test_risk)
print(f"Final Model ROC AUC on test data: {test_auc:.4f}")
print(f"Final Model AUPRC on test data: {test_auprc:.4f}")

###############################################################################
# Save the final model.
###############################################################################
model_output_path = "/well/clifton/users/ncu080/proteo_fs/outputs/models/xgb_final_model.json"
final_model.save_model(model_output_path)
print(f"\nFinal model saved to: {model_output_path}")
