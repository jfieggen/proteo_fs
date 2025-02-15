#!/usr/bin/env python
"""
lasso.py

This script performs feature selection using a Lasso-based logistic regression model.
It loads the processed training data from:
    /well/clifton/users/ncu080/proteo_fs/data/train_features.csv
    /well/clifton/users/ncu080/proteo_fs/data/train_outcome.csv

A pipeline is built that standardizes the features and fits a logistic regression model
with an L1 penalty. GridSearchCV (with 5-fold CV) is used to tune the inverse regularization
parameter C. After training, the script extracts the nonzero coefficients, ranks them by
absolute value, and retains the top 20 features. The selected features are saved to:
    /well/clifton/users/ncu080/proteo_fs/data/lasso_selected_features.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def load_data():
    data_dir = "/well/clifton/users/ncu080/proteo_fs/data"
    X = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    # Squeeze converts single-column DataFrame to a Series.
    y = pd.read_csv(os.path.join(data_dir, "train_outcome.csv")).squeeze()
    return X, y

def main():
    X, y = load_data()
    
    # Build a pipeline: scaling then logistic regression with L1 penalty.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(penalty='l1', solver='liblinear',
                                        random_state=42, max_iter=10000))
    ])
    
    # Define a grid for the inverse regularization parameter C.
    param_grid = {
        'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                               scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print(f"Best CV ROC AUC Score: {best_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Extract coefficients from the logistic regression step.
    coef = best_model.named_steps['logistic'].coef_.ravel()
    feature_names = X.columns
    # Identify features with nonzero coefficients.
    nonzero_mask = coef != 0
    selected_features = feature_names[nonzero_mask]
    selected_coefs = coef[nonzero_mask]
    
    # If more than 20 features are selected, choose the top 20 by absolute coefficient value.
    if len(selected_features) > 20:
        sorted_idx = np.argsort(np.abs(selected_coefs))[::-1]
        top_features = selected_features[sorted_idx][:20]
    else:
        top_features = selected_features
    
    print("\nSelected Features (Top 20 by coefficient magnitude):")
    for feat in top_features:
        print(feat)
    
    # Save the selected features to a file.
    output_file = "/well/clifton/users/ncu080/proteo_fs/data/lasso_selected_features.txt"
    with open(output_file, "w") as f:
        for feat in top_features:
            f.write(f"{feat}\n")
    print(f"\nSelected features saved to: {output_file}")

if __name__ == "__main__":
    main()
