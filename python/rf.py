#!/usr/bin/env python
"""
rf.py

This script performs feature selection using a Random Forest classifier.
It loads the processed training data from:
    /well/clifton/users/ncu080/proteo_fs/data/train_features.csv
    /well/clifton/users/ncu080/proteo_fs/data/train_outcome.csv

GridSearchCV with 5-fold CV is used to tune hyperparameters of the Random Forest.
After training, the script extracts feature importances, ranks them, and selects the top 20 features.
The selected features are saved to:
    /well/clifton/users/ncu080/proteo_fs/data/rf_selected_features.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_data():
    data_dir = "/well/clifton/users/ncu080/proteo_fs/data"
    X = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    # Squeeze converts single-column DataFrame to a Series.
    y = pd.read_csv(os.path.join(data_dir, "train_outcome.csv")).squeeze()
    return X, y

def main():
    X, y = load_data()
    
    # Define a RandomForestClassifier.
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Define a parameter grid for grid search.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=5,
                               scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    best_rf = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print(f"Best CV ROC AUC Score: {best_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Extract and rank feature importances.
    importances = best_rf.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    top_features = feature_names[sorted_idx][:20]
    
    print("\nTop 20 features by importance:")
    for feat in top_features:
        print(feat)
    
    # Save the selected features to a file.
    output_file = "/well/clifton/users/ncu080/proteo_fs/data/rf_selected_features.txt"
    with open(output_file, "w") as f:
        for feat in top_features:
            f.write(f"{feat}\n")
    print(f"\nSelected features saved to: {output_file}")

if __name__ == "__main__":
    main()
