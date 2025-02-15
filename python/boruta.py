#!/usr/bin/env python
"""
boruta.py

This script uses the Boruta feature selection method to identify the most important features 
from the processed training data. It loads the processed training features and outcome from:
    /well/clifton/users/ncu080/proteo_fs/data/train_features.csv
    /well/clifton/users/ncu080/proteo_fs/data/train_outcome.csv

It uses a RandomForestClassifier as the estimator for Boruta and selects features deemed relevant. 
If more than 20 features are selected, it ranks them by importance and retains only the top 20.

The selected features are printed and saved to:
    /well/clifton/users/ncu080/proteo_fs/data/boruta_selected_features.txt
"""

import os
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """Load processed training features and outcome."""
    data_dir = "/well/clifton/users/ncu080/proteo_fs/data"
    train_features_file = os.path.join(data_dir, "train_features.csv")
    train_outcome_file = os.path.join(data_dir, "train_outcome.csv")
    
    X = pd.read_csv(train_features_file)
    # Read the outcome; squeeze converts a single-column DataFrame to a Series.
    y = pd.read_csv(train_outcome_file).squeeze()
    return X, y

def main():
    X, y = load_data()
    
    # Set up a RandomForestClassifier as the estimator for Boruta.
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight='balanced',
        max_depth=5,
        random_state=42
    )
    
    # Initialize BorutaPy with the random forest.
    boruta_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )
    
    print("Running Boruta feature selection...")
    # Note: BorutaPy expects numpy arrays.
    boruta_selector.fit(X.values, y.values)
    
    # Get the boolean mask of selected features.
    selected_mask = boruta_selector.support_
    selected_features = X.columns[selected_mask]
    
    print(f"\nBoruta selected {len(selected_features)} features.")
    
    # If more than 20 features are selected, rank them by importance and take the top 20.
    if len(selected_features) > 20:
        # The underlying estimator (refitted on the selected features) provides feature importances.
        importances = boruta_selector.estimator_.feature_importances_
        # Create a DataFrame to rank features.
        df_feat = pd.DataFrame({
            'feature': selected_features,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        top_features = df_feat['feature'].head(20).tolist()
        print("\nTop 20 features selected by Boruta (ranked by importance):")
        for feat in top_features:
            print(feat)
    else:
        top_features = selected_features.tolist()
        print("\nSelected features by Boruta:")
        for feat in top_features:
            print(feat)
    
    # Save the selected features to a file.
    output_file = os.path.join("/well/clifton/users/ncu080/proteo_fs/outputs/top_features", "boruta_selected_features.txt")
    with open(output_file, "w") as f:
        for feat in top_features:
            f.write(f"{feat}\n")
    print(f"\nSelected features saved to: {output_file}")

if __name__ == "__main__":
    main()
