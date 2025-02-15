#!/usr/bin/env python
"""
process_data_parallel.py

This module loads the training and test datasets and processes the data by:
  - Dropping specified columns,
  - Binarizing the outcome (1 if Myeloma event, 0 if censored),
  - Removing features with > missing_threshold proportion of missing data,
  - Imputing missing values using a Gibbs sampler–style approach via IterativeImputer.
  
In this parallelized version, we use a RandomForestRegressor (with n_jobs=-1) as the estimator,
so that the underlying regression models are fit in parallel. Progress messages are logged
so you can monitor the imputation steps in your SLURM log output.
"""

import os
import time
import logging
import pandas as pd
import numpy as np

# Configure logging to print progress messages with time stamps.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def impute(train_df, test_df=None):
    """
    Impute missing values using IterativeImputer with a parallelized estimator.
    
    This version uses RandomForestRegressor with n_jobs=-1 to speed up the regression steps.
    
    Parameters:
        train_df (DataFrame): Training features with missing values.
        test_df (DataFrame, optional): Test features with missing values.
        
    Returns:
        If test_df is provided, returns (train_imputed_df, test_imputed_df); otherwise returns train_imputed_df.
    """
    logging.info("Starting imputation process...")
    
    # Enable the experimental IterativeImputer.
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor

    # Use a RandomForestRegressor that uses all cores.
    estimator = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    # Set verbose > 0 to get iteration-level progress from the imputer.
    imputer = IterativeImputer(random_state=42, estimator=estimator, max_iter=10, verbose=2)
    
    logging.info("Fitting imputer on training data...")
    start_time = time.time()
    imputed_train = imputer.fit_transform(train_df)
    elapsed = time.time() - start_time
    logging.info("Training imputation completed in %.2f seconds.", elapsed)
    
    train_imputed_df = pd.DataFrame(imputed_train, columns=train_df.columns, index=train_df.index)
    
    if test_df is not None:
        logging.info("Transforming test data using the fitted imputer...")
        start_time = time.time()
        imputed_test = imputer.transform(test_df)
        elapsed = time.time() - start_time
        logging.info("Test data imputation completed in %.2f seconds.", elapsed)
        
        test_imputed_df = pd.DataFrame(imputed_test, columns=test_df.columns, index=test_df.index)
        return train_imputed_df, test_imputed_df
    else:
        return train_imputed_df

def load_data(
    train_file='/well/clifton/projects/ukb_v2/derived/prot_final/train_data.csv',
    test_file='/well/clifton/projects/ukb_v2/derived/prot_final/test_data.csv',
    missing_threshold=0.30
):
    """
    Loads and processes training and test data.

    Processing steps:
      1. Load CSV files.
      2. Binarize outcome: 1 if Myeloma==1 (event), 0 otherwise.
      3. Drop specified columns.
      4. Remove features with > missing_threshold missing data (based on training data).
      5. Impute missing values.
      
    Returns:
        dtrain, y_train, dtest, y_test
    """
    logging.info("Loading training data from: %s", train_file)
    train_data = pd.read_csv(train_file)
    logging.info("Loading test data from: %s", test_file)
    test_data = pd.read_csv(test_file)
    
    logging.info("Binarizing outcomes...")
    y_train = (train_data['Myeloma'] == 1).astype(int)
    y_test = (test_data['Myeloma'] == 1).astype(int)
    
    exclude_columns = [
        'Age', 'Sex', 'PRO_Consort_Participant', 'PRO_WellSampleRun', 
        'PRO_PlateSampleRun', 'PRO_NProteinMeasured', 'Elevated Waist-Hip Ratio', 
        'Ethnic Group', 'Haemoglobin', 'MCV', 'Platelet Count', 'White Blood Cell Count', 
        'Hemaaoglobin (Binary)', 'Anemia', 'Back Pain', 'Chest Pain', 'CRP', 'Calcium', 
        'TP Result', 'Smoking Status', 'Alcohol Status', 'Cholesterol Result', 'C10AA', 
        'ins_index', 'ID', 'Myeloma', 'time'
    ]
    
    logging.info("Dropping specified columns from the datasets...")
    dtrain = train_data.drop(columns=exclude_columns)
    dtest = test_data.drop(columns=exclude_columns)
    
    logging.info("Removing features with more than %.0f%% missing values...", missing_threshold*100)
    missing_percentage = dtrain.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    logging.info("Columns to drop: %s", list(columns_to_drop))
    dtrain = dtrain.drop(columns=columns_to_drop)
    dtest = dtest.drop(columns=columns_to_drop)
    
    logging.info("Starting missing value imputation...")
    start_time = time.time()
    dtrain, dtest = impute(dtrain, dtest)
    elapsed = time.time() - start_time
    logging.info("Imputation finished in %.2f seconds.", elapsed)
    
    return dtrain, y_train, dtest, y_test

if __name__ == "__main__":
    logging.info("Starting data processing...")
    dtrain, y_train, dtest, y_test = load_data()
    
    logging.info("First 50 column names in training features:")
    logging.info(dtrain.columns[:50].tolist())
    
    logging.info("Training features (first 5 rows):\n%s", dtrain.head())
    logging.info("Training outcome (first 5 rows):\n%s", y_train.head())
    logging.info("Test features (first 5 rows):\n%s", dtest.head())
    logging.info("Test outcome (first 5 rows):\n%s", y_test.head())
    
    pos_train = y_train.sum()
    neg_train = y_train.shape[0] - pos_train
    pos_test = y_test.sum()
    neg_test = y_test.shape[0] - pos_test
    
    logging.info("Training outcome: %d events and %d censored values", pos_train, neg_train)
    logging.info("Test outcome: %d events and %d censored values", pos_test, neg_test)
    
    logging.info("Data shapes after processing: dtrain: %s, dtest: %s", dtrain.shape, dtest.shape)
    
    # Write the processed data to disk.
    output_dir = "/well/clifton/users/ncu080/proteo_fs/data"
    logging.info("Writing processed data to directory: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    train_features_file = os.path.join(output_dir, "train_features.csv")
    train_outcome_file = os.path.join(output_dir, "train_outcome.csv")
    test_features_file = os.path.join(output_dir, "test_features.csv")
    test_outcome_file = os.path.join(output_dir, "test_outcome.csv")
    
    dtrain.to_csv(train_features_file, index=False)
    y_train.to_csv(train_outcome_file, index=False)
    dtest.to_csv(test_features_file, index=False)
    y_test.to_csv(test_outcome_file, index=False)
    
    logging.info("Processed training features written to: %s", train_features_file)
    logging.info("Processed training outcome written to: %s", train_outcome_file)
    logging.info("Processed test features written to: %s", test_features_file)
    logging.info("Processed test outcome written to: %s", test_outcome_file)
    
    logging.info("Data processing completed.")
