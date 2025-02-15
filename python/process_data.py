#!/usr/bin/env python
"""
process_data_parallel.py

This module loads the training and test datasets and processes the data by:
  - Dropping specified columns,
  - Creating three different outcome datasets:
      1. Log loss outcome (only the binary event),
      2. Cox outcome (time and event columns),
      3. Cox loss outcome (time positive for event and negative for no event),
  - Removing features with > missing_threshold proportion of missing data.
  
Note: Missing value imputation has been removed from this script.
      It will be performed separately during each bootstrap of the model building process.
"""

import os
import time
import logging
import pandas as pd
import numpy as np

# Configure logging to print progress messages with time stamps.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(
    train_file='/well/clifton/projects/ukb_v2/derived/prot_final/train_data.csv',
    test_file='/well/clifton/projects/ukb_v2/derived/prot_final/test_data.csv',
    missing_threshold=0.30
):
    """
    Loads and processes training and test data.

    Processing steps:
      1. Load CSV files.
      2. Create three outcome datasets:
           - Log loss outcome: 1 if Myeloma==1 (event), 0 otherwise.
           - Cox outcome: DataFrame with 'time' and binary outcome columns.
           - Cox loss outcome: time positive if event occurred, negative if censored.
      3. Drop specified columns.
      4. Remove features with > missing_threshold missing data (based on training data).
      5. (Note: Missing value imputation is not performed here.)

    Returns:
        dtrain, (y_train_logloss, y_train_cox, y_train_cox_loss),
        dtest,  (y_test_logloss,  y_test_cox,  y_test_cox_loss)
    """
    logging.info("Loading training data from: %s", train_file)
    train_data = pd.read_csv(train_file)
    logging.info("Loading test data from: %s", test_file)
    test_data = pd.read_csv(test_file)
    
    # Create outcomes for different loss functions before dropping columns.
    logging.info("Creating outcome datasets...")
    # For log loss: outcome only.
    y_train_logloss = (train_data['Myeloma'] == 1).astype(int)
    y_test_logloss = (test_data['Myeloma'] == 1).astype(int)
    
    # For Cox model: DataFrame with 'time' and binary outcome columns.
    y_train_cox = pd.DataFrame({
        'time': train_data['time'],
        'outcome': y_train_logloss
    })
    y_test_cox = pd.DataFrame({
        'time': test_data['time'],
        'outcome': y_test_logloss
    })
    
    # For Cox loss: time is positive if event occurred and negative if censored.
    y_train_cox_loss = pd.Series(
        np.where(train_data['Myeloma'] == 1, train_data['time'], -train_data['time']),
        index=train_data.index
    )
    y_test_cox_loss = pd.Series(
        np.where(test_data['Myeloma'] == 1, test_data['time'], -test_data['time']),
        index=test_data.index
    )
    
    # Exclude columns that are not used as features.
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
    
    logging.info("Removing features with more than %.0f%% missing values...", missing_threshold * 100)
    missing_percentage = dtrain.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    logging.info("Columns to drop: %s", list(columns_to_drop))
    dtrain = dtrain.drop(columns=columns_to_drop)
    dtest = dtest.drop(columns=columns_to_drop)
    
    return dtrain, (y_train_logloss, y_train_cox, y_train_cox_loss), dtest, (y_test_logloss, y_test_cox, y_test_cox_loss)

if __name__ == "__main__":
    logging.info("Starting data processing...")
    dtrain, (y_train_logloss, y_train_cox, y_train_cox_loss), dtest, (y_test_logloss, y_test_cox, y_test_cox_loss) = load_data()
    
    logging.info("First 50 column names in training features:")
    logging.info(dtrain.columns[:50].tolist())
    
    logging.info("Training features (first 5 rows):\n%s", dtrain.head())
    logging.info("Training log loss outcome (first 5 rows):\n%s", y_train_logloss.head())
    logging.info("Training Cox outcome (first 5 rows):\n%s", y_train_cox.head())
    logging.info("Training Cox loss outcome (first 5 rows):\n%s", y_train_cox_loss.head())
    
    logging.info("Test features (first 5 rows):\n%s", dtest.head())
    logging.info("Test log loss outcome (first 5 rows):\n%s", y_test_logloss.head())
    logging.info("Test Cox outcome (first 5 rows):\n%s", y_test_cox.head())
    logging.info("Test Cox loss outcome (first 5 rows):\n%s", y_test_cox_loss.head())
    
    pos_train = y_train_logloss.sum()
    neg_train = y_train_logloss.shape[0] - pos_train
    pos_test = y_test_logloss.sum()
    neg_test = y_test_logloss.shape[0] - pos_test
    
    logging.info("Training outcome (log loss): %d events and %d censored values", pos_train, neg_train)
    logging.info("Test outcome (log loss): %d events and %d censored values", pos_test, neg_test)
    
    logging.info("Data shapes after processing: dtrain: %s, dtest: %s", dtrain.shape, dtest.shape)
    
    # Write the processed data to disk.
    output_dir = "/well/clifton/users/ncu080/proteo_fs/data"
    logging.info("Writing processed data to directory: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Features files.
    train_features_file = os.path.join(output_dir, "train_features.csv")
    test_features_file = os.path.join(output_dir, "test_features.csv")
    
    # Outcome files for log loss.
    train_outcome_logloss_file = os.path.join(output_dir, "train_outcome_logloss.csv")
    test_outcome_logloss_file = os.path.join(output_dir, "test_outcome_logloss.csv")
    
    # Outcome files for Cox models.
    train_outcome_cox_file = os.path.join(output_dir, "train_outcome_cox.csv")
    test_outcome_cox_file = os.path.join(output_dir, "test_outcome_cox.csv")
    
    # Outcome files for Cox loss.
    train_outcome_cox_loss_file = os.path.join(output_dir, "train_outcome_cox_loss.csv")
    test_outcome_cox_loss_file = os.path.join(output_dir, "test_outcome_cox_loss.csv")
    
    dtrain.to_csv(train_features_file, index=False)
    dtest.to_csv(test_features_file, index=False)
    
    y_train_logloss.to_csv(train_outcome_logloss_file, index=False)
    y_test_logloss.to_csv(test_outcome_logloss_file, index=False)
    
    y_train_cox.to_csv(train_outcome_cox_file, index=False)
    y_test_cox.to_csv(test_outcome_cox_file, index=False)
    
    y_train_cox_loss.to_csv(train_outcome_cox_loss_file, index=False)
    y_test_cox_loss.to_csv(test_outcome_cox_loss_file, index=False)
    
    logging.info("Processed training features written to: %s", train_features_file)
    logging.info("Processed test features written to: %s", test_features_file)
    logging.info("Processed training log loss outcome written to: %s", train_outcome_logloss_file)
    logging.info("Processed test log loss outcome written to: %s", test_outcome_logloss_file)
    logging.info("Processed training Cox outcome written to: %s", train_outcome_cox_file)
    logging.info("Processed test Cox outcome written to: %s", test_outcome_cox_file)
    logging.info("Processed training Cox loss outcome written to: %s", train_outcome_cox_loss_file)
    logging.info("Processed test Cox loss outcome written to: %s", test_outcome_cox_loss_file)
    
    logging.info("Data processing completed.")
