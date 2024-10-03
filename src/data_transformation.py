"""
Script: Data Transformation for Cryptocurrency Dataset

Description:
This script processes cryptocurrency CSV files to:
1. Compute the percent variation of every row with respect to the previous one for each feature.
2. Normalize the dataset using Robust Scaling.

Features are transformed, excluding the class label ('Anomaly'), and the resulting dataset is saved as new CSV files.

Execution:
To execute this script, from the root folder, run the following command:

    python src/data_transformation.py --input_folder /path/to/input/folder --output_folder /path/to/output/folder

Example:

    python src/data_transformation.py --input_folder data/with_indicators --output_folder data/transformed

"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
from sklearn.preprocessing import RobustScaler

def compute_percent_variation(df, exclude_columns):
    """
    Computes the percent variation of each feature with respect to the previous row.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    exclude_columns (list): List of columns to exclude from the computation.

    Returns:
    pandas.DataFrame: DataFrame with percent variation computed.
    """
    df = df.copy()
    # Columns to compute percent variation on
    columns = [col for col in df.columns if col not in exclude_columns]

    # Compute percent variation
    for col in columns:
        df[f"{col}_pct_change"] = df[col].pct_change()

    # Drop the first row which will have NaN values after pct_change
    df.dropna(inplace=True)

    return df

def robust_scaling(df, exclude_columns):
    """
    Scales the features using Robust Scaler.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    exclude_columns (list): List of columns to exclude from scaling.

    Returns:
    pandas.DataFrame: Scaled DataFrame.
    """
    df = df.copy()
    # Columns to scale
    columns = [col for col in df.columns if col not in exclude_columns]

    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Fit and transform the data
    df[columns] = scaler.fit_transform(df[columns])

    return df

def process_file(file_path, output_folder):
    """
    Process a single CSV file to compute percent variation and normalize the data.

    Parameters:
    file_path (str): Path to the CSV file.
    output_folder (str): Path to the output folder.

    Returns:
    None
    """
    try:
        df = pd.read_csv(file_path)

        # Ensure datetime column is in datetime format
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            print(f"No 'Datetime' or 'Date' column found in '{file_path}'. Skipping file.")
            return

        # Exclude columns from transformations
        # Also the Volumne is excluded to avoid
        # Input X contains infinity or a value too large for dtype('float64'), which can be caused by
        # the Volume column having a value of 0 and so the pct_change is infinite
        exclude_columns_pct = ['Datetime', 'Date', 'Anomaly', 'Volume']
        exclude_columns_scaler = ['Datetime', 'Date', 'Anomaly']

        # Compute percent variation
        df = compute_percent_variation(df, exclude_columns_pct)

        # Normalize using Robust Scaling
        df_robust_scaled = robust_scaling(df, exclude_columns_scaler)

        # Save the transformed dataset
        base_filename = os.path.basename(file_path).split('.')[0]
        output_file_robust = os.path.join(output_folder, f"{base_filename}_robust_scaled.csv")

        df_robust_scaled.to_csv(output_file_robust, index=False)

        print(f"Processed and saved file for '{base_filename}':")
        print(f" - Robust Scaled: {output_file_robust}")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute percent variation and normalize cryptocurrency data.')
    parser.add_argument('--input_folder', required=True, help='Folder containing the CSV files to process.')
    parser.add_argument('--output_folder', required=True, help='Folder where transformed CSV files will be saved.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except Exception as e:
            print(f"Error creating output folder '{output_folder}': {e}")
            return

    # Process each CSV file in the input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    for file_path in csv_files:
        print(f"Processing file: {os.path.basename(file_path)}")
        # Avoid all the files with full_data in the name
        if 'full_data' in file_path:
            continue
        process_file(file_path, output_folder)

if __name__ == '__main__':
    main()
