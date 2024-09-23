"""
Script: Dataset Construction for Cryptocurrency Anomaly Detection

Description:
This script processes cryptocurrency data CSV files obtained from the data acquisition phase. It calculates the hourly close price variations, labels anomalies based on specified thresholds, applies curve shifting to label preceding hours, and handles interleaved anomalies by labeling them as stable. The goal is to prepare a dataset suitable for machine learning models to predict anomalies in cryptocurrency prices.

Requirements:
A conda environment is required to run this script. You can create a new conda environment by running the following command:

    conda env create -f crypto_anomalies.yaml

This will create a new conda environment named 'crypto_anomalies' with all the required dependencies.

Execution:
To execute this script from the root folder, run the following command in your terminal:

    python src/dataset_construction.py --input_folder /path/to/input/folder --output_folder /path/to/output/folder --threshold 1.0 --shift_hours 4

- `--input_folder`: The path to the folder containing the raw CSV files to process.
- `--output_folder`: The path to the folder where the processed CSV files will be saved.
- `--threshold`: The percentage threshold for anomaly detection (e.g., 1.0 for 1% price variation).
- `--shift_hours`: The number of hours for curve shifting (e.g., 4 hours preceding an anomaly).

Example:

    python src/dataset_construction.py --input_folder data/raw --output_folder data/processed --threshold 1.0 --shift_hours 4

This command processes all CSV files in `data/raw`, labels anomalies based on a 1% price variation threshold, applies a curve shifting of 4 hours, and saves the processed files to `data/processed`.
"""


import pandas as pd
import argparse
import os
import glob

def calculate_price_variation(df):
    """
    Calculate the percentage variation in the close price for each hour.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the cryptocurrency data.

    Returns:
    pandas.DataFrame: DataFrame with an additional 'Price_Variation' column.
    """
    df = df.copy()
    # Calculate percentage variation of close prices
    df['Price_Variation'] = df['Close'].pct_change() * 100
    return df

def label_anomalies(df, threshold=1.0):
    """
    Label anomalies based on the price variation thresholds.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the cryptocurrency data with 'Price_Variation' column.
    threshold (float): The percentage threshold to consider for anomalies.

    Returns:
    pandas.DataFrame: DataFrame with an additional 'Anomaly' column.
    """
    df = df.copy()
    df['Anomaly'] = 0  # Initialize anomaly column with 0 (stable)

    # Identify upward anomalies
    upward_anomalies = df['Price_Variation'].shift(-1) > threshold
    df.loc[upward_anomalies, 'Anomaly'] = 1  # Label previous hour as upward anomaly

    # Identify downward anomalies
    downward_anomalies = df['Price_Variation'].shift(-1) < -threshold
    df.loc[downward_anomalies, 'Anomaly'] = 2  # Label previous hour as downward anomaly

    return df

def apply_curve_shifting(df, shift_hours=4):
    """
    Apply curve shifting to label the previous n hours preceding any anomaly.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the cryptocurrency data with 'Anomaly' column.
    shift_hours (int): Number of hours to shift the anomaly labels backward.

    Returns:
    pandas.DataFrame: DataFrame with updated 'Anomaly' column after curve shifting.
    """
    df = df.copy()
    anomaly_indices = df.index[df['Anomaly'] > 0].tolist()

    for idx in anomaly_indices:
        # Get the label of the current anomaly (1 or 2)
        anomaly_label = df.at[idx, 'Anomaly']
        # Apply curve shifting to the previous n hours
        start_idx = max(0, idx - shift_hours)
        df.loc[start_idx:idx, 'Anomaly'] = anomaly_label

    return df

def handle_interleaved_anomalies(df):
    """
    Handle interleaved anomalies by labeling them as stable.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the cryptocurrency data with 'Anomaly' column.

    Returns:
    pandas.DataFrame: DataFrame with updated 'Anomaly' column after handling interleaved anomalies.
    """
    df = df.copy()
    anomaly_series = df['Anomaly']
    for i in range(1, len(anomaly_series)):
        if anomaly_series[i] != 0 and anomaly_series[i-1] != 0:
            if anomaly_series[i] != anomaly_series[i-1]:
                # Set both current and previous anomalies to 0 (stable)
                df.at[i, 'Anomaly'] = 0
                df.at[i-1, 'Anomaly'] = 0
    return df

def process_file(file_path, threshold, shift_hours):
    """
    Process a single CSV file to label anomalies.

    Parameters:
    file_path (str): Path to the CSV file.
    threshold (float): The percentage threshold to consider for anomalies.
    shift_hours (int): Number of hours to shift the anomaly labels backward.

    Returns:
    pandas.DataFrame: Processed DataFrame with anomalies labeled.
    """
    try:
        df = pd.read_csv(file_path)
        df = calculate_price_variation(df)
        df = label_anomalies(df, threshold=threshold)
        df = apply_curve_shifting(df, shift_hours=shift_hours)
        df = handle_interleaved_anomalies(df)
        return df
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Construct dataset by labeling anomalies in cryptocurrency data.')
    parser.add_argument('--input_folder', required=True, help='Folder containing the CSV files to process.')
    parser.add_argument('--output_folder', required=True, help='Folder where processed CSV files will be saved.')
    parser.add_argument('--threshold', type=float, default=1.0, help='Percentage threshold for anomaly detection.')
    parser.add_argument('--shift_hours', type=int, default=4, help='Number of hours for curve shifting.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    threshold = args.threshold
    shift_hours = args.shift_hours

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
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        df_processed = process_file(file_path, threshold, shift_hours)

        if df_processed is not None:
            output_file = os.path.join(output_folder, file_name)
            try:
                df_processed.to_csv(output_file, index=False)
                print(f"Processed data saved to '{output_file}'.")
            except Exception as e:
                print(f"Error saving processed data to '{output_file}': {e}")
        else:
            print(f"Skipping file '{file_name}' due to processing error.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
