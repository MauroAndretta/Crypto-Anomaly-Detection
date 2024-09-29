"""
Script: Technical Indicators and Sentiment Integration for Cryptocurrency Data

Description:
This script processes cryptocurrency data CSV files, calculates various technical indicators, fetches the Fear & Greed Index sentiment data, and integrates both into the existing CSV files.
The updated CSV files will contain new columns for each technical indicator and the sentiment index.

Requirements:
A conda environment is required to run this script. You can create a new conda environment by running the following command:

    conda env create -f crypto_anomalies.yaml

This will create a new conda environment named 'crypto_anomalies' with all the required dependencies.

Execution:
To execute this script from the root folder, run the following command in your terminal:

    python integrate_indicators.py --input_folder /path/to/input/folder --output_folder /path/to/output/folder

- `--input_folder`: The path to the folder containing the CSV files to process.
- `--output_folder`: The path to the folder where the updated CSV files will be saved.

Example:

    python integrate_indicators.py --input_folder data/processed --output_folder data/with_indicators
"""

import pandas as pd
import pandas_ta as ta
import argparse
import os
import glob
import requests
from datetime import datetime, timedelta
import numpy as np


def integrate_technical_indicators(df):
    """
    Calculates technical indicators and adds them to the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the cryptocurrency data.

    Returns:
    pandas.DataFrame: DataFrame with technical indicators added.
    """
    df = df.copy()
    # Ensure datetime column is in datetime format
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        # Remove timezone information to avoid warning 
        # UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
        # while using the ta.vwap function
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        df.set_index('Datetime', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.tz_localize(None)
        df.set_index('Date', inplace=True)
    else:
        print("No 'Datetime' or 'Date' column found.")
        return df

    # Calculate SMA for different periods
    sma_periods = [5,12,13,14,20,21,26,30,50,100,200]
    for period in sma_periods:
        df[f'SMA_{period}'] = ta.sma(df['Close'], length=period)

    # Calculate EMA for different periods
    ema_periods = sma_periods  # Using the same periods as SMA
    for period in ema_periods:
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)

    # Calculate MACD
    df['MACD'] = ta.macd(df['Close']).iloc[:, 0]  # MACD line
    df['MACD_signal'] = ta.macd(df['Close']).iloc[:, 1]  # Signal line
    df['MACD_diff'] = ta.macd(df['Close']).iloc[:, 2]  # MACD histogram

    # Calculate RSI for different periods
    rsi_periods = sma_periods  # Using the same periods
    for period in rsi_periods:
        df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period)

    # Calculate Momentum (MOM)
    df['MOM'] = ta.mom(df['Close'])

    # Calculate Chande Momentum Oscillator (CMO)
    cmo_periods = sma_periods
    for period in cmo_periods:
        df[f'CMO_{period}'] = ta.cmo(df['Close'], length=period)

    # Calculate Ultimate Oscillator (UO)
    df['UO'] = ta.uo(df['High'], df['Low'], df['Close'])

    # Calculate Bollinger Bands (BBANDS)
    bbands = ta.bbands(df['Close'])
    df = df.join(bbands)

    # Reset index to have date as a column again
    df.reset_index(inplace=True)

    return df


def process_file(file_path):
    """
    Process a single CSV file to integrate technical indicators and sentiment data.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Processed DataFrame with indicators and sentiment data.
    """
    try:
        df = pd.read_csv(file_path)
        df = integrate_technical_indicators(df)
        return df
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Integrate technical indicators and sentiment data into cryptocurrency CSV files.')
    parser.add_argument('--input_folder', required=True, help='Folder containing the CSV files to process.')
    parser.add_argument('--output_folder', required=True, help='Folder where updated CSV files will be saved.')
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
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        df_processed = process_file(file_path)

        # once all the technical indicators are calculated, it is necessary to delete all the rows with NaN values
        # since the indicators are calculated based on historical data, the first rows will have NaN values, for example the SMA_200
        # Alternatively, drop all rows with any missing values
        if df_processed is not None:
            df_clean = df_processed.dropna()

        if df_clean is not None:
            output_file = os.path.join(output_folder, file_name)
            try:
                df_clean.to_csv(output_file, index=False)
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
