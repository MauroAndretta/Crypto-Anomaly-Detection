"""
Script: Cryptocurrency Data Downloader and Cleaner

Description:
This script downloads cryptocurrency data from Yahoo Finance, cleans the data by 
handling missing values (using forward-fill), and saves the cleaned data into CSV files.
The user must specify the cryptocurrency tickers, the period, the interval, and the 
output folder where the CSV files will be saved.

Requirements:
A conda environment is required to run this script. You can create a new conda environment by running the following command:

conda env create -f crypto_anomalies.yaml

This will create a new conda environment named 'crypto_anomalies' with all the required dependencies.

Execution:
To execute this script from the root folder, run the following command in your terminal:

    python dataset_acquisition.py --tickers BTC ETH --period 7d --interval 1h --output_folder /path/to/output/folder

- --tickers: Space-separated cryptocurrency tickers (e.g., BTC ETH).
- --period: Time period to retrieve data (e.g., '7d', '1mo').
- --interval: Data interval (e.g., '1h', '1d').
- --output_folder: The path to the folder where the CSV files will be saved.

Example:
    python src/dataset_acquisition.py --tickers BTC BTS DGB XMR DASH DOGE ETH LTC MAID MONA NAV VTC XCP XRP SYS XLM --output_folder data/raw --period ytd --interval 1h
"""


import yfinance as yf
import pandas as pd
import argparse
import os


def download_crypto_data(ticker, period='ytd', interval='1h'):
    """
    Download historical data for a given cryptocurrency ticker from Yahoo Finance.
    
    Parameters:
    ticker (str): Cryptocurrency ticker symbol.
    period (str): Data period to download.
    interval (str): Data interval (e.g., '1h' for hourly data).
    
    Returns:
    pandas.DataFrame or None: DataFrame containing the historical data, or None if data is not available.
    """
    # Construct Yahoo Finance ticker symbol (e.g., 'BTC-USD')
    yahoo_ticker = f"{ticker}-USD"
    
    try:
        # Download data from Yahoo Finance
        data = yf.download(tickers=yahoo_ticker, period=period, interval=interval)
        
        # Check if data is empty
        if data.empty:
            print(f"No data found for {yahoo_ticker}.")
            return None
        else:
            # Reset index to have DateTime as a column
            data.reset_index(inplace=True)
            return data
    except Exception as e:
        print(f"Error downloading data for {yahoo_ticker}: {e}")
        return None

def fill_missing_values(data):
    """
    Fill missing values in the DataFrame by propagating the last valid observation forward.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing data with missing values.
    
    Returns:
    pandas.DataFrame: DataFrame with missing values filled.
    """
    try:
        # Forward fill to propagate last valid observation forward
        data_filled = data.ffill()
        
        # Backward fill to fill any remaining NaNs at the start
        data_filled = data_filled.bfill()
        
        return data_filled
    except Exception as e:
        print(f"Error filling missing values: {e}")
        return data

def main():
    """
    Main function to download and process data for a list of cryptocurrency tickers.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download and process cryptocurrency data.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of cryptocurrency ticker symbols.')
    parser.add_argument('--output_folder', required=True, help='Folder where CSV files will be saved.')
    parser.add_argument('--period', default='ytd', help='Data period to download (e.g., "ytd").')
    parser.add_argument('--interval', default='1h', help='Data interval (e.g., "1h" for hourly data").')
    
    # Parse arguments
    args = parser.parse_args()
    
    tickers = args.tickers
    output_folder = args.output_folder
    period = args.period
    interval = args.interval
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except Exception as e:
            print(f"Error creating output folder '{output_folder}': {e}")
            return
    
    # Dictionary to store data for each ticker
    all_data = {}
    
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        
        # Download data
        data = download_crypto_data(ticker, period=period, interval=interval)
        
        if data is not None:
            # Fill missing values
            data_filled = fill_missing_values(data)
            
            # Store the cleaned data
            all_data[ticker] = data_filled
            
            # Save data to CSV
            filename = os.path.join(output_folder, f"{ticker}_data.csv")
            try:
                data_filled.to_csv(filename, index=False)
                print(f"Data for {ticker} saved to {filename}.")
            except Exception as e:
                print(f"Error saving data for {ticker} to '{filename}': {e}")
        else:
            print(f"Skipping {ticker} due to lack of data.")
    
    # Combine all data into a single DataFrame or perform further processing and analysis
    full_data = pd.concat(all_data.values(), keys=all_data.keys(), names=['Ticker']).reset_index()
    # Save the full data to a CSV file
    full_filename = os.path.join(output_folder, "full_data.csv")
    try:
        full_data.to_csv(full_filename, index=False)
        print(f"Full data saved to {full_filename}.")
    except Exception as e:
        print(f"Error saving full data to '{full_filename}': {e}")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

