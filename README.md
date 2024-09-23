# Crypto-Anomaly-Detection

## Overview 

This project predicts anomalies in cryptocurrency markets using machine learning and time-series analysis. It identifies unusual price movements and deviations from normal patterns, offering real-time detection of potential market anomalies for further analysis.

## Table of Contents
- [Code Directory](#code-directory)
- [Dataset Acquisition](#dataset-acquisition)
- [Dataset Construction](#dataset-construction)


### Code directory
```
Crypto-Anomaly-Detection
    ├── conda          # All the conda environments
    ├── data           # All the data
    ├── images         # All the images
    ├── notebooks      # All the notebooks
    ├── src            # All the scripts for the analysis
    ├── .gitignore
    ├── README.md
    └── requirements.txt

```

## Dataset Acquisition
The first step to exploit our work is to select a dataset that best suits our needs.
The decision was about crypocurrencies due to their higher volatilty with respect to stock market. 

As we’re going to see in the next sections, we will need to use some technical analysis indicators which are commonly used by traders to decide whether to sell or buy an asset. Traders use two strategies:
1. The first is the use of these **technical indicators** to determine through graphs of the price possible situation where the asset they’re examining is oversold or overbought; 
2. The other strategy that traders use is **fundamental analysis**, this strategy is used mostly with stocks prices and aims at analyzing the main characteristics of a firm, for instance, the revenue, total debt, price/earnings ratio, etc.

It is behind the scope of this analysis to investigate deeply all those analysis indicator.

The data were acquired from [Yahoo Finance](https://finance.yahoo.com/) using the [yfinance](https://github.com/ranaroussi/yfinance) python package. To get as much sample as possible the price variation of 1 hour has been considered. 

Below the features extracted, from each single crypto, in this data acquisition steps:
- Date
- Open price for the hour
- Close price for that hour
- The Higest price in the hour
- The Lowest price in the hour
- The volume of crypto traded in the hour

![Data Acquisition image](images/data_acquisition.png)

After all the cryptos' information are acquired a data cleaning pahse is performed, replacing all the missing values with the previous available value. 

Below an example of how execute the data_acquisition.py script, additional information can be found looking at the first row of the script. 

```bash
    python src/dataset_acquisition.py --tickers BTC-USD BTS-USD DGB-USD XMR-USD DASH-USD DOGE-USD ETH-USD LTC-USD MAID-USD MONA-USD NAV-USD VTC-USD XCP-USD XRP-USD SYS-USD XLM-USD --period ytd --interval 1h --output_folder data/raw
```

## Dataset Construction

After acquiring and cleaning the data, the next step is to construct the dataset by labeling anomalies based on price variations.

### Anomaly Definition

An anomaly is defined based on the percentage change in the close price of each hour:

- **Upward Anomaly**: If the price variation is greater than 1%, we label the previous hour as an upward anomaly.
- **Downward Anomaly**: If the price variation is less than -1%, we label the previous hour as a downward anomaly.
- **Stable**: If the price variation is between -1% and 1%, we label it as stable.

This labeling helps in identifying significant price movements in the market.

![Data Acquisition Anomaly image](images/data_acquisition_anomaly.PNG)

### Curve Shifting Technique

Labeling only the hour before a price variation as anomalous wasn't sufficient for our analysis. To improve the results, we applied a technique called *curve shifting*.

**Curve Shifting involves** labeling the previous n hours preceding any anomaly. In our case, we chose a curve shifting of 4 hours. This means that the 4 hours leading up to an anomaly are also labeled as anomalies.

This approach accounts for patterns or signals that may occur before significant price movements, allowing our models to learn from the lead-up to anomalies.

To visualize the impact of curve shifting on the class distribution, we plotted the number of observations for each class before and after applying curve shifting.

#### Bar Plot of Class Distribution:

![Data Acquisition Anomaly image](images/bar_plot_cs_before_after.png)

#### Line Plot of Class Trend Over Time:

The image below shows the trends for the 3 classes (stable, upward and downward anomaly) in the considered time period **before** the application of the curve shifting: 

![Data Acquisition Anomaly image](images/lp_cs_before.png)

The image below shows the trends for the 3 classes (stable, upward and downward anomaly) in the considered time period **after** the application of the curve shifting: 

![Data Acquisition Anomaly image](images/lp_cs_after.png)

These plots illustrate that, after applying curve shifting, the dataset becomes more balanced, with more samples labeled as anomalies. This helps in training machine learning models more effectively by providing sufficient examples of each class.

### Executing the Dataset Construction Script

Below is an example of how to execute the `dataset_construction.py` script from the root folder, which processes the raw data to label anomalies and apply curve shifting.

```bash
    python src/dataset_construction.py --input_folder data/raw --output_folder data/processed --threshold 1.0 --shift_hours 4
```

- --input_folder: Path to the folder containing the raw CSV files.
- --output_folder: Path where the processed CSV files will be saved.
- --threshold: Percentage threshold for anomaly detection (e.g., 1.0 for 1% price variation).
- --shift_hours: Number of hours to shift the anomaly labels backward (e.g., 4).



## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](LICENSE) - see the `LICENSE` file for details.