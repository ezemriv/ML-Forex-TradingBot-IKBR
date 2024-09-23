import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis

class FinancialInstrumentAnalyzer:
    def __init__(self, tickers, start_date, end_date, threshold):
        """
        Initialize the analyzer with a list of tickers and a date range.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.extreme_threshold = threshold
        self.data = {}          # To store adjusted close prices
        self.log_returns = {}   # To store logarithmic returns
        self.stats = {}         # To store computed statistics
        self.trading_days = 252 # Number of trading days in a year
        # Define a color palette for consistent coloring across plots
        self.colors = sns.color_palette('tab10', len(self.tickers))
        self._download_data()

    def _download_data(self):
        """
        Download historical data for each ticker and compute log returns.
        """
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            if df.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue
            self.data[ticker] = df['Adj Close']
            # Compute logarithmic returns
            self.log_returns[ticker] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()

    def compute_statistics(self):
        """
        Compute statistical measures for each instrument.
        """
        for ticker in self.tickers:
            if ticker not in self.log_returns:
                continue
            ret = self.log_returns[ticker]
            stats = {}
            # Daily statistics
            stats['mean_daily_log_return'] = ret.mean()
            stats['daily_volatility'] = ret.std()
            # Annualized statistics
            stats['annualized_log_return'] = ret.mean() * self.trading_days
            stats['annualized_volatility'] = ret.std() * np.sqrt(self.trading_days)
            # Convert annualized log return to annualized simple return for comparison
            stats['annualized_return'] = np.exp(stats['annualized_log_return']) - 1
            # Additional statistics
            stats['skewness'] = skew(ret)
            stats['kurtosis'] = kurtosis(ret)
            adf_result = adfuller(ret)
            stats['adf_statistic'] = adf_result[0]
            stats['adf_pvalue'] = adf_result[1]
            self.stats[ticker] = stats
        stats_df = pd.DataFrame(self.stats).T  # Transpose for better readability
        return stats_df

    def plot_normalized_price_series(self):
        """
        Plot the normalized price series for each instrument, starting from 1.
        """
        plt.figure(figsize=(12, 6))
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.data:
                continue
            normalized_prices = self.data[ticker] / self.data[ticker].iloc[0]
            plt.plot(normalized_prices, label=ticker, color=self.colors[idx])
        plt.legend()
        plt.title('Normalized Price Series')
        plt.xlabel('Date')
        plt.ylabel('Normalized Adjusted Close Price')
        plt.show()

    def plot_log_return_series(self):
        """
        Plot the log return series for each instrument.
        """
        plt.figure(figsize=(12, 6))
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.log_returns:
                continue
            plt.plot(self.log_returns[ticker], label=ticker, color=self.colors[idx])
        plt.legend()
        plt.title('Log Return Series')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.show()

    def plot_return_distribution(self):
        """
        Plot the distribution of log returns for each instrument.
        """
        plt.figure(figsize=(12, 6))
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.log_returns:
                continue
            sns.histplot(self.log_returns[ticker], label=ticker, kde=False, bins=50, 
                         alpha=0.6, color=self.colors[idx])
        plt.legend()
        plt.title('Log Return Distribution')
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
        plt.show()

    def plot_acf_pacf(self, ticker, lags=20):
        """
        Plot the ACF and PACF for a given instrument.
        """
        if ticker not in self.log_returns:
            print(f"No log returns data for {ticker}. Skipping ACF and PACF plots.")
            return
        ret = self.log_returns[ticker]
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(ret, lags=lags, ax=ax[0])
        plot_pacf(ret, lags=lags, ax=ax[1])
        ax[0].set_title(f'Autocorrelation Function (ACF) for {ticker}')
        ax[1].set_title(f'Partial Autocorrelation Function (PACF) for {ticker}')
        plt.tight_layout()
        plt.show()

    def compute_correlations(self):
        """
        Compute and plot the correlation matrix between instruments.
        """
        returns_df = pd.DataFrame(self.log_returns)
        if returns_df.empty:
            print("No log returns data available for correlation analysis.")
            return
        corr_matrix = returns_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Log Returns')
        plt.show()
        return corr_matrix

    def plot_extreme_return_histograms(self):
        """
        Plot histograms of extreme log returns (absolute value greater than threshold)
        for each instrument on the same plot, using the same colors as the return distribution.

        Also, after the plot, print for each ticker the number of days per year with extreme log returns.
        """
        threshold = self.extreme_threshold

        plt.figure(figsize=(12, 6))
        has_extremes = False
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.log_returns:
                continue
            extreme_returns = self.log_returns[ticker][np.abs(self.log_returns[ticker]) > threshold]
            if extreme_returns.empty:
                print(f"No extreme returns (|return| > {threshold}) found for {ticker}.")
                continue
            has_extremes = True
            sns.histplot(extreme_returns, bins=20, alpha=0.5, color=self.colors[idx],
                         label=ticker, kde=False, stat='count')
            # Calculate number of days per year with extreme returns
            num_extreme_days = len(extreme_returns)
            num_years = (self.log_returns[ticker].index[-1] - self.log_returns[ticker].index[0]).days / 365.25
            extreme_days_per_year = num_extreme_days / num_years
            print(f"{ticker}: {extreme_days_per_year:.2f} extreme days per year (threshold = {threshold})")
        if not has_extremes:
            print(f"No extreme returns (|return| > {threshold}) found for any ticker.")
        plt.legend()
        plt.title(f'Extreme Log Returns Histogram (|return| > {threshold})')
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
        plt.show()

    def analyze(self):
        """
        Perform the full analysis.
        """
        stats_df = self.compute_statistics()
        if not stats_df.empty:
            print("Statistical Summary:")
            print(stats_df, '\n')
            print("SKEWNESS Positive indicates a longer right tail (more positive returns), while negative skew indicates a longer left tail (more negative returns).")
            print("Higher KURTOSIS indicates more extreme return values (heavy tails).")
            print("ADF test checks for stationarity in the time series. A low p-value (< 0.05) suggests that the series is stationary.")
        self.plot_normalized_price_series()
        self.plot_log_return_series()
        self.plot_return_distribution()
        self.plot_extreme_return_histograms()
        self.compute_correlations()
        for ticker in self.tickers:
            self.plot_acf_pacf(ticker)