import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import optuna
import yfinance as yf
from datetime import datetime, timedelta

class PeakValleyStrategy:
    def __init__(self, df):
        """
        Initialize the PeakValleyStrategy class with a DataFrame.
        """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")
        self.df = df.copy()

    def detect_peaks_valleys(self, distance=5, prominence=1, plot=True):
        """
        Detect peaks and valleys using signal processing and create trade signals.
        """
        peaks_indices, _ = find_peaks(self.df['Close'], distance=distance, prominence=prominence)
        valleys_indices, _ = find_peaks(-self.df['Close'], distance=distance, prominence=prominence)

        self.df['Peak'] = np.nan
        self.df['Valley'] = np.nan
        self.df.iloc[peaks_indices, self.df.columns.get_loc('Peak')] = self.df['Close'].iloc[peaks_indices]
        self.df.iloc[valleys_indices, self.df.columns.get_loc('Valley')] = self.df['Close'].iloc[valleys_indices]

        # Create log returns column
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))

        # Create trade signal column
        self.df['Signal'] = np.nan
        self.df.loc[self.df['Peak'].notnull(), 'Signal'] = -1  # Sell signal at peaks
        self.df.loc[self.df['Valley'].notnull(), 'Signal'] = 1  # Buy signal at valleys
        self.df['Signal'] = self.df['Signal'].ffill().fillna(0)  # Forward fill signals

        if plot:
            plt.figure(figsize=(14, 7))
            plt.plot(self.df.index, self.df['Close'], label='Closing Price', color='blue')
            plt.scatter(self.df.index[peaks_indices], self.df['Close'].iloc[peaks_indices],
                        label='Peaks', color='red', marker='v', s=100)
            plt.scatter(self.df.index[valleys_indices], self.df['Close'].iloc[valleys_indices],
                        label='Valleys', color='green', marker='^', s=100)
            plt.title('Stock Price with Peaks and Valleys')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

        total_trades = self.df['Signal'].diff().fillna(0).abs().sum()
        print(f'Total number of trades: {total_trades}')
        total_trading_days = self.df['Close'].count()
        trading_weeks = total_trading_days / 5
        print(f"Approx nr Trades per Week: {round(total_trades / trading_weeks, 3)}")

    def compare_strategy_returns(self, returns_col='Log_Returns', signal_col='Signal'):
        """
        Compare buy-and-hold returns vs. strategy returns based on signals.
        """
        if returns_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{returns_col}' column.")
        if signal_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{signal_col}' column.")

        self.df['Cumulative_BH_Returns'] = self.df[returns_col].cumsum()
        self.df['Strategy_Returns'] = self.df[returns_col] * self.df[signal_col].shift(1).fillna(0)
        self.df['Cumulative_Strategy_Returns'] = self.df['Strategy_Returns'].cumsum()

        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Cumulative_BH_Returns'], label='Buy and Hold', color='blue')
        plt.plot(self.df.index, self.df['Cumulative_Strategy_Returns'], label='Strategy', color='orange')
        plt.title('Cumulative Log Returns: Buy and Hold vs. Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

        total_bh_return = (np.exp(self.df['Cumulative_BH_Returns'].iloc[-1]) - 1) * 100
        total_strategy_return = (np.exp(self.df['Cumulative_Strategy_Returns'].iloc[-1]) - 1) * 100
        total_trades = self.df['Signal'].diff().fillna(0).abs().sum()

        print(f"Total Buy and Hold Return: {total_bh_return:.2f}%")
        print(f"Total Strategy Return: {total_strategy_return:.2f}% ({(total_strategy_return/total_bh_return)/1000:.2f} 1000s Fold B&H)")
        print(f"Total Number of Trades: {total_trades}")
        total_trading_days = self.df['Close'].count()
        trading_weeks = total_trading_days / 5
        print(f"Approx nr Trades per Week: {round(total_trades / trading_weeks, 3)}")

    def optimize_parameters(self, n_trials=100, penalty=False, trade_off_max=0.02, min_dist=1, min_prominence=0.1):
        """
        Optimize 'distance' and 'prominence' parameters using Optuna.
        """
        def objective(trial):
            distance = trial.suggest_int('distance', min_dist, 10)
            prominence = trial.suggest_float('prominence', min_prominence, 5.0)

            if penalty:
                trade_off = trial.suggest_float('trade_off', 0.0, trade_off_max)
            else:
                trade_off = 0

            data = self.df.copy()
            peaks_indices, _ = find_peaks(data['Close'], distance=distance, prominence=prominence)
            valleys_indices, _ = find_peaks(-data['Close'], distance=distance, prominence=prominence)

            data['Signal'] = np.nan
            data.loc[data.index[peaks_indices], 'Signal'] = -1
            data.loc[data.index[valleys_indices], 'Signal'] = 1
            data['Signal'] = data['Signal'].ffill().fillna(0)

            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
            data['Strategy_Returns'] = data['Log_Returns'] * data['Signal'].shift(1).fillna(0)
            total_strategy_return = data['Strategy_Returns'].sum()
            total_trades = data['Signal'].diff().fillna(0).abs().sum()
            return_volatility = data['Strategy_Returns'].std()

            return -total_strategy_return + (trade_off * total_trades) + return_volatility

        sampler = optuna.samplers.TPESampler(seed=23)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        self.optimized_distance = study.best_params['distance']
        self.optimized_prominence = study.best_params['prominence']
        self.optimized = True

        print(f"Optimized distance: {self.optimized_distance}")
        print(f"Optimized prominence: {self.optimized_prominence:.2f}")
        self.detect_peaks_valleys(distance=self.optimized_distance, prominence=self.optimized_prominence)

    def get_signal_dataframe(self):
        """
        Return the DataFrame with 'Close', 'Signal', 'Log_Returns', 'Peak', 'Valley', and 'train_target'.
        The 'train_target' column is 1 for valleys (buy signal), -1 for peaks (sell signal), and 0 otherwise.
        """
        # Create the 'train_target' column based on peaks and valleys
        self.df['train_target'] = 0
        self.df.loc[self.df['Valley'].notnull(), 'train_target'] = 1   # Buy signal at valleys
        self.df.loc[self.df['Peak'].notnull(), 'train_target'] = -1    # Sell signal at peaks

        return self.df[['Close', 'Signal', 'Log_Returns', 'Peak', 'Valley', 'train_target']]


def plot_price_signals(df, start_date=None, end_date=None, signal_col='Signal'):
    """
    Plot the stock price and signals (peaks, valleys) for a specified period or the entire DataFrame.

    Parameters:
    - df: pandas DataFrame, output from strategy.get_signal_dataframe()
    - start_date: str, datetime, or None. Start date of the plot (inclusive). If None, includes all data from the start.
    - end_date: str, datetime, or None. End date of the plot (inclusive). If None, includes all data up to the end.
    - signal_col: str, name of the column containing the signals (default: 'Signal').
    """
    # Filter the DataFrame for the specified date range, if start_date or end_date is provided
    if start_date is not None and end_date is not None:
        df_period = df.loc[start_date:end_date]
        plot_title = f"Stock Price and Signals from {start_date} to {end_date}"
    else:
        df_period = df  # Plot the entire DataFrame
        plot_title = "Stock Price and Signals"

    # Plot the close price
    plt.figure(figsize=(12, 6))
    plt.plot(df_period.index, df_period['Close'], label='Close Price', color='blue')

    # Plot buy signals (valleys)
    plt.scatter(df_period.index[df_period[signal_col] == 1], df_period['Close'][df_period[signal_col] == 1], 
                label='Buy Signal', marker='^', color='green', s=100)

    # Plot sell signals (peaks)
    plt.scatter(df_period.index[df_period[signal_col] == -1], df_period['Close'][df_period[signal_col] == -1], 
                label='Sell Signal', marker='v', color='red', s=100)

    # Add labels and legend
    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
