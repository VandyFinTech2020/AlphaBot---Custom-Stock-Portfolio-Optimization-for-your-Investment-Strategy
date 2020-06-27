
def trading_indicators(data_df):
    """Generates trading signals for a given dataset."""
    # Grab just the `date` and `close` from the dataset
    signals_df = data_df.loc[:, ["date", "close"]].copy()

    # Set the `date` column as the index
    signals_df = signals_df.set_index("date", drop=True)

    # Set the short window and long windows
    short_window = 50
    long_window = 200

    # Generate the short and long moving averages (50 and 200 days, respectively)
    signals_df["SMA50"] = signals_df["close"].rolling(window=short_window).mean()
    signals_df["SMA200"] = signals_df["close"].rolling(window=long_window).mean()
    signals_df["Cross"] = 0.0

    # Generate the trading signal 0 or 1,
    # Death Cross Zone = 0 is when the SMA50 < SMA200
    # Golden Cross Zone = 1 is when the SMA50 > SMA200
    signals_df["Cross"][short_window:] = np.where(
        signals_df["SMA50"][short_window:] > signals_df["SMA200"][short_window:],
        1.0,
        0.0,
    )

    # Calculate the points in time at which a position should be taken, 1 or -1
    signals_df["Entry/Exit"] = signals_df["Cross"].diff()
    #Set Bollinger Band window to 20 (standard lag)
    bollinger_window = 20

# Calculate rolling mean and standard deviation
    signals_df['bollinger_mid_band'] = signals_df['Close'].rolling(window=bollinger_window).mean()
    signals_df['bollinger_std'] = signals_df['Close'].rolling(window=20).std()

# Calculate upper and lowers bands of bollinger band. Range set to 2 standard deviations instead of 1
    signals_df['bollinger_upper_band']  = signals_df['bollinger_mid_band'] + (signals_df['bollinger_std'] * 2)
    signals_df['bollinger_lower_band']  = signals_df['bollinger_mid_band'] - (signals_df['bollinger_std'] * 2)

# Calculate bollinger band trading signal
    signals_df['bollinger_long'] = np.where(signals_df['Close'] < signals_df['bollinger_lower_band'], 1.0, 0.0)
    signals_df['bollinger_short'] = np.where(signals_df['Close'] > signals_df['bollinger_upper_band'], -1.0, 0.0)
    signals_df['bollinger_signal'] = signals_df['bollinger_long'] + signals_df['bollinger_short']
        
    
    return signals_df