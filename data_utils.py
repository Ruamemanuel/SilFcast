import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def get_sliding_windows(window_size, series):
    """Generate sliding windows for a given time series."""
    n = len(series)
    window_size += 1  # Include target column
    windows = np.array([series[i:i + window_size] for i in range(n - window_size + 1)])
    columns = [f't_{window_size - i - 1}' for i in range(window_size - 1)] + ['target']
    return pd.DataFrame(windows, columns=columns)

def normalize_series(series, feature_range=(0.1, 0.9)):
    """Normalize a series using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=feature_range)
    series_norm = scaler.fit_transform(series.values.reshape(-1, 1))
    return pd.Series(series_norm.flatten(), index=series.index), scaler

def cosine_similarity(distances):
    """Convert cosine distances to similarities."""
    return 1 - distances
