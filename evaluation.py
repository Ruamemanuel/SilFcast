import numpy as np

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def rmse(y_true, y_pred):
    """Calculate Root Mean Square Error (RMSE)."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
