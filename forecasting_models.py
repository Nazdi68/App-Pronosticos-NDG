# forecasting_models.py
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm

def calculate_metrics(y_true, y_pred):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    valid_indices = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_clean = y_true_arr[valid_indices]
    y_pred_clean = y_pred_arr[valid_indices]
    if len(y_true_clean) == 0:
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return rmse, mae

def train_test_split_series(series, test_size_periods):
    if not isinstance(series, pd.Series): series = pd.Series(series)
    if test_size_periods <= 0 or test_size_periods >= len(series):
        return series, pd.Series(dtype=series.dtype, index=pd.DatetimeIndex([])) 
    train = series[:-test_size_periods]; test = series[-test_size_periods:]
    return train, test
    