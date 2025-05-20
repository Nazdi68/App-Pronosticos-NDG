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

# --- Modelos de Línea Base ---
def historical_average_forecast(train_series, test_series, h_future):
    model_name = "Promedio Histórico"
    nan_forecast_future = np.full(h_future, np.nan)
    try:
        if train_series.empty:
            return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: Train series vacía)"
        mean_val_train = train_series.mean()
        
        test_forecast_values = np.full(len(test_series), mean_val_train) if not test_series.empty else np.array([])
        rmse, mae = calculate_metrics(test_series, test_forecast_values)

        full_series = pd.concat([train_series, test_series])
        if full_series.empty:
             return nan_forecast_future, None, rmse, mae, f"{model_name} (Error: Full series vacía para pron. futuro)"
        full_series_mean = full_series.mean()
        future_forecast_values = np.full(h_future, full_series_mean)
        return future_forecast_values, None, rmse, mae, model_name
    except Exception as e:
        error_message = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return nan_forecast_future, None, np.nan, np.nan, error_message

def naive_forecast(train_series, test_series, h_future):
    model_name = "Ingénuo (Último Valor)"
    nan_forecast_future = np.full(h_future, np.nan)
    try:
        if train_series.empty:
            return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: Train series vacía)"
        last_train_val = train_series.iloc[-1]
        
        test_forecast_values = np.full(len(test_series), last_train_val) if not test_series.empty else np.array([])
        rmse, mae = calculate_metrics(test_series, test_forecast_values)
        
        full_series = pd.concat([train_series, test_series])
        if full_series.empty:
            return nan_forecast_future, None, rmse, mae, f"{model_name} (Error: Full series vacía para pron. futuro)"
        last_full_series_val = full_series.iloc[-1]
        future_forecast_values = np.full(h_future, last_full_series_val)
        return future_forecast_values, None, rmse, mae, model_name
    except Exception as e:
        error_message = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return nan_forecast_future, None, np.nan, np.nan, error_message

def seasonal_naive_forecast(train_series, test_series, h_future, seasonal_period):
    model_name = f"Estacional Ingénuo (P:{seasonal_period})"
    nan_forecast_future = np.full(h_future, np.nan)
    try:
        if seasonal_period <= 0 or train_series.empty or seasonal_period > len(train_series):
            return nan_forecast_future, None, np.nan, np.nan, model_name + " (Error: Inválido o datos insuf.)"

        test_forecast_values = np.zeros(len(test_series))
        if not test_series.empty:
            for i in range(len(test_series)):
                 # CORRECCIÓN: Eliminado el ':' extra al final de la subcondición
                 if len(train_series) - seasonal_period + (i % seasonal_period) < len(train_series) and \
                    len(train_series) - seasonal_period + (i % seasonal_period) >= 0:
                    test_forecast_values[i] = train_series.iloc[len(train_series) - seasonal_period + (i % seasonal_period)]
                 else: 
                    test_forecast_values[i] = np.nan 
        else: test_forecast_values = np.array([])
            
        rmse, mae = calculate_metrics(test_series, test_forecast_values)
        
        full_series = pd.concat([train_series, test_series])
        future_forecast_values = np.zeros(h_future)
        if not full_series.empty and seasonal_period <= len(full_series):
            for i in range(h_future):
                future_forecast_values[i] = full_series.iloc[len(full_series) - seasonal_period + (i % seasonal_period)]
        else:
            future_forecast_values.fill(np.nan)
            if seasonal_period > len(full_series): model_name += " (Error: Pron. Futuro Inválido)"
        return future_forecast_values, None, rmse, mae, model_name
    except Exception as e:
        error_message = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return nan_forecast_future, None, np.nan, np.nan, error_message

def moving_average_forecast(train_series, test_series, h_future, window):
    model_name = f"Promedio Móvil (Ventana {window})"
    nan_forecast_future = np.full(h_future, np.nan)
    try:
        if window <= 0 or train_series.empty or window > len(train_series): 
            return nan_forecast_future, None, np.nan, np.nan, model_name + " (Error: Ventana Inválida o Datos Insuf. en Train)"

        test_fc_val = np.full(len(test_series), train_series.iloc[-window:].mean()) if not test_series.empty and len(train_series) >= window else (np.full(len(test_series), train_series.mean()) if not test_series.empty else np.array([]))
        rmse, mae = calculate_metrics(test_series, test_fc_val)
        
        full_series = pd.concat([train_series, test_series])
        if full_series.empty or window > len(full_series):
            return nan_forecast_future, None, rmse, mae, model_name + " (Error: Datos Insuf. para Pron. Futuro)"
        
        future_fc_val = np.zeros(h_future); series_for_ma = full_series.copy()
        for i in range(h_future):
            current_ma_fc = series_for_ma.iloc[-window:].mean() if len(series_for_ma) >= window else (series_for_ma.mean() if not series_for_ma.empty else np.nan)
            future_fc_val[i] = current_ma_fc
            if not np.isnan(current_ma_fc) and isinstance(series_for_ma.index, pd.DatetimeIndex) and not series_for_ma.empty:
                last_dt = series_for_ma.index[-1]; freq_inf = pd.infer_freq(series_for_ma.index)
                try:
                    next_dt = last_dt + pd.tseries.frequencies.to_offset(freq_inf) if freq_inf else (last_dt + (series_for_ma.index[-1] - series_for_ma.index[-2]) if len(series_for_ma.index) > 1 else last_dt + pd.Timedelta(days=1))
                    new_pt = pd.Series([current_ma_fc], index=[next_dt])
                    series_for_ma = pd.concat([series_for_ma, new_pt])
                except Exception: 
                    future_fc_val[i] = np.nan; break 
            else: future_fc_val[i] = np.nan; break 
        return future_fc_val, None, rmse, mae, model_name
    except Exception as e:
        error_message = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return nan_forecast_future, None, np.nan, np.nan, error_message

# --- Modelos de Statsmodels ---
def forecast_with_statsmodels(train_series, test_series, h_future, model_name_short, seasonal_period=None, holt_params=None, hw_params=None):
    min_data_needed = 5
    if model_name_short == "Holt-Winters" and seasonal_period and seasonal_period > 1: min_data_needed = max(min_data_needed, 2 * seasonal_period + 1) 
    
    model_display_name = model_name_short 
    if model_name_short == "SES": model_display_name = "Suav. Exp. Simple"
    elif model_name_short == "Holt": 
        damped_h = holt_params.get('damped_trend', False) if holt_params else False
        model_display_name = "Holt" + (" Amort." if damped_h else "")
    elif model_name_short == "Holt-Winters":
        if not seasonal_period or seasonal_period <= 1: return np.full(h_future, np.nan), None, np.nan, np.nan, "HW (Error: Período estacional > 1 requerido)."
        trend_hw = hw_params.get('trend', 'add') if hw_params else 'add'
        seasonal_hw = hw_params.get('seasonal', 'add') if hw_params else 'add'
        damped_hw = hw_params.get('damped_trend', False) if hw_params else False
        use_bc_hw = hw_params.get('use_boxcox', False) if hw_params else False
        model_display_name = f"HW (T:{trend_hw},S:{seasonal_hw},P:{seasonal_period}{',Damp' if damped_hw else ''}{',BC' if use_bc_hw else ''})"
    
    try:
        if train_series.empty or len(train_series) < min_data_needed :
            msg_err = f"{model_display_name} (Error: Datos insuficientes - {len(train_series)} obs, necesita {min_data_needed})"
            return np.full(h_future, np.nan), None, np.nan, np.nan, msg_err

        fit_model_train = None
        if model_name_short == "SES":
            fit_model_train = SimpleExpSmoothing(train_series, initialization_method="estimated").fit()
        elif model_name_short == "Holt":
            damped = holt_params.get('damped_trend', False) if holt_params else False
            fit_model_train = Holt(train_series, damped_trend=damped, initialization_method="estimated").fit()
        elif model_name_short == "Holt-Winters":
            trend = hw_params.get('trend', 'add') if hw_params else 'add'
            seasonal = hw_params.get('seasonal', 'add') if hw_params else 'add'
            damped = hw_params.get('damped_trend', False) if hw_params else False
            use_bc = hw_params.get('use_boxcox', False) if hw_params else False
            fit_model_train = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period, damped_trend=damped, use_boxcox=use_bc, initialization_method="estimated").fit()
        else: return np.full(h_future, np.nan), None, np.nan, np.nan, f"'{model_name_short}' (Error: Modelo Statsmodels no reconocido)."

        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty: test_fc_vals = fit_model_train.forecast(len(test_series)); rmse_test, mae_test = calculate_metrics(test_series, test_fc_vals)
        
        full_series = pd.concat([train_series, test_series])
        if full_series.empty or len(full_series) < min_data_needed:
            future_fc_vals_arr = fit_model_train.forecast(h_future).values if h_future > 0 else np.array([])
            future_ci_df_res = None
            if h_future > 0: try: pred_obj = fit_model_train.get_prediction(start=len(train_series), end=len(train_series) + h_future -1); future_ci_df_res = pred_obj.conf_int(alpha=0.05) except: pass
            return future_fc_vals_arr, future_ci_df_res, rmse_test, mae_test, model_display_name + " (Pron. de modelo de train)"
        
        fit_model_full = type(fit_model_train.model)(full_series, **fit_model_train.model.kwargs).fit(**fit_model_train.params_formatted)
        
        future_fc_vals_arr = fit_model_full.forecast(h_future).values if h_future > 0 else np.array([])
        future_ci_df_res = None
        if h_future > 0: try: pred_obj_full = fit_model_full.get_prediction(start=len(full_series), end=len(full_series) + h_future - 1); future_ci_df_res = pred_obj_full.conf_int(alpha=0.05) except: future_ci_df_res = None
        
        return future_fc_vals_arr, future_ci_df_res, rmse_test, mae_test, model_display_name
    except Exception as e:
        error_message_detail = f"{model_display_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, error_message_detail

# --- AutoARIMA ---
def forecast_with_auto_arima(train_series, test_series, h_future, seasonal_period, arima_params=None):
    model_base_name = "AutoARIMA"
    try:
        min_samples = 10
        if seasonal_period > 1: min_samples = max(min_samples, 2 * seasonal_period + 1)
        if train_series.empty or len(train_series) < min_samples:
            return np.full(h_future, np.nan), None, np.nan, np.nan, f"{model_base_name} (Error: Datos insuficientes - {len(train_series)} obs, necesita {min_samples})"

        m_val = seasonal_period if seasonal_period > 1 else 1; seasonal_flag = True if seasonal_period > 1 else False; ap = arima_params or {}
        auto_arima_model_train = pm.auto_arima(train_series,start_p=1,start_q=1,max_p=ap.get('max_p',3),max_q=ap.get('max_q',3),max_d=ap.get('max_d',2),m=m_val,start_P=0,seasonal=seasonal_flag,max_P=ap.get('max_P',1),max_Q=ap.get('max_Q',1),max_D=ap.get('max_D',1), test='adf', information_criterion='aic', trace=False,error_action='ignore', suppress_warnings=True, stepwise=True)
        model_name_display = f"{model_base_name} {auto_arima_model_train.order}{auto_arima_model_train.seasonal_order if seasonal_flag and auto_arima_model_train.seasonal_order != (0,0,0,0) else ''}"
        
        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty: test_fc = auto_arima_model_train.predict(n_periods=len(test_series)); rmse_test, mae_test = calculate_metrics(test_series, test_fc)
        
        full_series = pd.concat([train_series, test_series])
        future_fc_output = np.array([])
        future_ci_df_output = None

        if full_series.empty or len(full_series) < min_samples:
            if h_future > 0:
                future_fc_output, future_ci_arr_output = auto_arima_model_train.predict(n_periods=h_future, return_conf_int=True)
                if future_ci_arr_output is not None:
                    future_ci_df_output = pd.DataFrame(future_ci_arr_output, columns=['lower', 'upper'])
            return future_fc_output, future_ci_df_output, rmse_test, mae_test, model_name_display + " (Pron. de modelo de train)"

        final_model_fit = pm.ARIMA(order=auto_arima_model_train.order, seasonal_order=auto_arima_model_train.seasonal_order if seasonal_flag else (0,0,0,0),suppress_warnings=True).fit(full_series)
        if h_future > 0:
            future_fc_output, future_ci_arr_output = final_model_fit.predict(n_periods=h_future, return_conf_int=True)
            if future_ci_arr_output is not None:
                future_ci_df_output = pd.DataFrame(future_ci_arr_output, columns=['lower', 'upper'])
        
        return future_fc_output, future_ci_df_output, rmse_test, mae_test, model_name_display
    except Exception as e:
        error_message_detail = f"{model_base_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, error_message_detail
