# forecasting_models.py (v2.2)
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm

def calculate_metrics(y_true, y_pred):
    # ... (sin cambios)
    y_true_arr = np.array(y_true); y_pred_arr = np.array(y_pred)
    valid_indices = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_clean = y_true_arr[valid_indices]; y_pred_clean = y_pred_arr[valid_indices]
    if len(y_true_clean) == 0: return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return rmse, mae

def train_test_split_series(series, test_size_periods):
    # ... (sin cambios)
    if not isinstance(series, pd.Series): series = pd.Series(series)
    if test_size_periods <= 0 or test_size_periods >= len(series):
        return series, pd.Series(dtype=series.dtype, index=pd.DatetimeIndex([])) 
    train = series[:-test_size_periods]; test = series[-test_size_periods:]
    return train, test

# --- Modelos de Línea Base ---
def historical_average_forecast(train_series, test_series, h_future):
    # ... (sin cambios)
    model_name = "Promedio Histórico"; nan_forecast_future = np.full(h_future, np.nan)
    try:
        if train_series.empty: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: Train series vacía)"
        mean_val_train = train_series.mean()
        test_fc = np.full(len(test_series), mean_val_train) if not test_series.empty else np.array([])
        rmse, mae = calculate_metrics(test_series, test_fc)
        full_series = pd.concat([train_series, test_series])
        if full_series.empty: return nan_forecast_future, None, rmse, mae, f"{model_name} (Error: Full series vacía)"
        full_mean = full_series.mean(); future_fc = np.full(h_future, full_mean)
        return future_fc, None, rmse, mae, model_name
    except Exception as e: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"

def naive_forecast(train_series, test_series, h_future):
    # ... (sin cambios)
    model_name = "Ingénuo (Último Valor)"; nan_forecast_future = np.full(h_future, np.nan)
    try:
        if train_series.empty: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: Train series vacía)"
        last_train = train_series.iloc[-1]
        test_fc = np.full(len(test_series), last_train) if not test_series.empty else np.array([])
        rmse, mae = calculate_metrics(test_series, test_fc)
        full_series = pd.concat([train_series, test_series])
        if full_series.empty: return nan_forecast_future, None, rmse, mae, f"{model_name} (Error: Full series vacía)"
        last_full = full_series.iloc[-1]; future_fc = np.full(h_future, last_full)
        return future_fc, None, rmse, mae, model_name
    except Exception as e: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"

def seasonal_naive_forecast(train_series, test_series, h_future, seasonal_period):
    # ... (sin cambios)
    model_name = f"Estacional Ingénuo (P:{seasonal_period})"; nan_forecast_future = np.full(h_future, np.nan)
    try:
        if seasonal_period <= 0 or train_series.empty or seasonal_period > len(train_series): return nan_forecast_future, None, np.nan, np.nan, model_name + " (Error: Inválido o datos insuf.)"
        test_fc = np.zeros(len(test_series))
        if not test_series.empty:
            for i in range(len(test_series)): test_fc[i] = train_series.iloc[len(train_series) - seasonal_period + (i % seasonal_period)] if len(train_series) - seasonal_period + (i % seasonal_period) < len(train_series) and len(train_series) - seasonal_period + (i % seasonal_period) >= 0 else np.nan 
        else: test_fc = np.array([])
        rmse, mae = calculate_metrics(test_series, test_fc)
        full_series = pd.concat([train_series, test_series]); future_fc = np.zeros(h_future)
        if not full_series.empty and seasonal_period <= len(full_series):
            for i in range(h_future): future_fc[i] = full_series.iloc[len(full_series) - seasonal_period + (i % seasonal_period)]
        else: future_fc.fill(np.nan); model_name += " (Error: Pron. Futuro Inválido)" if seasonal_period > len(full_series) else ""
        return future_fc, None, rmse, mae, model_name
    except Exception as e: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"

def moving_average_forecast(train_series, test_series, h_future, window):
    # ... (sin cambios)
    model_name = f"Promedio Móvil (Ventana {window})"; nan_forecast_future = np.full(h_future, np.nan)
    try:
        if window <= 0 or train_series.empty or window > len(train_series): return nan_forecast_future, None, np.nan, np.nan, model_name + " (Error: Ventana Inválida o Datos Insuf. en Train)"
        test_fc = np.full(len(test_series), train_series.iloc[-window:].mean()) if not test_series.empty and len(train_series) >= window else (np.full(len(test_series), train_series.mean()) if not test_series.empty else np.array([]))
        rmse, mae = calculate_metrics(test_series, test_fc)
        full_series = pd.concat([train_series, test_series])
        if full_series.empty or window > len(full_series): return nan_forecast_future, None, rmse, mae, model_name + " (Error: Datos Insuf. para Pron. Futuro)"
        future_fc = np.zeros(h_future); series_ma = full_series.copy()
        for i in range(h_future):
            curr_ma = series_ma.iloc[-window:].mean() if len(series_ma) >= window else (series_ma.mean() if not series_ma.empty else np.nan)
            future_fc[i] = curr_ma
            if not np.isnan(curr_ma) and isinstance(series_ma.index, pd.DatetimeIndex) and not series_ma.empty:
                last_dt = series_ma.index[-1]; freq = pd.infer_freq(series_ma.index)
                try:
                    next_dt = last_dt + pd.tseries.frequencies.to_offset(freq) if freq else (last_dt + (series_ma.index[-1] - series_ma.index[-2]) if len(series_ma.index) > 1 else last_dt + pd.Timedelta(days=1))
                    series_ma = pd.concat([series_ma, pd.Series([curr_ma], index=[next_dt])])
                except: future_fc[i] = np.nan; break 
            else: future_fc[i] = np.nan; break 
        return future_fc, None, rmse, mae, model_name
    except Exception as e: return nan_forecast_future, None, np.nan, np.nan, f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"

# --- Modelos de Statsmodels ---
def forecast_with_statsmodels(train_series, test_series, h_future, model_name_short, seasonal_period=None, holt_params=None, hw_params=None):
    min_data_needed = 5
    if model_name_short == "Holt-Winters" and seasonal_period and seasonal_period > 1: min_data_needed = max(min_data_needed, 2 * seasonal_period + 1) 
    
    model_display_name = model_name_short 
    # Construir nombre ANTES del try, para usarlo en el mensaje de error
    if model_name_short == "SES": model_display_name = "Suav. Exp. Simple"
    elif model_name_short == "Holt": 
        damped_h_cfg = holt_params.get('damped_trend', False) if holt_params else False
        model_display_name = "Holt" + (" Amort." if damped_h_cfg else "")
    elif model_name_short == "Holt-Winters":
        if not seasonal_period or seasonal_period <= 1: return np.full(h_future, np.nan), None, np.nan, np.nan, "HW (Error: Período estacional > 1 requerido)."
        trend_hw_cfg = hw_params.get('trend', 'add') if hw_params else 'add'
        seasonal_hw_cfg = hw_params.get('seasonal', 'add') if hw_params else 'add'
        damped_hw_cfg = hw_params.get('damped_trend', False) if hw_params else False
        use_bc_hw_cfg = hw_params.get('use_boxcox', False) if hw_params else False
        model_display_name = f"HW (T:{trend_hw_cfg},S:{seasonal_hw_cfg},P:{seasonal_period}{',Damp' if damped_hw_cfg else ''}{',BC' if use_bc_hw_cfg else ''})"
    
    try:
        if train_series.empty or len(train_series) < min_data_needed :
            return np.full(h_future, np.nan), None, np.nan, np.nan, f"{model_display_name} (Error: Datos insuf. - {len(train_series)} obs, necesita {min_data_needed})"

        # Ajustar en train_series
        fit_model_train = None
        if model_name_short == "SES":
            fit_model_train = SimpleExpSmoothing(train_series, initialization_method="estimated").fit()
        elif model_name_short == "Holt":
            damped_train = holt_params.get('damped_trend', False) if holt_params else False
            fit_model_train = Holt(train_series, damped_trend=damped_train, initialization_method="estimated").fit()
        elif model_name_short == "Holt-Winters":
            trend_train = hw_params.get('trend', 'add') if hw_params else 'add'
            seasonal_train = hw_params.get('seasonal', 'add') if hw_params else 'add'
            damped_train = hw_params.get('damped_trend', False) if hw_params else False
            use_bc_train = hw_params.get('use_boxcox', False) if hw_params else False
            fit_model_train = ExponentialSmoothing(train_series, trend=trend_train, seasonal=seasonal_train, seasonal_periods=seasonal_period, damped_trend=damped_train, use_boxcox=use_bc_train, initialization_method="estimated").fit()
        else: return np.full(h_future, np.nan), None, np.nan, np.nan, f"'{model_name_short}' (Error: Modelo Statsmodels no reconocido)."

        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty: 
            test_fc_values = fit_model_train.forecast(len(test_series))
            rmse_test, mae_test = calculate_metrics(test_series, test_fc_values)
        
        full_series = pd.concat([train_series, test_series])
        if full_series.empty or len(full_series) < min_data_needed: # Si la serie completa es muy corta
            future_fc_values = fit_model_train.forecast(h_future).values if h_future > 0 else np.array([])
            future_ci_df = None
            if h_future > 0:
                try:
                    pred_obj = fit_model_train.get_prediction(start=len(train_series), end=len(train_series) + h_future -1)
                    future_ci_df = pred_obj.conf_int(alpha=0.05)
                except Exception: pass # Ignorar si falla el PI aquí
            return future_fc_values, future_ci_df, rmse_test, mae_test, model_display_name + " (Pron. de modelo de train)"
        
        # Re-ajustar en full_series con los mismos parámetros
        fit_model_full = None
        if model_name_short == "SES":
            fit_model_full = SimpleExpSmoothing(full_series, initialization_method="estimated").fit()
        elif model_name_short == "Holt":
            damped_full = holt_params.get('damped_trend', False) if holt_params else False
            fit_model_full = Holt(full_series, damped_trend=damped_full, initialization_method="estimated").fit()
        elif model_name_short == "Holt-Winters":
            trend_full = hw_params.get('trend', 'add') if hw_params else 'add'
            seasonal_full = hw_params.get('seasonal', 'add') if hw_params else 'add'
            damped_full = hw_params.get('damped_trend', False) if hw_params else False
            use_bc_full = hw_params.get('use_boxcox', False) if hw_params else False
            fit_model_full = ExponentialSmoothing(full_series, trend=trend_full, seasonal=seasonal_full, seasonal_periods=seasonal_period, damped_trend=damped_full, use_boxcox=use_bc_full, initialization_method="estimated").fit()
        
        future_fc_values = fit_model_full.forecast(h_future).values if h_future > 0 else np.array([])
        future_ci_df = None
        if h_future > 0: 
            try:
                pred_obj_full = fit_model_full.get_prediction(start=len(full_series), end=len(full_series) + h_future - 1)
                future_ci_df = pred_obj_full.conf_int(alpha=0.05)
            except Exception: future_ci_df = None # Asegurar que es None si falla
        
        return future_fc_values, future_ci_df, rmse_test, mae_test, model_display_name
    except Exception as e:
        error_message_detail = f"{model_display_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, error_message_detail

# --- AutoARIMA ---
def forecast_with_auto_arima(train_series, test_series, h_future, seasonal_period, arima_params=None):
    model_base_name = "AutoARIMA"
    try:
        min_samples = 10
        if seasonal_period > 1: min_samples = max(min_samples, 2 * seasonal_period + 2) # Un poco más para SARIMA
        if train_series.empty or len(train_series) < min_samples:
            return np.full(h_future, np.nan), None, np.nan, np.nan, f"{model_base_name} (Error: Datos insuficientes - {len(train_series)} obs, necesita {min_samples})"

        m_val = seasonal_period if seasonal_period > 1 else 1; seasonal_flag = True if seasonal_period > 1 else False; ap = arima_params or {}
        
        # Parámetros para auto_arima, permitiendo que el usuario los controle
        # Si un parámetro no está en ap, auto_arima usará su default.
        auto_arima_model_train = pm.auto_arima(
            train_series,
            start_p=ap.get('start_p', 1), start_q=ap.get('start_q', 1),
            max_p=ap.get('max_p',3), max_q=ap.get('max_q',3), max_d=ap.get('max_d',2),
            m=m_val, 
            start_P=ap.get('start_P', 0), seasonal=seasonal_flag,
            max_P=ap.get('max_P',1), max_Q=ap.get('max_Q',1), max_D=ap.get('max_D',1), 
            test='adf', information_criterion='aic', trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        model_name_display = f"{model_base_name} {auto_arima_model_train.order}{auto_arima_model_train.seasonal_order if seasonal_flag and auto_arima_model_train.seasonal_order != (0,0,0,0) and auto_arima_model_train.seasonal_order != (0,0,0,1) else ''}" # No mostrar (0,0,0,1)
        
        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty: 
            test_fc = auto_arima_model_train.predict(n_periods=len(test_series))
            rmse_test, mae_test = calculate_metrics(test_series, test_fc)
        
        full_series = pd.concat([train_series, test_series])
        future_fc_output = np.array([])
        future_ci_df_output = None

        if full_series.empty or len(full_series) < min_samples: # Si la serie completa es muy corta
            if h_future > 0:
                future_fc_output_raw, future_ci_arr_output_raw = auto_arima_model_train.predict(n_periods=h_future, return_conf_int=True)
                future_fc_output = np.array(future_fc_output_raw) 
                if future_ci_arr_output_raw is not None:
                    future_ci_df_output = pd.DataFrame(future_ci_arr_output_raw, columns=['lower', 'upper'])
            return future_fc_output, future_ci_df_output, rmse_test, mae_test, model_name_display + " (Pron. de modelo de train)"

        # Re-ajustar en full_series con los órdenes encontrados
        final_model_fit = pm.ARIMA(
            order=auto_arima_model_train.order, 
            seasonal_order=auto_arima_model_train.seasonal_order if seasonal_flag else (0,0,0,0),
            suppress_warnings=True
        ).fit(full_series)
        
        if h_future > 0:
            future_fc_output_raw, future_ci_arr_output_raw = final_model_fit.predict(n_periods=h_future, return_conf_int=True)
            future_fc_output = np.array(future_fc_output_raw) 
            if future_ci_arr_output_raw is not None:
                future_ci_df_output = pd.DataFrame(future_ci_arr_output_raw, columns=['lower', 'upper'])
        
        return future_fc_output, future_ci_df_output, rmse_test, mae_test, model_name_display
    except Exception as e:
        error_message_detail = f"{model_base_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, error_message_detail    