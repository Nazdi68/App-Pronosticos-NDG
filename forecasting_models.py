# forecasting_models.py
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm

def calculate_metrics(y_true, y_pred):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Asegurar que no haya NaNs o Infs que rompan las métricas
    valid_indices = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_clean = y_true_arr[valid_indices]
    y_pred_clean = y_pred_arr[valid_indices]

    if len(y_true_clean) == 0:
        return np.nan, np.nan
        
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return rmse, mae

def train_test_split_series(series, test_size_periods):
    if not isinstance(series, pd.Series): # Asegurar que sea Series
        series = pd.Series(series)

    if test_size_periods <= 0 or test_size_periods >= len(series):
        # Devolver toda la serie como entrenamiento y una serie vacía para prueba
        return series, pd.Series(dtype=series.dtype, index=pd.DatetimeIndex([])) 
    
    train = series[:-test_size_periods]
    test = series[-test_size_periods:]
    return train, test

# --- Modelos de Línea Base ---
def historical_average_forecast(train_series, test_series, h_future):
    model_name = "Promedio Histórico"
    mean_val_train = train_series.mean() if not train_series.empty else np.nan
    
    test_forecast_values = np.full(len(test_series), mean_val_train) if not test_series.empty else np.array([])
    rmse, mae = calculate_metrics(test_series, test_forecast_values)

    full_series = pd.concat([train_series, test_series])
    full_series_mean = full_series.mean() if not full_series.empty else np.nan
    future_forecast_values = np.full(h_future, full_series_mean)
    
    return future_forecast_values, None, rmse, mae, model_name

def naive_forecast(train_series, test_series, h_future):
    model_name = "Ingénuo (Último Valor)"
    last_train_val = train_series.iloc[-1] if not train_series.empty else np.nan
    
    test_forecast_values = np.full(len(test_series), last_train_val) if not test_series.empty else np.array([])
    rmse, mae = calculate_metrics(test_series, test_forecast_values)
    
    full_series = pd.concat([train_series, test_series])
    last_full_series_val = full_series.iloc[-1] if not full_series.empty else np.nan
    future_forecast_values = np.full(h_future, last_full_series_val)
    
    return future_forecast_values, None, rmse, mae, model_name

def seasonal_naive_forecast(train_series, test_series, h_future, seasonal_period):
    model_name = f"Estacional Ingénuo (P:{seasonal_period})"
    nan_forecast_future = np.full(h_future, np.nan)

    if seasonal_period <= 0 or train_series.empty or seasonal_period > len(train_series):
        return nan_forecast_future, None, np.nan, np.nan, model_name + " (Inválido)"

    test_forecast_values = np.zeros(len(test_series))
    if not test_series.empty:
        for i in range(len(test_series)):
            # Índice en train_series para el valor estacional correspondiente
            # (i % seasonal_period) da el offset dentro del ciclo estacional para el punto i-ésimo del test set
            # len(train_series) - seasonal_period + (i % seasonal_period) podría ser negativo si i es pequeño
            # Una forma más robusta: tomar el último ciclo completo de train_series
            if len(train_series) >= seasonal_period:
                test_forecast_values[i] = train_series.iloc[len(train_series) - seasonal_period + (i % seasonal_period)]
            else: # No hay un ciclo completo en train para tomar como referencia
                test_forecast_values[i] = np.nan 
    else:
        test_forecast_values = np.array([]) # Vacío si test_series es vacía
        
    rmse, mae = calculate_metrics(test_series, test_forecast_values)
    
    full_series = pd.concat([train_series, test_series])
    future_forecast_values = np.zeros(h_future)
    if not full_series.empty and seasonal_period <= len(full_series):
        for i in range(h_future):
            future_forecast_values[i] = full_series.iloc[len(full_series) - seasonal_period + (i % seasonal_period)]
    else:
        future_forecast_values.fill(np.nan)
        if seasonal_period > len(full_series): model_name += " (Pron. Futuro Inválido)"

    return future_forecast_values, None, rmse, mae, model_name

# --- Modelos de Statsmodels ---
def forecast_with_statsmodels(train_series, test_series, h_future, model_name_short, seasonal_period=None, holt_params=None, hw_params=None):
    min_data_needed = 5
    if model_name_short == "Holt-Winters" and seasonal_period and seasonal_period > 1:
        min_data_needed = max(min_data_needed, 2 * seasonal_period + 1) # Holt-Winters necesita al menos 2 ciclos completos

    if train_series.empty or len(train_series) < min_data_needed :
        msg = f"{model_name_short}: Datos insuficientes ({len(train_series)} obs, necesita {min_data_needed})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, msg

    fit_model_train = None
    model_display_name = model_name_short
    
    try:
        if model_name_short == "SES":
            model_display_name = "Suav. Exp. Simple"
            fit_model_train = SimpleExpSmoothing(train_series, initialization_method="estimated").fit()
        elif model_name_short == "Holt":
            damped = holt_params.get('damped_trend', False) if holt_params else False
            model_display_name = "Holt" + (" Amort." if damped else "")
            fit_model_train = Holt(train_series, damped_trend=damped, initialization_method="estimated").fit()
        elif model_name_short == "Holt-Winters":
            if not seasonal_period or seasonal_period <= 1:
                return np.full(h_future, np.nan), None, np.nan, np.nan, "HW: Período estacional > 1 requerido."
            trend = hw_params.get('trend', 'add'); seasonal = hw_params.get('seasonal', 'add')
            damped = hw_params.get('damped_trend', False); use_bc = hw_params.get('use_boxcox', False)
            model_display_name = f"HW (T:{trend},S:{seasonal},P:{seasonal_period}{',Damp' if damped else ''}{',BC' if use_bc else ''})"
            fit_model_train = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period, damped_trend=damped, use_boxcox=use_bc, initialization_method="estimated").fit()
        else:
            return np.full(h_future, np.nan), None, np.nan, np.nan, f"Modelo Statsmodels '{model_name_short}' no reconocido."

        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty:
            test_fc_vals = fit_model_train.forecast(len(test_series))
            rmse_test, mae_test = calculate_metrics(test_series, test_fc_vals)

        full_series = pd.concat([train_series, test_series])
        if full_series.empty or len(full_series) < min_data_needed: # Si la serie completa es muy corta
            future_fc_vals = fit_model_train.forecast(h_future) if h_future > 0 else np.array([])
            future_ci_df = None # PI no es crítico aquí
            if h_future > 0:
                try:
                    pred_obj = fit_model_train.get_prediction(start=len(train_series), end=len(train_series) + h_future -1)
                    future_ci_df = pred_obj.conf_int(alpha=0.05)
                except: pass
            return future_fc_vals.values if isinstance(future_fc_vals, pd.Series) else future_fc_vals, future_ci_df, rmse_test, mae_test, model_display_name + " (Pron. de modelo de train)"
        
        fit_model_full = type(fit_model_train.model)(full_series, **fit_model_train.model.kwargs).fit(**fit_model_train.params_formatted) # Re-fit con mismos params

        future_fc_vals = fit_model_full.forecast(h_future) if h_future > 0 else np.array([])
        future_ci_df = None
        if h_future > 0:
            try:
                pred_obj_full = fit_model_full.get_prediction(start=len(full_series), end=len(full_series) + h_future - 1)
                future_ci_df = pred_obj_full.conf_int(alpha=0.05)
            except: future_ci_df = None
        
        return future_fc_vals.values if isinstance(future_fc_vals, pd.Series) else future_fc_vals, future_ci_df, rmse_test, mae_test, model_display_name
    except Exception as e:
        return np.full(h_future, np.nan), None, np.nan, np.nan, f"{model_name_short}: Error ({str(e)[:30]}...)"

# --- AutoARIMA ---
def forecast_with_auto_arima(train_series, test_series, h_future, seasonal_period, arima_params=None):
    model_base_name = "AutoARIMA"
    min_samples = 10
    if seasonal_period > 1: min_samples = max(min_samples, 2 * seasonal_period + 1)

    if train_series.empty or len(train_series) < min_samples:
        msg = f"{model_base_name}: Datos insuficientes ({len(train_series)} obs, necesita {min_samples})"
        return np.full(h_future, np.nan), None, np.nan, np.nan, msg

    try:
        m_val = seasonal_period if seasonal_period > 1 else 1
        seasonal_flag = True if seasonal_period > 1 else False
        
        ap = arima_params or {} # Default a dict vacío
        auto_arima_model_train = pm.auto_arima(
            train_series,
            start_p=1, start_q=1,
            max_p=ap.get('max_p', 3), max_q=ap.get('max_q', 3), max_d=ap.get('max_d', 2),
            m=m_val, start_P=0, seasonal=seasonal_flag,
            max_P=ap.get('max_P', 1), max_Q=ap.get('max_Q', 1), max_D=ap.get('max_D', 1), 
            test='adf', information_criterion='aic', trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        model_name = f"{model_base_name} {auto_arima_model_train.order}{auto_arima_model_train.seasonal_order if seasonal_flag and auto_arima_model_train.seasonal_order != (0,0,0,0) else ''}"

        rmse_test, mae_test = np.nan, np.nan
        if not test_series.empty:
            test_fc = auto_arima_model_train.predict(n_periods=len(test_series))
            rmse_test, mae_test = calculate_metrics(test_series, test_fc)

        full_series = pd.concat([train_series, test_series])
        if full_series.empty or len(full_series) < min_samples: # Si la serie completa es muy corta
            future_fc, future_ci_arr = auto_arima_model_train.predict(n_periods=h_future, return_conf_int=True) if h_future > 0 else (np.array([]), None)
            future_ci_df = pd.DataFrame(future_ci_arr, columns=['lower', 'upper']) if future_ci_arr is not None else None
            return future_fc, future_ci_df, rmse_test, mae_test, model_name + " (Pron. de modelo de train)"

        # Re-fit on full_series with found orders
        final_model = pm.ARIMA(order=auto_arima_model_train.order, 
                               seasonal_order=auto_arima_model_train.seasonal_order if seasonal_flag else (0,0,0,0),
                               suppress_warnings=True).fit(full_series)
        
        future_fc, future_ci_arr = final_model.predict(n_periods=h_future, return_conf_int=True) if h_future > 0 else (np.array([]), None)
        future_ci_df = pd.DataFrame(future_ci_arr, columns=['lower', 'upper']) if future_ci_arr is not None else None

        return future_fc, future_ci_df, rmse_test, mae_test, model_name
    except Exception as e:
        return np.full(h_future, np.nan), None, np.nan, np.nan, f"{model_base_name}: Error ({str(e)[:30]}...)"
