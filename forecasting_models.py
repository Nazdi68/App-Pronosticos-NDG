  # forecasting_models.py (v3.0 - Simplificado)
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
    if len(y_true_clean) == 0 or len(y_pred_clean) == 0 or len(y_true_clean) != len(y_pred_clean): # Chequeo adicional
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return rmse, mae

# --- Modelos Simples para el Nuevo Flujo ---
# Estas funciones ahora toman la serie completa, ajustan, y devuelven:
# forecast_future, conf_int_future, fitted_values, model_name

def run_ses_simple(series, h_future):
    model_name = "Suavización Exponencial Simple (SES)"
    try:
        if series.empty or len(series) < 5: # Mínimo de datos
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes)"
        
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        
        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception: # Ignorar si PI falla
            pass
            
        return forecast_values, conf_int_df, fitted_values, model_name
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg

def run_holt_simple(series, h_future, damped=False):
    model_name = "Holt (Tendencia Lineal)" + (" Amortiguada" if damped else "")
    try:
        if series.empty or len(series) < 10: # Holt necesita un poco más
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes)"

        model = Holt(series, damped_trend=damped, initialization_method="estimated")
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        
        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception:
            pass
            
        return forecast_values, conf_int_df, fitted_values, model_name
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg

def run_hw_simple(series, h_future, seasonal_periods, trend='add', seasonal='add', damped=False, use_boxcox=False):
    model_name_base = "Holt-Winters"
    model_name = f"{model_name_base} (T:{trend},S:{seasonal},P:{seasonal_periods}{',Damp' if damped else ''}{',BC' if use_boxcox else ''})"
    try:
        if seasonal_periods <= 1:
             return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Período estacional debe ser > 1)"
        if series.empty or len(series) < 2 * seasonal_periods +1 : # Necesita al menos 2 ciclos
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes para P={seasonal_periods})"

        model = ExponentialSmoothing(
            series, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods, 
            damped_trend=damped,
            use_boxcox=use_boxcox,
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        
        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception:
            pass
            
        return forecast_values, conf_int_df, fitted_values, model_name
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg

def run_autoarima_simple(series, h_future, seasonal_period, arima_params=None):
    model_base_name = "AutoARIMA"
    try:
        min_samples = 15 # AutoARIMA puede necesitar más, especialmente si es estacional
        if seasonal_period > 1: min_samples = max(min_samples, 2 * seasonal_period + 5) 
        if series.empty or len(series) < min_samples:
            return np.full(h_future, np.nan), None, None, f"{model_base_name} (Error: Datos insuficientes - {len(series)} obs, necesita {min_samples})"

        m_val = seasonal_period if seasonal_period > 1 else 1
        seasonal_flag = True if seasonal_period > 1 else False
        ap = arima_params or {} # Usa defaults de auto_arima si no se proveen

        model = pm.auto_arima(
            series,
            start_p=ap.get('start_p', 1), start_q=ap.get('start_q', 1),
            max_p=ap.get('max_p',3), max_q=ap.get('max_q',3), max_d=ap.get('max_d',2),
            m=m_val, 
            start_P=ap.get('start_P', 0), seasonal=seasonal_flag,
            max_P=ap.get('max_P',1), max_Q=ap.get('max_Q',1), max_D=ap.get('max_D',1), 
            test='adf', information_criterion='aic', trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        
        model_name_display = f"{model_base_name} {model.order}{model.seasonal_order if seasonal_flag and model.seasonal_order != (0,0,0,0) and model.seasonal_order != (0,0,0,1) else ''}"
        
        forecast_values = np.array([])
        conf_int_df = None
        if h_future > 0:
            fc_raw, ci_raw = model.predict(n_periods=h_future, return_conf_int=True)
            forecast_values = np.array(fc_raw)
            if ci_raw is not None:
                conf_int_df = pd.DataFrame(ci_raw, columns=['lower', 'upper'])
        
        fitted_values = model.predict_in_sample(X=None, start=0, end=len(series)-1) # Ajustado para obtener todos los fitted
            
        return forecast_values, conf_int_df, fitted_values, model_name_display
    except Exception as e:
        error_msg = f"{model_base_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg

# --- Funciones de Baseline (pueden seguir usando la estructura anterior si se desea, 
# ya que no tienen "fittedvalues" de la misma manera, o se adaptan)

def historical_average_simple(series, h_future):
    model_name = "Promedio Histórico (Simple)"
    try:
        if series.empty:
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Serie vacía)"
        mean_val = series.mean()
        forecast_values = np.full(h_future, mean_val)
        fitted_values = np.full(len(series), mean_val) # El "ajuste" es la media constante
        return forecast_values, None, fitted_values, model_name
    except Exception as e:
        return np.full(h_future, np.nan), None, None, f"{model_name} (Error: {type(e).__name__})"

def naive_simple(series, h_future):
    model_name = "Ingénuo (Último Valor - Simple)"
    try:
        if series.empty:
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Serie vacía)"
        last_val = series.iloc[-1]
        forecast_values = np.full(h_future, last_val)
        fitted_values = series.shift(1).fillna(method='bfill').values # y_hat_t = y_t-1
        return forecast_values, None, fitted_values, model_name
    except Exception as e:
        return np.full(h_future, np.nan), None, None, f"{model_name} (Error: {type(e).__name__})"

# Podrías añadir versiones simplificadas de Seasonal Naive y Moving Average si es necesario,
# o ajustar las llamadas en app.py para que usen estas nuevas funciones simples.
# Por ahora, el app.py v4.0 asume que llamarás a estas nuevas funciones "run_..._simple".