# forecasting_models.py (v3.1 - Añadir devolución de parámetros)
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
    if len(y_true_clean) == 0 or len(y_pred_clean) == 0 or len(y_true_clean) != len(y_pred_clean): return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return rmse, mae

# --- Modelos Simples para el Nuevo Flujo ---
# Ahora devuelven: forecast_future, conf_int_future, fitted_values, model_name, model_params_dict

def run_ses_simple(series, h_future):
    model_name = "Suavización Exponencial Simple (SES)"
    model_params = {}
    try:
        if series.empty or len(series) < 5:
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes)", model_params
        
        model = SimpleExpSmoothing(series, initialization_method="estimated")
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        model_params = {key: round(val, 4) for key, val in fit.params.items() if "smoothing_level" in key or "initial_level" in key} # Redondear
        
        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception: pass
            
        return forecast_values, conf_int_df, fitted_values, model_name, model_params
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg, model_params

def run_holt_simple(series, h_future, damped=False):
    model_name_base = "Holt (Tendencia Lineal)"
    model_name = model_name_base + (" Amortiguada" if damped else "")
    model_params = {}
    try:
        if series.empty or len(series) < 10: 
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes)", model_params

        model = Holt(series, damped_trend=damped, initialization_method="estimated")
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        model_params = {key: round(val, 4) for key, val in fit.params.items() if "smoothing_level" in key or "smoothing_trend" in key or "initial_level" in key or "initial_trend" in key}
        if damped: model_params['phi (damping_trend)'] = round(fit.params.get('damping_trend', np.nan), 4)


        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception: pass
            
        return forecast_values, conf_int_df, fitted_values, model_name, model_params
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg, model_params

def run_hw_simple(series, h_future, seasonal_periods, trend='add', seasonal='add', damped=False, use_boxcox=False):
    model_name_base = "Holt-Winters"
    model_name = f"{model_name_base} (T:{trend},S:{seasonal},P:{seasonal_periods}{',Damp' if damped else ''}{',BC' if use_boxcox else ''})"
    model_params = {}
    try:
        if seasonal_periods <= 1:
             return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Período estacional debe ser > 1)", model_params
        if series.empty or len(series) < 2 * seasonal_periods + 1 : 
            return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Datos insuficientes para P={seasonal_periods})", model_params

        model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, 
                                     damped_trend=damped, use_boxcox=use_boxcox, initialization_method="estimated")
        fit = model.fit()
        forecast_values = fit.forecast(h_future).values
        fitted_values = fit.fittedvalues.values
        
        param_keys = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 
                      'initial_level', 'initial_trend', 'initial_seasons', 'damping_trend']
        model_params = {key: (round(val, 4) if isinstance(val, (int, float)) else val) for key, val in fit.params.items() if key in param_keys}
        if 'initial_seasons' in model_params and isinstance(model_params['initial_seasons'], np.ndarray):
            model_params['initial_seasons'] = [round(s, 4) for s in model_params['initial_seasons'][:4]] # Mostrar solo los primeros 4 para brevedad
            if len(fit.params['initial_seasons']) > 4: model_params['initial_seasons'].append("...")


        conf_int_df = None
        try:
            pred_obj = fit.get_prediction(start=len(series), end=len(series) + h_future - 1)
            conf_int_df = pred_obj.conf_int(alpha=0.05)
        except Exception: pass
            
        return forecast_values, conf_int_df, fitted_values, model_name, model_params
    except Exception as e:
        error_msg = f"{model_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg, model_params

def run_autoarima_simple(series, h_future, seasonal_period, arima_params=None):
    model_base_name = "AutoARIMA"
    model_params = {}
    try:
        min_samples = 15 
        if seasonal_period > 1: min_samples = max(min_samples, 2 * seasonal_period + 5) 
        if series.empty or len(series) < min_samples:
            return np.full(h_future, np.nan), None, None, f"{model_base_name} (Error: Datos insuficientes - {len(series)} obs, necesita {min_samples})", model_params

        m_val = seasonal_period if seasonal_period > 1 else 1
        seasonal_flag = True if seasonal_period > 1 else False
        ap = arima_params or {} 

        model = pm.auto_arima(series, start_p=ap.get('start_p',1), start_q=ap.get('start_q',1),
                              max_p=ap.get('max_p',3), max_q=ap.get('max_q',3), max_d=ap.get('max_d',2),
                              m=m_val, start_P=ap.get('start_P',0), seasonal=seasonal_flag,
                              max_P=ap.get('max_P',1), max_Q=ap.get('max_Q',1), max_D=ap.get('max_D',1), 
                              test='adf', information_criterion='aic', trace=False,
                              error_action='ignore', suppress_warnings=True, stepwise=True)
        
        model_name_display = f"{model_base_name} {model.order}{model.seasonal_order if seasonal_flag and model.seasonal_order != (0,0,0,0) and model.seasonal_order != (0,0,0,1) else ''}"
        model_params['order (p,d,q)'] = str(model.order)
        if seasonal_flag and model.seasonal_order != (0,0,0,0) and model.seasonal_order != (0,0,0,1) :
            model_params['seasonal_order (P,D,Q,m)'] = str(model.seasonal_order)
        # model_params['coefficients'] = {k: round(v,4) for k,v in model.params().items()} # Puede ser muy largo

        forecast_values = np.array([])
        conf_int_df = None
        if h_future > 0:
            fc_raw, ci_raw = model.predict(n_periods=h_future, return_conf_int=True)
            forecast_values = np.array(fc_raw)
            if ci_raw is not None: conf_int_df = pd.DataFrame(ci_raw, columns=['lower', 'upper'])
        
        fitted_values = model.predict_in_sample(X=None, start=0, end=len(series)-1)
            
        return forecast_values, conf_int_df, fitted_values, model_name_display, model_params
    except Exception as e:
        error_msg = f"{model_base_name} (Error: {type(e).__name__} - {str(e)[:100]})"
        return np.full(h_future, np.nan), None, None, error_msg, model_params

# --- Baselines Simples ---
def historical_average_simple(series, h_future):
    model_name = "Promedio Histórico" # Simplificado el nombre
    try:
        if series.empty: return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Serie vacía)", {}
        mean_val = series.mean(); forecast_values = np.full(h_future, mean_val)
        fitted_values = np.full(len(series), mean_val)
        return forecast_values, None, fitted_values, model_name, {"mean": round(mean_val, 4)}
    except Exception as e: return np.full(h_future, np.nan),None,None,f"{model_name} (Error: {type(e).__name__})",{}

def naive_simple(series, h_future):
    model_name = "Ingénuo (Último Valor)" # Simplificado el nombre
    try:
        if series.empty: return np.full(h_future, np.nan), None, None, f"{model_name} (Error: Serie vacía)", {}
        last_val = series.iloc[-1]; forecast_values = np.full(h_future, last_val)
        fitted_values = series.shift(1).fillna(method='bfill').values
        return forecast_values, None, fitted_values, model_name, {"last_value": round(last_val,4) if isinstance(last_val, (int,float)) else last_val}
    except Exception as e: return np.full(h_future, np.nan),None,None,f"{model_name} (Error: {type(e).__name__})",{}