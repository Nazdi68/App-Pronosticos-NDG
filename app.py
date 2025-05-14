# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

import data_handler
import visualization
import forecasting_models
import recommendations

st.set_page_config(page_title="Asistente de Pronósticos PRO", layout="wide")

def init_session_state():
    defaults = {
        'df_loaded': None, 'current_file_name': None, 'df_processed': None, 
        'selected_date_col': None, 'selected_value_col': None, 
        'original_target_column_name': "Valor", 
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'moving_avg_window': 3, 
        'model_results': [], 'best_model_name_auto': None,
        'selected_model_for_manual_explore': None,
        'use_train_test_split': True, 'test_split_size': 12, 
        'train_series_for_plot': None, 'test_series_for_plot': None,
        'run_autoarima': True,
        'arima_max_p': 3, 'arima_max_q': 3, 'arima_max_d': 2,
        'arima_max_P': 1, 'arima_max_Q': 1, 'arima_max_D': 1,
        'holt_damped': False, 'hw_trend': 'add', 
        'hw_seasonal': 'add', 'hw_damped': False, 'hw_boxcox': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state()

def to_excel(df):
    output = BytesIO(); df.to_excel(output, index=True, sheet_name='Pronostico')
    return output.getvalue()

def reset_on_file_change():
    keys_to_reset = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name', 'data_diagnosis_report', 'acf_fig', 
        'model_results', 'best_model_name_auto', 
        'selected_model_for_manual_explore', 'train_series_for_plot', 
        'test_series_for_plot', 'auto_seasonal_period'
    ]
    for key in keys_to_reset:
        if key in st.session_state: del st.session_state[key]
    st.session_state.df_loaded = None
    init_session_state()

def reset_model_related_state(): # Usado al cambiar params de preproc o antes de generar modelos
    st.session_state.model_results = []
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    # No reseteamos df_processed aquí si es solo por cambio de params de modelo
    # Pero si es por cambio de params de PREPROCESAMIENTO, df_processed SÍ debería resetearse
    # Esto se maneja en el botón "Aplicar Preprocesamiento"

def prepare_forecast_display_data(model_data, series_full_idx, horizon):
    if model_data is None or model_data.get('forecast_future') is None: return None, None, None
    last_date_hist = series_full_idx.max(); freq = pd.infer_freq(series_full_idx)
    if freq is None and len(series_full_idx) > 1:
        diffs = series_full_idx.to_series().diff().dropna()
        if not diffs.empty: freq = diffs.min()
    if freq is None: freq = 'D'; st.warning(f"Frecuencia no inferida, usando '{freq}'.")
    
    # Asegurar que el horizonte sea al menos 1 para evitar error en date_range
    actual_horizon = max(1, horizon)
    forecast_dates = pd.date_range(start=last_date_hist, periods=actual_horizon + 1, freq=freq)[1:]
    
    forecast_values = model_data['forecast_future']
    # Asegurar que forecast_values tenga la longitud correcta
    if len(forecast_values) != len(forecast_dates) and len(forecast_dates) > 0 :
        # st.warning(f"Discrepancia en longitud de pronóstico y fechas para {model_data.get('name', 'un modelo')}. Ajustando pronóstico.")
        # Esto puede ser problemático, idealmente las longitudes coinciden.
        # Si son diferentes, tomamos el mínimo de las dos longitudes.
        min_len = min(len(forecast_values), len(forecast_dates))
        forecast_values = forecast_values[:min_len]
        forecast_dates = forecast_dates[:min_len]
        if min_len == 0: return None, None, None # No hay nada que mostrar

    conf_int_df_raw = model_data.get('conf_int_future')
    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}
    pi_display_df = None
    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        pi_indexed = conf_int_df_raw.copy()
        # Asegurar que pi_indexed tenga la misma longitud que forecast_dates
        if len(pi_indexed) == len(forecast_dates):
            pi_indexed.index = forecast_dates
            export_dict['Limite Inferior PI'] = pi_indexed['lower'].values
            export_dict['Limite Superior PI'] = pi_indexed['upper'].values
            pi_display_df = pi_indexed[['lower', 'upper']]
        # else:
            # st.warning(f"Discrepancia en longitud de P.I. para {model_data.get('name', 'un modelo')}. No se mostrarán P.I.")
            
    final_export_df = pd.DataFrame(export_dict).set_index('Fecha')
    forecast_series_for_plot = final_export_df['Pronostico'] if not final_export_df.empty else pd.Series(dtype='float64')
    return final_export_df, forecast_series_for_plot, pi_display_df

st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="file_uploader_v5", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None

if st.session_state.get('df_loaded') is not None:
    df_input = st.session_state.df_loaded.copy()
    date_col_options = df_input.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    
    sel_date_idx = date_col_options.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options else dt_col_guess_idx
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_v5")

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_v5")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_v5", on_change=reset_model_related_state)
    desired_freq = freq_map[freq_label]

    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_v5", on_change=reset_model_related_state)
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_v5"):
        # Resetear df_processed y resultados de modelos si se vuelve a preprocesar
        st.session_state.df_processed = None
        st.session_state.model_results = [] 
        st.session_state.best_model_name_auto = None
        st.session_state.selected_model_for_manual_explore = None
        st.session_state.data_diagnosis_report = None
        st.session_state.acf_fig = None

        date_col = st.session_state.get('selected_date_col')
        value_col = st.session_state.get('selected_value_col')
        valid = True
        if not date_col or date_col not in df_input.columns: st.sidebar.error("Seleccione columna de fecha."); valid = False
        if not value_col or value_col not in df_input.columns: st.sidebar.error("Seleccione columna de valor."); valid = False
        elif valid and not pd.api.types.is_numeric_dtype(df_input[value_col].dropna()):
             st.sidebar.error(f"Columna '{value_col}' no es numérica."); valid = False
        
        if valid:
            with st.spinner("Preprocesando..."):
                proc_df, msg = data_handler.preprocess_data(df_input.copy(), date_col, value_col, desired_freq, imp_code)
            if proc_df is not None:
                st.session_state.df_processed = proc_df; st.session_state.original_target_column_name = value_col
                st.success(f"Preprocesamiento OK. {msg}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(proc_df, value_col)
                if not proc_df.empty:
                    series_acf = proc_df[value_col]
                    lags_acf = min(len(series_acf)//2 -1, 60)
                    if lags_acf > 5: st.session_state.acf_fig = data_handler.plot_acf_pacf(series_acf, lags_acf, value_col)
                    else: st.session_state.acf_fig = None
                    _, auto_s = data_handler.get_series_frequency_and_period(proc_df.index)
                    st.session_state.auto_seasonal_period = auto_s
                    if st.session_state.user_seasonal_period == 1 or st.session_state.user_seasonal_period != auto_s:
                        st.session_state.user_seasonal_period = auto_s
            else: st.error(f"Fallo en preprocesamiento: {msg}"); st.session_state.df_processed = None

if st.session_state.get('df_processed') is not None and not st.session_state.get('df_processed').empty and st.session_state.get('original_target_column_name'):
    target_col = st.session_state.original_target_column_name
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag, col2_acf = st.columns(2)
    with col1_diag: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf: st.subheader("Autocorrelación"); st.pyplot(st.session_state.acf_fig) if st.session_state.acf_fig else st.info("ACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col in st.session_state.df_processed.columns:
        fig_hist = visualization.plot_historical_data(st.session_state.df_processed, target_col, f"Histórico de '{target_col}'")
        if fig_hist: st.pyplot(fig_hist)
    st.markdown("---")

    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v5")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v5", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=len(st.session_state.df_processed)//2 if st.session_state.df_processed is not None and not st.session_state.df_processed.empty else 2, step=1, key="ma_win_key_v5")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v5")
    if st.session_state.use_train_test_split:
        min_train = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test = len(st.session_state.df_processed) - min_train; max_test = max(1, max_test)
        def_test = min(max(1, st.session_state.forecast_horizon), max_test)
        if 'test_split_size' not in st.session_state or st.session_state.test_split_size > max_test or st.session_state.test_split_size <=0:
            st.session_state.test_split_size = def_test
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=st.session_state.test_split_size, min_value=1, max_value=max_test, step=1, key="test_size_key_v5", help=f"Máx: {max_test}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v5")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1a,c2a=st.columns(2); st.session_state.arima_max_p=c1a.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k"); st.session_state.arima_max_q=c2a.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k") #... (resto de params arima)
    with st.sidebar.expander("Parámetros Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v5") #... (resto de params HW)

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v5"):
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None; st.session_state.selected_model_for_manual_explore = None
        df_pr = st.session_state.get('df_processed'); target_c = st.session_state.get('original_target_column_name')
        if df_pr is None or target_c is None or target_c not in df_pr.columns: st.error("🔴 Datos no preprocesados. Aplique preprocesamiento."); 
        else:
            series_full = df_pr[target_c].copy(); h = st.session_state.forecast_horizon; s_period = st.session_state.user_seasonal_period; ma_win = st.session_state.moving_avg_window
            train_s, test_s = series_full, pd.Series(dtype=series_full.dtype)
            if st.session_state.use_train_test_split:
                min_tr_s = max(5, 2 * s_period + 1 if s_period > 1 else 5)
                curr_test_s = st.session_state.get('test_split_size', 12)
                if curr_test_s < (len(series_full) - min_tr_s) and curr_test_s > 0 : train_s, test_s = forecasting_models.train_test_split_series(series_full, curr_test_s)
                else: st.warning(f"No posible split con test_size={curr_test_s}. Usando toda la serie."); st.session_state.use_train_test_split = False
            st.session_state.train_series_for_plot = train_s; st.session_state.test_series_for_plot = test_s
            
            with st.spinner("Calculando modelos..."):
                models_to_run_specs = [
                    ("Promedio Histórico", forecasting_models.historical_average_forecast, [train_s, test_s, h]),
                    ("Ingénuo (Último Valor)", forecasting_models.naive_forecast, [train_s, test_s, h]),
                    (f"Promedio Móvil (Ventana {ma_win})", forecasting_models.moving_average_forecast, [train_s, test_s, h, ma_win]),
                ]
                if s_period > 1: models_to_run_specs.append((f"Estacional Ingénuo (P:{s_period})", forecasting_models.seasonal_naive_forecast, [train_s, test_s, h, s_period]))
                
                holt_p = {'damped_trend': st.session_state.holt_damped}
                hw_p = {'trend': st.session_state.hw_trend, 'seasonal': st.session_state.hw_seasonal, 'damped_trend': st.session_state.hw_damped, 'use_boxcox': st.session_state.hw_boxcox}
                stats_specs = [("SES", {}), ("Holt", holt_p)]
                if s_period > 1: stats_specs.append(("Holt-Winters", hw_p))
                for name_short, params_dict in stats_specs:
                    models_to_run_specs.append((None, forecasting_models.forecast_with_statsmodels, [train_s, test_s, h, name_short, s_period if name_short == "Holt-Winters" else None, params_dict if name_short == "Holt" else None, params_dict if name_short == "Holt-Winters" else None]))
                
                if st.session_state.run_autoarima:
                    arima_p = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                    models_to_run_specs.append((None, forecasting_models.forecast_with_auto_arima, [train_s, test_s, h, s_period, arima_p]))

                for model_disp_name_override, func, args_list in models_to_run_specs:
                    try:
                        fc, ci, rmse, mae, name_from_func = func(*args_list)
                        final_name = model_disp_name_override or name_from_func
                        # Lógica para obtener forecast_on_test (esto es un placeholder, necesitas la lógica real)
                        fc_on_test = None 
                        if not test_s.empty and not ("Error" in final_name or "Insuf" in final_name):
                            # Placeholder: necesitas una forma de obtener fc_on_test para CADA modelo
                            # Por ejemplo, re-ajustando solo en train_s y prediciendo len(test_s)
                            # Esto se complica si la función original no lo devuelve.
                            # Para los baselines, es más fácil. Para statsmodels/ARIMA, puede requerir re-fit.
                            if "Promedio Histórico" in final_name and not train_s.empty: fc_on_test = np.full(len(test_s), train_s.mean())
                            elif "Ingénuo" in final_name and not train_s.empty : fc_on_test = np.full(len(test_s), train_s.iloc[-1])
                            # Para otros, necesitarías una lógica más compleja o que la función de modelo lo devuelva.

                        st.session_state.model_results.append({'name': final_name, 'rmse': rmse, 'mae': mae, 'forecast_future': fc, 'conf_int_future': ci, 'forecast_on_test': fc_on_test})
                    except Exception as e: st.warning(f"Error ejecutando {func.__name__}: {e}")
            
            valid_res = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r['forecast_future'])==h]
            if valid_res: st.session_state.best_model_name_auto = min(valid_res, key=lambda x: x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido."); st.session_state.best_model_name_auto = None


df_proc_check = st.session_state.get('df_processed')
if df_proc_check is not None and not df_proc_check.empty and \
   st.session_state.get('original_target_column_name') and \
   st.session_state.get('model_results'):
    target_col_tabs = st.session_state.original_target_column_name
    st.header("Resultados del Modelado y Pronóstico")
    tab_rec, tab_comp, tab_manual, tab_diag = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    with tab_rec: # ... (Lógica de la pestaña Modelo Recomendado)
        pass
    with tab_comp: # ... (Lógica de la pestaña Comparación General)
        pass
    with tab_manual: # ... (Lógica de la pestaña Explorar Manualmente)
        pass
    with tab_diag: # ... (Lógica de la pestaña Diagnóstico y Guía)
        pass

elif uploaded_file is None: st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') and (st.session_state.get('df_processed') is None or st.session_state.get('df_processed').empty):
    st.warning("⚠️ Aplique preprocesamiento a los datos cargados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.6")