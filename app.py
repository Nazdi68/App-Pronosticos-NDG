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
        'moving_avg_window': 3, # Default para la ventana del promedio móvil
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
    init_session_state() # Re-inicializa para asegurar defaults en las claves borradas

def reset_model_related_state():
    st.session_state.df_processed = None; st.session_state.model_results = []
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None
    st.session_state.train_series_for_plot = None; st.session_state.test_series_for_plot = None

def prepare_forecast_display_data(model_data, series_full_idx, horizon):
    # ... (código sin cambios de la versión anterior) ...
    if model_data is None or model_data.get('forecast_future') is None: return None, None, None
    last_date_hist = series_full_idx.max(); freq = pd.infer_freq(series_full_idx)
    if freq is None and len(series_full_idx) > 1:
        diffs = series_full_idx.to_series().diff().dropna()
        if not diffs.empty: freq = diffs.min()
    if freq is None: freq = 'D'; st.warning(f"Frecuencia no inferida, usando '{freq}'.")
    forecast_dates = pd.date_range(start=last_date_hist, periods=horizon + 1, freq=freq)[1:]
    forecast_values = model_data['forecast_future']; conf_int_df_raw = model_data.get('conf_int_future')
    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}; pi_display_df = None
    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        pi_indexed = conf_int_df_raw.copy(); pi_indexed.index = forecast_dates
        export_dict['Limite Inferior PI'] = pi_indexed['lower'].values
        export_dict['Limite Superior PI'] = pi_indexed['upper'].values
        pi_display_df = pi_indexed[['lower', 'upper']]
    final_export_df = pd.DataFrame(export_dict).set_index('Fecha')
    forecast_series_for_plot = final_export_df['Pronostico']
    return final_export_df, forecast_series_for_plot, pi_display_df

st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v4", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None

if st.session_state.get('df_loaded') is not None:
    df_input = st.session_state.df_loaded.copy()
    date_col_options = df_input.columns.tolist()
    # ... (lógica de guess para date y value col con manejo de listas vacías) ...
    dt_col_guess_idx = 0; val_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=dt_col_guess_idx, key="date_sel_key_v4")
    
    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=val_col_guess_idx, key="val_sel_key_v4")
    
    freq_options_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    selected_freq_label_val = st.sidebar.selectbox("Frecuencia (Remuestreo):", options=list(freq_options_map.keys()), key="freq_sel_key_v4", on_change=reset_model_related_state)
    desired_freq_code_to_use = freq_options_map[selected_freq_label_val]

    imputation_methods_list_val = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    selected_imputation_label_val = st.sidebar.selectbox("Imputación de Faltantes:", imputation_methods_list_val, index=1, key="imp_sel_key_v4", on_change=reset_model_related_state)
    imputation_method_code_to_use = None if selected_imputation_label_val == "No imputar" else selected_imputation_label_val.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preprocess_btn_key_v4"):
        reset_model_related_state()
        date_col_use = st.session_state.get('selected_date_col')
        value_col_use = st.session_state.get('selected_value_col')
        valid_preproc = True
        if not date_col_use or date_col_use not in df_input.columns: st.sidebar.error("Seleccione columna de fecha válida."); valid_preproc = False
        if not value_col_use or value_col_use not in df_input.columns: st.sidebar.error("Seleccione columna de valor válida."); valid_preproc = False
        elif valid_preproc and not pd.api.types.is_numeric_dtype(df_input[value_col_use].dropna()):
             st.sidebar.error(f"Columna '{value_col_use}' no es numérica."); valid_preproc = False
        
        if valid_preproc:
            with st.spinner("Preprocesando..."):
                proc_df, msg_proc = data_handler.preprocess_data(df_input.copy(), date_col_use, value_col_use, desired_freq_code_to_use, imputation_method_code_to_use)
            if proc_df is not None:
                st.session_state.df_processed = proc_df; st.session_state.original_target_column_name = value_col_use
                st.success(f"Preprocesamiento OK. {msg_proc}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(st.session_state.df_processed, value_col_use)
                if not st.session_state.df_processed.empty:
                    series_acf_plot = st.session_state.df_processed[value_col_use]
                    lags_acf_plot = min(len(series_acf_plot) // 2 -1, 60)
                    if lags_acf_plot > 5: st.session_state.acf_fig = data_handler.plot_acf_pacf(series_acf_plot, lags_acf_plot, value_col_use)
                    else: st.session_state.acf_fig = None
                    _, auto_s_period_val = data_handler.get_series_frequency_and_period(st.session_state.df_processed.index)
                    st.session_state.auto_seasonal_period = auto_s_period_val
                    if st.session_state.user_seasonal_period == 1 or st.session_state.user_seasonal_period != auto_s_period_val:
                        st.session_state.user_seasonal_period = auto_s_period_val
            else: st.error(f"Fallo en preprocesamiento: {msg_proc}"); st.session_state.df_processed = None

if st.session_state.get('df_processed') is not None and st.session_state.get('original_target_column_name'):
    target_col_main_display = st.session_state.original_target_column_name
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    # ... (Mostrar diagnóstico, ACF, serie histórica como antes) ...

    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v4")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v4", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Promedio Móvil:", value=st.session_state.moving_avg_window, min_value=2, step=1, key="ma_win_key_v4", help="Para el modelo de Promedio Móvil.")
    
    st.sidebar.subheader("Evaluación del Modelo")
    # ... (Checkbox Train/Test y NumberInput para test_split_size con keys únicas) ...
    
    st.sidebar.subheader("Configuración de Modelos Específicos")
    # ... (Checkbox AutoARIMA y Expanders para parámetros de modelos con keys únicas) ...

    if st.sidebar.button("📊 Generar y Evaluar Todos los Modelos", key="gen_models_key_v4"):
        reset_model_related_state() # Resetear antes de correr modelos de nuevo
        df_proc_run_btn = st.session_state.get('df_processed')
        target_col_run_btn = st.session_state.get('original_target_column_name')

        if df_proc_run_btn is None or target_col_run_btn is None or target_col_run_btn not in df_proc_run_btn.columns:
            st.error("🔴 Datos no preprocesados correctamente. Aplique preprocesamiento.")
        else:
            series_full_run_btn = df_proc_run_btn[target_col_run_btn].copy()
            h_run_btn = st.session_state.forecast_horizon
            s_period_run_btn = st.session_state.user_seasonal_period
            ma_window_run_btn = st.session_state.moving_avg_window
            
            # ... (Lógica de Train/Test split como antes) ...
            train_series_run_btn, test_series_run_btn = series_full_run_btn, pd.Series(dtype=series_full_run_btn.dtype) # Default
            # ... (tu lógica completa de split aquí) ...
            st.session_state.train_series_for_plot = train_series_run_btn
            st.session_state.test_series_for_plot = test_series_run_btn
            
            with st.spinner("Calculando modelos..."):
                # Baselines
                # ... (Promedio Histórico, Naive, Seasonal Naive) ...
                # Promedio Móvil
                fc_ma, _, rmse_ma, mae_ma, name_ma = forecasting_models.moving_average_forecast(train_series_run_btn, test_series_run_btn, h_run_btn, ma_window_run_btn)
                # Lógica para obtener fc_on_test para MA (simplificada)
                fc_on_test_ma_val = None
                if not train_series_run_btn.empty and not test_series_run_btn.empty and len(train_series_run_btn) >= ma_window_run_btn :
                    fc_on_test_ma_val = np.full(len(test_series_run_btn), train_series_run_btn.iloc[-ma_window_run_btn:].mean())

                st.session_state.model_results.append({'name': name_ma, 'rmse': rmse_ma, 'mae': mae_ma, 'forecast_future': fc_ma, 'conf_int_future': None, 'forecast_on_test': fc_on_test_ma_val})

                # Statsmodels (SES, Holt, HW)
                # ... (Tu bucle para modelos statsmodels) ...
                # AutoARIMA
                # ... (Tu lógica para AutoARIMA si está habilitado) ...

            # Determinar mejor modelo
            # ... (Tu lógica para determinar st.session_state.best_model_name_auto) ...


# --- Sección de Resultados y Pestañas ---
# Corrección de la condición if:
df_proc_check_tabs = st.session_state.get('df_processed')
if df_proc_check_tabs is not None and not df_proc_check_tabs.empty and \
   st.session_state.get('original_target_column_name') and \
   st.session_state.get('model_results'): # Verificar que model_results no esté vacío también

    target_col_for_tabs_final = st.session_state.original_target_column_name
    st.header("Resultados del Modelado y Pronóstico")
    # ... (Resto de la lógica de las pestañas como la tenías, usando prepare_forecast_display_data,
    #      visualization, y recommendations. Asegúrate de que los títulos de los modelos se pasen
    #      correctamente y que los botones de descarga tengan keys únicas si se repiten en pestañas.) ...

elif uploaded_file is None:
    st.info("👋 ¡Bienvenido! Por favor, cargue un archivo de datos para comenzar.")
elif st.session_state.get('df_loaded') is not None and (st.session_state.get('df_processed') is None or st.session_state.get('df_processed').empty):
    st.warning("⚠️ Por favor, aplique el preprocesamiento a los datos cargados y asegúrese de que sea exitoso.")

st.sidebar.markdown("---")
st.sidebar.info("Asistente de Pronósticos PRO v3.5")