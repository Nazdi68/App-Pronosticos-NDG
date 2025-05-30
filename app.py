# app.py (v4.1 - Flujo Simplificado Completo con Parámetros de Modelo)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models # Debe ser v3.1 (con devolución de model_params)
import recommendations   # Debe ser v2.1 (con parámetro model_params)

st.set_page_config(page_title="Asistente de Pronósticos Simplificado", layout="wide")

# --- Estado de la Sesión ---
def init_session_state_v4_1(): # Renombrado para evitar conflicto si se corre localmente después de otra versión
    defaults = {
        'df_loaded': None, 'current_file_name': None, 'df_processed': None, 
        'selected_date_col': None, 'selected_value_col': None, 
        'original_target_column_name': "Valor", 
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'sugerencia_modelo': None, 'sugerencia_explicacion': None,
        'modelo_ejecutado_info': None, # Guardará {'name', 'rmse', 'mae', 'forecast_future', 'conf_int_future', 'model_params'}
        'forecast_df_final': None 
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state_v4_1()

def reset_on_file_change_v4_1(): # Renombrado
    keys_to_reset = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name', 'data_diagnosis_report', 'acf_fig', 
        'sugerencia_modelo', 'sugerencia_explicacion',
        'modelo_ejecutado_info', 'forecast_df_final',
        'auto_seasonal_period', 'user_seasonal_period'
    ]
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state: del st.session_state[key_to_del]
    st.session_state.df_loaded = None
    init_session_state_v4_1() 

def reset_after_preproc_params_change_v4_1(): # Renombrado
    st.session_state.df_processed = None 
    st.session_state.sugerencia_modelo = None; st.session_state.sugerencia_explicacion = None
    st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None

def sugerir_tipo_modelo_simple(serie_procesada, auto_seasonal_period=1):
    if serie_procesada is None or serie_procesada.empty:
        return "N/A", "No hay datos suficientes para analizar y sugerir un modelo.", []
    sugerencias_log = []; modelo_sugerido_tipo = "SES" 
    explicacion = "Para series cortas o sin patrones claros, SES es un buen punto de partida."
    if len(serie_procesada) < 15:
        explicacion = "Datos muy limitados. Se sugiere un modelo baseline simple: SES."
        sugerencias_log.append("Datos muy limitados.")
        return "SES", explicacion, sugerencias_log
    tiene_tendencia_visible = False; tiene_estacionalidad_visible = False
    if len(serie_procesada) >= 20: 
        primera_mitad_mean = serie_procesada.iloc[:len(serie_procesada)//2].mean()
        segunda_mitad_mean = serie_procesada.iloc[len(serie_procesada)//2:].mean()
        denominador_tendencia = abs(primera_mitad_mean) if abs(primera_mitad_mean) > 1e-9 else 1.0
        diff_relativa_tendencia = abs(segunda_mitad_mean - primera_mitad_mean) / denominador_tendencia
        if diff_relativa_tendencia > 0.15: sugerencias_log.append("Posible tendencia detectada."); tiene_tendencia_visible = True
    if auto_seasonal_period > 1 and len(serie_procesada) >= 2 * auto_seasonal_period:
        sugerencias_log.append(f"Posible estacionalidad con período {auto_seasonal_period}."); tiene_estacionalidad_visible = True
    if tiene_tendencia_visible and tiene_estacionalidad_visible: modelo_sugerido_tipo = "Holt-Winters"; explicacion = "Detectada posible tendencia y estacionalidad. Sugerido: Holt-Winters. Verifique 'Período Estacional'."
    elif tiene_tendencia_visible: modelo_sugerido_tipo = "Holt"; explicacion = "Detectada posible tendencia sin estacionalidad clara. Sugerido: Holt."
    elif tiene_estacionalidad_visible: modelo_sugerido_tipo = "Holt-Winters (sin tendencia)"; explicacion = "Detectada posible estacionalidad sin tendencia fuerte. Sugerido: Holt-Winters (sin tendencia) o AutoARIMA con estacionalidad."
    elif len(serie_procesada) > 25: modelo_sugerido_tipo = "AutoARIMA"; explicacion = "No se detectaron patrones fuertes. AutoARIMA intentará encontrar un modelo ARIMA/SARIMA."
    sugerencias_log.append(f"Sugerencia final: {modelo_sugerido_tipo}. Razón: {explicacion}")
    return modelo_sugerido_tipo, explicacion, sugerencias_log

def prepare_forecast_display_data(model_data, series_full_idx, horizon):
    if model_data is None or model_data.get('forecast_future') is None: return None, None, None
    if series_full_idx is None or series_full_idx.empty : return None, None, None
    last_date_hist = series_full_idx.max(); freq = pd.infer_freq(series_full_idx)
    if freq is None and len(series_full_idx) > 1:
        diffs = series_full_idx.to_series().diff().dropna()
        if not diffs.empty: freq = diffs.min()
    if freq is None: freq = 'D'; st.warning(f"Frecuencia no inferida, usando '{freq}'.")
    actual_horizon = max(1, horizon)
    try: forecast_dates = pd.date_range(start=last_date_hist, periods=actual_horizon + 1, freq=freq)[1:]
    except ValueError as e_date_range: st.warning(f"Error al generar fechas: {e_date_range}."); return None, None, None
    forecast_values_raw = model_data['forecast_future']
    if forecast_values_raw is None : return None, None, None
    forecast_values = np.array(forecast_values_raw)
    min_len = len(forecast_dates)
    if len(forecast_values) != len(forecast_dates):
        min_len = min(len(forecast_values), len(forecast_dates))
        forecast_values = forecast_values[:min_len]
        forecast_dates = forecast_dates[:min_len]
    if min_len == 0: return pd.DataFrame(columns=['Fecha','Pronostico']).set_index('Fecha'), pd.Series(dtype='float64'), None
    conf_int_df_raw = model_data.get('conf_int_future')
    if len(forecast_values) != len(forecast_dates): return None, None, None 
    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}; pi_display_df = None
    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        pi_indexed = conf_int_df_raw.copy()
        if len(pi_indexed) == len(forecast_dates):
            pi_indexed.index = forecast_dates; export_dict['Limite Inferior PI'] = pi_indexed['lower'].values; export_dict['Limite Superior PI'] = pi_indexed['upper'].values; pi_display_df = pi_indexed[['lower', 'upper']]
    final_export_df = pd.DataFrame(export_dict)
    if not final_export_df.empty: final_export_df = final_export_df.set_index('Fecha'); forecast_series_for_plot = final_export_df['Pronostico']
    else: forecast_series_for_plot = pd.Series(dtype='float64'); final_export_df = pd.DataFrame(columns=['Fecha','Pronostico']).set_index('Fecha')
    return final_export_df, forecast_series_for_plot, pi_display_df

# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos 1")
st.markdown("Herramienta guiada para generar pronósticos de series de tiempo.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo (CSV o Excel)", type=["csv", "xlsx", "xls"], key="uploader_key_v4_1", on_change=reset_on_file_change_v4_1)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb_v41 = st.session_state.get('df_loaded')

if df_input_sb_v41 is not None:
    date_col_options_sb_v41 = df_input_sb_v41.columns.tolist()
    dt_col_guess_idx_v41 = 0
    if date_col_options_sb_v41:
        for i, col in enumerate(date_col_options_sb_v41):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_v41 = i; break
    sel_date_idx_v41 = 0
    if date_col_options_sb_v41 : 
        sel_date_idx_v41 = date_col_options_sb_v41.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb_v41 else dt_col_guess_idx_v41
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb_v41, index=sel_date_idx_v41, key="date_sel_key_v4_1")

    value_col_options_sb_v41 = [col for col in df_input_sb_v41.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_v41 = 0
    if value_col_options_sb_v41:
        for i, col in enumerate(value_col_options_sb_v41):
            if pd.api.types.is_numeric_dtype(df_input_sb_v41[col].dropna()): val_col_guess_idx_v41 = i; break
    sel_val_idx_v41 = 0
    if value_col_options_sb_v41:
        sel_val_idx_v41 = value_col_options_sb_v41.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb_v41 else val_col_guess_idx_v41
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb_v41, index=sel_val_idx_v41, key="val_sel_key_v4_1")
    
    freq_map_sb_v41 = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb_v41 = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb_v41.keys()), key="freq_sel_key_v4_1", on_change=reset_after_preproc_params_change_v4_1) # Corregido nombre de función
    desired_freq_sb_v41 = freq_map_sb_v41[freq_label_sb_v41]
    imp_list_sb_v41 = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_v41 = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb_v41, index=1, key="imp_sel_key_v4_1", on_change=reset_after_preproc_params_change_v4_1) # Corregido nombre de función
    imp_code_sb_v41 = None if imp_label_sb_v41 == "No imputar" else imp_label_sb_v41.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v4_1"):
        st.session_state.df_processed = None; reset_after_preproc_params_change_v4_1() # Corregido nombre de función
        date_col_btn_v41 = st.session_state.get('selected_date_col'); value_col_btn_v41 = st.session_state.get('selected_value_col'); valid_btn_v41 = True
        if not date_col_btn_v41 or date_col_btn_v41 not in df_input_sb_v41.columns: st.sidebar.error("Seleccione fecha."); valid_btn_v41=False
        if not value_col_btn_v41 or value_col_btn_v41 not in df_input_sb_v41.columns: st.sidebar.error("Seleccione valor."); valid_btn_v41=False
        elif valid_btn_v41 and not pd.api.types.is_numeric_dtype(df_input_sb_v41[value_col_btn_v41].dropna()): st.sidebar.error(f"'{value_col_btn_v41}' no numérica."); valid_btn_v41=False
        if valid_btn_v41:
            with st.spinner("Preprocesando..."): proc_df_v41,msg_raw_v41 = data_handler.preprocess_data(df_input_sb_v41.copy(),date_col_btn_v41,value_col_btn_v41,desired_freq_sb_v41,imp_code_sb_v41)
            msg_disp_v41 = msg_raw_v41; 
            if msg_raw_v41: 
                if "MS" in msg_raw_v41: msg_disp_v41=msg_raw_v41.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_v41: msg_disp_v41=msg_raw_v41.replace(" D."," D (Diario).")
                elif msg_raw_v41.strip().endswith("D"): msg_disp_v41=msg_raw_v41.replace("D", "D (Diario)")
            if proc_df_v41 is not None and not proc_df_v41.empty:
                st.session_state.df_processed=proc_df_v41; st.session_state.original_target_column_name=value_col_btn_v41; st.success(f"Preproc. OK. {msg_disp_v41}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_v41,value_col_btn_v41)
                if not proc_df_v41.empty:
                    s_acf_v41=proc_df_v41[value_col_btn_v41];l_acf_v41=min(len(s_acf_v41)//2-1,60)
                    if l_acf_v41 > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_v41,l_acf_v41,value_col_btn_v41)
                    else: st.session_state.acf_fig=None
                    _,auto_s_v41_val=data_handler.get_series_frequency_and_period(proc_df_v41.index)
                    st.session_state.auto_seasonal_period=auto_s_v41_val
                    st.session_state.user_seasonal_period = auto_s_v41_val 
                    st.session_state.sugerencia_modelo, st.session_state.sugerencia_explicacion, _ = sugerir_tipo_modelo_simple(s_acf_v41, auto_seasonal_period=auto_s_v41_val)
            else: st.error(f"Fallo preproc: {msg_raw_v41 or 'DataFrame vacío.'}"); st.session_state.df_processed=None

df_processed_main_v41_disp = st.session_state.get('df_processed')
target_col_main_v41_disp = st.session_state.get('original_target_column_name')

if df_processed_main_v41_disp is not None and not df_processed_main_v41_disp.empty and target_col_main_v41_disp:
    st.header("1. Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_v41, col2_acf_v41 = st.columns(2)
    with col1_diag_v41: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v41: 
        st.subheader("Autocorrelación (ACF/PACF)")
        acf_fig_v41_plot = st.session_state.get('acf_fig')
        if acf_fig_v41_plot is not None: 
            try: st.pyplot(acf_fig_v41_plot)
            except Exception as e_acf_v41_plot: st.error(f"Error al mostrar ACF/PACF: {e_acf_v41_plot}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie de Tiempo Preprocesada")
    if target_col_main_v41_disp in df_processed_main_v41_disp.columns:
        fig_hist_v41_plot = visualization.plot_historical_data(df_processed_main_v41_disp, target_col_main_v41_disp, f"Histórico de '{target_col_main_v41_disp}'")
        if fig_hist_v41_plot: st.pyplot(fig_hist_v41_plot)
    st.markdown("---")

    if st.session_state.get('sugerencia_modelo') and st.session_state.get('sugerencia_explicacion'):
        st.subheader("🧠 Sugerencia de Modelo Inicial")
        st.info(f"**Tipo de Modelo Sugerido:** {st.session_state.sugerencia_modelo}\n\n**Razón:** {st.session_state.sugerencia_explicacion}")
        st.markdown("Esta es una sugerencia basada en un análisis simple. La aplicación intentará ajustar este tipo de modelo o AutoARIMA si es apropiado.")
    st.markdown("---")

    st.sidebar.header("2. Configuración del Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v4_1_simple")
    if st.session_state.get('sugerencia_modelo') and ("Holt-Winters" in st.session_state.sugerencia_modelo or "AutoARIMA" in st.session_state.sugerencia_modelo):
        st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional (si aplica):", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v4_1_simple", help=f"Detectado: {st.session_state.auto_seasonal_period}")
    
    if st.sidebar.button("🚀 Generar Pronóstico", key="gen_forecast_btn_v4_1_action"):
        st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
        
        # Leer del estado de sesión DENTRO del botón
        df_proc_fc_v41 = st.session_state.get('df_processed')
        target_col_fc_v41 = st.session_state.get('original_target_column_name')
        modelo_a_usar_v41 = st.session_state.get('sugerencia_modelo', "AutoARIMA")
        h_pron_v41 = st.session_state.forecast_horizon
        s_period_pron_v41 = st.session_state.get('user_seasonal_period', 1)
        
        if df_proc_fc_v41 is None or target_col_fc_v41 is None or target_col_fc_v41 not in df_proc_fc_v41.columns:
            st.error("🔴 Error interno: Datos preprocesados no disponibles para generar pronóstico.")
        else:
            serie_pron_v41 = df_proc_fc_v41[target_col_fc_v41].copy()
            fc_vals_res_v41, ci_df_res_v41, fitted_vals_res_v41, nombre_modelo_res_v41, params_modelo_res_v41 = None, None, None, f"{modelo_a_usar_v41} (No Ejecutado)", {}

            with st.spinner(f"Ajustando '{modelo_a_usar_v41}' y generando pronóstico..."):
                try:
                    if modelo_a_usar_v41 == "SES": fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.run_ses_simple(serie_pron_v41, h_pron_v41)
                    elif modelo_a_usar_v41 == "Holt": fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.run_holt_simple(serie_pron_v41, h_pron_v41, damped=False)
                    elif "Holt-Winters" in modelo_a_usar_v41:
                        trend_hw_run_v41 = 'add' if "(sin tendencia)" not in modelo_a_usar_v41 else None
                        fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.run_hw_simple(serie_pron_v41,h_pron_v41,s_period_pron_v41,trend=trend_hw_run_v41,seasonal='add')
                    elif modelo_a_usar_v41 == "AutoARIMA":
                        arima_params_run_simple_v41 = {'max_p':3,'max_q':3,'max_d':2,'max_P':1,'max_Q':1,'max_D':1}
                        fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.run_autoarima_simple(serie_pron_v41,h_pron_v41,s_period_pron_v41,arima_params=arima_params_run_simple_v41)
                    elif modelo_a_usar_v41 == "Promedio Histórico": fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.historical_average_simple(serie_pron_v41,h_pron_v41)
                    elif modelo_a_usar_v41 == "Ingénuo": fc_vals_res_v41,ci_df_res_v41,fitted_vals_res_v41,nombre_modelo_res_v41,params_modelo_res_v41 = forecasting_models.naive_simple(serie_pron_v41,h_pron_v41)
                    else: st.error(f"Modelo '{modelo_a_usar_v41}' no implementado."); nombre_modelo_res_v41=f"{modelo_a_usar_v41} (No Imp.)"

                    if fc_vals_res_v41 is not None and fitted_vals_res_v41 is not None:
                        rmse_calc_v41, mae_calc_v41 = np.nan, np.nan
                        if len(fitted_vals_res_v41) == len(serie_pron_v41): rmse_calc_v41,mae_calc_v41 = forecasting_models.calculate_metrics(serie_pron_v41,fitted_vals_res_v41)
                        else:
                            offset_v41 = len(serie_pron_v41) - len(fitted_vals_res_v41)
                            if offset_v41 >= 0 and len(fitted_vals_res_v41)>0: rmse_calc_v41,mae_calc_v41 = forecasting_models.calculate_metrics(serie_pron_v41[offset_v41:], fitted_vals_res_v41)
                        st.session_state.modelo_ejecutado_info = {"name":nombre_modelo_res_v41,"rmse":rmse_calc_v41,"mae":mae_calc_v41,"forecast_future":fc_vals_res_v41,"conf_int_future":ci_df_res_v41, "model_params": params_modelo_res_v41}
                        idx_fc_final_v41 = serie_pron_v41.index
                        df_fc_final_res_v41,_,_ = prepare_forecast_display_data(st.session_state.modelo_ejecutado_info,idx_fc_final_v41,h_pron_v41)
                        st.session_state.forecast_df_final = df_fc_final_res_v41
                        st.success(f"Pronóstico generado con: {nombre_modelo_res_v41}")
                    else: st.error(f"Modelo '{nombre_modelo_res_v41 or modelo_a_usar_v41}' no generó pronóstico."); st.session_state.modelo_ejecutado_info = {"name": (nombre_modelo_res_v41 or modelo_a_usar_v41) + " (FALLÓ)", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None, "model_params": {}}
                except Exception as e_model_run_v41: st.error(f"Error ejecutando '{modelo_a_usar_v41}': {e_model_run_v41}"); st.session_state.modelo_ejecutado_info = {"name": modelo_a_usar_v41 + f" (FALLÓ: {type(e_model_run_v41).__name__})", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None, "model_params": {}}

# --- Mostrar Resultados del Pronóstico ---
if st.session_state.get('modelo_ejecutado_info') and st.session_state.get('forecast_df_final') is not None:
    st.header("2. Resultados del Pronóstico")
    info_modelo_final_v41 = st.session_state.modelo_ejecutado_info
    df_pronostico_final_v41 = st.session_state.forecast_df_final
    # Asegurar que target_col_main_v41_disp esté definido y sea válido
    target_col_for_results_v41 = st.session_state.get('original_target_column_name')
    df_proc_for_results_v41 = st.session_state.get('df_processed')

    if df_proc_for_results_v41 is not None and target_col_for_results_v41 in df_proc_for_results_v41.columns:
        serie_historica_final_v41 = df_proc_for_results_v41[target_col_for_results_v41]

        st.subheader(f"Modelo Utilizado: {info_modelo_final_v41['name']}")
        if info_modelo_final_v41.get('model_params') and isinstance(info_modelo_final_v41['model_params'], dict) and info_modelo_final_v41['model_params']: # Check if dict and not empty
            with st.expander("Parámetros Estimados del Modelo"):
                st.json(info_modelo_final_v41['model_params'])
        if pd.notna(info_modelo_final_v41.get('rmse')) and pd.notna(info_modelo_final_v41.get('mae')):
            st.markdown(f"**Métricas de Ajuste (In-Sample):** RMSE = {info_modelo_final_v41['rmse']:.2f}, MAE = {info_modelo_final_v41['mae']:.2f}")
        
        pi_df_plot_final_v41 = None
        if 'Limite Inferior PI' in df_pronostico_final_v41.columns and 'Limite Superior PI' in df_pronostico_final_v41.columns:
            pi_df_plot_final_v41 = df_pronostico_final_v41[['Limite Inferior PI', 'Limite Superior PI']]

        fig_pronostico_final_v41 = visualization.plot_final_forecast(serie_historica_final_v41, df_pronostico_final_v41['Pronostico'],pi_df_plot_final_v41, model_name=info_modelo_final_v41['name'],value_col_name=target_col_for_results_v41)
        if fig_pronostico_final_v41: st.pyplot(fig_pronostico_final_v41)
        else: st.warning("No se pudo generar el gráfico del pronóstico.")

        st.markdown("##### Valores del Pronóstico"); st.dataframe(df_pronostico_final_v41.style.format("{:.2f}"))
# — Generación de Excel con gráfico y recomendaciones —
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    # 1) Escribimos el pronóstico
    df_pronostico_final_v41.to_excel(writer, sheet_name='Pronóstico', index=True, startrow=0)
    wb = writer.book
    ws = writer.sheets['Pronóstico']

    # 2) Insertamos gráfico
    chart = wb.add_chart({'type':'line'})
    rows = len(df_pronostico_final_v41)
    chart.add_series({
        'name': info_modelo_final_v41['name'],
        'categories': ['Pronóstico', 1, 0, rows, 0],
        'values':     ['Pronóstico', 1, 1, rows, 1],
        'marker':     {'type':'circle','size':4},
    })
    chart.set_title({'name': f"Pronóstico ({info_modelo_final_v41['name']}) a {st.session_state.forecast_horizon} periodos"})
    chart.set_x_axis({'name':'Fecha'})
    chart.set_y_axis({'name': target_col_for_results_v41})
    ws.insert_chart('D2', chart, {'x_scale':1.2, 'y_scale':1.2})

    # 3) Información adicional
    start = rows + 3
    info_lines = [
        "Informe: Asistente de Pronósticos PRO v4.1",
        f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Modelo: {info_modelo_final_v41['name']}",
        f"Horizonte: {st.session_state.forecast_horizon}"
    ]
    for i, line in enumerate(info_lines):
        ws.write(start + i, 0, line)

    # 4) Recomendaciones
    reco_text = recommendations.generate_recommendations_simple(
        selected_model_name   = info_modelo_final_v41['name'],
        data_diag_summary     = st.session_state.data_diagnosis_report,
        has_pis               = (pi_df_plot_final_v41 is not None and not pi_df_plot_final_v41.empty),
        target_column_name    = target_col_for_results_v41,
        model_rmse            = info_modelo_final_v41.get('rmse'),
        model_mae             = info_modelo_final_v41.get('mae'),
        forecast_horizon      = st.session_state.forecast_horizon,
        model_params          = info_modelo_final_v41.get('model_params'),
    )
    reco_lines = reco_text.split("\n")
    rec_start = start + len(info_lines) + 2
    ws.write(rec_start, 0, "Recomendaciones:")
    for j, line in enumerate(reco_lines):
        ws.write(rec_start + 1 + j, 0, line)

# 5) Botón de descarga
output.seek(0)
st.download_button(
    "⬇️ Descargar Excel completo",
    data      = output.getvalue(),
    file_name = f"pronostico_{target_col_for_results_v41}.xlsx",
    mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
    st.subheader("💡 Recomendaciones y Próximos Pasos")
    st.markdown(
        recommendations.generate_recommendations_simple(
            selected_model_name=info_modelo_final_v41['name'],
            data_diag_summary=st.session_state.data_diagnosis_report,
            has_pis=(pi_df_plot_final_v41 is not None and not pi_df_plot_final_v41.empty),
            target_column_name=target_col_for_results_v41,
            model_rmse=info_modelo_final_v41.get('rmse'),
            model_mae=info_modelo_final_v41.get('mae'),
            forecast_horizon=st.session_state.forecast_horizon,
            model_params=info_modelo_final_v41.get('model_params')
        )
    )
    else:
        st.warning("No se pueden mostrar los resultados del pronóstico porque los datos históricos o la columna objetivo no están disponibles.")


elif uploaded_file is None and st.session_state.get('df_loaded') is None : 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') is not None and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
elif st.session_state.get('df_loaded') is not None and \
     st.session_state.get('df_processed') is not None and \
     not st.session_state.get('df_processed').empty and \
     st.session_state.get('modelo_ejecutado_info') is None:
     if st.session_state.get('sugerencia_modelo'):
         st.info(f"Datos preprocesados. El modelo sugerido es **{st.session_state.sugerencia_modelo}**. Configure el horizonte y haga clic en 'Generar Pronóstico'.")
     else:
         st.info("Datos preprocesados. Configure el horizonte y haga clic en 'Generar Pronóstico'.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v4.1") # Actualizada la versión
# prueba
