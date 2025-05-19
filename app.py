# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models 
import recommendations 

st.set_page_config(page_title="Asistente de Pronósticos PRO", layout="wide")

# --- Estado de la Sesión ---
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

# --- Funciones Auxiliares ---
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
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state: del st.session_state[key_to_del]
    st.session_state.df_loaded = None
    init_session_state() 

def reset_sidebar_config_dependent_state():
    st.session_state.df_processed = None 
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None
    st.session_state.train_series_for_plot = None; st.session_state.test_series_for_plot = None

def reset_model_execution_results():
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None

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
    else: forecast_series_for_plot = pd.Series(dtype='float64')
    return final_export_df, forecast_series_for_plot, pi_display_df

# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v17", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb_v17 = st.session_state.get('df_loaded')

if df_input_sb_v17 is not None:
    # ... (Lógica de selectores de columna, frecuencia, imputación como en v3.15, con keys _v17) ...
    # ... (Asegúrate de que esta sección esté completa y correcta) ...
    date_col_options_sb_v17 = df_input_sb_v17.columns.tolist()
    dt_col_guess_idx_sb_v17 = 0
    if date_col_options_sb_v17:
        for i, col in enumerate(date_col_options_sb_v17):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb_v17 = i; break
    sel_date_idx_sb_v17 = 0
    if date_col_options_sb_v17 : 
        sel_date_idx_sb_v17 = date_col_options_sb_v17.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb_v17 else dt_col_guess_idx_sb_v17
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb_v17, index=sel_date_idx_sb_v17, key="date_sel_key_v17")

    value_col_options_sb_v17 = [col for col in df_input_sb_v17.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb_v17 = 0
    if value_col_options_sb_v17:
        for i, col in enumerate(value_col_options_sb_v17):
            if pd.api.types.is_numeric_dtype(df_input_sb_v17[col].dropna()): val_col_guess_idx_sb_v17 = i; break
    sel_val_idx_sb_v17 = 0
    if value_col_options_sb_v17:
        sel_val_idx_sb_v17 = value_col_options_sb_v17.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb_v17 else val_col_guess_idx_sb_v17
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb_v17, index=sel_val_idx_sb_v17, key="val_sel_key_v17")
    
    freq_map_sb_v17 = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb_v17 = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb_v17.keys()), key="freq_sel_key_v17", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb_v17 = freq_map_sb_v17[freq_label_sb_v17]
    imp_list_sb_v17 = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_v17 = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb_v17, index=1, key="imp_sel_key_v17", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb_v17 = None if imp_label_sb_v17 == "No imputar" else imp_label_sb_v17.split('(')[0].strip()


    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v17"):
        st.session_state.df_processed = None; reset_sidebar_config_dependent_state() 
        date_col_btn_v17 = st.session_state.get('selected_date_col'); value_col_btn_v17 = st.session_state.get('selected_value_col'); valid_btn_v17 = True
        if not date_col_btn_v17 or date_col_btn_v17 not in df_input_sb_v17.columns: st.sidebar.error("Seleccione fecha."); valid_btn_v17=False
        if not value_col_btn_v17 or value_col_btn_v17 not in df_input_sb_v17.columns: st.sidebar.error("Seleccione valor."); valid_btn_v17=False
        elif valid_btn_v17 and not pd.api.types.is_numeric_dtype(df_input_sb_v17[value_col_btn_v17].dropna()): st.sidebar.error(f"'{value_col_btn_v17}' no numérica."); valid_btn_v17=False
        if valid_btn_v17:
            with st.spinner("Preprocesando..."): proc_df_v17,msg_raw_v17 = data_handler.preprocess_data(df_input_sb_v17.copy(),date_col_btn_v17,value_col_btn_v17,desired_freq_sb_v17,imp_code_sb_v17)
            msg_disp_v17 = msg_raw_v17; 
            if msg_raw_v17: 
                if "MS" in msg_raw_v17: msg_disp_v17=msg_raw_v17.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_v17: msg_disp_v17=msg_raw_v17.replace(" D."," D (Diario).")
                elif msg_raw_v17.endswith("D"): msg_disp_v17=msg_raw_v17.replace("D", "D (Diario)")
            if proc_df_v17 is not None and not proc_df_v17.empty:
                st.session_state.df_processed=proc_df_v17; st.session_state.original_target_column_name=value_col_btn_v17; st.success(f"Preproc. OK. {msg_disp_v17}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_v17,value_col_btn_v17)
                if not proc_df_v17.empty:
                    s_acf_v17=proc_df_v17[value_col_btn_v17];l_acf_v17=min(len(s_acf_v17)//2-1,60)
                    if l_acf_v17 > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_v17,l_acf_v17,value_col_btn_v17)
                    else: st.session_state.acf_fig=None
                    _,auto_s_v17_val=data_handler.get_series_frequency_and_period(proc_df_v17.index)
                    st.session_state.auto_seasonal_period=auto_s_v17_val
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s_v17_val: st.session_state.user_seasonal_period=auto_s_v17_val
            else: st.error(f"Fallo preproc: {msg_raw_v17 or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main_view_v17 = st.session_state.get('df_processed')
target_col_main_view_v17 = st.session_state.get('original_target_column_name')

if df_processed_main_view_v17 is not None and not df_processed_main_view_v17.empty and target_col_main_view_v17:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    # ... (código para mostrar diagnóstico, ACF, serie preprocesada como en v3.15) ...
    col1_diag_v17, col2_acf_v17 = st.columns(2)
    with col1_diag_v17: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v17: 
        st.subheader("Autocorrelación")
        acf_fig_v17 = st.session_state.get('acf_fig')
        if acf_fig_v17 is not None: 
            try: st.pyplot(acf_fig_v17)
            except Exception as e_acf_v17: st.error(f"Error al mostrar ACF/PACF: {e_acf_v17}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main_view_v17 in df_processed_main_view_v17.columns:
        fig_hist_v17 = visualization.plot_historical_data(df_processed_main_view_v17, target_col_main_view_v17, f"Histórico de '{target_col_main_view_v17}'")
        if fig_hist_v17: st.pyplot(fig_hist_v17)
    st.markdown("---")


    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    # ... (widgets de la sidebar para horizonte, periodo, ventana MA, train/test, AutoARIMA, Holt, HW como en v3.15, CON KEYS ÚNICAS _v17) ...
    # EJEMPLO (DEBES COMPLETAR TODOS):
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v17")
    # ... (resto de los widgets de configuración)


    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v17_action"):
        reset_model_execution_results()
        
        # CORRECCIÓN: Leer del estado de sesión DENTRO del botón
        df_processed_for_models = st.session_state.get('df_processed')
        target_col_for_models = st.session_state.get('original_target_column_name')

        if df_processed_for_models is None or target_col_for_models is None or \
           target_col_for_models not in df_processed_for_models.columns: 
            st.error("🔴 Datos no preprocesados correctamente. Por favor, aplique preprocesamiento primero."); 
        else:
            series_full_models_run = df_processed_for_models[target_col_for_models].copy(); # Usar variables correctas
            h_models_run = st.session_state.forecast_horizon; 
            s_period_models_run = st.session_state.user_seasonal_period; 
            ma_win_models_run = st.session_state.moving_avg_window
            
            train_s_models_run, test_s_models_run = series_full_models_run, pd.Series(dtype=series_full_models_run.dtype)
            if st.session_state.use_train_test_split:
                min_tr_models_run = max(5, 2*s_period_models_run+1 if s_period_models_run>1 else 5); 
                curr_test_models_run = st.session_state.get('test_split_size', 12)
                if len(series_full_models_run) > min_tr_models_run + curr_test_models_run and curr_test_models_run > 0 : 
                    train_s_models_run,test_s_models_run = forecasting_models.train_test_split_series(series_full_models_run, curr_test_models_run)
                else: 
                    st.warning(f"No fue posible el split con test_size={curr_test_models_run}. Evaluando in-sample."); 
                    st.session_state.use_train_test_split=False
                    train_s_models_run, test_s_models_run = series_full_models_run, pd.Series(dtype=series_full_models_run.dtype)
            
            st.session_state.train_series_for_plot = train_s_models_run; 
            st.session_state.test_series_for_plot = test_s_models_run
                
            with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
                model_execution_list_final_run = []
                # ... (Lógica para construir model_execution_list_final_run como en v3.15) ...
                # Ejemplo:
                model_execution_list_final_run.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_models_run, test_s_models_run, h_models_run], "name_override": "Promedio Histórico", "type":"baseline"})
                # ... (AÑADE TODOS TUS MODELOS AQUÍ)

                for spec_item_loop in model_execution_list_final_run:
                    try:
                        # CORRECCIÓN DE NOMBRES DE VARIABLES
                        fc_future_item, ci_future_item, rmse_item, mae_item, name_from_func_item = spec_item_loop["func"](*spec_item_loop["args"])
                        name_display_item = spec_item_loop["name_override"] or name_from_func_item
                        fc_on_test_item_result = None # Placeholder - Implementar lógica robusta
                        # ... (Lógica para calcular fc_on_test_item_result como en v3.15) ...
                        st.session_state.model_results.append({
                            'name':name_display_item,
                            'rmse':rmse_item, # Usar variable correcta
                            'mae':mae_item,    # Usar variable correcta
                            'forecast_future':fc_future_item, # Usar variable correcta
                            'conf_int_future':ci_future_item, # Usar variable correcta
                            'forecast_on_test':fc_on_test_item_result
                        })
                    except Exception as e_model_loop: 
                        st.warning(f"Error procesando {spec_item_loop.get('name_override',spec_item_loop['func'].__name__)}: {str(e_model_loop)[:150]}")
                
                # --- DEBUG: Mostrar contenido de model_results ---
                st.write("--- DEBUG: Contenido de st.session_state.model_results ---")
                if isinstance(st.session_state.model_results, list) and st.session_state.model_results:
                    for i_debug_final_v16, res_debug_final_v16 in enumerate(st.session_state.model_results):
                        st.write(f"Modelo {i_debug_final_v16+1}:")
                        st.json(res_debug_final_v16) 
                else:
                    st.write("st.session_state.model_results está vacío o no es una lista.")
                st.write(f"Horizonte (h_models_run) usado para filtrar: {h_models_run}") 
                st.write("--- FIN DEBUG ---")

            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_results_final_run_list_v16 = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and isinstance(r.get('forecast_future'), np.ndarray) and len(r.get('forecast_future'))==h_models_run]
            if valid_results_final_run_list_v16: st.session_state.best_model_name_auto = min(valid_results_final_run_list_v16, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido de los resultados válidos."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_v16 = st.session_state.get('df_processed')
target_col_for_tabs_v16 = st.session_state.get('original_target_column_name')
model_results_exist_v16 = st.session_state.get('model_results')

if df_proc_for_tabs_v16 is not None and not df_proc_for_tabs_v16.empty and \
   target_col_for_tabs_v16 and \
   model_results_exist_v16 is not None and isinstance(model_results_exist_v16, list) and \
   len(model_results_exist_v16) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    # ... (Lógica de las pestañas como en v3.15, usando las variables con sufijo _v16
    #      y asegurando que los placeholders de contenido de pestañas estén completos)
    # Ejemplo para la pestaña Recomendado:
    # with tab_rec_v16:
    #    best_model_v16 = st.session_state.best_model_name_auto
    #    if best_model_v16 and ... :
    #        # ... mostrar detalles ...
    pass # Placeholder para el contenido de las pestañas

# --- Mensajes si las pestañas no se muestran (Sección final CORREGIDA) ---
else: 
    df_loaded_is_present_final_v16 = st.session_state.get('df_loaded') is not None
    df_processed_value_final_v16 = st.session_state.get('df_processed')
    model_results_list_final_v16 = st.session_state.get('model_results')
    df_processed_is_empty_or_none_v16 = (df_processed_value_final_v16 is None or (isinstance(df_processed_value_final_v16, pd.DataFrame) and df_processed_value_final_v16.empty))
    model_results_is_empty_or_none_v16 = (model_results_list_final_v16 is None or (isinstance(model_results_list_final_v16, list) and not model_results_list_final_v16))

    if uploaded_file is None and not df_loaded_is_present_final_v16: 
        st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
    elif df_loaded_is_present_final_v16 and df_processed_is_empty_or_none_v16: 
        st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
    elif df_loaded_is_present_final_v16 and not df_processed_is_empty_or_none_v16 and model_results_is_empty_or_none_v16:
        st.info("Datos preprocesados. Por favor, genere los modelos para ver los resultados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.16")