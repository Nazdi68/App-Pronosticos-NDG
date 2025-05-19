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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v15", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb_v15 = st.session_state.get('df_loaded')

if df_input_sb_v15 is not None:
    date_col_options_sb_v15 = df_input_sb_v15.columns.tolist()
    dt_col_guess_idx_sb_v15 = 0
    if date_col_options_sb_v15:
        for i, col in enumerate(date_col_options_sb_v15):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb_v15 = i; break
    sel_date_idx_sb_v15 = 0
    if date_col_options_sb_v15 : 
        sel_date_idx_sb_v15 = date_col_options_sb_v15.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb_v15 else dt_col_guess_idx_sb_v15
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb_v15, index=sel_date_idx_sb_v15, key="date_sel_key_v15")

    value_col_options_sb_v15 = [col for col in df_input_sb_v15.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb_v15 = 0
    if value_col_options_sb_v15:
        for i, col in enumerate(value_col_options_sb_v15):
            if pd.api.types.is_numeric_dtype(df_input_sb_v15[col].dropna()): val_col_guess_idx_sb_v15 = i; break
    sel_val_idx_sb_v15 = 0
    if value_col_options_sb_v15:
        sel_val_idx_sb_v15 = value_col_options_sb_v15.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb_v15 else val_col_guess_idx_sb_v15
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb_v15, index=sel_val_idx_sb_v15, key="val_sel_key_v15")
    
    freq_map_sb_v15 = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb_v15 = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb_v15.keys()), key="freq_sel_key_v15", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb_v15 = freq_map_sb_v15[freq_label_sb_v15]
    imp_list_sb_v15 = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_v15 = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb_v15, index=1, key="imp_sel_key_v15", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb_v15 = None if imp_label_sb_v15 == "No imputar" else imp_label_sb_v15.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v15"):
        st.session_state.df_processed = None; reset_sidebar_config_dependent_state() 
        date_col_btn_v15 = st.session_state.get('selected_date_col'); value_col_btn_v15 = st.session_state.get('selected_value_col'); valid_btn_v15 = True
        if not date_col_btn_v15 or date_col_btn_v15 not in df_input_sb_v15.columns: st.sidebar.error("Seleccione fecha."); valid_btn_v15=False
        if not value_col_btn_v15 or value_col_btn_v15 not in df_input_sb_v15.columns: st.sidebar.error("Seleccione valor."); valid_btn_v15=False
        elif valid_btn_v15 and not pd.api.types.is_numeric_dtype(df_input_sb_v15[value_col_btn_v15].dropna()): st.sidebar.error(f"'{value_col_btn_v15}' no numérica."); valid_btn_v15=False
        if valid_btn_v15:
            with st.spinner("Preprocesando..."): proc_df_v15,msg_raw_v15 = data_handler.preprocess_data(df_input_sb_v15.copy(),date_col_btn_v15,value_col_btn_v15,desired_freq_sb_v15,imp_code_sb_v15)
            msg_disp_v15 = msg_raw_v15; 
            if msg_raw_v15: 
                if "MS" in msg_raw_v15: msg_disp_v15=msg_raw_v15.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_v15: msg_disp_v15=msg_raw_v15.replace(" D."," D (Diario).")
                elif msg_raw_v15.endswith("D"): msg_disp_v15=msg_raw_v15.replace("D", "D (Diario)")
            if proc_df_v15 is not None and not proc_df_v15.empty:
                st.session_state.df_processed=proc_df_v15; st.session_state.original_target_column_name=value_col_btn_v15; st.success(f"Preproc. OK. {msg_disp_v15}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_v15,value_col_btn_v15)
                if not proc_df_v15.empty:
                    s_acf_v15=proc_df_v15[value_col_btn_v15];l_acf_v15=min(len(s_acf_v15)//2-1,60)
                    if l_acf_v15 > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_v15,l_acf_v15,value_col_btn_v15)
                    else: st.session_state.acf_fig=None
                    _,auto_s_v15_val=data_handler.get_series_frequency_and_period(proc_df_v15.index)
                    st.session_state.auto_seasonal_period=auto_s_v15_val
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s_v15_val: st.session_state.user_seasonal_period=auto_s_v15_val
            else: st.error(f"Fallo preproc: {msg_raw_v15 or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_display_v15 = st.session_state.get('df_processed')
target_col_display_v15 = st.session_state.get('original_target_column_name')

if df_processed_display_v15 is not None and not df_processed_display_v15.empty and target_col_display_v15:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_v15, col2_acf_v15 = st.columns(2)
    with col1_diag_v15: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v15: 
        st.subheader("Autocorrelación")
        acf_fig_v15 = st.session_state.get('acf_fig')
        if acf_fig_v15: 
            try: st.pyplot(acf_fig_v15)
            except Exception as e_acf_v15: st.error(f"Error al mostrar ACF/PACF: {e_acf_v15}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_display_v15 in df_processed_display_v15.columns:
        fig_hist_v15 = visualization.plot_historical_data(df_processed_display_v15, target_col_display_v15, f"Histórico de '{target_col_display_v15}'")
        if fig_hist_v15: st.pyplot(fig_hist_v15)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v15")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v15", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_v15 = len(df_processed_display_v15)//2 if df_processed_display_v15 is not None and not df_processed_display_v15.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_v15), step=1, key="ma_win_key_v15")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v15", on_change=reset_model_execution_results) # Corregido a reset_model_execution_results
    if st.session_state.use_train_test_split:
        min_train_v15 = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_v15 = len(df_processed_display_v15) - min_train_v15; max_test_v15 = max(1, max_test_v15)
        def_test_v15 = min(max(1, st.session_state.forecast_horizon), max_test_v15)
        current_test_v15 = st.session_state.get('test_split_size', def_test_v15)
        if current_test_v15 > max_test_v15 or current_test_v15 <=0 : current_test_v15 = def_test_v15
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_v15, min_value=1, max_value=max_test_v15, step=1, key="test_size_key_v15", help=f"Máx: {max_test_v15}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v15")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        # --- Placeholder - COMPLETA ESTO con tus st.columns y st.number_input para ARIMA ---
        # Ejemplo: c1ar,c2ar=st.columns(2); st.session_state.arima_max_p=c1ar.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v15"); ...etc
        st.caption("Configuraciones para AutoARIMA (max_p, max_q, etc.)")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        # --- Placeholder - COMPLETA ESTO con tus st.checkbox y st.selectbox para Holt/HW ---
        # Ejemplo: st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", ..., key="hd_k_v15") ... etc
        st.caption("Configuraciones para Holt y Holt-Winters (damped, trend, seasonal, boxcox)")


    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v15_action"):
        reset_model_execution_results() # Limpiar resultados de modelos anteriores
        
        # Leer los datos necesarios directamente del estado de sesión DENTRO de la lógica del botón
        df_processed_for_models = st.session_state.get('df_processed')
        target_col_for_models = st.session_state.get('original_target_column_name')

        if df_processed_for_models is None or target_col_for_models is None or target_col_for_models not in df_processed_for_models.columns: 
            st.error("🔴 Datos no preprocesados correctamente. Por favor, asegúrese de cargar datos y aplicar el preprocesamiento primero."); 
        else:
            series_full_models_run = df_processed_for_models[target_col_for_models].copy(); 
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
                model_execution_specs_list_final = []
                # Baselines
                model_execution_specs_list_final.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_models_run, test_s_models_run, h_models_run], "name_override": "Promedio Histórico", "type":"baseline"})
                model_execution_specs_list_final.append({"func": forecasting_models.naive_forecast, "args": [train_s_models_run, test_s_models_run, h_models_run], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
                model_execution_specs_list_final.append({"func": forecasting_models.moving_average_forecast, "args": [train_s_models_run, test_s_models_run, h_models_run, ma_win_models_run], "name_override": None, "type":"baseline"})
                if s_period_models_run > 1: model_execution_specs_list_final.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s_models_run, test_s_models_run, h_models_run, s_period_models_run], "name_override": None, "type":"baseline"})
                
                # Statsmodels
                holt_p_exec_final_run = {'damped_trend': st.session_state.holt_damped}
                hw_p_exec_final_run = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
                stats_configs_final_run = [("SES", {}), ("Holt", holt_p_exec_final_run)]
                if s_period_models_run > 1: stats_configs_final_run.append(("Holt-Winters", hw_p_exec_final_run))
                for name_s_final_run_item, params_s_final_run_item in stats_configs_final_run:
                    model_execution_specs_list_final.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s_models_run, test_s_models_run, h_models_run, name_s_final_run_item, s_period_models_run if name_s_final_run_item=="Holt-Winters" else None, params_s_final_run_item if name_s_final_run_item=="Holt" else None, params_s_final_run_item if name_s_final_run_item=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_final_run_item, "params_dict": params_s_final_run_item})
                
                if st.session_state.run_autoarima:
                    arima_p_final_run_item = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                    model_execution_specs_list_final.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s_models_run, test_s_models_run, h_models_run, s_period_models_run, arima_p_final_run_item], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_final_run_item})

                for spec_final_item_run_loop in model_execution_specs_list_final:
                    try:
                        fc_res_item, ci_res_item, rmse_res_item, mae_res_item, name_res_item = spec_final_item_run_loop["func"](*spec_final_item_run_loop["args"])
                        name_disp_item = spec_final_item_run_loop["name_override"] or name_res_item
                        fc_on_test_item_val = None
                        if st.session_state.use_train_test_split and not test_s_models_run.empty and not any(err_str in name_disp_item for err_str in ["Error","Insuf","Inválido","FALLÓ"]):
                            # --- INICIO: Lógica para fc_on_test_item_val ---
                            if spec_final_item_run_loop["type"] == "baseline":
                                if "Promedio Histórico" in name_disp_item and not train_s_models_run.empty: fc_on_test_item_val = np.full(len(test_s_models_run), train_s_models_run.mean())
                                elif "Ingénuo" in name_disp_item and not train_s_models_run.empty: fc_on_test_item_val = np.full(len(test_s_models_run), train_s_models_run.iloc[-1])
                                elif "Promedio Móvil" in name_disp_item and not train_s_models_run.empty and len(train_s_models_run) >= ma_win_models_run : fc_on_test_item_val = np.full(len(test_s_models_run), train_s_models_run.iloc[-ma_win_models_run:].mean())
                                elif "Estacional Ingénuo" in name_disp_item and not train_s_models_run.empty and len(train_s_models_run) >= s_period_models_run:
                                    temp_fc_test_val_loop = np.zeros(len(test_s_models_run)); 
                                    for i_fc_val_loop in range(len(test_s_models_run)): temp_fc_test_val_loop[i_fc_val_loop] = train_s_models_run.iloc[len(train_s_models_run) - s_period_models_run + (i_fc_val_loop % s_period_models_run)]
                                    fc_on_test_item_val = temp_fc_test_val_loop
                            elif spec_final_item_run_loop["type"] == "statsmodels":
                                sm_name_test_run_loop = spec_final_item_run_loop["model_short_name"]; sm_params_test_run_loop = spec_final_item_run_loop["params_dict"]; sm_fit_test_item_run_loop = None
                                try:
                                    if sm_name_test_run_loop=="SES": sm_fit_test_item_run_loop=forecasting_models.SimpleExpSmoothing(train_s_models_run,initialization_method="estimated").fit()
                                    elif sm_name_test_run_loop=="Holt": sm_fit_test_item_run_loop=forecasting_models.Holt(train_s_models_run,damped_trend=sm_params_test_run_loop.get('damped_trend',False),initialization_method="estimated").fit()
                                    elif sm_name_test_run_loop=="Holt-Winters": sm_fit_test_item_run_loop=forecasting_models.ExponentialSmoothing(train_s_models_run,trend=sm_params_test_run_loop.get('trend'),seasonal=sm_params_test_run_loop.get('seasonal'),seasonal_periods=s_period_models_run,damped_trend=sm_params_test_run_loop.get('damped_trend',False),use_boxcox=sm_params_test_run_loop.get('use_boxcox',False),initialization_method="estimated").fit()
                                    if sm_fit_test_item_run_loop: fc_on_test_item_val = sm_fit_test_item_run_loop.forecast(len(test_s_models_run)).values
                                except Exception as e_sm_fc_test_item_loop: st.caption(f"Warn fc_test {name_disp_item}: {e_sm_fc_test_item_loop}")
                            elif spec_final_item_run_loop["type"] == "autoarima":
                                arima_cfg_test_run_loop = spec_final_item_run_loop["arima_params_dict"]; arima_fit_test_item_run_loop = None
                                try:
                                    arima_fit_test_item_run_loop = forecasting_models.pm.auto_arima(train_s_models_run,max_p=arima_cfg_test_run_loop.get('max_p',3),max_q=arima_cfg_test_run_loop.get('max_q',3),m=s_period_models_run if s_period_models_run>1 else 1,seasonal=s_period_models_run>1,suppress_warnings=True,error_action='ignore',stepwise=True, trace=False, **{'max_d':arima_cfg_test_run_loop.get('max_d',2), 'max_P':arima_cfg_test_run_loop.get('max_P',1), 'max_Q':arima_cfg_test_run_loop.get('max_Q',1), 'max_D':arima_cfg_test_run_loop.get('max_D',1)})
                                    if arima_fit_test_item_run_loop: fc_on_test_item_val = arima_fit_test_item_run_loop.predict(n_periods=len(test_s_models_run))
                                except Exception as e_arima_test_fc_item_loop: st.caption(f"Warn fc_test {name_disp_item}: {e_arima_test_fc_item_loop}")
                            # --- FIN: Lógica para fc_on_test_item_val ---
                        st.session_state.model_results.append({'name':name_disp_item,'rmse':rmse_res_item,'mae':mae_res_item,'forecast_future':fc_res_item,'conf_int_future':ci_res_item,'forecast_on_test':fc_on_test_item_val})
                    except Exception as e_model_item_loop: st.warning(f"Error procesando {spec_final_item_run_loop.get('name_override',spec_final_item_run_loop['func'].__name__)}: {str(e_model_item_loop)[:150]}")
            
            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_results_final_loop = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and isinstance(r.get('forecast_future'), np.ndarray) and len(r.get('forecast_future'))==h_models_run]
            if valid_results_final_loop: st.session_state.best_model_name_auto = min(valid_results_final_loop, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido de los resultados válidos."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_render_final = st.session_state.get('df_processed')
target_col_for_tabs_render_final = st.session_state.get('original_target_column_name')
model_results_exist_render_final = st.session_state.get('model_results')

if df_proc_for_tabs_render_final is not None and not df_proc_for_tabs_render_final.empty and \
   target_col_for_tabs_render_final and \
   model_results_exist_render_final is not None and isinstance(model_results_exist_render_final, list) and len(model_results_exist_render_final) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_final_render_view, tab_comp_final_render_view, tab_man_final_render_view, tab_diag_final_render_view = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    historical_series_for_tabs_final_plot_view = None
    if target_col_for_tabs_render_final in df_proc_for_tabs_render_final.columns:
        historical_series_for_tabs_final_plot_view = df_proc_for_tabs_render_final[target_col_for_tabs_render_final]

    with tab_rec_final_render_view: # Pestaña Modelo Recomendado
        best_model_rec_render_final_view = st.session_state.best_model_name_auto
        if best_model_rec_render_final_view and "Error" not in best_model_rec_render_final_view and "FALLÓ" not in best_model_rec_render_final_view and historical_series_for_tabs_final_plot_view is not None:
            st.subheader(f"Modelo Recomendado: {best_model_rec_render_final_view}")
            model_data_rec_final_view = next((item for item in st.session_state.model_results if item["name"] == best_model_rec_render_final_view), None)
            if model_data_rec_final_view:
                final_df_r_render_final_view, fc_s_r_render_final_view, pi_df_r_render_final_view = prepare_forecast_display_data(model_data_rec_final_view, historical_series_for_tabs_final_plot_view.index, st.session_state.forecast_horizon)
                if final_df_r_render_final_view is not None and fc_s_r_render_final_view is not None and not fc_s_r_render_final_view.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_rec_final_view.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); 
                        fc_test_r_s_final_view = pd.Series(model_data_rec_final_view['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_rec_final_view['forecast_on_test'],(np.ndarray, list)) and len(model_data_rec_final_view['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_rec_final_view.get('forecast_on_test'); 
                        if fc_test_r_s_final_view is not None and not (isinstance(fc_test_r_s_final_view, float) and np.isnan(fc_test_r_s_final_view)): 
                            fig_vr_final_view=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_r_s_final_view,best_model_rec_render_final_view,target_col_for_tabs_render_final); 
                            if fig_vr_final_view: st.pyplot(fig_vr_final_view)
                    st.markdown(f"##### Pronóstico Futuro"); fig_fr_final_view=visualization.plot_final_forecast(historical_series_for_tabs_final_plot_view,fc_s_r_render_final_view,pi_df_r_render_final_view,best_model_rec_render_final_view,target_col_for_tabs_render_final); st.pyplot(fig_fr_final_view) if fig_fr_final_view else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_r_render_final_view.style.format("{:.2f}")); 
                    dl_key_rec_v15_btn = f"dl_rec_{best_model_rec_render_final_view[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF_v15"
                    st.download_button(f"📥 Descargar ({best_model_rec_render_final_view})",to_excel(final_df_r_render_final_view),f"fc_{best_model_rec_render_final_view.replace(' ','_')}.xlsx",key=dl_key_rec_v15_btn)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model_rec_render_final_view,st.session_state.data_diagnosis_report,True,(pi_df_r_render_final_view is not None and not pi_df_r_render_final_view.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_render_final))
                else: st.warning(f"No se pudo preparar visualización para '{best_model_rec_render_final_view}'.")
            else: st.info(f"Datos del modelo '{best_model_rec_render_final_view}' no encontrados.")
        elif not historical_series_for_tabs_final_plot_view : st.warning("Datos históricos no disponibles.")
        else: st.info("No se ha determinado un modelo recomendado. Genere los modelos o verifique errores.")

    with tab_comp_final_view: # Pestaña Comparación
        st.subheader("Comparación de Modelos")
        metrics_list_comp_v15 = [{'Modelo': r.get('name','N/A'), 'RMSE': r.get('rmse'), 'MAE': r.get('mae')} for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None]
        if metrics_list_comp_v15:
            metrics_df_comp_v15 = pd.DataFrame(metrics_list_comp_v15).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_v15(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_comp_v15.style.format({'RMSE':"{:.3f}", 'MAE':"{:.3f}"}).apply(highlight_best_v15, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Sugerido: **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay métricas de modelos para mostrar.")

    with tab_man_final_view: # Pestaña Explorar Manualmente
        st.subheader("Explorar Modelo Manualmente")
        valid_manual_models_v15 = [r['name'] for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r.get('rmse'))]
        if valid_manual_models_v15 and historical_series_for_tabs_final_plot_view is not None:
            sel_idx_man_v15 = 0
            if st.session_state.selected_model_for_manual_explore in valid_manual_models_v15: sel_idx_man_v15 = valid_manual_models_v15.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in valid_manual_models_v15: sel_idx_man_v15 = valid_manual_models_v15.index(st.session_state.best_model_name_auto)
            st.session_state.selected_model_for_manual_explore = st.selectbox("Modelo:", valid_manual_models_v15, index=sel_idx_man_v15, key="man_sel_key_v15_final")
            model_data_man_v15 = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)
            if model_data_man_v15:
                final_df_m_v15, fc_s_m_v15, pi_df_m_v15 = prepare_forecast_display_data(model_data_man_v15, historical_series_for_tabs_final_plot_view.index, st.session_state.forecast_horizon)
                if final_df_m_v15 is not None and fc_s_m_v15 is not None and not fc_s_m_v15.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_man_v15.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); 
                        fc_test_m_s_v15 = pd.Series(model_data_man_v15['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_man_v15['forecast_on_test'],(np.ndarray,list)) and len(model_data_man_v15['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_man_v15.get('forecast_on_test');
                        if fc_test_m_s_v15 is not None and not (isinstance(fc_test_m_s_v15, float) and np.isnan(fc_test_m_s_v15)): 
                            fig_vm_v15=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_m_s_v15,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final_view); st.pyplot(fig_vm_v15) if fig_vm_v15 else st.caption("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fm_v15=visualization.plot_final_forecast(historical_series_for_tabs_final_plot_view,fc_s_m_v15,pi_df_m_v15,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final_view); st.pyplot(fig_fm_v15) if fig_fm_v15 else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_m_v15.style.format("{:.2f}")); 
                    dl_key_m_v15_btn = f"dl_man_{st.session_state.selected_model_for_manual_explore[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF_final"
                    st.download_button(f"📥 Descargar ({st.session_state.selected_model_for_manual_explore})",to_excel(final_df_m_v15),f"fc_man_{target_col_for_tabs_final_view}.xlsx",key=dl_key_m_v15_btn)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(st.session_state.selected_model_for_manual_explore,st.session_state.data_diagnosis_report,True,(pi_df_m_v15 is not None and not pi_df_m_v15.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_final_view))
                else: st.warning(f"No se pudo preparar visualización para '{st.session_state.selected_model_for_manual_explore}'.")
        elif not historical_series_for_tabs_final_plot_view : st.warning("Datos históricos no disponibles.")
        else: st.warning("No hay modelos válidos para exploración. Genere los modelos.")
        
    with tab_diag_final_view:
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor.\n- Intervalos de Predicción: Incertidumbre.\n- Calidad de Datos: Crucial.")

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') is not None and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
elif st.session_state.get('df_loaded') is not None and st.session_state.get('df_processed') is not None and not st.session_state.get('df_processed').empty and \
     (st.session_state.get('model_results') is None or (isinstance(st.session_state.get('model_results'), list) and not st.session_state.get('model_results'))):
    st.info("Datos preprocesados. Por favor, genere los modelos para ver los resultados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.15")