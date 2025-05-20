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

def reset_on_file_change(): # Llamado por on_change del file_uploader
    keys_to_reset = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name', 'data_diagnosis_report', 'acf_fig', 
        'model_results', 'best_model_name_auto', 
        'selected_model_for_manual_explore', 'train_series_for_plot', 
        'test_series_for_plot', 'auto_seasonal_period'
    ]
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state: del st.session_state[key_to_del]
    st.session_state.df_loaded = None # Forzar recarga
    init_session_state() # Re-establecer defaults para claves borradas

def reset_sidebar_config_dependent_state(): # Llamado al cambiar Frecuencia o Imputación
    st.session_state.df_processed = None 
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None
    st.session_state.train_series_for_plot = None; st.session_state.test_series_for_plot = None

def reset_model_execution_results(): # Llamado al cambiar "Usar Train/Test split" o ANTES de "Generar Modelos"
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

# --- Sección 1: Carga y Preprocesamiento de Datos (Sidebar) ---
st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v20", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb = st.session_state.get('df_loaded') # Usar un nombre más genérico para el df de input de la sidebar

if df_input_sb is not None:
    date_col_options_sb = df_input_sb.columns.tolist()
    dt_col_guess_idx_sb = 0
    if date_col_options_sb:
        for i, col in enumerate(date_col_options_sb):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb = i; break
    sel_date_idx_sb = 0
    if date_col_options_sb : 
        sel_date_idx_sb = date_col_options_sb.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb else dt_col_guess_idx_sb
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb, index=sel_date_idx_sb, key="date_sel_key_v20")

    value_col_options_sb = [col for col in df_input_sb.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb = 0
    if value_col_options_sb:
        for i, col in enumerate(value_col_options_sb):
            if pd.api.types.is_numeric_dtype(df_input_sb[col].dropna()): val_col_guess_idx_sb = i; break
    sel_val_idx_sb = 0
    if value_col_options_sb:
        sel_val_idx_sb = value_col_options_sb.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb else val_col_guess_idx_sb
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb, index=sel_val_idx_sb, key="val_sel_key_v20")
    
    freq_map_sb = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb.keys()), key="freq_sel_key_v20", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb_val = freq_map_sb[freq_label_sb] # Nombre de variable diferente
    imp_list_sb = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_val = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb, index=1, key="imp_sel_key_v20", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb_val = None if imp_label_sb_val == "No imputar" else imp_label_sb_val.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v20"):
        st.session_state.df_processed = None; reset_sidebar_config_dependent_state() # Resetear df_processed y todo lo que depende de él
        
        date_col_from_state = st.session_state.get('selected_date_col')
        value_col_from_state = st.session_state.get('selected_value_col')
        is_valid_input = True
        
        if not date_col_from_state or date_col_from_state not in df_input_sb.columns: st.sidebar.error("Seleccione columna de fecha válida."); is_valid_input=False
        if not value_col_from_state or value_col_from_state not in df_input_sb.columns: st.sidebar.error("Seleccione columna de valor válida."); is_valid_input=False
        elif is_valid_input and not pd.api.types.is_numeric_dtype(df_input_sb[value_col_from_state].dropna()): st.sidebar.error(f"Columna '{value_col_from_state}' no es numérica."); is_valid_input=False
        
        if is_valid_input:
            with st.spinner("Preprocesando..."): 
                processed_df_result, msg_raw_from_preproc = data_handler.preprocess_data(
                    df_input_sb.copy(),
                    date_col_from_state,
                    value_col_from_state,
                    desired_freq_sb_val, # Usar la variable correcta
                    imp_code_sb_val      # Usar la variable correcta
                )
            msg_display_from_preproc = msg_raw_from_preproc
            if msg_raw_from_preproc: 
                if "MS" in msg_raw_from_preproc: msg_display_from_preproc=msg_raw_from_preproc.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_from_preproc: msg_display_from_preproc=msg_raw_from_preproc.replace(" D."," D (Diario).")
                elif msg_raw_from_preproc.endswith("D"): msg_display_from_preproc=msg_raw_from_preproc.replace("D", "D (Diario)")
            
            if processed_df_result is not None and not processed_df_result.empty:
                st.session_state.df_processed = processed_df_result
                st.session_state.original_target_column_name = value_col_from_state # Guardar la columna usada
                st.success(f"Preprocesamiento OK. {msg_display_from_preproc}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(processed_df_result,value_col_from_state)
                if not processed_df_result.empty:
                    series_for_acf_plot = processed_df_result[value_col_from_state]
                    lags_for_acf_plot = min(len(series_for_acf_plot)//2-1,60)
                    if lags_for_acf_plot > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(series_for_acf_plot,lags_for_acf_plot,value_col_from_state)
                    else: st.session_state.acf_fig=None
                    _,auto_seasonal_period_val=data_handler.get_series_frequency_and_period(processed_df_result.index)
                    st.session_state.auto_seasonal_period=auto_seasonal_period_val
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_seasonal_period_val: 
                        st.session_state.user_seasonal_period=auto_seasonal_period_val
            else: 
                st.error(f"Fallo en preprocesamiento: {msg_raw_from_preproc or 'El DataFrame resultante está vacío o es None.'}")
                st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_for_display_v20 = st.session_state.get('df_processed')
target_col_for_display_v20 = st.session_state.get('original_target_column_name')

if df_processed_for_display_v20 is not None and not df_processed_for_display_v20.empty and target_col_for_display_v20:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_v20, col2_acf_v20 = st.columns(2)
    with col1_diag_v20: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v20: 
        st.subheader("Autocorrelación")
        acf_fig_v20 = st.session_state.get('acf_fig')
        if acf_fig_v20 is not None: 
            try: st.pyplot(acf_fig_v20)
            except Exception as e_acf_v20: st.error(f"Error al mostrar ACF/PACF: {e_acf_v20}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_for_display_v20 in df_processed_for_display_v20.columns:
        fig_hist_v20 = visualization.plot_historical_data(df_processed_for_display_v20, target_col_for_display_v20, f"Histórico de '{target_col_for_display_v20}'")
        if fig_hist_v20: st.pyplot(fig_hist_v20)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v20")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v20", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_v20 = len(df_processed_for_display_v20)//2 if df_processed_for_display_v20 is not None and not df_processed_for_display_v20.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_v20), step=1, key="ma_win_key_v20")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v20", on_change=reset_model_execution_results)
    if st.session_state.use_train_test_split:
        min_train_v20 = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_v20 = len(df_processed_for_display_v20) - min_train_v20; max_test_v20 = max(1, max_test_v20)
        def_test_v20 = min(max(1, st.session_state.forecast_horizon), max_test_v20)
        current_test_v20 = st.session_state.get('test_split_size', def_test_v20)
        if current_test_v20 > max_test_v20 or current_test_v20 <=0 : current_test_v20 = def_test_v20
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_v20, min_value=1, max_value=max_test_v20, step=1, key="test_size_key_v20", help=f"Máx: {max_test_v20}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v20")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v20,c2ar_v20=st.columns(2); st.session_state.arima_max_p=c1ar_v20.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v20"); st.session_state.arima_max_q=c2ar_v20.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v20"); st.session_state.arima_max_d=c1ar_v20.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v20"); st.session_state.arima_max_P=c2ar_v20.number_input("max_P (est.)",0,3,st.session_state.arima_max_P,key="aP_k_v20"); st.session_state.arima_max_Q=c1ar_v20.number_input("max_Q (est.)",0,3,st.session_state.arima_max_Q,key="aQ_k_v20"); st.session_state.arima_max_D=c2ar_v20.number_input("max_D (est.)",0,2,st.session_state.arima_max_D,key="aD_k_v20")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v20")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v20"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v20"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v20"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v20")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v20_action"):
        reset_model_execution_results()
        
        df_proc_for_models_btn = st.session_state.get('df_processed')
        target_col_for_models_btn = st.session_state.get('original_target_column_name')

        if df_proc_for_models_btn is None or target_col_for_models_btn is None or \
           target_col_for_models_btn not in df_proc_for_models_btn.columns: 
            st.error("🔴 Datos no preprocesados. Aplique preprocesamiento."); 
        else:
            series_full_for_btn_run = df_proc_for_models_btn[target_col_for_models_btn].copy(); 
            h_for_btn_run = st.session_state.forecast_horizon; 
            s_period_for_btn_run = st.session_state.user_seasonal_period; 
            ma_win_for_btn_run = st.session_state.moving_avg_window
            
            train_s_for_btn_run, test_s_for_btn_run = series_full_for_btn_run, pd.Series(dtype=series_full_for_btn_run.dtype)
            if st.session_state.use_train_test_split:
                min_tr_for_btn_run = max(5, 2*s_period_for_btn_run+1 if s_period_for_btn_run>1 else 5); 
                curr_test_for_btn_run = st.session_state.get('test_split_size', 12)
                if len(series_full_for_btn_run) > min_tr_for_btn_run + curr_test_for_btn_run and curr_test_for_btn_run > 0 : 
                    train_s_for_btn_run,test_s_for_btn_run = forecasting_models.train_test_split_series(series_full_for_btn_run, curr_test_for_btn_run)
                else: 
                    st.warning(f"No fue posible el split con test_size={curr_test_for_btn_run}. Evaluando in-sample."); 
                    st.session_state.use_train_test_split=False
                    train_s_for_btn_run, test_s_for_btn_run = series_full_for_btn_run, pd.Series(dtype=series_full_for_btn_run.dtype)
            
            st.session_state.train_series_for_plot = train_s_for_btn_run; 
            st.session_state.test_series_for_plot = test_s_for_btn_run
                
            with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
                model_execution_list = [] # Nombre de lista consistente
                
                # Baselines
                model_execution_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run], "name_override": "Promedio Histórico", "type":"baseline"})
                model_execution_list.append({"func": forecasting_models.naive_forecast, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
                model_execution_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run, ma_win_for_btn_run], "name_override": None, "type":"baseline"})
                if s_period_for_btn_run > 1: model_execution_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run, s_period_for_btn_run], "name_override": None, "type":"baseline"})
                
                # Statsmodels
                holt_p_exec_btn = {'damped_trend': st.session_state.holt_damped}
                hw_p_exec_btn = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
                stats_configs_btn = [("SES", {}), ("Holt", holt_p_exec_btn)]
                if s_period_for_btn_run > 1: stats_configs_btn.append(("Holt-Winters", hw_p_exec_btn))
                for name_s_btn_item, params_s_btn_item in stats_configs_btn:
                    model_execution_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run, name_s_btn_item, s_period_for_btn_run if name_s_btn_item=="Holt-Winters" else None, params_s_btn_item if name_s_btn_item=="Holt" else None, params_s_btn_item if name_s_btn_item=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_btn_item, "params_dict": params_s_btn_item})
                
                if st.session_state.run_autoarima:
                    arima_p_btn_item = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                    model_execution_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s_for_btn_run, test_s_for_btn_run, h_for_btn_run, s_period_for_btn_run, arima_p_btn_item], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_btn_item})

                for spec_item_btn_loop in model_execution_list: 
                    try:
                        fc_future_btn_item, ci_future_btn_item, rmse_btn_item, mae_btn_item, name_from_func_btn_item = spec_item_btn_loop["func"](*spec_item_btn_loop["args"])
                        name_display_btn_item = spec_item_btn_loop["name_override"] or name_from_func_btn_item
                        fc_on_test_btn_item_final = None
                        if st.session_state.use_train_test_split and not test_s_for_btn_run.empty and not any(err in name_display_btn_item for err in ["Error","Insuf","Inválido","FALLÓ"]):
                            if spec_item_btn_loop["type"] == "baseline":
                                if "Promedio Histórico" in name_display_btn_item and not train_s_for_btn_run.empty: fc_on_test_btn_item_final = np.full(len(test_s_for_btn_run), train_s_for_btn_run.mean())
                                elif "Ingénuo" in name_display_btn_item and not train_s_for_btn_run.empty: fc_on_test_btn_item_final = np.full(len(test_s_for_btn_run), train_s_for_btn_run.iloc[-1])
                                elif "Promedio Móvil" in name_display_btn_item and not train_s_for_btn_run.empty and len(train_s_for_btn_run) >= ma_win_for_btn_run : fc_on_test_btn_item_final = np.full(len(test_s_for_btn_run), train_s_for_btn_run.iloc[-ma_win_for_btn_run:].mean())
                                elif "Estacional Ingénuo" in name_display_btn_item and not train_s_for_btn_run.empty and len(train_s_for_btn_run) >= s_period_for_btn_run:
                                    temp_fc_test_btn_item = np.zeros(len(test_s_for_btn_run)); 
                                    for i_fc_btn_item in range(len(test_s_for_btn_run)): temp_fc_test_btn_item[i_fc_btn_item] = train_s_for_btn_run.iloc[len(train_s_for_btn_run) - s_period_for_btn_run + (i_fc_btn_item % s_period_for_btn_run)]
                                    fc_on_test_btn_item_final = temp_fc_test_btn_item
                            elif spec_item_btn_loop["type"] == "statsmodels":
                                sm_name_test_btn = spec_item_btn_loop["model_short_name"]; sm_params_test_btn = spec_item_btn_loop["params_dict"]; sm_fit_test_btn_item = None
                                try: 
                                    if sm_name_test_btn=="SES": sm_fit_test_btn_item=forecasting_models.SimpleExpSmoothing(train_s_for_btn_run,initialization_method="estimated").fit()
                                    elif sm_name_test_btn=="Holt": sm_fit_test_btn_item=forecasting_models.Holt(train_s_for_btn_run,damped_trend=sm_params_test_btn.get('damped_trend',False),initialization_method="estimated").fit()
                                    elif sm_name_test_btn=="Holt-Winters": sm_fit_test_btn_item=forecasting_models.ExponentialSmoothing(train_s_for_btn_run,trend=sm_params_test_btn.get('trend'),seasonal=sm_params_test_btn.get('seasonal'),seasonal_periods=s_period_for_btn_run,damped_trend=sm_params_test_btn.get('damped_trend',False),use_boxcox=sm_params_test_btn.get('use_boxcox',False),initialization_method="estimated").fit()
                                    if sm_fit_test_btn_item: fc_on_test_btn_item_final = sm_fit_test_btn_item.forecast(len(test_s_for_btn_run)).values
                                except Exception as e_sm_fc_test_btn_item: st.caption(f"Warn fc_test {name_display_btn_item}: {e_sm_fc_test_btn_item}")
                            elif spec_item_btn_loop["type"] == "autoarima":
                                arima_cfg_test_btn = spec_item_btn_loop["arima_params_dict"]; arima_fit_test_btn_item = None
                                try: 
                                    arima_fit_test_btn_item = forecasting_models.pm.auto_arima(train_s_for_btn_run,max_p=arima_cfg_test_btn.get('max_p',3),max_q=arima_cfg_test_btn.get('max_q',3),m=s_period_for_btn_run if s_period_for_btn_run>1 else 1,seasonal=s_period_for_btn_run>1,suppress_warnings=True,error_action='ignore',stepwise=True, trace=False, **{'max_d':arima_cfg_test_btn.get('max_d',2), 'max_P':arima_cfg_test_btn.get('max_P',1), 'max_Q':arima_cfg_test_btn.get('max_Q',1), 'max_D':arima_cfg_test_btn.get('max_D',1)})
                                    if arima_fit_test_btn_item: fc_on_test_btn_item_final = arima_fit_test_btn_item.predict(n_periods=len(test_s_for_btn_run))
                                except Exception as e_arima_test_fc_btn_item: st.caption(f"Warn fc_test {name_display_btn_item}: {e_arima_test_fc_btn_item}")
                        st.session_state.model_results.append({'name':name_display_btn_item,'rmse':rmse_btn_item,'mae':mae_btn_item,'forecast_future':fc_future_btn_item,'conf_int_future':ci_future_btn_item,'forecast_on_test':fc_on_test_btn_item_final})
                    except Exception as e_model_btn_loop_final: st.warning(f"Error procesando {spec_item_btn_loop.get('name_override',spec_item_btn_loop['func'].__name__)}: {str(e_model_btn_loop_final)[:150]}")
                
                st.write("--- DEBUG: Contenido de st.session_state.model_results ---")
                if isinstance(st.session_state.model_results, list) and st.session_state.model_results:
                    for i_debug_v18_final_loop, res_debug_v18_final_loop in enumerate(st.session_state.model_results):
                        st.write(f"Modelo {i_debug_v18_final_loop+1}:")
                        st.json(res_debug_v18_final_loop) 
                else: st.write("st.session_state.model_results está vacío o no es una lista.")
                st.write(f"Horizonte (h_for_run) usado para filtrar: {h_for_run}") 
                st.write("--- FIN DEBUG ---")

            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_results_final_v18_loop = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and isinstance(r.get('forecast_future'), (np.ndarray, list)) and len(r.get('forecast_future'))==h_for_run]
            if valid_results_final_v18_loop: st.session_state.best_model_name_auto = min(valid_results_final_v18_loop, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido de los resultados válidos."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_v18_render = st.session_state.get('df_processed')
target_col_for_tabs_v18_render = st.session_state.get('original_target_column_name')
model_results_exist_v18_render = st.session_state.get('model_results')

if df_proc_for_tabs_v18_render is not None and not df_proc_for_tabs_v18_render.empty and \
   target_col_for_tabs_v18_render and \
   model_results_exist_v18_render is not None and isinstance(model_results_exist_v18_render, list) and \
   len(model_results_exist_v18_render) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_v18_render_view, tab_comp_v18_render_view, tab_man_v18_render_view, tab_diag_v18_render_view = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    historical_series_for_tabs_v18_render_plot = None
    if target_col_for_tabs_v18_render in df_proc_for_tabs_v18_render.columns:
        historical_series_for_tabs_v18_render_plot = df_proc_for_tabs_v18_render[target_col_for_tabs_v18_render]

    # --- Pestaña 1: Modelo Recomendado ---
    with tab_rec_v18_render_view:
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "RECOMENDADO" DE LA VERSIÓN ANTERIOR v3.18)
        # ... (Asegúrate de usar las variables con sufijo _v18 donde sea apropiado si las cambiaste arriba)
        best_model_rec_v18_view = st.session_state.best_model_name_auto
        if best_model_rec_v18_view and "Error" not in best_model_rec_v18_view and "FALLÓ" not in best_model_rec_v18_view and historical_series_for_tabs_v18_render_plot is not None:
            st.subheader(f"Modelo Recomendado: {best_model_rec_v18_view}")
            # ... (El resto de la lógica de la pestaña Recomendado)
        else:
            st.info("No se ha determinado un modelo recomendado o hubo un error. Genere los modelos.")


    # --- Pestaña 2: Comparación General ---
    with tab_comp_v18_render_view:
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "COMPARACIÓN" DE LA VERSIÓN ANTERIOR v3.18)
        st.subheader("Comparación de Modelos")
        # ... (Mostrar tabla de métricas)

    # --- Pestaña 3: Explorar y Seleccionar Manualmente ---
    with tab_man_v18_render_view:
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "EXPLORAR" DE LA VERSIÓN ANTERIOR v3.18)
        st.subheader("Explorar Modelo Manualmente")
        # ... (Selector y visualización del modelo manual)
        
    # --- Pestaña 4: Diagnóstico y Guía ---
    with tab_diag_v18_render_view:
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor.\n- Intervalos de Predicción: Incertidumbre.\n- Calidad de Datos: Crucial.")

# --- Mensajes si las pestañas no se muestran (Sección final CORREGIDA) ---
else: 
    df_loaded_is_present_final_v18_else = st.session_state.get('df_loaded') is not None
    df_processed_value_final_v18_else = st.session_state.get('df_processed')
    model_results_list_final_v18_else = st.session_state.get('model_results')
    df_processed_is_empty_or_none_v18_else = (df_processed_value_final_v18_else is None or (isinstance(df_processed_value_final_v18_else, pd.DataFrame) and df_processed_value_final_v18_else.empty))
    model_results_is_empty_or_none_v18_else = (model_results_list_final_v18_else is None or (isinstance(model_results_list_final_v18_else, list) and not model_results_list_final_v18_else))

    if uploaded_file is None and not df_loaded_is_present_final_v18_else: 
        st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
    elif df_loaded_is_present_final_v18_else and df_processed_is_empty_or_none_v18_else: 
        st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
    elif df_loaded_is_present_final_v18_else and not df_processed_is_empty_or_none_v18_else and model_results_is_empty_or_none_v18_else:
        st.info("Datos preprocesados. Por favor, genere los modelos para ver los resultados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.19")
