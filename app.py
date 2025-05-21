# app.py (v3.22)
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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v22", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb_v22 = st.session_state.get('df_loaded')

if df_input_sb_v22 is not None:
    date_col_options_sb_v22 = df_input_sb_v22.columns.tolist()
    dt_col_guess_idx_sb_v22 = 0
    if date_col_options_sb_v22:
        for i, col in enumerate(date_col_options_sb_v22):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb_v22 = i; break
    sel_date_idx_sb_v22 = 0
    if date_col_options_sb_v22 : 
        sel_date_idx_sb_v22 = date_col_options_sb_v22.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb_v22 else dt_col_guess_idx_sb_v22
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb_v22, index=sel_date_idx_sb_v22, key="date_sel_key_v22")

    value_col_options_sb_v22 = [col for col in df_input_sb_v22.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb_v22 = 0
    if value_col_options_sb_v22:
        for i, col in enumerate(value_col_options_sb_v22):
            if pd.api.types.is_numeric_dtype(df_input_sb_v22[col].dropna()): val_col_guess_idx_sb_v22 = i; break
    sel_val_idx_sb_v22 = 0
    if value_col_options_sb_v22:
        sel_val_idx_sb_v22 = value_col_options_sb_v22.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb_v22 else val_col_guess_idx_sb_v22
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb_v22, index=sel_val_idx_sb_v22, key="val_sel_key_v22")
    
    freq_map_sb_v22 = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb_v22 = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb_v22.keys()), key="freq_sel_key_v22", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb_v22 = freq_map_sb_v22[freq_label_sb_v22]
    imp_list_sb_v22 = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_v22 = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb_v22, index=1, key="imp_sel_key_v22", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb_v22 = None if imp_label_sb_v22 == "No imputar" else imp_label_sb_v22.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v22"):
        st.session_state.df_processed = None; reset_sidebar_config_dependent_state() 
        date_col_btn_v22 = st.session_state.get('selected_date_col'); value_col_btn_v22 = st.session_state.get('selected_value_col'); valid_btn_v22 = True
        if not date_col_btn_v22 or date_col_btn_v22 not in df_input_sb_v22.columns: st.sidebar.error("Seleccione fecha."); valid_btn_v22=False
        if not value_col_btn_v22 or value_col_btn_v22 not in df_input_sb_v22.columns: st.sidebar.error("Seleccione valor."); valid_btn_v22=False
        elif valid_btn_v22 and not pd.api.types.is_numeric_dtype(df_input_sb_v22[value_col_btn_v22].dropna()): st.sidebar.error(f"'{value_col_btn_v22}' no numérica."); valid_btn_v22=False
        if valid_btn_v22:
            with st.spinner("Preprocesando..."): proc_df_v22,msg_raw_v22 = data_handler.preprocess_data(df_input_sb_v22.copy(),date_col_btn_v22,value_col_btn_v22,desired_freq_sb_v22,imp_code_sb_v22)
            msg_disp_v22 = msg_raw_v22; 
            if msg_raw_v22: 
                if "MS" in msg_raw_v22: msg_disp_v22=msg_raw_v22.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_v22: msg_disp_v22=msg_raw_v22.replace(" D."," D (Diario).")
                elif msg_raw_v22.endswith("D"): msg_disp_v22=msg_raw_v22.replace("D", "D (Diario)")
            if proc_df_v22 is not None and not proc_df_v22.empty:
                st.session_state.df_processed=proc_df_v22; st.session_state.original_target_column_name=value_col_btn_v22; st.success(f"Preproc. OK. {msg_disp_v22}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_v22,value_col_btn_v22)
                if not proc_df_v22.empty:
                    s_acf_v22=proc_df_v22[value_col_btn_v22];l_acf_v22=min(len(s_acf_v22)//2-1,60)
                    if l_acf_v22 > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_v22,l_acf_v22,value_col_btn_v22)
                    else: st.session_state.acf_fig=None
                    _,auto_s_v22_val=data_handler.get_series_frequency_and_period(proc_df_v22.index)
                    st.session_state.auto_seasonal_period=auto_s_v22_val
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s_v22_val: st.session_state.user_seasonal_period=auto_s_v22_val
            else: st.error(f"Fallo preproc: {msg_raw_v22 or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main_display = st.session_state.get('df_processed')       # Renombrado para claridad
target_col_main_display = st.session_state.get('original_target_column_name') # Renombrado

if df_processed_main_display is not None and not df_processed_main_display.empty and target_col_main_display:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_display_v22, col2_acf_display_v22 = st.columns(2)
    with col1_diag_display_v22: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_display_v22: 
        st.subheader("Autocorrelación")
        acf_fig_display_v22 = st.session_state.get('acf_fig')
        if acf_fig_display_v22 is not None: 
            try: st.pyplot(acf_fig_display_v22)
            except Exception as e_acf_display_v22: st.error(f"Error al mostrar ACF/PACF: {e_acf_display_v22}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main_display in df_processed_main_display.columns:
        fig_hist_display_v22 = visualization.plot_historical_data(df_processed_main_display, target_col_main_display, f"Histórico de '{target_col_main_display}'")
        if fig_hist_display_v22: st.pyplot(fig_hist_display_v22)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v22")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v22", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_v22 = len(df_processed_main_display)//2 if df_processed_main_display is not None and not df_processed_main_display.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_v22), step=1, key="ma_win_key_v22")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v22", on_change=reset_model_execution_results)
    if st.session_state.use_train_test_split:
        min_train_v22 = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_v22 = len(df_processed_main_display) - min_train_v22; max_test_v22 = max(1, max_test_v22)
        def_test_v22 = min(max(1, st.session_state.forecast_horizon), max_test_v22)
        current_test_v22 = st.session_state.get('test_split_size', def_test_v22)
        if current_test_v22 > max_test_v22 or current_test_v22 <=0 : current_test_v22 = def_test_v22
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_v22, min_value=1, max_value=max_test_v22, step=1, key="test_size_key_v22", help=f"Máx: {max_test_v22}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v22")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v22,c2ar_v22=st.columns(2); st.session_state.arima_max_p=c1ar_v22.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v22"); st.session_state.arima_max_q=c2ar_v22.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v22"); st.session_state.arima_max_d=c1ar_v22.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v22"); st.session_state.arima_max_P=c2ar_v22.number_input("max_P (est.)",0,3,st.session_state.arima_max_P,key="aP_k_v22"); st.session_state.arima_max_Q=c1ar_v22.number_input("max_Q (est.)",0,3,st.session_state.arima_max_Q,key="aQ_k_v22"); st.session_state.arima_max_D=c2ar_v22.number_input("max_D (est.)",0,2,st.session_state.arima_max_D,key="aD_k_v22")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v22")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v22"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v22"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v22"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v22")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v22_action"):
        reset_model_execution_results()
        
        # Usar las variables definidas al principio de la sección if df_processed_main_display is not None...
        df_proc_for_models_exec_btn = df_processed_main_display # Corregido
        target_col_for_models_exec_btn = target_col_main_display # Corregido

        # La validación ya se hizo antes de este if, pero una comprobación rápida no hace daño
        if df_proc_for_models_exec_btn is None or target_col_for_models_exec_btn is None or \
           target_col_for_models_exec_btn not in df_proc_for_models_exec_btn.columns: 
            st.error("🔴 Error interno: Datos preprocesados no disponibles para la ejecución de modelos."); 
        else:
            series_full_for_exec_btn = df_proc_for_models_exec_btn[target_col_for_models_exec_btn].copy(); 
            h_for_exec_btn = st.session_state.forecast_horizon; 
            s_period_for_exec_btn = st.session_state.user_seasonal_period; 
            ma_win_for_exec_btn = st.session_state.moving_avg_window
            
            train_s_for_exec_btn, test_s_for_exec_btn = series_full_for_exec_btn, pd.Series(dtype=series_full_for_exec_btn.dtype)
            if st.session_state.use_train_test_split:
                min_tr_for_exec_btn = max(5, 2*s_period_for_exec_btn+1 if s_period_for_exec_btn>1 else 5); 
                curr_test_for_exec_btn = st.session_state.get('test_split_size', 12)
                if len(series_full_for_exec_btn) > min_tr_for_exec_btn + curr_test_for_exec_btn and curr_test_for_exec_btn > 0 : 
                    train_s_for_exec_btn,test_s_for_exec_btn = forecasting_models.train_test_split_series(series_full_for_exec_btn, curr_test_for_exec_btn)
                else: 
                    st.warning(f"No fue posible el split con test_size={curr_test_for_exec_btn}. Evaluando in-sample."); 
                    st.session_state.use_train_test_split=False
                    train_s_for_exec_btn, test_s_for_exec_btn = series_full_for_exec_btn, pd.Series(dtype=series_full_for_exec_btn.dtype)
            
            st.session_state.train_series_for_plot = train_s_for_exec_btn; 
            st.session_state.test_series_for_plot = test_s_for_exec_btn
                
            with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
                model_execution_list = [] # Inicialización correcta
                
                # Construcción de model_execution_list (como en v3.19)
                model_execution_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn], "name_override": "Promedio Histórico", "type":"baseline"})
                model_execution_list.append({"func": forecasting_models.naive_forecast, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
                model_execution_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn, ma_win_for_exec_btn], "name_override": None, "type":"baseline"})
                if s_period_for_exec_btn > 1: model_execution_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn, s_period_for_exec_btn], "name_override": None, "type":"baseline"})
                holt_p_exec_v22 = {'damped_trend': st.session_state.holt_damped}
                hw_p_exec_v22 = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
                stats_configs_v22 = [("SES", {}), ("Holt", holt_p_exec_v22)]
                if s_period_for_exec_btn > 1: stats_configs_v22.append(("Holt-Winters", hw_p_exec_v22))
                for name_s_v22_item, params_s_v22_item in stats_configs_v22:
                    model_execution_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn, name_s_v22_item, s_period_for_exec_btn if name_s_v22_item=="Holt-Winters" else None, params_s_v22_item if name_s_v22_item=="Holt" else None, params_s_v22_item if name_s_v22_item=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_v22_item, "params_dict": params_s_v22_item})
                if st.session_state.run_autoarima:
                    arima_p_v22_item = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                    model_execution_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s_for_exec_btn, test_s_for_exec_btn, h_for_exec_btn, s_period_for_exec_btn, arima_p_v22_item], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_v22_item})

                # Bucle de ejecución de modelos
                for spec_item_v22_loop in model_execution_list: 
                    try:
                        fc_future_v22, ci_future_v22, rmse_v22, mae_v22, name_from_func_v22 = spec_item_v22_loop["func"](*spec_item_v22_loop["args"])
                        name_display_v22 = spec_item_v22_loop["name_override"] or name_from_func_v22
                        fc_on_test_v22 = None
                        # --- INICIO: Lógica para fc_on_test_v22 ---
                        if st.session_state.use_train_test_split and not test_s_for_exec_btn.empty and not any(err in name_display_v22 for err in ["Error","Insuf","Inválido","FALLÓ"]):
                            if spec_item_v22_loop["type"] == "baseline":
                                if "Promedio Histórico" in name_display_v22 and not train_s_for_exec_btn.empty: fc_on_test_v22 = np.full(len(test_s_for_exec_btn), train_s_for_exec_btn.mean())
                                elif "Ingénuo" in name_display_v22 and not train_s_for_exec_btn.empty: fc_on_test_v22 = np.full(len(test_s_for_exec_btn), train_s_for_exec_btn.iloc[-1])
                                elif "Promedio Móvil" in name_display_v22 and not train_s_for_exec_btn.empty and len(train_s_for_exec_btn) >= ma_win_for_exec_btn : fc_on_test_v22 = np.full(len(test_s_for_exec_btn), train_s_for_exec_btn.iloc[-ma_win_for_exec_btn:].mean()) # ma_win_for_exec_btn debe estar definido
                                elif "Estacional Ingénuo" in name_display_v22 and not train_s_for_exec_btn.empty and len(train_s_for_exec_btn) >= s_period_for_exec_btn: # s_period_for_exec_btn debe estar definido
                                    temp_fc_test_v22_calc = np.zeros(len(test_s_for_exec_btn)); 
                                    for i_fc_v22_calc in range(len(test_s_for_exec_btn)): temp_fc_test_v22_calc[i_fc_v22_calc] = train_s_for_exec_btn.iloc[len(train_s_for_exec_btn) - s_period_for_exec_btn + (i_fc_v22_calc % s_period_for_exec_btn)]
                                    fc_on_test_v22 = temp_fc_test_v22_calc
                            elif spec_item_v22_loop["type"] == "statsmodels":
                                sm_name_test_v22_calc = spec_item_v22_loop["model_short_name"]; sm_params_test_v22_calc = spec_item_v22_loop["params_dict"]; sm_fit_test_v22_calc = None
                                try: 
                                    if sm_name_test_v22_calc=="SES": sm_fit_test_v22_calc=forecasting_models.SimpleExpSmoothing(train_s_for_exec_btn,initialization_method="estimated").fit()
                                    elif sm_name_test_v22_calc=="Holt": sm_fit_test_v22_calc=forecasting_models.Holt(train_s_for_exec_btn,damped_trend=sm_params_test_v22_calc.get('damped_trend',False),initialization_method="estimated").fit()
                                    elif sm_name_test_v22_calc=="Holt-Winters": sm_fit_test_v22_calc=forecasting_models.ExponentialSmoothing(train_s_for_exec_btn,trend=sm_params_test_v22_calc.get('trend'),seasonal=sm_params_test_v22_calc.get('seasonal'),seasonal_periods=s_period_for_exec_btn,damped_trend=sm_params_test_v22_calc.get('damped_trend',False),use_boxcox=sm_params_test_v22_calc.get('use_boxcox',False),initialization_method="estimated").fit()
                                    if sm_fit_test_v22_calc: fc_on_test_v22 = sm_fit_test_v22_calc.forecast(len(test_s_for_exec_btn)).values
                                except Exception as e_sm_fc_test_v22_calc: st.caption(f"Warn fc_test {name_display_v22}: {e_sm_fc_test_v22_calc}")
                            elif spec_item_v22_loop["type"] == "autoarima":
                                arima_cfg_test_v22_calc = spec_item_v22_loop["arima_params_dict"]; arima_fit_test_v22_calc = None
                                try: 
                                    arima_fit_test_v22_calc = forecasting_models.pm.auto_arima(train_s_for_exec_btn,max_p=arima_cfg_test_v22_calc.get('max_p',3),max_q=arima_cfg_test_v22_calc.get('max_q',3),m=s_period_for_exec_btn if s_period_for_exec_btn>1 else 1,seasonal=s_period_for_exec_btn>1,suppress_warnings=True,error_action='ignore',stepwise=True, trace=False, **{'max_d':arima_cfg_test_v22_calc.get('max_d',2), 'max_P':arima_cfg_test_v22_calc.get('max_P',1), 'max_Q':arima_cfg_test_v22_calc.get('max_Q',1), 'max_D':arima_cfg_test_v22_calc.get('max_D',1)})
                                    if arima_fit_test_v22_calc: fc_on_test_v22 = arima_fit_test_v22_calc.predict(n_periods=len(test_s_for_exec_btn))
                                except Exception as e_arima_test_fc_v22_calc: st.caption(f"Warn fc_test {name_display_v22}: {e_arima_test_fc_v22_calc}")
                        # --- FIN: Lógica para fc_on_test_v22 ---
                        st.session_state.model_results.append({'name':name_display_v22,'rmse':rmse_v22,'mae':mae_v22,'forecast_future':fc_future_v22,'conf_int_future':ci_future_v22,'forecast_on_test':fc_on_test_v22})
                    except Exception as e_model_v22_loop: st.warning(f"Error procesando {spec_item_v22_loop.get('name_override',spec_item_v22_loop['func'].__name__)}: {str(e_model_v22_loop)[:150]}")
                
                # --- DEBUG: Mostrar contenido de st.session_state.model_results ---
                st.write("--- DEBUG: Contenido de st.session_state.model_results ---")
                if isinstance(st.session_state.model_results, list) and st.session_state.model_results:
                    for i_debug_v22, res_debug_v22 in enumerate(st.session_state.model_results):
                        st.write(f"Modelo {i_debug_v22+1}:")
                        st.json(res_debug_v22) 
                else: st.write("st.session_state.model_results está vacío o no es una lista.")
                st.write(f"Horizonte (h_for_exec_btn) usado para filtrar: {h_for_exec_btn}") 
                st.write("--- FIN DEBUG ---")

            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_results_final_v22 = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and isinstance(r.get('forecast_future'), (np.ndarray, list)) and len(r.get('forecast_future'))==h_for_exec_btn]
            if valid_results_final_v22: st.session_state.best_model_name_auto = min(valid_results_final_v22, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido de los resultados válidos."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_v22_render = st.session_state.get('df_processed')
target_col_for_tabs_v22_render = st.session_state.get('original_target_column_name')
model_results_exist_v22_render = st.session_state.get('model_results')

if df_proc_for_tabs_v22_render is not None and not df_proc_for_tabs_v22_render.empty and \
   target_col_for_tabs_v22_render and \
   model_results_exist_v22_render is not None and isinstance(model_results_exist_v22_render, list) and \
   len(model_results_exist_v22_render) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_v22, tab_comp_v22, tab_man_v22, tab_diag_v22 = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    historical_series_for_tabs_v22_plot = None
    if target_col_for_tabs_v22_render in df_proc_for_tabs_v22_render.columns:
        historical_series_for_tabs_v22_plot = df_proc_for_tabs_v22_render[target_col_for_tabs_v22_render]

    with tab_rec_v22: # Pestaña Modelo Recomendado
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "RECOMENDADO" DE LA VERSIÓN v3.19)
        pass 

    with tab_comp_v22: # Pestaña Comparación
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "COMPARACIÓN" DE LA VERSIÓN v3.19)
        pass 

    with tab_man_v22: # Pestaña Explorar Manualmente
        # ... (COPIA AQUÍ LA LÓGICA COMPLETA DE LA PESTAÑA "EXPLORAR" DE LA VERSIÓN v3.19)
        pass
        
    with tab_diag_v22: # Pestaña Diagnóstico
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor.\n- Intervalos de Predicción: Incertidumbre.\n- Calidad de Datos: Crucial.")

# --- Mensajes si las pestañas no se muestran (Sección final CORREGIDA) ---
else: 
    df_loaded_is_present_final_v22_else = st.session_state.get('df_loaded') is not None
    df_processed_value_final_v22_else = st.session_state.get('df_processed')
    model_results_list_final_v22_else = st.session_state.get('model_results')
    df_processed_is_empty_or_none_v22_else = (df_processed_value_final_v22_else is None or (isinstance(df_processed_value_final_v22_else, pd.DataFrame) and df_processed_value_final_v22_else.empty))
    model_results_is_empty_or_none_v22_else = (model_results_list_final_v22_else is None or (isinstance(model_results_list_final_v22_else, list) and not model_results_list_final_v22_else))

    if uploaded_file is None and not df_loaded_is_present_final_v22_else: 
        st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
    elif df_loaded_is_present_final_v22_else and df_processed_is_empty_or_none_v22_else: 
        st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
    elif df_loaded_is_present_final_v22_else and not df_processed_is_empty_or_none_v22_else and model_results_is_empty_or_none_v22_else:
        st.info("Datos preprocesados. Por favor, genere los modelos para ver los resultados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.22")