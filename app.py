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
    for key in keys_to_reset:
        if key in st.session_state: del st.session_state[key]
    st.session_state.df_loaded = None
    init_session_state() 

def reset_sidebar_config_dependent_state():
    st.session_state.df_processed = None 
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None
    st.session_state.train_series_for_plot = None; st.session_state.test_series_for_plot = None

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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v13", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

if st.session_state.get('df_loaded') is not None:
    df_input_sb = st.session_state.df_loaded.copy() # sb para sidebar
    date_col_options_sb = df_input_sb.columns.tolist()
    dt_col_guess_idx_sb = 0
    if date_col_options_sb: # Evitar error si no hay columnas
        for i, col in enumerate(date_col_options_sb):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb = i; break
    
    # Manejar caso donde la lista de opciones es vacía o la selección previa no existe
    sel_date_idx_sb = 0
    if date_col_options_sb:
        sel_date_idx_sb = date_col_options_sb.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb else dt_col_guess_idx_sb
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb, index=sel_date_idx_sb, key="date_sel_key_v13")

    value_col_options_sb = [col for col in df_input_sb.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb = 0
    if value_col_options_sb: # Evitar error si no hay opciones
        for i, col in enumerate(value_col_options_sb):
            if pd.api.types.is_numeric_dtype(df_input_sb[col].dropna()): val_col_guess_idx_sb = i; break
    
    sel_val_idx_sb = 0
    if value_col_options_sb:
        sel_val_idx_sb = value_col_options_sb.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb else val_col_guess_idx_sb
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb, index=sel_val_idx_sb, key="val_sel_key_v13")
    
    freq_map_sb = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb.keys()), key="freq_sel_key_v13", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb = freq_map_sb[freq_label_sb]

    imp_list_sb = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb, index=1, key="imp_sel_key_v13", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb = None if imp_label_sb == "No imputar" else imp_label_sb.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v13"):
        st.session_state.df_processed = None; reset_model_related_state() # Resetear antes de preprocesar
        date_col_btn_sb = st.session_state.get('selected_date_col'); value_col_btn_sb = st.session_state.get('selected_value_col'); valid_btn_sb = True
        if not date_col_btn_sb or date_col_btn_sb not in df_input_sb.columns: st.sidebar.error("Seleccione columna de fecha válida."); valid_btn_sb=False
        if not value_col_btn_sb or value_col_btn_sb not in df_input_sb.columns: st.sidebar.error("Seleccione columna de valor válida."); valid_btn_sb=False
        elif valid_btn_sb and not pd.api.types.is_numeric_dtype(df_input_sb[value_col_btn_sb].dropna()): st.sidebar.error(f"Columna '{value_col_btn_sb}' no es numérica."); valid_btn_sb=False
        
        if valid_btn_sb:
            with st.spinner("Preprocesando..."): proc_df_res_sb,msg_raw_sb = data_handler.preprocess_data(df_input_sb.copy(),date_col_btn_sb,value_col_btn_sb,desired_freq_sb,imp_code_sb)
            msg_disp_sb = msg_raw_sb; 
            if msg_raw_sb: 
                if "MS" in msg_raw_sb: msg_disp_sb=msg_raw_sb.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_sb: msg_disp_sb=msg_raw_sb.replace(" D."," D (Diario).")
                elif msg_raw_sb.endswith("D"): msg_disp_sb=msg_raw_sb.replace("D", "D (Diario)")
            if proc_df_res_sb is not None and not proc_df_res_sb.empty:
                st.session_state.df_processed=proc_df_res_sb; st.session_state.original_target_column_name=value_col_btn_sb; st.success(f"Preproc. OK. {msg_disp_sb}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_res_sb,value_col_btn_sb)
                if not proc_df_res_sb.empty:
                    s_acf_sb=proc_df_res_sb[value_col_btn_sb];l_acf_sb=min(len(s_acf_sb)//2-1,60)
                    if l_acf_sb > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_sb,l_acf_sb,value_col_btn_sb)
                    else: st.session_state.acf_fig=None
                    _,auto_s_val_sb=data_handler.get_series_frequency_and_period(proc_df_res_sb.index)
                    st.session_state.auto_seasonal_period=auto_s_val_sb
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s_val_sb: st.session_state.user_seasonal_period=auto_s_val_sb
            else: st.error(f"Fallo preproc: {msg_raw_sb or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main_view = st.session_state.get('df_processed')
target_col_main_view = st.session_state.get('original_target_column_name')

if df_processed_main_view is not None and not df_processed_main_view.empty and target_col_main_view:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_view, col2_acf_view = st.columns(2)
    with col1_diag_view: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_view: 
        st.subheader("Autocorrelación")
        acf_fig_view = st.session_state.get('acf_fig')
        if acf_fig_view: 
            try: st.pyplot(acf_fig_view)
            except Exception as e_acf_view: st.error(f"Error al mostrar ACF/PACF: {e_acf_view}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main_view in df_processed_main_view.columns:
        fig_hist_view = visualization.plot_historical_data(df_processed_main_view, target_col_main_view, f"Histórico de '{target_col_main_view}'")
        if fig_hist_view: st.pyplot(fig_hist_view)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v13")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v13", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_cfg_v13 = len(df_processed_main_view)//2 if df_processed_main_view is not None and not df_processed_main_view.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_cfg_v13), step=1, key="ma_win_key_v13")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v13", on_change=reset_model_related_state)
    if st.session_state.use_train_test_split:
        min_train_cfg_v13 = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_cfg_v13 = len(df_processed_main_view) - min_train_cfg_v13; max_test_cfg_v13 = max(1, max_test_cfg_v13)
        def_test_cfg_v13 = min(max(1, st.session_state.forecast_horizon), max_test_cfg_v13)
        current_test_cfg_v13 = st.session_state.get('test_split_size', def_test_cfg_v13)
        if current_test_cfg_v13 > max_test_cfg_v13 or current_test_cfg_v13 <=0 : current_test_cfg_v13 = def_test_cfg_v13
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_cfg_v13, min_value=1, max_value=max_test_cfg_v13, step=1, key="test_size_key_v13", help=f"Máx: {max_test_cfg_v13}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v13")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v13,c2ar_v13=st.columns(2); st.session_state.arima_max_p=c1ar_v13.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v13"); st.session_state.arima_max_q=c2ar_v13.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v13"); st.session_state.arima_max_d=c1ar_v13.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v13"); st.session_state.arima_max_P=c2ar_v13.number_input("max_P",0,3,st.session_state.arima_max_P,key="aP_k_v13"); st.session_state.arima_max_Q=c1ar_v13.number_input("max_Q",0,3,st.session_state.arima_max_Q,key="aQ_k_v13"); st.session_state.arima_max_D=c2ar_v13.number_input("max_D",0,2,st.session_state.arima_max_D,key="aD_k_v13")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v13")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v13"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v13"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v13"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v13")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v13"):
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None; st.session_state.selected_model_for_manual_explore = None
        
        series_full_run = df_processed_display_main[target_col_main_view].copy(); h_run = st.session_state.forecast_horizon; s_period_run = st.session_state.user_seasonal_period; ma_win_run = st.session_state.moving_avg_window
        train_s_run, test_s_run = series_full_run, pd.Series(dtype=series_full_run.dtype)
        if st.session_state.use_train_test_split:
            min_tr_run = max(5, 2*s_period_run+1 if s_period_run>1 else 5); curr_test_run = st.session_state.get('test_split_size', 12)
            if len(series_full_run) > min_tr_run + curr_test_run and curr_test_run > 0 : train_s_run,test_s_run = forecasting_models.train_test_split_series(series_full_run, curr_test_run)
            else: st.warning(f"No split con test_size={curr_test_run}."); st.session_state.use_train_test_split=False
        st.session_state.train_series_for_plot = train_s_run; st.session_state.test_series_for_plot = test_s_run
            
        with st.spinner("Calculando modelos..."):
            model_execution_list = []
            model_execution_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_run, test_s_run, h_run], "name_override": "Promedio Histórico", "type":"baseline"})
            model_execution_list.append({"func": forecasting_models.naive_forecast, "args": [train_s_run, test_s_run, h_run], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
            model_execution_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s_run, test_s_run, h_run, ma_win_run], "name_override": None, "type":"baseline"}) # Nombre se genera en la función
            if s_period_run > 1: model_execution_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s_run, test_s_run, h_run, s_period_run], "name_override": None, "type":"baseline"}) # Nombre se genera en la función
            
            holt_p_config = {'damped_trend': st.session_state.holt_damped}
            hw_p_config = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
            stats_model_configurations = [("SES", {}), ("Holt", holt_p_config)]
            if s_period_run > 1: stats_model_configurations.append(("Holt-Winters", hw_p_config))
            for name_s_config, params_s_config in stats_model_configurations:
                model_execution_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s_run, test_s_run, h_run, name_s_config, s_period_run if name_s_config=="Holt-Winters" else None, params_s_config if name_s_config=="Holt" else None, params_s_config if name_s_config=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_config, "params_dict": params_s_config})
            
            if st.session_state.run_autoarima:
                arima_p_config = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                model_execution_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s_run, test_s_run, h_run, s_period_run, arima_p_config], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_config})

            for spec_run_item in model_execution_list:
                try:
                    fc_future, ci_future, rmse_test, mae_test, model_name_from_func = spec_run_item["func"](*spec_run_item["args"])
                    display_name = spec_run_item["name_override"] or model_name_from_func
                    
                    fc_on_test_values = None
                    if st.session_state.use_train_test_split and not test_s_run.empty and not any(err_str in display_name for err_str in ["Error", "Insuf", "Inválido", "FALLÓ"]):
                        if spec_run_item["type"] == "baseline":
                            if "Promedio Histórico" in display_name and not train_s_run.empty: fc_on_test_values = np.full(len(test_s_run), train_s_run.mean())
                            elif "Ingénuo" in display_name and not train_s_run.empty: fc_on_test_values = np.full(len(test_s_run), train_s_run.iloc[-1])
                            elif "Promedio Móvil" in display_name and not train_s_run.empty and len(train_s_run) >= ma_win_run: fc_on_test_values = np.full(len(test_s_run), train_s_run.iloc[-ma_win_run:].mean())
                            elif "Estacional Ingénuo" in display_name and not train_s_run.empty and len(train_s_run) >= s_period_run:
                                temp_fc_test = np.zeros(len(test_s_run)); 
                                for i_test_fc_val in range(len(test_s_run)): temp_fc_test[i_test_fc_val] = train_s_run.iloc[len(train_s_run) - s_period_run + (i_test_fc_val % s_period_run)]
                                fc_on_test_values = temp_fc_test
                        elif spec_run_item["type"] == "statsmodels":
                            sm_name = spec_run_item["model_short_name"]; sm_params = spec_run_item["params_dict"]; sm_fit_test = None
                            try:
                                if sm_name=="SES": sm_fit_test=forecasting_models.SimpleExpSmoothing(train_s_run,initialization_method="estimated").fit()
                                elif sm_name=="Holt": sm_fit_test=forecasting_models.Holt(train_s_run,damped_trend=sm_params.get('damped_trend',False),initialization_method="estimated").fit()
                                elif sm_name=="Holt-Winters": sm_fit_test=forecasting_models.ExponentialSmoothing(train_s_run,trend=sm_params.get('trend'),seasonal=sm_params.get('seasonal'),seasonal_periods=s_period_run,damped_trend=sm_params.get('damped_trend',False),use_boxcox=sm_params.get('use_boxcox',False),initialization_method="estimated").fit()
                                if sm_fit_test: fc_on_test_values = sm_fit_test.forecast(len(test_s_run)).values
                            except Exception as e_sm_test_fc: st.caption(f"Warn fc_test {display_name}: {e_sm_test_fc}")
                        elif spec_run_item["type"] == "autoarima":
                            arima_cfg_test = spec_run_item["arima_params_dict"]; arima_fit_test = None
                            try:
                                arima_fit_test = forecasting_models.pm.auto_arima(train_s_run,max_p=arima_cfg_test.get('max_p',3),max_q=arima_cfg_test.get('max_q',3),m=s_period_run if s_period_run>1 else 1,seasonal=s_period_run>1,suppress_warnings=True,error_action='ignore',stepwise=True, trace=False) # Simplificado
                                if arima_fit_test: fc_on_test_values = arima_fit_test.predict(n_periods=len(test_s_run))
                            except Exception as e_arima_test_fc: st.caption(f"Warn fc_test {display_name}: {e_arima_test_fc}")
                    st.session_state.model_results.append({'name':display_name,'rmse':rmse_test,'mae':mae_test,'forecast_future':fc_future,'conf_int_future':ci_future,'forecast_on_test':fc_on_test_values})
                except Exception as e_model_run_item: st.warning(f"Error {spec_run_item.get('name_override',spec_run_item['func'].__name__)}: {e_model_run_item}")
            
            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_results_final = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h_run]
            if valid_results_final: st.session_state.best_model_name_auto = min(valid_results_final, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_view_final = st.session_state.get('df_processed')
target_col_for_tabs_view_final = st.session_state.get('original_target_column_name')
model_results_exist_view_final = st.session_state.get('model_results')

if df_proc_for_tabs_view_final is not None and not df_proc_for_tabs_view_final.empty and \
   target_col_for_tabs_view_final and \
   model_results_exist_view_final is not None and isinstance(model_results_exist_view_final, list) and len(model_results_exist_view_final) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_view_final, tab_comp_view_final, tab_manual_view_final, tab_diag_view_final = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    historical_series_for_tabs_plot_final = None
    if target_col_for_tabs_view_final in df_proc_for_tabs_view_final.columns:
        historical_series_for_tabs_plot_final = df_proc_for_tabs_view_final[target_col_for_tabs_view_final]

    with tab_rec_view_final:
        best_model_rec_name_final = st.session_state.best_model_name_auto
        if best_model_rec_name_final and "Error" not in best_model_rec_name_final and "FALLÓ" not in best_model_rec_name_final and historical_series_for_tabs_plot_final is not None:
            st.subheader(f"Modelo Recomendado: {best_model_rec_name_final}")
            model_data_rec_final = next((item for item in st.session_state.model_results if item["name"] == best_model_rec_name_final), None)
            if model_data_rec_final:
                final_df_r_view, fc_s_r_view, pi_df_r_view = prepare_forecast_display_data(model_data_rec_final, historical_series_for_tabs_plot_final.index, st.session_state.forecast_horizon)
                if final_df_r_view is not None and fc_s_r_view is not None and not fc_s_r_view.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_rec_final.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); 
                        fc_test_r_s_final = pd.Series(model_data_rec_final['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_rec_final['forecast_on_test'],np.ndarray) and len(model_data_rec_final['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_rec_final.get('forecast_on_test'); 
                        if fc_test_r_s_final is not None: fig_vr_final=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_r_s_final,best_model_rec_name_final,target_col_for_tabs_view_final); st.pyplot(fig_vr_final) if fig_vr_final else st.caption("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fr_final=visualization.plot_final_forecast(historical_series_for_tabs_plot_final,fc_s_r_view,pi_df_r_view,best_model_rec_name_final,target_col_for_tabs_view_final); st.pyplot(fig_fr_final) if fig_fr_final else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_r_view.style.format("{:.2f}")); dl_key_rec_final_btn = f"dl_rec_{best_model_rec_name_final[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF_final"; st.download_button(f"📥 Descargar ({best_model_rec_name_final})",to_excel(final_df_r_view),f"fc_{best_model_rec_name_final.replace(' ','_')}.xlsx",key=dl_key_rec_final_btn)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model_rec_name_final,st.session_state.data_diagnosis_report,True,(pi_df_r_view is not None and not pi_df_r_view.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_view_final))
                else: st.warning(f"No se pudo preparar visualización para '{best_model_rec_name_final}'.")
            else: st.info(f"Datos del modelo '{best_model_rec_name_final}' no encontrados.")
        elif not historical_series_for_tabs_plot_final : st.warning("Datos históricos no disponibles.")
        else: st.info("No se ha determinado un modelo recomendado. Genere los modelos o verifique errores.")

    with tab_comp_final:
        st.subheader("Comparación de Modelos")
        metrics_list_comp_final_tab = [{'Modelo': r.get('name','N/A'), 'RMSE': r.get('rmse'), 'MAE': r.get('mae')} for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None]
        if metrics_list_comp_final_tab:
            metrics_df_comp_final_tab = pd.DataFrame(metrics_list_comp_final_tab).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_final_tab(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_comp_final_tab.style.format({'RMSE':"{:.3f}", 'MAE':"{:.3f}"}).apply(highlight_best_final_tab, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Sugerido: **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay métricas de modelos para mostrar.")

    with tab_man_final:
        st.subheader("Explorar Modelo Manualmente")
        valid_manual_models_final_list_tab = [r['name'] for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r.get('rmse'))]
        if valid_manual_models_final_list_tab and historical_series_for_tabs_plot_final is not None:
            sel_idx_man_final_val_tab = 0
            if st.session_state.selected_model_for_manual_explore in valid_manual_models_final_list_tab: sel_idx_man_final_val_tab = valid_manual_models_final_list_tab.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in valid_manual_models_final_list_tab: sel_idx_man_final_val_tab = valid_manual_models_final_list_tab.index(st.session_state.best_model_name_auto)
            st.session_state.selected_model_for_manual_explore = st.selectbox("Modelo:", valid_manual_models_final_list_tab, index=sel_idx_man_final_val_tab, key="man_sel_key_v12_final_tab")
            model_data_man_final_view_tab = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)
            if model_data_man_final_view_tab:
                final_df_m_final_view_tab, fc_s_m_final_view_tab, pi_df_m_final_view_tab = prepare_forecast_display_data(model_data_man_final_view_tab, historical_series_for_tabs_plot_final.index, st.session_state.forecast_horizon)
                if final_df_m_final_view_tab is not None and fc_s_m_final_view_tab is not None and not fc_s_m_final_view_tab.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_man_final_view_tab.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); 
                        fc_test_m_s_final_tab = pd.Series(model_data_man_final_view_tab['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_man_final_view_tab['forecast_on_test'],np.ndarray) and len(model_data_man_final_view_tab['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_man_final_view_tab.get('forecast_on_test');
                        if fc_test_m_s_final_tab is not None: fig_vm_final_tab=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_m_s_final_tab,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final_view); st.pyplot(fig_vm_final_tab) if fig_vm_final_tab else st.caption("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fm_final_tab=visualization.plot_final_forecast(historical_series_for_tabs_final_plot,fc_s_m_final_view_tab,pi_df_m_final_view_tab,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final_view); st.pyplot(fig_fm_final_tab) if fig_fm_final_tab else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_m_final_view_tab.style.format("{:.2f}")); dl_key_m_final_tab = f"dl_man_{st.session_state.selected_model_for_manual_explore[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF_final"; st.download_button(f"📥 Descargar ({st.session_state.selected_model_for_manual_explore})",to_excel(final_df_m_final_view_tab),f"fc_man_{target_col_for_tabs_final_view}.xlsx",key=dl_key_m_final_tab)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(st.session_state.selected_model_for_manual_explore,st.session_state.data_diagnosis_report,True,(pi_df_m_final_view_tab is not None and not pi_df_m_final_view_tab.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_final_view))
                else: st.warning(f"No se pudo preparar visualización para '{st.session_state.selected_model_for_manual_explore}'.")
        elif not historical_series_for_tabs_plot_final : st.warning("Datos históricos no disponibles.")
        else: st.warning("No hay modelos válidos para exploración. Genere los modelos.")
        
    with tab_diag_final:
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

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.12")