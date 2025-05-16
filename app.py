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
    for key_to_del in keys_to_reset: # Usar un nombre de variable diferente para evitar conflicto
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

# --- Sección 1: Carga y Preprocesamiento de Datos (Sidebar) ---
st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v14", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sidebar_v14 = st.session_state.get('df_loaded') # Usar una variable local para claridad

if df_input_sidebar_v14 is not None:
    date_col_options_sb_v14 = df_input_sidebar_v14.columns.tolist()
    dt_col_guess_idx_sb_v14 = 0
    if date_col_options_sb_v14:
        for i, col in enumerate(date_col_options_sb_v14):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx_sb_v14 = i; break
    
    sel_date_idx_sb_v14 = 0
    if date_col_options_sb_v14 : # Solo si hay opciones
        sel_date_idx_sb_v14 = date_col_options_sb_v14.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb_v14 else dt_col_guess_idx_sb_v14
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb_v14, index=sel_date_idx_sb_v14, key="date_sel_key_v14")

    value_col_options_sb_v14 = [col for col in df_input_sidebar_v14.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx_sb_v14 = 0
    if value_col_options_sb_v14:
        for i, col in enumerate(value_col_options_sb_v14):
            if pd.api.types.is_numeric_dtype(df_input_sidebar_v14[col].dropna()): val_col_guess_idx_sb_v14 = i; break
    
    sel_val_idx_sb_v14 = 0
    if value_col_options_sb_v14:
        sel_val_idx_sb_v14 = value_col_options_sb_v14.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb_v14 else val_col_guess_idx_sb_v14
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb_v14, index=sel_val_idx_sb_v14, key="val_sel_key_v14")
    
    freq_map_sb_v14 = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label_sb_v14 = st.sidebar.selectbox("Frecuencia:", options=list(freq_map_sb_v14.keys()), key="freq_sel_key_v14", on_change=reset_sidebar_config_dependent_state)
    desired_freq_sb_v14 = freq_map_sb_v14[freq_label_sb_v14]

    imp_list_sb_v14 = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label_sb_v14 = st.sidebar.selectbox("Imputación Faltantes:", imp_list_sb_v14, index=1, key="imp_sel_key_v14", on_change=reset_sidebar_config_dependent_state)
    imp_code_sb_v14 = None if imp_label_sb_v14 == "No imputar" else imp_label_sb_v14.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v14"): # Esta es la línea 145 en tu última captura
        st.session_state.df_processed = None; reset_sidebar_config_dependent_state() # Línea 146 (antes reset_model_related_state)
        
        # Las siguientes líneas (antes 147 en adelante) deben estar indentadas DENTRO de este if del botón
        date_col_btn_v14_val = st.session_state.get('selected_date_col')
        value_col_btn_v14_val = st.session_state.get('selected_value_col')
        valid_btn_v14_check = True
        
        if not date_col_btn_v14_val or date_col_btn_v14_val not in df_input_sidebar_v14.columns: 
            st.sidebar.error("Seleccione columna de fecha válida.")
            valid_btn_v14_check=False
        if not value_col_btn_v14_val or value_col_btn_v14_val not in df_input_sidebar_v14.columns: 
            st.sidebar.error("Seleccione columna de valor válida.")
            valid_btn_v14_check=False
        elif valid_btn_v14_check and not pd.api.types.is_numeric_dtype(df_input_sidebar_v14[value_col_btn_v14_val].dropna()): 
            st.sidebar.error(f"Columna '{value_col_btn_v14_val}' no es numérica.")
            valid_btn_v14_check=False
        
        if valid_btn_v14_check:
            with st.spinner("Preprocesando..."): 
                proc_df_res_v14,msg_raw_v14 = data_handler.preprocess_data(
                    df_input_sidebar_v14.copy(),
                    date_col_btn_v14_val,
                    value_col_btn_v14_val,
                    desired_freq_sb_v14,
                    imp_code_sb_v14
                )
            msg_disp_v14 = msg_raw_v14; 
            if msg_raw_v14: 
                if "MS" in msg_raw_v14: msg_disp_v14=msg_raw_v14.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw_v14: msg_disp_v14=msg_raw_v14.replace(" D."," D (Diario).")
                elif msg_raw_v14.endswith("D"): msg_disp_v14=msg_raw_v14.replace("D", "D (Diario)")
            
            if proc_df_res_v14 is not None and not proc_df_res_v14.empty:
                st.session_state.df_processed=proc_df_res_v14
                st.session_state.original_target_column_name=value_col_btn_v14_val
                st.success(f"Preproc. OK. {msg_disp_v14}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df_res_v14,value_col_btn_v14_val)
                if not proc_df_res_v14.empty:
                    s_acf_v14=proc_df_res_v14[value_col_btn_v14_val];l_acf_v14=min(len(s_acf_v14)//2-1,60)
                    if l_acf_v14 > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf_v14,l_acf_v14,value_col_btn_v14_val)
                    else: st.session_state.acf_fig=None
                    _,auto_s_v14_val=data_handler.get_series_frequency_and_period(proc_df_res_v14.index)
                    st.session_state.auto_seasonal_period=auto_s_v14_val
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s_v14_val: 
                        st.session_state.user_seasonal_period=auto_s_v14_val
            else: 
                st.error(f"Fallo preproc: {msg_raw_v14 or 'DataFrame vacío.'}")
                st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main_display_v14 = st.session_state.get('df_processed')
target_col_main_display_v14 = st.session_state.get('original_target_column_name')

if df_processed_main_display_v14 is not None and not df_processed_main_display_v14.empty and target_col_main_display_v14:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_v14, col2_acf_v14 = st.columns(2)
    with col1_diag_v14: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v14: 
        st.subheader("Autocorrelación")
        acf_fig_v14 = st.session_state.get('acf_fig')
        if acf_fig_v14: 
            try: st.pyplot(acf_fig_v14)
            except Exception as e_acf_v14: st.error(f"Error al mostrar ACF/PACF: {e_acf_v14}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main_display_v14 in df_processed_main_display_v14.columns:
        fig_hist_v14 = visualization.plot_historical_data(df_processed_main_display_v14, target_col_main_display_v14, f"Histórico de '{target_col_main_display_v14}'")
        if fig_hist_v14: st.pyplot(fig_hist_v14)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v14")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v14", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_v14 = len(df_processed_main_display_v14)//2 if df_processed_main_display_v14 is not None and not df_processed_main_display_v14.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_v14), step=1, key="ma_win_key_v14")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v14", on_change=reset_model_related_state) # Resetear modelos si cambia esto
    if st.session_state.use_train_test_split:
        min_train_v14 = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_v14 = len(df_processed_main_display_v14) - min_train_v14; max_test_v14 = max(1, max_test_v14)
        def_test_v14 = min(max(1, st.session_state.forecast_horizon), max_test_v14)
        current_test_v14 = st.session_state.get('test_split_size', def_test_v14)
        if current_test_v14 > max_test_v14 or current_test_v14 <=0 : current_test_v14 = def_test_v14
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_v14, min_value=1, max_value=max_test_v14, step=1, key="test_size_key_v14", help=f"Máx: {max_test_v14}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v14")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v14,c2ar_v14=st.columns(2); st.session_state.arima_max_p=c1ar_v14.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v14"); st.session_state.arima_max_q=c2ar_v14.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v14"); st.session_state.arima_max_d=c1ar_v14.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v14"); st.session_state.arima_max_P=c2ar_v14.number_input("max_P (est.)",0,3,st.session_state.arima_max_P,key="aP_k_v14"); st.session_state.arima_max_Q=c1ar_v14.number_input("max_Q (est.)",0,3,st.session_state.arima_max_Q,key="aQ_k_v14"); st.session_state.arima_max_D=c2ar_v14.number_input("max_D (est.)",0,2,st.session_state.arima_max_D,key="aD_k_v14")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v14")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v14"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v14"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v14"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v14")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v14"):
        reset_model_execution_results()
        
        series_full_models = df_processed_main_display_v14[target_col_main_display_v14].copy(); 
        h_models = st.session_state.forecast_horizon; 
        s_period_models = st.session_state.user_seasonal_period; 
        ma_win_models = st.session_state.moving_avg_window
        
        train_s_models, test_s_models = series_full_models, pd.Series(dtype=series_full_models.dtype)
        if st.session_state.use_train_test_split:
            min_tr_models = max(5, 2*s_period_models+1 if s_period_models>1 else 5); 
            curr_test_models = st.session_state.get('test_split_size', 12)
            if len(series_full_models) > min_tr_models + curr_test_models and curr_test_models > 0 : 
                train_s_models,test_s_models = forecasting_models.train_test_split_series(series_full_models, curr_test_models)
            else: 
                st.warning(f"No fue posible el split con test_size={curr_test_models}. Evaluando in-sample."); 
                st.session_state.use_train_test_split=False
                train_s_models, test_s_models = series_full_models, pd.Series(dtype=series_full_models.dtype)
        
        st.session_state.train_series_for_plot = train_s_models; 
        st.session_state.test_series_for_plot = test_s_models
            
        with st.spinner("Calculando modelos..."):
            model_exec_list_final = []
            # Baselines
            model_exec_list_final.append({"func": forecasting_models.historical_average_forecast, "args": [train_s_models, test_s_models, h_models], "name_override": "Promedio Histórico", "type":"baseline"})
            model_exec_list_final.append({"func": forecasting_models.naive_forecast, "args": [train_s_models, test_s_models, h_models], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
            model_exec_list_final.append({"func": forecasting_models.moving_average_forecast, "args": [train_s_models, test_s_models, h_models, ma_win_models], "name_override": None, "type":"baseline"})
            if s_period_models > 1: model_exec_list_final.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s_models, test_s_models, h_models, s_period_models], "name_override": None, "type":"baseline"})
            
            # Statsmodels
            holt_p_exec_final = {'damped_trend': st.session_state.holt_damped}
            hw_p_exec_final = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
            stats_configs_final = [("SES", {}), ("Holt", holt_p_exec_final)]
            if s_period_models > 1: stats_configs_final.append(("Holt-Winters", hw_p_exec_final))
            for name_s_final, params_s_final in stats_configs_final:
                model_exec_list_final.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s_models, test_s_models, h_models, name_s_final, s_period_models if name_s_final=="Holt-Winters" else None, params_s_final if name_s_final=="Holt" else None, params_s_final if name_s_final=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_final, "params_dict": params_s_final})
            
            if st.session_state.run_autoarima:
                arima_p_final = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                model_exec_list_final.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s_models, test_s_models, h_models, s_period_models, arima_p_final], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_final})

            for spec_final_item in model_exec_list_final:
                try:
                    fc_res_item, ci_res_item, rmse_res_item, mae_res_item, name_res_item = spec_final_item["func"](*spec_final_item["args"])
                    name_disp_item = spec_final_item["name_override"] or name_res_item
                    fc_on_test_item = None
                    if st.session_state.use_train_test_split and not test_s_models.empty and not any(err_str in name_disp_item for err_str in ["Error","Insuf","Inválido","FALLÓ"]):
                        # Lógica para fc_on_test (simplificada, necesita robustez para modelos complejos)
                        if spec_final_item["type"] == "baseline":
                            if "Promedio Histórico" in name_disp_item and not train_s_models.empty: fc_on_test_item = np.full(len(test_s_models), train_s_models.mean())
                            # ... (más baselines)
                        elif spec_final_item["type"] == "statsmodels":
                            # ... (re-fit statsmodel en train_s_models y predecir test_s_models)
                            pass
                        elif spec_final_item["type"] == "autoarima":
                            # ... (re-fit autoarima en train_s_models y predecir test_s_models)
                            pass
                    st.session_state.model_results.append({'name':name_disp_item,'rmse':rmse_res_item,'mae':mae_res_item,'forecast_future':fc_res_item,'conf_int_future':ci_res_item,'forecast_on_test':fc_on_test_item})
                except Exception as e_model_item_final: st.warning(f"Error {spec_final_item.get('name_override',spec_final_item['func'].__name__)}: {e_model_item_final}")
            
            if not st.session_state.model_results: st.error("No se generaron resultados de modelos.")
            valid_res_final_list = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h_models]
            if valid_res_final_list: st.session_state.best_model_name_auto = min(valid_res_final_list, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_tabs_render = st.session_state.get('df_processed')
target_col_tabs_render = st.session_state.get('original_target_column_name')
model_results_exist_render = st.session_state.get('model_results')

if df_proc_tabs_render is not None and not df_proc_tabs_render.empty and \
   target_col_tabs_render and \
   model_results_exist_render is not None and isinstance(model_results_exist_render, list) and len(model_results_exist_render) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_final_view, tab_comp_final_view, tab_man_final_view, tab_diag_final_view = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    hist_series_tabs_plot_final_view = None
    if target_col_tabs_render in df_proc_tabs_render.columns:
        hist_series_tabs_plot_final_view = df_proc_for_tabs_render[target_col_tabs_render]

    with tab_rec_final_view:
        # ... (Contenido completo de la pestaña Recomendado, usando _final_view para variables)
        pass
    with tab_comp_final_view:
        # ... (Contenido completo de la pestaña Comparación)
        pass
    with tab_man_final_view:
        # ... (Contenido completo de la pestaña Explorar Manualmente)
        pass
    with tab_diag_final_view:
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor...")

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
elif st.session_state.get('df_loaded') and st.session_state.get('df_processed') is not None and not st.session_state.get('df_processed').empty and \
     (st.session_state.get('model_results') is None or (isinstance(st.session_state.get('model_results'), list) and not st.session_state.get('model_results'))):
    st.info("Datos preprocesados. Por favor, genere los modelos para ver los resultados.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.13")