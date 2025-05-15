# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models 
import recommendations # Asegúrate que este archivo esté actualizado

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

st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v12", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

if st.session_state.get('df_loaded') is not None:
    df_input = st.session_state.df_loaded.copy()
    date_col_options = df_input.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    sel_date_idx = date_col_options.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options else dt_col_guess_idx
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_key_v12")

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_key_v12")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v12", on_change=reset_sidebar_config_dependent_state)
    desired_freq = freq_map[freq_label]
    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v12", on_change=reset_sidebar_config_dependent_state)
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v12"):
        st.session_state.df_processed = None; reset_model_related_state()
        date_col = st.session_state.get('selected_date_col'); value_col = st.session_state.get('selected_value_col'); valid = True
        if not date_col or date_col not in df_input.columns: st.sidebar.error("Seleccione fecha."); valid=False
        if not value_col or value_col not in df_input.columns: st.sidebar.error("Seleccione valor."); valid=False
        elif valid and not pd.api.types.is_numeric_dtype(df_input[value_col].dropna()): st.sidebar.error(f"'{value_col}' no numérica."); valid=False
        if valid:
            with st.spinner("Preprocesando..."): proc_df,msg_raw = data_handler.preprocess_data(df_input.copy(),date_col,value_col,desired_freq,imp_code)
            msg_disp = msg_raw; 
            if msg_raw: 
                if "MS" in msg_raw: msg_disp=msg_raw.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw: msg_disp=msg_raw.replace(" D."," D (Diario).") # Chequear espacio antes de D.
                elif msg_raw.endswith("D"): msg_disp=msg_raw.replace("D", "D (Diario)")


            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed=proc_df; st.session_state.original_target_column_name=value_col; st.success(f"Preprocesamiento OK. {msg_disp}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df,value_col)
                if not proc_df.empty:
                    s_acf=proc_df[value_col];l_acf=min(len(s_acf)//2-1,60)
                    if l_acf > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf,l_acf,value_col)
                    else: st.session_state.acf_fig=None
                    _,auto_s=data_handler.get_series_frequency_and_period(proc_df.index)
                    st.session_state.auto_seasonal_period=auto_s
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s: st.session_state.user_seasonal_period=auto_s
            else: st.error(f"Fallo preproc: {msg_raw or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main = st.session_state.get('df_processed')
target_col_main = st.session_state.get('original_target_column_name')

if df_processed_main is not None and not df_processed_main.empty and target_col_main:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_main, col2_acf_main = st.columns(2)
    with col1_diag_main: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_main: 
        st.subheader("Autocorrelación")
        acf_fig_main = st.session_state.get('acf_fig')
        if acf_fig_main: 
            try: st.pyplot(acf_fig_main)
            except Exception as e_acf: st.error(f"Error al mostrar ACF/PACF: {e_acf}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main in df_processed_main.columns:
        fig_hist_main = visualization.plot_historical_data(df_processed_main, target_col_main, f"Histórico de '{target_col_main}'")
        if fig_hist_main: st.pyplot(fig_hist_main)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v12")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v12", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win_cfg_val = len(df_processed_main)//2 if df_processed_main is not None and not df_processed_main.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win_cfg_val), step=1, key="ma_win_key_v12")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v12", on_change=reset_model_related_state)
    if st.session_state.use_train_test_split:
        min_train_cfg_val = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_cfg_val = len(df_processed_main) - min_train_cfg_val; max_test_cfg_val = max(1, max_test_cfg_val)
        def_test_cfg_val = min(max(1, st.session_state.forecast_horizon), max_test_cfg_val)
        current_test_cfg_val = st.session_state.get('test_split_size', def_test_cfg_val)
        if current_test_cfg_val > max_test_cfg_val or current_test_cfg_val <=0 : current_test_cfg_val = def_test_cfg_val
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_cfg_val, min_value=1, max_value=max_test_cfg_val, step=1, key="test_size_key_v12", help=f"Máx: {max_test_cfg_val}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v12")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v12,c2ar_v12=st.columns(2); st.session_state.arima_max_p=c1ar_v12.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v12"); st.session_state.arima_max_q=c2ar_v12.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v12"); st.session_state.arima_max_d=c1ar_v12.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v12"); st.session_state.arima_max_P=c2ar_v12.number_input("max_P",0,3,st.session_state.arima_max_P,key="aP_k_v12"); st.session_state.arima_max_Q=c1ar_v12.number_input("max_Q",0,3,st.session_state.arima_max_Q,key="aQ_k_v12"); st.session_state.arima_max_D=c2ar_v12.number_input("max_D",0,2,st.session_state.arima_max_D,key="aD_k_v12")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v12")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v12"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v12"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v12"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v12")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v12"):
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None; st.session_state.selected_model_for_manual_explore = None
        
        series_full = df_processed_main_check[target_col_main].copy(); h = st.session_state.forecast_horizon; s_period = st.session_state.user_seasonal_period; ma_win = st.session_state.moving_avg_window
        train_s, test_s = series_full, pd.Series(dtype=series_full.dtype)
        if st.session_state.use_train_test_split:
            min_tr = max(5, 2*s_period+1 if s_period>1 else 5); curr_test = st.session_state.get('test_split_size', 12)
            if len(series_full) > min_tr + curr_test and curr_test > 0 : train_s,test_s = forecasting_models.train_test_split_series(series_full, curr_test)
            else: st.warning(f"No split con test_size={curr_test}."); st.session_state.use_train_test_split=False
        st.session_state.train_series_for_plot = train_s; st.session_state.test_series_for_plot = test_s
            
        with st.spinner("Calculando modelos..."):
            model_exec_list = []
            model_exec_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s, test_s, h], "name_override": "Promedio Histórico", "type":"baseline"})
            model_exec_list.append({"func": forecasting_models.naive_forecast, "args": [train_s, test_s, h], "name_override": "Ingénuo (Último Valor)", "type":"baseline"})
            model_exec_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s, test_s, h, ma_win], "name_override": None, "type":"baseline"})
            if s_period > 1: model_exec_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s, test_s, h, s_period], "name_override": None, "type":"baseline"})
            
            holt_p_cfg = {'damped_trend': st.session_state.holt_damped}
            hw_p_cfg = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
            stats_cfg_list = [("SES", {}), ("Holt", holt_p_cfg)]
            if s_period > 1: stats_cfg_list.append(("Holt-Winters", hw_p_cfg))
            for name_s_cfg_item, params_s_cfg_item in stats_cfg_list:
                model_exec_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s, test_s, h, name_s_cfg_item, s_period if name_s_cfg_item=="Holt-Winters" else None, params_s_cfg_item if name_s_cfg_item=="Holt" else None, params_s_cfg_item if name_s_cfg_item=="Holt-Winters" else None], "name_override": None, "type":"statsmodels", "model_short_name": name_s_cfg_item, "params_dict": params_s_cfg_item})
            
            if st.session_state.run_autoarima:
                arima_p_cfg = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                model_exec_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s, test_s, h, s_period, arima_p_cfg], "name_override": None, "type":"autoarima", "arima_params_dict": arima_p_cfg})

            for spec in model_exec_list:
                try:
                    fc, ci, rmse, mae, name_f = spec["func"](*spec["args"])
                    name = spec["name_override"] or name_f
                    fc_on_test = None
                    if st.session_state.use_train_test_split and not test_s.empty and not ("Error" in name or "Insuf" in name or "Inválido" in name):
                        if spec["type"] == "baseline":
                            if "Promedio Histórico" in name and not train_s.empty: fc_on_test = np.full(len(test_s), train_s.mean())
                            elif "Ingénuo" in name and not train_s.empty: fc_on_test = np.full(len(test_s), train_s.iloc[-1])
                            elif "Promedio Móvil" in name and not train_s.empty and len(train_s) >= ma_win: fc_on_test = np.full(len(test_s), train_s.iloc[-ma_win:].mean())
                            elif "Estacional Ingénuo" in name and not train_s.empty and len(train_s) >= s_period:
                                temp_fc = np.zeros(len(test_s)); 
                                for i_fc in range(len(test_s)): temp_fc[i_fc] = train_s.iloc[len(train_s) - s_period + (i_fc % s_period)]
                                fc_on_test = temp_fc
                        elif spec["type"] == "statsmodels":
                            m_short = spec["model_short_name"]; p_dict = spec["params_dict"]; temp_fit = None
                            try:
                                if m_short=="SES": temp_fit=forecasting_models.SimpleExpSmoothing(train_s,initialization_method="estimated").fit()
                                elif m_short=="Holt": temp_fit=forecasting_models.Holt(train_s,damped_trend=p_dict.get('damped_trend',False),initialization_method="estimated").fit()
                                elif m_short=="Holt-Winters": temp_fit=forecasting_models.ExponentialSmoothing(train_s,trend=p_dict.get('trend'),seasonal=p_dict.get('seasonal'),seasonal_periods=s_period,damped_trend=p_dict.get('damped_trend',False),use_boxcox=p_dict.get('use_boxcox',False),initialization_method="estimated").fit()
                                if temp_fit: fc_on_test = temp_fit.forecast(len(test_s)).values
                            except Exception as e_sm_fc_test: st.caption(f"Warn fc_on_test {name}: {e_sm_fc_test}")
                        elif spec["type"] == "autoarima":
                            arima_p_fc_test = spec["arima_params_dict"]; temp_arima = None
                            try:
                                temp_arima = forecasting_models.pm.auto_arima(train_s,max_p=arima_p_fc_test.get('max_p',3),max_q=arima_p_fc_test.get('max_q',3),m=s_period if s_period>1 else 1,seasonal=s_period>1,suppress_warnings=True,error_action='ignore',stepwise=True)
                                if temp_arima: fc_on_test = temp_arima.predict(n_periods=len(test_s))
                            except Exception as e_arima_fc_test: st.caption(f"Warn fc_on_test {name}: {e_arima_fc_test}")
                    st.session_state.model_results.append({'name':name,'rmse':rmse,'mae':mae,'forecast_future':fc,'conf_int_future':ci,'forecast_on_test':fc_on_test})
                except Exception as e: st.warning(f"Error {spec.get('name_override',spec['func'].__name__)}: {e}")
            
            valid_r_list = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h]
            if valid_r_list: st.session_state.best_model_name_auto = min(valid_r_list, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_final_view = st.session_state.get('df_processed')
target_col_for_tabs_final_view = st.session_state.get('original_target_column_name')
model_results_exist_final_view = st.session_state.get('model_results')

if df_proc_for_tabs_final_view is not None and not df_proc_for_tabs_final_view.empty and \
   target_col_for_tabs_final_view and \
   model_results_exist_final_view is not None and isinstance(model_results_exist_final_view, list) and len(model_results_exist_final_view) > 0:

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec, tab_comp, tab_man, tab_diag = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    hist_series_tabs_plot = None
    if target_col_for_tabs_final_view in df_proc_for_tabs_final_view.columns:
        hist_series_tabs_plot = df_proc_for_tabs_final_view[target_col_for_tabs_final_view]

    with tab_rec:
        best_model_rec_view = st.session_state.best_model_name_auto
        if best_model_rec_view and "Error" not in best_model_rec_view and "FALLÓ" not in best_model_rec_view and hist_series_tabs_plot is not None:
            st.subheader(f"Modelo Recomendado: {best_model_rec_view}")
            model_data_rec_view = next((item for item in st.session_state.model_results if item["name"] == best_model_rec_view), None)
            if model_data_rec_view:
                final_df_r, fc_s_r, pi_df_r = prepare_forecast_display_data(model_data_rec_view, hist_series_tabs_plot.index, st.session_state.forecast_horizon)
                if final_df_r is not None and fc_s_r is not None and not fc_s_r.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_rec_view.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); fc_test_r_s = pd.Series(model_data_rec_view['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_rec_view['forecast_on_test'],np.ndarray) and len(model_data_rec_view['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_rec_view.get('forecast_on_test'); 
                        if fc_test_r_s is not None: fig_vr=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_r_s,best_model_rec_view,target_col_for_tabs_final_view); st.pyplot(fig_vr) if fig_vr else st.caption("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fr=visualization.plot_final_forecast(hist_series_tabs_plot,fc_s_r,pi_df_r,best_model_rec_view,target_col_for_tabs_final_view); st.pyplot(fig_fr) if fig_fr else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_r.style.format("{:.2f}")); dl_key_r_final = f"dl_r_{best_model_rec_view[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF"; st.download_button(f"📥 Descargar ({best_model_rec_view})",to_excel(final_df_r),f"fc_{best_model_rec_view.replace(' ','_')}.xlsx",key=dl_key_r_final)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model_rec_view,st.session_state.data_diagnosis_report,True,(pi_df_r is not None and not pi_df_r.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_final_view))
                else: st.warning(f"No se pudo preparar visualización para '{best_model_rec_view}'.")
            else: st.info(f"Datos del modelo '{best_model_rec_view}' no encontrados.")
        elif not hist_series_tabs_plot is not None: st.warning("Datos históricos no disponibles.")
        else: st.info("No se ha determinado un modelo recomendado. Genere los modelos o verifique errores.")

    with tab_comp:
        st.subheader("Comparación de Modelos")
        metrics_list_comp_final = [{'Modelo': r.get('name','N/A'), 'RMSE': r.get('rmse'), 'MAE': r.get('mae')} for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None]
        if metrics_list_comp_final:
            metrics_df_comp_final = pd.DataFrame(metrics_list_comp_final).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_final(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_comp_final.style.format({'RMSE':"{:.3f}", 'MAE':"{:.3f}"}).apply(highlight_best_final, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Sugerido: **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay métricas de modelos para mostrar.")

    with tab_man:
        st.subheader("Explorar Modelo Manualmente")
        valid_manual_models_final_list = [r['name'] for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r.get('rmse'))]
        if valid_manual_models_final_list and hist_series_tabs_plot is not None:
            sel_idx_man_final_val = 0
            if st.session_state.selected_model_for_manual_explore in valid_manual_models_final_list: sel_idx_man_final_val = valid_manual_models_final_list.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in valid_manual_models_final_list: sel_idx_man_final_val = valid_manual_models_final_list.index(st.session_state.best_model_name_auto)
            st.session_state.selected_model_for_manual_explore = st.selectbox("Modelo:", valid_manual_models_final_list, index=sel_idx_man_final_val, key="man_sel_key_v12_final") # Key única
            model_data_man_final_view = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)
            if model_data_man_final_view:
                final_df_m_final_view, fc_s_m_final_view, pi_df_m_final_view = prepare_forecast_display_data(model_data_man_final_view, hist_series_tabs_plot.index, st.session_state.forecast_horizon)
                if final_df_m_final_view is not None and fc_s_m_final_view is not None and not fc_s_m_final_view.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_man_final_view.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); 
                        fc_test_m_s = pd.Series(model_data_man_final_view['forecast_on_test'], index=st.session_state.test_series_for_plot.index) if isinstance(model_data_man_final_view['forecast_on_test'],np.ndarray) and len(model_data_man_final_view['forecast_on_test'])==len(st.session_state.test_series_for_plot) else model_data_man_final_view.get('forecast_on_test');
                        if fc_test_m_s is not None: fig_vm_final=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,fc_test_m_s,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final); st.pyplot(fig_vm_final) if fig_vm_final else st.caption("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fm_final=visualization.plot_final_forecast(hist_series_tabs_plot,fc_s_m_final_view,pi_df_m_final_view,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_final); st.pyplot(fig_fm_final) if fig_fm_final else st.caption("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_m_final_view.style.format("{:.2f}")); dl_key_m_final = f"dl_man_{st.session_state.selected_model_for_manual_explore[:15].replace(' ','_').replace('(','').replace(')','').replace(':','_').replace('[','').replace(']','').replace('.','_')}_vF"; st.download_button(f"📥 Descargar ({st.session_state.selected_model_for_manual_explore})",to_excel(final_df_m_final_view),f"fc_man_{target_col_for_tabs_final}.xlsx",key=dl_key_m_final)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(st.session_state.selected_model_for_manual_explore,st.session_state.data_diagnosis_report,True,(pi_df_m_final_view is not None and not pi_df_m_final_view.empty),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty, model_results_list=st.session_state.model_results, target_column_name=target_col_for_tabs_final))
                else: st.warning(f"No se pudo preparar visualización para '{st.session_state.selected_model_for_manual_explore}'.")
        elif not hist_series_tabs_plot : st.warning("Datos históricos no disponibles.")
        else: st.warning("No hay modelos válidos para exploración. Genere los modelos.")
        
    with tab_diag:
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor.\n- Intervalos de Predicción: Incertidumbre.\n- Calidad de Datos: Crucial.")

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
# No añadir un else aquí que podría ocultar el mensaje de "Genere modelos" si df_processed está OK pero model_results está vacío

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.11")