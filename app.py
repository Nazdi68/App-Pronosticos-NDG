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

def reset_model_related_state():
    # Llamado cuando cambian parámetros de preproc o antes de generar modelos
    # Si solo cambian params de modelo, df_processed se mantiene.
    # df_processed se resetea explícitamente en el botón "Aplicar Preprocesamiento"
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    # No resetear data_diagnosis_report y acf_fig aquí si df_processed no cambia

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

    forecast_values = model_data['forecast_future']
    if forecast_values is None : return None, None, None

    min_len = len(forecast_dates)
    if len(forecast_values) != len(forecast_dates):
        min_len = min(len(forecast_values), len(forecast_dates))
        forecast_values = np.array(forecast_values[:min_len]) # Asegurar que sea array para slicing
        forecast_dates = forecast_dates[:min_len]
    
    if min_len == 0: 
        return pd.DataFrame(columns=['Fecha', 'Pronostico']).set_index('Fecha'), pd.Series(dtype='float64'), None

    conf_int_df_raw = model_data.get('conf_int_future')
    # Asegurar que forecast_values tenga la longitud correcta de forecast_dates ahora
    if len(forecast_values) != len(forecast_dates):
        # Esto no debería suceder si min_len se calculó bien, pero es una salvaguarda
        return None, None, None 

    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}
    pi_display_df = None
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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v9", on_change=reset_on_file_change)

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
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_key_v9")

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_key_v9")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v9", on_change=reset_model_related_state)
    desired_freq = freq_map[freq_label]

    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v9", on_change=reset_model_related_state)
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v9"):
        st.session_state.df_processed = None # Resetear antes de intentar preprocesar
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None
        st.session_state.selected_model_for_manual_explore = None
        st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None

        date_col = st.session_state.get('selected_date_col')
        value_col = st.session_state.get('selected_value_col')
        valid = True
        if not date_col or date_col not in df_input.columns: st.sidebar.error("Seleccione columna de fecha válida."); valid = False
        if not value_col or value_col not in df_input.columns: st.sidebar.error("Seleccione columna de valor válida."); valid = False
        elif valid and not pd.api.types.is_numeric_dtype(df_input[value_col].dropna()):
             st.sidebar.error(f"Columna '{value_col}' no es numérica."); valid = False
        
        if valid:
            with st.spinner("Preprocesando..."):
                proc_df, msg_proc_raw = data_handler.preprocess_data(df_input.copy(), date_col, value_col, desired_freq, imp_code)
            
            msg_proc_display = msg_proc_raw
            if msg_proc_raw: 
                if "MS" in msg_proc_raw: msg_proc_display = msg_proc_raw.replace("MS", "MS (Inicio de Mes - Mensual)")
                elif "D" == msg_proc_raw.split(" ")[-1].replace(".",""): msg_proc_display = msg_proc_raw.replace("D", "D (Diario)")
            
            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed = proc_df; st.session_state.original_target_column_name = value_col
                st.success(f"Preprocesamiento OK. {msg_proc_display}")
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
            else: 
                st.error(f"Fallo en preprocesamiento: {msg_proc_raw if msg_proc_raw else 'El DataFrame resultante está vacío o es None.'}")
                st.session_state.df_processed = None

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
df_processed_main_check = st.session_state.get('df_processed')
target_col_main_check = st.session_state.get('original_target_column_name')

if df_processed_main_check is not None and not df_processed_main_check.empty and target_col_main_check:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag, col2_acf = st.columns(2)
    with col1_diag: 
        st.subheader("Diagnóstico")
        st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf: 
        st.subheader("Autocorrelación")
        acf_fig_to_plot = st.session_state.get('acf_fig')
        if acf_fig_to_plot is not None:
            try: st.pyplot(acf_fig_to_plot)
            except Exception as e_plot_acf: st.error(f"Error al mostrar ACF/PACF: {e_plot_acf}")
        else: st.info("ACF/PACF no disponible.")
    
    st.subheader("Serie Preprocesada")
    if target_col_main_check in df_processed_main_check.columns:
        fig_hist = visualization.plot_historical_data(df_processed_main_check, target_col_main_check, f"Histórico de '{target_col_main_check}'")
        if fig_hist: st.pyplot(fig_hist)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v9")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v9", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_window_val = len(df_processed_main_check)//2 if df_processed_main_check is not None and not df_processed_main_check.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_window_val), step=1, key="ma_win_key_v9")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v9", on_change=reset_model_related_state)
    if st.session_state.use_train_test_split:
        min_train_val = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_val = len(df_processed_main_check) - min_train_val; max_test_val = max(1, max_test_val)
        def_test_val = min(max(1, st.session_state.forecast_horizon), max_test_val)
        current_test_size_val = st.session_state.get('test_split_size', def_test_val)
        if current_test_size_val > max_test_val or current_test_size_val <=0 : current_test_size_val = def_test_val
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_size_val, min_value=1, max_value=max_test_val, step=1, key="test_size_key_v9", help=f"Máx: {max_test_val}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v9")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar_v9,c2ar_v9=st.columns(2); st.session_state.arima_max_p=c1ar_v9.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v9"); st.session_state.arima_max_q=c2ar_v9.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v9"); st.session_state.arima_max_d=c1ar_v9.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v9"); st.session_state.arima_max_P=c2ar_v9.number_input("max_P",0,3,st.session_state.arima_max_P,key="aP_k_v9"); st.session_state.arima_max_Q=c1ar_v9.number_input("max_Q",0,3,st.session_state.arima_max_Q,key="aQ_k_v9"); st.session_state.arima_max_D=c2ar_v9.number_input("max_D",0,2,st.session_state.arima_max_D,key="aD_k_v9")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v9")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v9"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v9"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v9"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v9")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v9"):
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None; st.session_state.selected_model_for_manual_explore = None
        
        df_pr_run = st.session_state.get('df_processed') # Usar el df_processed del estado
        target_c_run = st.session_state.get('original_target_column_name')

        if df_pr_run is None or target_c_run is None or target_c_run not in df_pr_run.columns: 
            st.error("🔴 Datos no preprocesados. Aplique preprocesamiento."); 
        else:
            series_full = df_pr_run[target_c_run].copy(); h = st.session_state.forecast_horizon; s_period = st.session_state.user_seasonal_period; ma_win = st.session_state.moving_avg_window
            train_s, test_s = series_full, pd.Series(dtype=series_full.dtype)
            if st.session_state.use_train_test_split:
                min_tr = max(5, 2*s_period+1 if s_period>1 else 5); curr_test = st.session_state.get('test_split_size', 12)
                if len(series_full) > min_tr + curr_test and curr_test > 0 : train_s,test_s = forecasting_models.train_test_split_series(series_full, curr_test)
                else: st.warning(f"No posible split con test_size={curr_test} y {len(series_full)} puntos."); st.session_state.use_train_test_split=False
            st.session_state.train_series_for_plot = train_s; st.session_state.test_series_for_plot = test_s
            
            with st.spinner("Calculando modelos..."):
                model_execution_list = []
                model_execution_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s, test_s, h], "name_override": "Promedio Histórico"})
                model_execution_list.append({"func": forecasting_models.naive_forecast, "args": [train_s, test_s, h], "name_override": "Ingénuo (Último Valor)"})
                model_execution_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s, test_s, h, ma_win], "name_override": f"Promedio Móvil (V:{ma_win})"})
                if s_period > 1: model_execution_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s, test_s, h, s_period], "name_override": f"Estacional Ingénuo (P:{s_period})"})
                
                holt_p_run = {'damped_trend': st.session_state.holt_damped}
                hw_p_run = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
                stats_model_configs_run = [("SES", {}), ("Holt", holt_p_run)]
                if s_period > 1: stats_model_configs_run.append(("Holt-Winters", hw_p_run))
                for name_s_run, params_s_run in stats_model_configs_run:
                    model_execution_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s, test_s, h, name_s_run, s_period if name_s_run=="Holt-Winters" else None, params_s_run if name_s_run=="Holt" else None, params_s_run if name_s_run=="Holt-Winters" else None], "name_override": None})
                
                if st.session_state.run_autoarima:
                    arima_p_run = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                    model_execution_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s, test_s, h, s_period, arima_p_run], "name_override": None})

                for model_spec_run in model_execution_list:
                    try:
                        fc_res_run, ci_res_run, rmse_res_run, mae_res_run, name_res_run = model_spec_run["func"](*model_spec_run["args"])
                        name_to_display_run = model_spec_run["name_override"] or name_res_run
                        
                        fc_on_test_res_run = None # Placeholder - Implementar lógica para cada tipo de modelo
                        if not test_s.empty and not ("Error" in name_to_display_run or "Insuf" in name_to_display_run or "Inválido" in name_to_display_run):
                            # Ejemplo simplificado para baselines
                            if "Promedio Histórico" in name_to_display_run and not train_s.empty: fc_on_test_res_run = np.full(len(test_s), train_s.mean())
                            elif "Ingénuo" in name_to_display_run and not train_s.empty : fc_on_test_res_run = np.full(len(test_s), train_s.iloc[-1])
                            # Para SES, Holt, HW, ARIMA necesitarías re-ajustar solo en train_s y predecir len(test_s)
                            # o modificar las funciones de modelo para que devuelvan esto.

                        st.session_state.model_results.append({'name': name_to_display_run, 'rmse': rmse_res_run, 'mae': mae_res_run, 'forecast_future': fc_res_run, 'conf_int_future': ci_res_run, 'forecast_on_test': fc_on_test_res_run})
                    except Exception as e_mod_run: st.warning(f"Error ejecutando {model_spec_run['func'].__name__ if model_spec_run['name_override'] is None else model_spec_run['name_override']}: {str(e_mod_run)[:100]}")
            
            valid_res_list_run = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h]
            if valid_res_list_run: st.session_state.best_model_name_auto = min(valid_res_list_run, key=lambda x: x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs_display = st.session_state.get('df_processed')
target_col_for_tabs_display = st.session_state.get('original_target_column_name')
model_results_exist_display = st.session_state.get('model_results')

if df_proc_for_tabs_display is not None and not df_proc_for_tabs_display.empty and \
   target_col_for_tabs_display and \
   model_results_exist_display is not None: # Solo verificar que la lista exista

    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_content_view, tab_comp_content_view, tab_manual_content_view, tab_diag_content_view = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    historical_series_for_tabs_view = None
    if target_col_for_tabs_display in df_proc_for_tabs_display.columns:
        historical_series_for_tabs_view = df_proc_for_tabs_display[target_col_for_tabs_display]

    with tab_rec_content_view:
        best_model_name_tab = st.session_state.best_model_name_auto
        if best_model_name_tab and "Error" not in best_model_name_tab and historical_series_for_tabs_view is not None:
            st.subheader(f"Modelo Recomendado: {best_model_name_tab}")
            model_data_tab = next((item for item in st.session_state.model_results if item["name"] == best_model_name_tab), None)
            if model_data_tab:
                final_df_tab, fc_s_tab, pi_df_tab = prepare_forecast_display_data(model_data_tab, historical_series_for_tabs_view.index, st.session_state.forecast_horizon)
                if final_df_tab is not None and fc_s_tab is not None and not fc_s_tab.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_tab.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); fig_v_tab=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,pd.Series(model_data_tab['forecast_on_test'], index=st.session_state.test_series_for_plot.index),best_model_name_tab,target_col_for_tabs_display); st.pyplot(fig_v_tab) if fig_v_tab else st.info("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_f_tab=visualization.plot_final_forecast(historical_series_for_tabs_view,fc_s_tab,pi_df_tab,best_model_name_tab,target_col_for_tabs_display); st.pyplot(fig_f_tab) if fig_f_tab else st.info("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df_tab.style.format("{:.2f}")); dl_key_r_tab = f"dl_r_{best_model_name_tab[:10].replace(' ','_').replace('(','').replace(')','').replace(':','')}"; st.download_button(f"📥 Descargar ({best_model_name_tab})",to_excel(final_df_tab),f"fc_{best_model_name_tab.replace(' ','_')}.xlsx",key=dl_key_r_tab)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model_name_tab,st.session_state.data_diagnosis_report,True,(pi_df_tab is not None),st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty))
                else: st.warning(f"No se pudo preparar visualización para '{best_model_name_tab}'.")
            else: st.info(f"Datos del modelo '{best_model_name_tab}' no encontrados en resultados.")
        elif not historical_series_for_tabs_view is not None: st.warning("Datos históricos no disponibles.")
        else: st.info("No se ha determinado un modelo recomendado o hubo un error. Genere los modelos.")

    with tab_comp_content_view:
        st.subheader("Comparación de Modelos")
        metrics_list_display = [{'Modelo': r['name'], 'RMSE': r['rmse'], 'MAE': r['mae']} for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None]
        if metrics_list_display:
            metrics_df_display = pd.DataFrame(metrics_list_display).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_comp(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_display.style.format({'RMSE':"{:.3f}", 'MAE':"{:.3f}"}).apply(highlight_best_comp, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Sugerido: **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay métricas de modelos para mostrar.")

    with tab_manual_content_view:
        st.subheader("Explorar Modelo Manualmente")
        valid_manual_models_list = [r['name'] for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r.get('rmse'))]
        if valid_manual_models_list:
            sel_idx_man_tab = 0
            if st.session_state.selected_model_for_manual_explore in valid_manual_models_list: sel_idx_man_tab = valid_manual_models_list.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in valid_manual_models_list: sel_idx_man_tab = valid_manual_models_list.index(st.session_state.best_model_name_auto)
            
            st.session_state.selected_model_for_manual_explore = st.selectbox("Modelo:", valid_manual_models_list, index=sel_idx_man_tab, key="man_sel_key_v9")
            model_data_manual_display = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)
            if model_data_manual_display and historical_series_for_tabs_view is not None:
                final_df_m_tab, fc_s_m_tab, pi_df_m_tab = prepare_forecast_display_data(model_data_manual_display, historical_series_for_tabs_view.index, st.session_state.forecast_horizon)
                if final_df_m_tab is not None and fc_s_m_tab is not None and not fc_s_m_tab.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_manual_display.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); fig_vm_tab=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,pd.Series(model_data_manual_display['forecast_on_test'], index=st.session_state.test_series_for_plot.index),st.session_state.selected_model_for_manual_explore,target_col_for_tabs_display); st.pyplot(fig_vm_tab) if fig_vm_tab else st.info("-")
                    st.markdown(f"##### Pronóstico Futuro"); fig_fm_tab=visualization.plot_final_forecast(historical_series_for_tabs_view,fc_s_m_tab,pi_df_m_tab,st.session_state.selected_model_for_manual_explore,target_col_for_tabs_display); st.pyplot(fig_fm_tab) if fig_fm_tab else st.info("-")
                    st.markdown("##### Valores"); st.dataframe(final_df_m_tab.style.format("{:.2f}")); dl_key_m_tab = f"dl_m_{st.session_state.selected_model_for_manual_explore[:10].replace(' ','_').replace('(','').replace(')','').replace(':','')}"; st.download_button(f"📥 Descargar ({st.session_state.selected_model_for_manual_explore})",to_excel(final_df_m_tab),f"fc_{st.session_state.selected_model_for_manual_explore}.xlsx",key=dl_key_m_tab)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(st.session_state.selected_model_for_manual_explore,st.session_state.data_diagnosis_report,True,(pi_df_m_tab is not None),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty))
                else: st.warning(f"No se pudo preparar visualización para '{st.session_state.selected_model_for_manual_explore}'.")
        else: st.warning("No hay modelos válidos para exploración manual.")
        
    with tab_diag_content_view:
        st.subheader("Diagnóstico de Datos (Post-Preprocesamiento)")
        st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General de Interpretación")
        st.markdown("""
        - **RMSE y MAE:** Métricas de error. Valores más bajos = mejor ajuste.
        - **Intervalos de Predicción:** Incertidumbre del pronóstico.
        - **Calidad de Datos:** Crucial para pronósticos fiables.
        - **Conocimiento del Dominio:** Combine resultados con su experiencia.
        """)

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') is not None and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.8")