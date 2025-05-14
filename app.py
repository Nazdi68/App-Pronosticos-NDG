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
    st.session_state.model_results = [] # Resetear solo los modelos, no df_processed aquí
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    # Si el preprocesamiento en sí cambia (ej. frecuencia), df_processed se resetea en el botón de preproc.
    # Si solo cambian parámetros de modelo, df_processed se mantiene.

def prepare_forecast_display_data(model_data, series_full_idx, horizon):
    # ... (Código de prepare_forecast_display_data como en la última versión que te di, está bien)
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
    min_len = 0
    if len(forecast_values) != len(forecast_dates) and len(forecast_dates) > 0 :
        min_len = min(len(forecast_values), len(forecast_dates)); forecast_values = forecast_values[:min_len]; forecast_dates = forecast_dates[:min_len]
        if min_len == 0: return None, None, None 
    elif len(forecast_dates) == 0 and len(forecast_values) > 0 : return None, None, None
    elif len(forecast_dates) == 0 and len(forecast_values) == 0: return pd.DataFrame(columns=['Fecha', 'Pronostico']).set_index('Fecha'), pd.Series(dtype='float64'), None
    conf_int_df_raw = model_data.get('conf_int_future')
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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v7", on_change=reset_on_file_change)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None

if st.session_state.get('df_loaded') is not None:
    df_input = st.session_state.df_loaded.copy()
    date_col_options = df_input.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    
    sel_date_idx = date_col_options.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options else dt_col_guess_idx
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_key_v7")

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_key_v7")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v7", on_change=reset_model_related_state) # Resetear modelos si cambia la frecuencia
    desired_freq = freq_map[freq_label]

    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v7", on_change=reset_model_related_state) # Resetear modelos si cambia imputación
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v7"):
        # Al aplicar preprocesamiento, siempre reseteamos df_processed y los modelos
        st.session_state.df_processed = None 
        reset_model_related_state() 

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
            
            # Interpretar el mensaje de frecuencia para ser más descriptivo
            msg_proc_display = msg_proc_raw
            if "MS" in msg_proc_raw:
                msg_proc_display = msg_proc_raw.replace("MS", "MS (Inicio de Mes - Mensual)")
            elif "D" == msg_proc_raw.split(" ")[-1].replace(".",""): # Un poco frágil, pero intenta capturar
                msg_proc_display = msg_proc_raw.replace("D", "D (Diario)")
            # Añadir más elif para otras frecuencias si es necesario (W, Q, A)

            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed = proc_df; st.session_state.original_target_column_name = value_col
                st.success(f"Preprocesamiento OK. {msg_proc_display}") # Mensaje mejorado
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

df_processed_check = st.session_state.get('df_processed')
target_col_check = st.session_state.get('original_target_column_name')

if df_processed_check is not None and not df_processed_check.empty and target_col_check:
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
    if target_col_check in df_processed_check.columns:
        fig_hist = visualization.plot_historical_data(df_processed_check, target_col_check, f"Histórico de '{target_col_check}'")
        if fig_hist: st.pyplot(fig_hist)
    st.markdown("---")

    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v7")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v7", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_window = len(df_processed_check)//2 if df_processed_check is not None and not df_processed_check.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_window), step=1, key="ma_win_key_v7")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v7", on_change=reset_model_related_state)
    if st.session_state.use_train_test_split:
        min_train = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test = len(df_processed_check) - min_train; max_test = max(1, max_test)
        def_test = min(max(1, st.session_state.forecast_horizon), max_test)
        current_test_size = st.session_state.get('test_split_size', def_test)
        if current_test_size > max_test or current_test_size <=0 : current_test_size = def_test
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_size, min_value=1, max_value=max_test, step=1, key="test_size_key_v7", help=f"Máx: {max_test}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v7")
    with st.sidebar.expander("Parámetros AutoARIMA"):
        c1ar,c2ar=st.columns(2); st.session_state.arima_max_p=c1ar.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v7"); st.session_state.arima_max_q=c2ar.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v7"); st.session_state.arima_max_d=c1ar.number_input("max_d",0,3,st.session_state.arima_max_d,key="ad_k_v7"); st.session_state.arima_max_P=c2ar.number_input("max_P",0,3,st.session_state.arima_max_P,key="aP_k_v7"); st.session_state.arima_max_Q=c1ar.number_input("max_Q",0,3,st.session_state.arima_max_Q,key="aQ_k_v7"); st.session_state.arima_max_D=c2ar.number_input("max_D",0,2,st.session_state.arima_max_D,key="aD_k_v7")
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"):
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v7")
        st.markdown("**Holt-Winters:**"); st.session_state.hw_trend = st.selectbox("HW: Tendencia", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hwt_k_v7"); st.session_state.hw_seasonal = st.selectbox("HW: Estacionalidad", ['add','mul',None], index=['add','mul',None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hws_k_v7"); st.session_state.hw_damped = st.checkbox("HW: Amortiguar Tendencia", value=st.session_state.hw_damped, key="hwd_k_v7"); st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hwbc_k_v7")

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v7"):
        st.session_state.model_results = []; st.session_state.best_model_name_auto = None; st.session_state.selected_model_for_manual_explore = None
        
        series_full = df_processed_check[target_col_check].copy(); h = st.session_state.forecast_horizon; s_period = st.session_state.user_seasonal_period; ma_win = st.session_state.moving_avg_window
        train_s, test_s = series_full, pd.Series(dtype=series_full.dtype)
        if st.session_state.use_train_test_split:
            min_tr = max(5, 2*s_period+1 if s_period>1 else 5); curr_test = st.session_state.get('test_split_size',12)
            if curr_test < (len(series_full)-min_tr) and curr_test > 0: train_s,test_s = forecasting_models.train_test_split_series(series_full, curr_test)
            else: st.warning(f"No split con test_size={curr_test}."); st.session_state.use_train_test_split=False
        st.session_state.train_series_for_plot = train_s; st.session_state.test_series_for_plot = test_s
            
        with st.spinner("Calculando modelos..."):
            # Definición de modelos y sus argumentos
            model_execution_list = []
            model_execution_list.append({"func": forecasting_models.historical_average_forecast, "args": [train_s, test_s, h], "name_override": None})
            model_execution_list.append({"func": forecasting_models.naive_forecast, "args": [train_s, test_s, h], "name_override": None})
            model_execution_list.append({"func": forecasting_models.moving_average_forecast, "args": [train_s, test_s, h, ma_win], "name_override": None})
            if s_period > 1: model_execution_list.append({"func": forecasting_models.seasonal_naive_forecast, "args": [train_s, test_s, h, s_period], "name_override": None})
            
            holt_p_run = {'damped_trend': st.session_state.holt_damped}
            hw_p_run = {'trend':st.session_state.hw_trend, 'seasonal':st.session_state.hw_seasonal, 'damped_trend':st.session_state.hw_damped, 'use_boxcox':st.session_state.hw_boxcox}
            stats_model_configs = [("SES", {}), ("Holt", holt_p_run)]
            if s_period > 1: stats_model_configs.append(("Holt-Winters", hw_p_run))
            for name_s_run, params_s_run in stats_model_configs:
                model_execution_list.append({"func": forecasting_models.forecast_with_statsmodels, "args": [train_s, test_s, h, name_s_run, s_period if name_s_run=="Holt-Winters" else None, params_s_run if name_s_run=="Holt" else None, params_s_run if name_s_run=="Holt-Winters" else None], "name_override": None})
            
            if st.session_state.run_autoarima:
                arima_p_run = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                model_execution_list.append({"func": forecasting_models.forecast_with_auto_arima, "args": [train_s, test_s, h, s_period, arima_p_run], "name_override": None})

            for model_spec in model_execution_list:
                try:
                    fc_res, ci_res, rmse_res, mae_res, name_res = model_spec["func"](*model_spec["args"])
                    name_to_display = model_spec["name_override"] or name_res
                    # Placeholder para fc_on_test - esto necesita lógica robusta
                    fc_on_test_res = None 
                    # ... (tu lógica para obtener fc_on_test para cada modelo) ...
                    st.session_state.model_results.append({'name': name_to_display, 'rmse': rmse_res, 'mae': mae_res, 'forecast_future': fc_res, 'conf_int_future': ci_res, 'forecast_on_test': fc_on_test_res})
                except Exception as e_mod: st.warning(f"Error ejecutando {model_spec['func'].__name__ if model_spec['name_override'] is None else model_spec['name_override']}: {str(e_mod)[:100]}")
            
            valid_res_list = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h]
            if valid_res_list: st.session_state.best_model_name_auto = min(valid_res_list, key=lambda x: x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_for_tabs = st.session_state.get('df_processed')
target_col_for_tabs = st.session_state.get('original_target_column_name')
model_results_exist = st.session_state.get('model_results')

if df_proc_for_tabs is not None and not df_proc_for_tabs.empty and target_col_for_tabs and model_results_exist:
    st.header("Resultados del Modelado y Pronóstico")
    tab_rec_view, tab_comp_view, tab_manual_view, tab_diag_view = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    hist_series_for_tabs = df_proc_for_tabs[target_col_for_tabs]

    with tab_rec_view:
        best_model_name_view = st.session_state.best_model_name_auto
        if best_model_name_view and "Error" not in best_model_name_view:
            st.subheader(f"Modelo Recomendado: {best_model_name_view}")
            model_data_view = next((item for item in st.session_state.model_results if item["name"] == best_model_name_view), None)
            if model_data_view:
                final_df, fc_s, pi_df = prepare_forecast_display_data(model_data_view, hist_series_for_tabs.index, st.session_state.forecast_horizon)
                if final_df is not None and fc_s is not None:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_view.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); fig_v=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,model_data_view['forecast_on_test'],best_model_name_view,target_col_for_tabs); st.pyplot(fig_v) if fig_v else st.info("-")
                    st.markdown(f"##### Pronóstico Futuro"); fig_f=visualization.plot_final_forecast(hist_series_for_tabs,fc_s,pi_df,best_model_name_view,target_col_for_tabs); st.pyplot(fig_f) if fig_f else st.info("-")
                    st.markdown("##### Valores"); st.dataframe(final_df.style.format("{:.2f}")); dl_key_r = f"dl_r_{best_model_name_view[:10].replace(' ','_')}"; st.download_button(f"📥 Descargar ({best_model_name_view})",to_excel(final_df),f"fc_{best_model_name_view}.xlsx",key=dl_key_r)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model_name_view,st.session_state.data_diagnosis_report,True,(pi_df is not None),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty))
                else: st.warning("No se pudo preparar visualización para modelo recomendado.")
        else: st.info("No se ha determinado un modelo recomendado o hubo un error.")

    with tab_comp_view:
        st.subheader("Comparación de Modelos")
        # ... (código de la pestaña de comparación)
        pass
    with tab_manual_view:
        st.subheader("Explorar Modelo Manualmente")
        # ... (código de la pestaña de exploración manual)
        pass
    with tab_diag_view:
        st.subheader("Diagnóstico y Guía")
        # ... (código de la pestaña de diagnóstico)
        pass

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') and (st.session_state.get('df_processed') is None or st.session_state.get('df_processed').empty):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.7")