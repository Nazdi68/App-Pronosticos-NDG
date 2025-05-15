# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models # Asegúrate que este archivo esté actualizado con moving_average_forecast
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
    st.session_state.model_results = [] 
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    # Si df_processed debe resetearse (ej. cambio de frecuencia), se hace en el botón de preproc.

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
    forecast_values = np.array(forecast_values_raw) # Asegurar que sea un array para slicing
    min_len = len(forecast_dates)
    if len(forecast_values) != len(forecast_dates):
        min_len = min(len(forecast_values), len(forecast_dates))
        forecast_values = forecast_values[:min_len]
        forecast_dates = forecast_dates[:min_len]
    if min_len == 0: return pd.DataFrame(columns=['Fecha', 'Pronostico']).set_index('Fecha'), pd.Series(dtype='float64'), None
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
uploaded_file = st.sidebar.file_uploader("Suba su archivo", type=["csv", "xlsx", "xls"], key="uploader_key_v10", on_change=reset_on_file_change)

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
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_key_v10")

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()): val_col_guess_idx = i; break
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_key_v10")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v10", on_change=lambda: setattr(st.session_state, 'df_processed', None) or reset_model_related_state())
    desired_freq = freq_map[freq_label]
    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v10", on_change=lambda: setattr(st.session_state, 'df_processed', None) or reset_model_related_state())
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v10"):
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
                elif msg_raw.endswith("D."): msg_disp=msg_raw.replace(" D."," D (Diario).") # Ajuste para el punto
            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed=proc_df; st.session_state.original_target_column_name=value_col; st.success(f"Preproc. OK. {msg_disp}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df,value_col)
                if not proc_df.empty:
                    s_acf=proc_df[value_col];l_acf=min(len(s_acf)//2-1,60)
                    if l_acf > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf,l_acf,value_col)
                    else: st.session_state.acf_fig=None
                    _,auto_s=data_handler.get_series_frequency_and_period(proc_df.index)
                    st.session_state.auto_seasonal_period=auto_s
                    if st.session_state.user_seasonal_period==1 or st.session_state.user_seasonal_period!=auto_s: st.session_state.user_seasonal_period=auto_s
            else: st.error(f"Fallo preproc: {msg_raw or 'DataFrame vacío.'}"); st.session_state.df_processed=None

df_proc_main = st.session_state.get('df_processed')
target_col_main = st.session_state.get('original_target_column_name')

if df_proc_main is not None and not df_proc_main.empty and target_col_main:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_main, col2_acf_main = st.columns(2)
    with col1_diag_main: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_main: 
        st.subheader("Autocorrelación")
        acf_fig_plot = st.session_state.get('acf_fig')
        if acf_fig_plot: 
            try: st.pyplot(acf_fig_plot)
            except Exception as e: st.error(f"Error al mostrar ACF/PACF: {e}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie Preprocesada")
    if target_col_main in df_proc_main.columns:
        fig_hist_main = visualization.plot_historical_data(df_proc_main, target_col_main, f"Histórico de '{target_col_main}'")
        if fig_hist_main: st.pyplot(fig_hist_main)
    st.markdown("---")

    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v10")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v10", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    max_ma_win = len(df_proc_main)//2 if df_proc_main is not None and not df_proc_main.empty else 2
    st.session_state.moving_avg_window = st.sidebar.number_input("Ventana Prom. Móvil:", value=st.session_state.moving_avg_window, min_value=2,max_value=max(2, max_ma_win), step=1, key="ma_win_key_v10")
    
    st.sidebar.subheader("Evaluación")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_key_v10", on_change=reset_model_related_state) # Resetear modelos si cambia esto
    if st.session_state.use_train_test_split:
        min_train_cfg = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_cfg = len(df_proc_main) - min_train_cfg; max_test_cfg = max(1, max_test_cfg)
        def_test_cfg = min(max(1, st.session_state.forecast_horizon), max_test_cfg)
        current_test_cfg = st.session_state.get('test_split_size', def_test_cfg)
        if current_test_cfg > max_test_cfg or current_test_cfg <=0 : current_test_cfg = def_test_cfg
        st.session_state.test_split_size = st.sidebar.number_input("Tamaño Test Set:", value=current_test_cfg, min_value=1, max_value=max_test_cfg, step=1, key="test_size_key_v10", help=f"Máx: {max_test_cfg}")

    st.sidebar.subheader("Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_key_v10")
    with st.sidebar.expander("Parámetros AutoARIMA"): # COMPLETA ESTO CON TUS number_input y keys únicas
        c1ar,c2ar=st.columns(2); st.session_state.arima_max_p=c1ar.number_input("max_p",1,5,st.session_state.arima_max_p,key="ap_k_v10"); st.session_state.arima_max_q=c2ar.number_input("max_q",1,5,st.session_state.arima_max_q,key="aq_k_v10"); #... etc
    with st.sidebar.expander("Parámetros Holt y Holt-Winters"): # COMPLETA ESTO
        st.session_state.holt_damped = st.checkbox("Holt: Amortiguar Tendencia", value=st.session_state.holt_damped, key="hd_k_v10") #... etc

    if st.sidebar.button("📊 Generar y Evaluar Modelos", key="gen_models_btn_key_v10"):
        reset_model_related_state() # Siempre resetear resultados antes de generar nuevos
        
        series_full_run = df_proc_main[target_col_main].copy(); h_run = st.session_state.forecast_horizon; s_period_run = st.session_state.user_seasonal_period; ma_win_run = st.session_state.moving_avg_window
        train_s_run, test_s_run = series_full_run, pd.Series(dtype=series_full_run.dtype)
        if st.session_state.use_train_test_split:
            min_tr_run = max(5, 2*s_period_run+1 if s_period_run>1 else 5); curr_test_run = st.session_state.get('test_split_size', 12)
            if len(series_full_run) > min_tr_run + curr_test_run and curr_test_run > 0 : train_s_run,test_s_run = forecasting_models.train_test_split_series(series_full_run, curr_test_run)
            else: st.warning(f"No split con test_size={curr_test_run}."); st.session_state.use_train_test_split=False
        st.session_state.train_series_for_plot = train_s_run; st.session_state.test_series_for_plot = test_s_run
            
        with st.spinner("Calculando modelos..."):
            model_exec_specs = []
            # Baselines
            model_exec_specs.append({"name_override":"Promedio Histórico", "func":forecasting_models.historical_average_forecast, "args":[train_s_run,test_s_run,h_run]})
            model_exec_specs.append({"name_override":"Ingénuo (Último Valor)", "func":forecasting_models.naive_forecast, "args":[train_s_run,test_s_run,h_run]})
            model_exec_specs.append({"name_override":f"Promedio Móvil (V:{ma_win_run})", "func":forecasting_models.moving_average_forecast, "args":[train_s_run,test_s_run,h_run,ma_win_run]})
            if s_period_run > 1: model_exec_specs.append({"name_override":f"Estacional Ingénuo (P:{s_period_run})", "func":forecasting_models.seasonal_naive_forecast, "args":[train_s_run,test_s_run,h_run,s_period_run]})
            
            # Statsmodels
            holt_p_exec = {'damped_trend': st.session_state.holt_damped}
            hw_p_exec = {'trend':st.session_state.hw_trend,'seasonal':st.session_state.hw_seasonal,'damped_trend':st.session_state.hw_damped,'use_boxcox':st.session_state.hw_boxcox}
            stats_configs = [("SES",{}), ("Holt",holt_p_exec)]
            if s_period_run > 1: stats_configs.append(("Holt-Winters",hw_p_exec))
            for name_s_exec, params_s_exec in stats_configs:
                model_exec_specs.append({"name_override":None, "func":forecasting_models.forecast_with_statsmodels, "args":[train_s_run,test_s_run,h_run,name_s_exec,s_period_run if name_s_exec=="Holt-Winters" else None,params_s_exec if name_s_exec=="Holt" else None,params_s_exec if name_s_exec=="Holt-Winters" else None]})
            
            if st.session_state.run_autoarima:
                arima_p_exec = {'max_p':st.session_state.arima_max_p, 'max_q':st.session_state.arima_max_q, 'max_d':st.session_state.arima_max_d, 'max_P':st.session_state.arima_max_P, 'max_Q':st.session_state.arima_max_Q, 'max_D':st.session_state.arima_max_D}
                model_exec_specs.append({"name_override":None, "func":forecasting_models.forecast_with_auto_arima, "args":[train_s_run,test_s_run,h_run,s_period_run,arima_p_exec]})

            for spec in model_exec_specs:
                try:
                    fc, ci, rmse, mae, name_f = spec["func"](*spec["args"])
                    name = spec["name_override"] or name_f
                    fc_on_test = None # Placeholder - Implement robust logic for each model type
                    if not test_s_run.empty and not ("Error" in name or "Insuf" in name or "Inválido" in name):
                        # Simplified example for baselines - needs proper implementation for others
                        if "Promedio Histórico" in name and not train_s_run.empty: fc_on_test = np.full(len(test_s_run), train_s_run.mean())
                        elif "Ingénuo" in name and not train_s_run.empty: fc_on_test = np.full(len(test_s_run), train_s_run.iloc[-1])
                        elif "Promedio Móvil" in name and not train_s_run.empty and len(train_s_run) >= ma_win_run: fc_on_test = np.full(len(test_s_run), train_s_run.iloc[-ma_win_run:].mean())
                        # For SES, Holt, HW, ARIMA, you'd typically re-fit on train_s_run and predict len(test_s_run)
                    st.session_state.model_results.append({'name':name,'rmse':rmse,'mae':mae,'forecast_future':fc,'conf_int_future':ci,'forecast_on_test':fc_on_test})
                except Exception as e: st.warning(f"Error {spec.get('name_override',spec['func'].__name__)}: {e}")
            
            valid_r = [r for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None and len(r.get('forecast_future'))==h_run]
            if valid_r: st.session_state.best_model_name_auto = min(valid_r, key=lambda x:x['rmse'])['name']
            else: st.error("No se pudo determinar un modelo sugerido."); st.session_state.best_model_name_auto = None

# --- Sección de Resultados y Pestañas ---
df_proc_tabs = st.session_state.get('df_processed')
target_col_tabs = st.session_state.get('original_target_column_name')
model_res_exist = st.session_state.get('model_results')

if df_proc_tabs is not None and not df_proc_tabs.empty and target_col_tabs and model_res_exist is not None: # Solo check if model_results key exists
    st.header("Resultados del Modelado y Pronóstico")
    tab_rec, tab_comp, tab_man, tab_diag = st.tabs(["⭐ Recomendado", "📊 Comparación", "⚙️ Explorar", "💡 Diagnóstico"])
    
    hist_series_tabs = None
    if target_col_tabs in df_proc_tabs.columns: hist_series_tabs = df_proc_tabs[target_col_tabs]

    with tab_rec:
        best_model = st.session_state.best_model_name_auto
        if best_model and "Error" not in best_model and hist_series_tabs is not None:
            st.subheader(f"Modelo Recomendado: {best_model}")
            model_data = next((m for m in st.session_state.model_results if m["name"] == best_model), None)
            if model_data:
                final_df, fc_s, pi_df = prepare_forecast_display_data(model_data, hist_series_tabs.index, st.session_state.forecast_horizon)
                if final_df is not None and fc_s is not None and not fc_s.empty:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Test"); fig_v=visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot,st.session_state.test_series_for_plot,pd.Series(model_data['forecast_on_test'], index=st.session_state.test_series_for_plot.index),best_model,target_col_tabs); st.pyplot(fig_v) if fig_v else st.info("No se pudo graficar validación.")
                    st.markdown(f"##### Pronóstico Futuro"); fig_f=visualization.plot_final_forecast(hist_series_tabs,fc_s,pi_df,best_model,target_col_tabs); st.pyplot(fig_f) if fig_f else st.info("No se pudo graficar pronóstico.")
                    st.markdown("##### Valores"); st.dataframe(final_df.style.format("{:.2f}")); dl_key = f"dl_rec_{best_model[:10].replace(' ','_')}"; st.download_button(f"📥 Descargar ({best_model})",to_excel(final_df),f"fc_rec_{target_col_tabs}.xlsx",key=dl_key)
                    st.markdown("##### Recomendaciones"); st.markdown(recommendations.generate_recommendations(best_model,st.session_state.data_diagnosis_report,True,(pi_df is not None),st.session_state.use_train_test_split and not st.session_state.test_series_for_plot.empty))
                else: st.warning(f"No se pudo preparar visualización para '{best_model}'.")
        elif not hist_series_tabs is not None: st.warning("Datos históricos no disponibles.")
        else: st.info("No se ha determinado un modelo recomendado. Genere los modelos.")

    with tab_comp:
        st.subheader("Comparación de Modelos")
        metrics = [{'Modelo': r['name'], 'RMSE': r['rmse'], 'MAE': r['mae']} for r in st.session_state.model_results if pd.notna(r.get('rmse')) and r.get('forecast_future') is not None]
        if metrics:
            df_m = pd.DataFrame(metrics).sort_values(by='RMSE').reset_index(drop=True)
            def highlight(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(df_m.style.format({'RMSE':"{:.3f}",'MAE':"{:.3f}"}).apply(highlight,axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Sugerido: **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay métricas de modelos para mostrar.")

    with tab_man:
        st.subheader("Explorar Modelo Manualmente")
        valid_man_models = [r['name'] for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r.get('rmse'))]
        if valid_man_models:
            sel_idx = 0
            if st.session_state.selected_model_for_manual_explore in valid_man_models: sel_idx = valid_man_models.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in valid_man_models: sel_idx = valid_man_models.index(st.session_state.best_model_name_auto)
            st.session_state.selected_model_for_manual_explore = st.selectbox("Modelo:", valid_man_models, index=sel_idx, key="man_sel_key_v10")
            model_data_m = next((m for m in st.session_state.model_results if m["name"] == st.session_state.selected_model_for_manual_explore), None)
            if model_data_m and hist_series_tabs is not None:
                final_df_m, fc_s_m, pi_df_m = prepare_forecast_display_data(model_data_m, hist_series_tabs.index, st.session_state.forecast_horizon)
                if final_df_m is not None and fc_s_m is not None and not fc_s_m.empty:
                    # ... (código de visualización y descarga para el modelo manual)
                    st.markdown(f"##### Pronóstico Futuro con {st.session_state.selected_model_for_manual_explore}") # ...
                    pass # Placeholder - añade tu lógica de visualización aquí
                else: st.warning(f"No se pudo preparar visualización para '{st.session_state.selected_model_for_manual_explore}'.")
        else: st.warning("No hay modelos válidos para exploración manual.")
        
    with tab_diag:
        st.subheader("Diagnóstico de Datos"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
        st.subheader("Guía General"); st.markdown("- RMSE y MAE: Error, menor es mejor...\n- Calidad de Datos: Crucial...")

elif uploaded_file is None: 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') and (st.session_state.get('df_processed') is None or (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Aplique preprocesamiento a los datos o verifique el resultado.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v3.9")