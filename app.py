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

# --- Configuración de la Página ---
st.set_page_config(page_title="Asistente de Pronósticos PRO", layout="wide")

# --- Estado de la Sesión ---
def init_session_state():
    defaults = {
        'df_loaded': None, 'current_file_name': None,
        'df_processed': None, 'original_target_column_name': "Valor", # Default por si acaso
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'model_results': [],
        'best_model_name_auto': None,
        'selected_model_for_manual_explore': None, # Para la pestaña de exploración manual
        # No necesitamos almacenar el pronóstico final globalmente, se genera en cada pestaña
        'use_train_test_split': True, 'test_split_size': 0.2, # Default a 20% o un número de periodos
        'train_series_for_plot': None, 'test_series_for_plot': None,
        # 'forecast_on_test_for_plot': None, # Esto será específico del modelo mostrado
        # 'model_for_validation_plot': None, # Esto será específico del modelo mostrado
        'run_autoarima': True,
        'arima_max_p': 3, 'arima_max_q': 3, 'arima_max_d': 2,
        'arima_max_P': 1, 'arima_max_Q': 1, 'arima_max_D': 1,
        'holt_damped': False,
        'hw_trend': 'add', 'hw_seasonal': 'add', 'hw_damped': False, 'hw_boxcox': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Funciones Auxiliares ---
def to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=True, sheet_name='Pronostico')
    processed_data = output.getvalue()
    return processed_data

def reset_all_except_load():
    keys_to_keep = ['df_loaded', 'current_file_name', 'forecast_horizon', 'user_seasonal_period',
                    'use_train_test_split', 'test_split_size', 'run_autoarima',
                    'arima_max_p', 'arima_max_q', 'arima_max_d', 'arima_max_P', 'arima_max_Q', 'arima_max_D',
                    'holt_damped', 'hw_trend', 'hw_seasonal', 'hw_damped', 'hw_boxcox']
    
    # Obtener todas las claves actuales en session_state
    current_keys = list(st.session_state.keys())
    
    for key in current_keys:
        if key not in keys_to_keep:
            del st.session_state[key]
    init_session_state() # Reinicializar con defaults las que se borraron

def prepare_forecast_display_data(model_data, series_full_idx, horizon, target_col_name):
    """Prepara los DataFrames necesarios para mostrar y exportar un pronóstico."""
    if model_data is None or model_data.get('forecast_future') is None:
        return None, None, None

    last_date_hist = series_full_idx.max()
    freq = pd.infer_freq(series_full_idx)
    if freq is None and len(series_full_idx) > 1: # Intenta inferir de las diferencias
        diffs = series_full_idx.to_series().diff().dropna()
        if not diffs.empty:
            freq = diffs.min() # Usa la diferencia mínima como frecuencia si no es regular
    if freq is None: # Fallback muy genérico si todo falla
        freq = 'D' 
        st.warning(f"No se pudo inferir una frecuencia regular para generar fechas de pronóstico. Usando '{freq}' (Diario) como default. Esto podría no ser correcto.")


    forecast_dates = pd.date_range(start=last_date_hist, periods=horizon + 1, freq=freq)[1:]
    
    forecast_values = model_data['forecast_future']
    conf_int_df_raw = model_data.get('conf_int_future')

    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}
    pi_display_df = None # Para el gráfico

    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        # Asegurar que los PIs tengan el mismo índice que forecast_dates
        pi_indexed = conf_int_df_raw.copy()
        pi_indexed.index = forecast_dates # Asignar el índice correcto
        export_dict['Limite Inferior PI'] = pi_indexed['lower'].values
        export_dict['Limite Superior PI'] = pi_indexed['upper'].values
        pi_display_df = pi_indexed[['lower', 'upper']]
    
    final_export_df = pd.DataFrame(export_dict).set_index('Fecha')
    forecast_series_for_plot = final_export_df['Pronostico']
    
    return final_export_df, forecast_series_for_plot, pi_display_df


# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico, incluyendo AutoARIMA.")

# --- Sección 1: Carga y Preprocesamiento de Datos ---
st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo (CSV o Excel)", type=["csv", "xlsx", "xls"], key="file_uploader_main")

if uploaded_file:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.df_loaded = None 
        st.session_state.current_file_name = uploaded_file.name
        reset_all_except_load()

    if st.session_state.df_loaded is None:
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)

if st.session_state.df_loaded is not None:
    df_input_display = st.session_state.df_loaded.copy()
    
    date_col_options_display = df_input_display.columns.tolist()
    dt_col_guess_idx_display = 0
    for i, col_display in enumerate(date_col_options_display):
        if any(keyword in col_display.lower() for keyword in ['date', 'fecha', 'time', 'period']):
            dt_col_guess_idx_display = i; break
    date_col_name_display = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_display, index=dt_col_guess_idx_display, key="date_col_main")

    value_col_options_display = [col for col in df_input_display.columns if col != date_col_name_display]
    val_col_guess_idx_display = 0
    if value_col_options_display: # Solo si hay opciones
        for i, col_val_display in enumerate(value_col_options_display):
            # Asegurar que la columna exista y sea numérica antes de intentar acceder
            if col_val_display in df_input_display.columns and pd.api.types.is_numeric_dtype(df_input_display[col_val_display].dropna()):
                val_col_guess_idx_display = i; break
    
    # Actualizar original_target_column_name en el estado de sesión
    st.session_state.original_target_column_name = st.sidebar.selectbox(
        "Columna a Pronosticar:", value_col_options_display, index=val_col_guess_idx_display, key="value_col_main"
    )
    
    freq_options_display = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    selected_freq_display_val = st.sidebar.selectbox("Frecuencia (Remuestreo):", options=list(freq_options_display.keys()), index=0, key="desired_freq_main")
    desired_freq_code_val = freq_options_display[selected_freq_display_val]

    imputation_methods_display = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imputation_method_display = st.sidebar.selectbox("Imputación de Faltantes:", imputation_methods_display, index=1, key="imputation_main")
    imputation_method_code_val = None if imputation_method_display == "No imputar" else imputation_method_display.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preprocess_btn_main"):
        reset_all_except_load()
        if st.session_state.original_target_column_name and st.session_state.original_target_column_name in df_input_display.columns:
            with st.spinner("Preprocesando datos..."):
                processed_series_df_res, msg_res = data_handler.preprocess_data(
                    df_input_display.copy(), date_col_name_display, st.session_state.original_target_column_name,
                    desired_freq=desired_freq_code_val, imputation_method=imputation_method_code_val
                )
            if processed_series_df_res is not None:
                st.session_state.df_processed = processed_series_df_res
                st.success(f"Preprocesamiento OK. {msg_res}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(st.session_state.df_processed, st.session_state.original_target_column_name)
                if not st.session_state.df_processed.empty:
                    series_for_acf_display = st.session_state.df_processed[st.session_state.original_target_column_name]
                    lags_acf_display = min(len(series_for_acf_display) // 2 -1, 60)
                    if lags_acf_display > 5: 
                        st.session_state.acf_fig = data_handler.plot_acf_pacf(series_for_acf_display, lags=lags_acf_display, target_column_name=st.session_state.original_target_column_name)
                    else: st.session_state.acf_fig = None
                    
                    final_freq_display, auto_s_period_display = data_handler.get_series_frequency_and_period(st.session_state.df_processed.index)
                    st.session_state.auto_seasonal_period = auto_s_period_display
                    if st.session_state.user_seasonal_period == 1 or st.session_state.user_seasonal_period != auto_s_period_display : # Actualizar si era default o cambió
                        st.session_state.user_seasonal_period = auto_s_period_display
            else: 
                st.error(f"Fallo en preprocesamiento: {msg_res}")
                st.session_state.df_processed = None
        else:
            st.sidebar.error("Por favor, seleccione una columna de valor válida.")

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
if st.session_state.df_processed is not None:
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_main, col2_acf_main = st.columns(2)
    with col1_diag_main:
        st.subheader("Diagnóstico de Datos")
        if st.session_state.data_diagnosis_report: st.markdown(st.session_state.data_diagnosis_report)
        else: st.info("Diagnóstico no disponible.")
    with col2_acf_main:
        st.subheader("Análisis de Autocorrelación")
        if st.session_state.acf_fig: st.pyplot(st.session_state.acf_fig)
        else: st.info("Gráfico ACF no disponible.")
    
    st.subheader("Serie de Tiempo Preprocesada")
    fig_hist_proc_main = visualization.plot_historical_data(st.session_state.df_processed, st.session_state.original_target_column_name, title=f"Histórico de '{st.session_state.original_target_column_name}' (Preprocesado)")
    if fig_hist_proc_main: st.pyplot(fig_hist_proc_main)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", min_value=1, value=st.session_state.forecast_horizon, step=1, key="horizon_cfg_main")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", min_value=1, value=st.session_state.user_seasonal_period, step=1, key="s_period_cfg_main", help=f"Sugerido por frecuencia/ACF: {st.session_state.auto_seasonal_period}")

    st.sidebar.subheader("Evaluación del Modelo")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_cfg_main")
    if st.session_state.use_train_test_split:
        max_test_size = len(st.session_state.df_processed) - (2 * st.session_state.user_seasonal_period) -1 if st.session_state.user_seasonal_period > 1 else len(st.session_state.df_processed) - 5
        max_test_size = max(1, max_test_size) # Asegurar que sea al menos 1
        
        # Asegurar que el valor por defecto del test_split_size no exceda max_test_size
        default_test_size = min(max(1, st.session_state.forecast_horizon), max_test_size)

        st.session_state.test_split_size = st.sidebar.number_input(
            "Tamaño Test Set (períodos):", 
            min_value=1, 
            value=default_test_size, 
            max_value=max_test_size, 
            step=1, key="test_size_cfg_main",
            help=f"Máximo sugerido: {max_test_size} para dejar suficientes datos para entrenamiento."
            )


    st.sidebar.subheader("Configuración de Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA (puede ser lento)", value=st.session_state.run_autoarima, key="run_arima_cfg_main")

    with st.sidebar.expander("Parámetros AutoARIMA", expanded=False):
        c1_arima, c2_arima = st.columns(2)
        st.session_state.arima_max_p = c1_arima.number_input("max_p", 1, 5, st.session_state.arima_max_p, key="arima_p", help="Máx orden AR no estacional")
        st.session_state.arima_max_q = c2_arima.number_input("max_q", 1, 5, st.session_state.arima_max_q, key="arima_q", help="Máx orden MA no estacional")
        st.session_state.arima_max_d = c1_arima.number_input("max_d", 0, 3, st.session_state.arima_max_d, key="arima_d", help="Máx orden Diff no estacional")
        st.session_state.arima_max_P = c2_arima.number_input("max_P", 0, 3, st.session_state.arima_max_P, key="arima_P", help="Máx orden SAR estacional")
        st.session_state.arima_max_Q = c1_arima.number_input("max_Q", 0, 3, st.session_state.arima_max_Q, key="arima_Q", help="Máx orden SMA estacional")
        st.session_state.arima_max_D = c2_arima.number_input("max_D", 0, 2, st.session_state.arima_max_D, key="arima_D", help="Máx orden Diff estacional")

    with st.sidebar.expander("Parámetros Holt y Holt-Winters", expanded=False):
        st.session_state.holt_damped = st.checkbox("Holt: Tendencia Amortiguada", value=st.session_state.holt_damped, key="holt_damped_cfg_main")
        st.markdown("**Holt-Winters:**")
        st.session_state.hw_trend = st.selectbox("HW: Tipo Tendencia", ['add', 'mul', None], index=['add', 'mul', None].index(st.session_state.hw_trend if st.session_state.hw_trend in ['add','mul',None] else 'add'), key="hw_trend_cfg_main")
        st.session_state.hw_seasonal = st.selectbox("HW: Tipo Estacionalidad", ['add', 'mul', None], index=['add', 'mul', None].index(st.session_state.hw_seasonal if st.session_state.hw_seasonal in ['add','mul',None] else 'add'), key="hw_seasonal_cfg_main")
        st.session_state.hw_damped = st.checkbox("HW: Tendencia Amortiguada", value=st.session_state.hw_damped, key="hw_damped_cfg_main")
        st.session_state.hw_boxcox = st.checkbox("HW: Usar Box-Cox", value=st.session_state.hw_boxcox, key="hw_boxcox_cfg_main", help="Puede ayudar si la varianza cambia con el nivel.")

    if st.sidebar.button("📊 Generar y Evaluar Todos los Modelos", key="generate_all_btn_main"):
        st.session_state.model_results = []
        st.session_state.best_model_name_auto = None
        st.session_state.selected_model_for_manual_explore = None

        series_full_run = st.session_state.df_processed[st.session_state.original_target_column_name].copy()
        h_run = st.session_state.forecast_horizon
        s_period_eff_run = st.session_state.user_seasonal_period

        train_series_run, test_series_run = series_full_run, pd.Series(dtype=series_full_run.dtype) # Default
        if st.session_state.use_train_test_split and st.session_state.test_split_size < len(series_full_run) and st.session_state.test_split_size > 0:
            # Asegurar que test_split_size no sea demasiado grande
            actual_test_size = min(st.session_state.test_split_size, len(series_full_run) - 5) # Dejar al menos 5 para train
            if actual_test_size > 0:
                train_series_run, test_series_run = forecasting_models.train_test_split_series(series_full_run, actual_test_size)
            else: # No es posible el split, usar toda la serie para entrenar
                st.session_state.use_train_test_split = False
        else:
            st.session_state.use_train_test_split = False
        
        st.session_state.train_series_for_plot = train_series_run
        st.session_state.test_series_for_plot = test_series_run

        with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
            # Baselines
            fc_avg, _, rmse_avg, mae_avg, name_avg = forecasting_models.historical_average_forecast(train_series_run, test_series_run, h_run)
            fc_on_test_avg = np.full(len(test_series_run), train_series_run.mean()) if not train_series_run.empty and not test_series_run.empty else None
            st.session_state.model_results.append({'name': name_avg, 'rmse': rmse_avg, 'mae': mae_avg, 'forecast_future': fc_avg, 'conf_int_future': None, 'forecast_on_test': fc_on_test_avg})

            fc_naive, _, rmse_naive, mae_naive, name_naive = forecasting_models.naive_forecast(train_series_run, test_series_run, h_run)
            fc_on_test_naive = np.full(len(test_series_run), train_series_run.iloc[-1]) if not train_series_run.empty and not test_series_run.empty else None
            st.session_state.model_results.append({'name': name_naive, 'rmse': rmse_naive, 'mae': mae_naive, 'forecast_future': fc_naive, 'conf_int_future': None, 'forecast_on_test': fc_on_test_naive})

            if s_period_eff_run > 1:
                fc_snaive, _, rmse_snaive, mae_snaive, name_snaive = forecasting_models.seasonal_naive_forecast(train_series_run, test_series_run, h_run, s_period_eff_run)
                snaive_test_fc_vals_run = None
                if not train_series_run.empty and not test_series_run.empty and s_period_eff_run <= len(train_series_run): # Condición añadida
                    snaive_test_fc_vals_run = np.zeros(len(test_series_run))
                    for i_test in range(len(test_series_run)):
                        idx_look_back = len(train_series_run) - s_period_eff_run + (i_test % s_period_eff_run)
                        if idx_look_back >= 0 and idx_look_back < len(train_series_run):
                             snaive_test_fc_vals_run[i_test] = train_series_run.iloc[idx_look_back]
                        else: snaive_test_fc_vals_run[i_test] = np.nan
                st.session_state.model_results.append({'name': name_snaive, 'rmse': rmse_snaive, 'mae': mae_snaive, 'forecast_future': fc_snaive, 'conf_int_future': None, 'forecast_on_test': snaive_test_fc_vals_run})
            
            # Statsmodels
            holt_p_run = {'damped_trend': st.session_state.holt_damped}
            hw_p_run = {'trend': st.session_state.hw_trend, 'seasonal': st.session_state.hw_seasonal, 'damped_trend': st.session_state.hw_damped, 'use_boxcox': st.session_state.hw_boxcox}
            models_stats_run = [("SES", {}), ("Holt", holt_p_run)]
            if s_period_eff_run > 1: models_stats_run.append(("Holt-Winters", hw_p_run))

            for model_s_name, model_s_params_dict in models_stats_run:
                fc_s, ci_s, rmse_s, mae_s, name_s = forecasting_models.forecast_with_statsmodels(
                    train_series_run, test_series_run, h_run, model_s_name, 
                    seasonal_period=s_period_eff_run if model_s_name == "Holt-Winters" else None,
                    holt_params=model_s_params_dict if model_s_name == "Holt" else None,
                    hw_params=model_s_params_dict if model_s_name == "Holt-Winters" else None
                )
                fc_on_test_s_run = None
                if not train_series_run.empty and not test_series_run.empty and not ("Error" in name_s or "insuficientes" in name_s):
                    try:
                        temp_model_s_fit = None
                        if model_s_name == "SES": temp_model_s_fit = SimpleExpSmoothing(train_series_run, initialization_method="estimated").fit()
                        elif model_s_name == "Holt": temp_model_s_fit = Holt(train_series_run, damped_trend=holt_p_run.get('damped_trend',False), initialization_method="estimated").fit()
                        elif model_s_name == "Holt-Winters": temp_model_s_fit = ExponentialSmoothing(train_series_run, trend=hw_p_run.get('trend'), seasonal=hw_p_run.get('seasonal'), seasonal_periods=s_period_eff_run, damped_trend=hw_p_run.get('damped_trend',False), use_boxcox=hw_p_run.get('use_boxcox',False), initialization_method="estimated").fit()
                        if temp_model_s_fit: fc_on_test_s_run = temp_model_s_fit.forecast(len(test_series_run)).values
                    except: fc_on_test_s_run = None
                st.session_state.model_results.append({'name': name_s, 'rmse': rmse_s, 'mae': mae_s, 'forecast_future': fc_s, 'conf_int_future': ci_s, 'forecast_on_test': fc_on_test_s_run})

            # AutoARIMA
            if st.session_state.run_autoarima:
                arima_p_run = {'max_p': st.session_state.arima_max_p, 'max_q': st.session_state.arima_max_q, 'max_d': st.session_state.arima_max_d, 'max_P': st.session_state.arima_max_P, 'max_Q': st.session_state.arima_max_Q, 'max_D': st.session_state.arima_max_D}
                fc_arima, ci_arima, rmse_arima, mae_arima, name_arima = forecasting_models.forecast_with_auto_arima(train_series_run, test_series_run, h_run, s_period_eff_run, arima_params=arima_p_run)
                fc_on_test_arima_run = None
                if not train_series_run.empty and not test_series_run.empty and not ("Error" in name_arima or "insuficientes" in name_arima):
                    try:
                        temp_model_arima_fit = pm.auto_arima(train_series_run, start_p=1, start_q=1, max_p=arima_p_run['max_p'], max_q=arima_p_run['max_q'], max_d=arima_p_run['max_d'], m=s_period_eff_run if s_period_eff_run > 1 else 1, start_P=0, seasonal=True if s_period_eff_run > 1 else False, max_P=arima_p_run['max_P'], max_Q=arima_p_run['max_Q'], max_D=arima_p_run['max_D'], trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
                        if temp_model_arima_fit: fc_on_test_arima_run = temp_model_arima_fit.predict(n_periods=len(test_series_run))
                    except: fc_on_test_arima_run = None
                st.session_state.model_results.append({'name': name_arima, 'rmse': rmse_arima, 'mae': mae_arima, 'forecast_future': fc_arima, 'conf_int_future': ci_arima, 'forecast_on_test': fc_on_test_arima_run})

        valid_results_sort = [res for res in st.session_state.model_results if pd.notna(res['rmse']) and res['forecast_future'] is not None and len(res['forecast_future']) == h_run]
        if valid_results_sort:
            best_model_auto_entry = min(valid_results_sort, key=lambda x: x['rmse'])
            st.session_state.best_model_name_auto = best_model_auto_entry['name']
        else:
            st.error("No se pudo determinar un modelo sugerido. Todos los modelos fallaron o no produjeron resultados válidos.")
            st.session_state.best_model_name_auto = "Error en modelos"


# --- Sección de Resultados y Pestañas ---
if st.session_state.df_processed is not None and st.session_state.model_results:
    st.header("Resultados del Modelado y Pronóstico")

    if not st.session_state.best_model_name_auto and st.session_state.model_results: # Intentar seleccionar el mejor si no se hizo
        valid_results_sort_late = [res for res in st.session_state.model_results if pd.notna(res['rmse']) and res.get('forecast_future') is not None and len(res['forecast_future']) == st.session_state.forecast_horizon]
        if valid_results_sort_late:
            st.session_state.best_model_name_auto = min(valid_results_sort_late, key=lambda x: x['rmse'])['name']

    tab_rec_main, tab_comp_main, tab_manual_main, tab_diag_guide_main = st.tabs([
        "⭐ Modelo Recomendado", "📊 Comparación General", 
        "⚙️ Explorar y Seleccionar Manualmente", "💡 Diagnóstico y Guía"
    ])

    # --- Pestaña 1: Modelo Recomendado (Automático) ---
    with tab_rec_main:
        st.subheader(f"Análisis Detallado del Modelo Recomendado: {st.session_state.best_model_name_auto or 'No determinado'}")
        model_data_auto_tab = next((item for item in st.session_state.model_results if item["name"] == st.session_state.best_model_name_auto), None)

        if model_data_auto_tab:
            final_export_df_auto, fc_series_auto, pi_df_auto = prepare_forecast_display_data(
                model_data_auto_tab, 
                st.session_state.df_processed[st.session_state.original_target_column_name].index,
                st.session_state.forecast_horizon,
                st.session_state.original_target_column_name
            )
            if final_export_df_auto is not None:
                if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_auto_tab.get('forecast_on_test') is not None:
                    st.markdown("##### Validación en Conjunto de Prueba")
                    fig_val_auto_tab = visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot, st.session_state.test_series_for_plot, pd.Series(model_data_auto_tab['forecast_on_test'], index=st.session_state.test_series_for_plot.index), st.session_state.best_model_name_auto, st.session_state.original_target_column_name)
                    if fig_val_auto_tab: st.pyplot(fig_val_auto_tab)
                
                st.markdown("##### Pronóstico Futuro")
                fig_fc_auto_tab = visualization.plot_final_forecast(st.session_state.df_processed[st.session_state.original_target_column_name], fc_series_auto, pi_df_auto, model_name=st.session_state.best_model_name_auto, value_col_name=st.session_state.original_target_column_name)
                if fig_fc_auto_tab: st.pyplot(fig_fc_auto_tab)

                st.markdown("##### Valores del Pronóstico")
                st.dataframe(final_export_df_auto.style.format("{:.2f}"))
                excel_data_auto_tab = to_excel(final_export_df_auto)
                st.download_button(f"📥 Descargar Pronóstico ({st.session_state.best_model_name_auto})", excel_data_auto_tab, f"pronostico_recomendado_{st.session_state.original_target_column_name}.xlsx", key="dl_auto_main")
                
                st.markdown("##### Recomendaciones")
                rec_text_auto_tab = recommendations.generate_recommendations(st.session_state.best_model_name_auto, st.session_state.data_diagnosis_report, True, (pi_df_auto is not None), (st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty))
                st.markdown(rec_text_auto_tab)
            else: st.warning("No se pudo preparar la visualización del pronóstico para el modelo recomendado.")
        else: st.info("El modelo recomendado no produjo un pronóstico válido o no se ha ejecutado el análisis.")

    # --- Pestaña 2: Comparación General ---
    with tab_comp_main:
        st.subheader("Métricas de Rendimiento de Todos los Modelos Probados")
        eval_type_comp_tab = "en Conjunto de Prueba" if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty else "In-Sample (toda la serie)"
        st.markdown(f"RMSE y MAE {eval_type_comp_tab}. Valores más bajos son mejores.")
        metrics_list_comp_tab = [{'Modelo': r['name'], 'RMSE': r['rmse'], 'MAE': r['mae']} for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r['rmse'])]
        if metrics_list_comp_tab:
            metrics_df_comp_tab = pd.DataFrame(metrics_list_comp_tab).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_tab(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_comp_tab.style.format({'RMSE': "{:.3f}", 'MAE': "{:.3f}"}).apply(highlight_best_tab, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Modelo Automáticamente Sugerido (menor RMSE): **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay resultados de modelos para mostrar.")

    # --- Pestaña 3: Explorar y Seleccionar Manualmente ---
    with tab_manual_main:
        st.subheader("Explorar Resultados y Seleccionar un Modelo Manualmente")
        available_models_manual_tab = [res['name'] for res in st.session_state.model_results if res.get('forecast_future') is not None and pd.notna(res['rmse'])]
        
        if not available_models_manual_tab:
            st.warning("No hay modelos válidos disponibles para seleccionar.")
        else:
            default_manual_idx = 0
            if st.session_state.selected_model_for_manual_explore in available_models_manual_tab:
                default_manual_idx = available_models_manual_tab.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in available_models_manual_tab: # Default al mejor si no hay selección previa
                default_manual_idx = available_models_manual_tab.index(st.session_state.best_model_name_auto)
            
            st.session_state.selected_model_for_manual_explore = st.selectbox("Seleccione un modelo:", options=available_models_manual_tab, index=default_manual_idx, key="manual_model_selector_main")
            model_data_manual_tab = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)

            if model_data_manual_tab:
                final_export_df_man, fc_series_man, pi_df_man = prepare_forecast_display_data(model_data_manual_tab, st.session_state.df_processed[st.session_state.original_target_column_name].index, st.session_state.forecast_horizon, st.session_state.original_target_column_name)
                if final_export_df_man is not None:
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_manual_tab.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Conjunto de Prueba")
                        fig_val_man_tab = visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot, st.session_state.test_series_for_plot, pd.Series(model_data_manual_tab['forecast_on_test'], index=st.session_state.test_series_for_plot.index), st.session_state.selected_model_for_manual_explore, st.session_state.original_target_column_name)
                        if fig_val_man_tab: st.pyplot(fig_val_man_tab)
                    
                    st.markdown("##### Pronóstico Futuro")
                    fig_fc_man_tab = visualization.plot_final_forecast(st.session_state.df_processed[st.session_state.original_target_column_name], fc_series_man, pi_df_man, model_name=st.session_state.selected_model_for_manual_explore, value_col_name=st.session_state.original_target_column_name)
                    if fig_fc_man_tab: st.pyplot(fig_fc_man_tab)

                    st.markdown("##### Valores del Pronóstico")
                    st.dataframe(final_export_df_man.style.format("{:.2f}"))
                    excel_data_man_tab = to_excel(final_export_df_man)
                    st.download_button(f"📥 Descargar Pronóstico ({st.session_state.selected_model_for_manual_explore})", excel_data_man_tab, f"pronostico_manual_{st.session_state.original_target_column_name}.xlsx", key="dl_manual_main")
                    
                    st.markdown("##### Notas sobre este Modelo")
                    rec_text_man_tab = recommendations.generate_recommendations(st.session_state.selected_model_for_manual_explore, st.session_state.data_diagnosis_report, True, (pi_df_man is not None), (st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty))
                    st.markdown(rec_text_man_tab)
                else: st.warning(f"No se pudo preparar la visualización del pronóstico para el modelo '{st.session_state.selected_model_for_manual_explore}'.")

    # --- Pestaña 4: Diagnóstico y Guía (Opcional) ---
    with tab_diag_guide_main:
        st.subheader("Diagnóstico de Datos (Post-Preprocesamiento)")
        if st.session_state.data_diagnosis_report: st.markdown(st.session_state.data_diagnosis_report)
        else: st.info("Ejecute el preprocesamiento de datos primero.")
        
        st.subheader("Guía General de Interpretación")
        st.markdown("""
        - **RMSE y MAE:** Métricas de error. Valores más bajos indican un mejor ajuste del modelo a los datos históricos (o de prueba).
        - **Intervalos de Predicción (Área Sombreada):** Representan la incertidumbre del pronóstico. Un intervalo más amplio significa más incertidumbre.
        - **Calidad de los Datos:** ¡Es crucial! Valores faltantes, atípicos o una frecuencia irregular pueden degradar significativamente la calidad del pronóstico.
        - **Conocimiento del Dominio:** Ningún modelo es perfecto. Siempre combine los resultados del modelo con su propio conocimiento del negocio y eventos externos.
        """)
else:
    st.info("👋 ¡Bienvenido! Por favor, cargue un archivo de datos para comenzar.")

# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.info("Asistente de Pronósticos PRO v3.1")
