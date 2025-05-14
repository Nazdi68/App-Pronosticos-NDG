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
        'df_processed': None, 
        'selected_date_col': None, 
        'selected_value_col': None, 
        'original_target_column_name': "Valor", 
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'model_results': [],
        'best_model_name_auto': None,
        'selected_model_for_manual_explore': None,
        'use_train_test_split': True, 'test_split_size': 12, 
        'train_series_for_plot': None, 'test_series_for_plot': None,
        'run_autoarima': True,
        'arima_max_p': 3, 'arima_max_q': 3, 'arima_max_d': 2,
        'arima_max_P': 1, 'arima_max_Q': 1, 'arima_max_D': 1,
        'holt_damped': False,
        'hw_trend': 'add', 'hw_seasonal': 'add', 'hw_damped': False, 'hw_boxcox': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state() # Llamar a la función al inicio del script

# --- Funciones Auxiliares ---
def to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=True, sheet_name='Pronostico')
    processed_data = output.getvalue()
    return processed_data

def reset_on_file_change():
    """Llamado por on_change del file_uploader para limpiar el estado relevante."""
    keys_to_reset_on_file_change = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name',
        'data_diagnosis_report', 'acf_fig', 'model_results', 
        'best_model_name_auto', 'selected_model_for_manual_explore',
        'train_series_for_plot', 'test_series_for_plot',
        'auto_seasonal_period'
    ]
    for key in keys_to_reset_on_file_change:
        if key in st.session_state:
            del st.session_state[key] # Eliminar para que init_session_state o la lógica posterior los restaure/recalcule
    
    st.session_state.df_loaded = None # Importante para forzar la recarga del nuevo archivo
    # Re-inicializar algunas claves explícitamente para claridad o si init_session_state no las cubre todas
    st.session_state.selected_date_col = None
    st.session_state.selected_value_col = None
    st.session_state.original_target_column_name = "Valor" # Resetear a un default
    st.session_state.model_results = []


def reset_model_related_state():
    """Llamado cuando cambian parámetros que invalidan modelos o datos procesados."""
    st.session_state.df_processed = None
    st.session_state.model_results = []
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.data_diagnosis_report = None
    st.session_state.acf_fig = None
    st.session_state.train_series_for_plot = None
    st.session_state.test_series_for_plot = None

def prepare_forecast_display_data(model_data, series_full_idx, horizon):
    if model_data is None or model_data.get('forecast_future') is None:
        return None, None, None
    last_date_hist = series_full_idx.max()
    freq = pd.infer_freq(series_full_idx)
    if freq is None and len(series_full_idx) > 1:
        diffs = series_full_idx.to_series().diff().dropna()
        if not diffs.empty: freq = diffs.min()
    if freq is None: freq = 'D'; st.warning(f"Frecuencia no inferida, usando '{freq}'.")

    forecast_dates = pd.date_range(start=last_date_hist, periods=horizon + 1, freq=freq)[1:]
    forecast_values = model_data['forecast_future']
    conf_int_df_raw = model_data.get('conf_int_future')
    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}
    pi_display_df = None
    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        pi_indexed = conf_int_df_raw.copy()
        pi_indexed.index = forecast_dates
        export_dict['Limite Inferior PI'] = pi_indexed['lower'].values
        export_dict['Limite Superior PI'] = pi_indexed['upper'].values
        pi_display_df = pi_indexed[['lower', 'upper']]
    final_export_df = pd.DataFrame(export_dict).set_index('Fecha')
    forecast_series_for_plot = final_export_df['Pronostico']
    return final_export_df, forecast_series_for_plot, pi_display_df

# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos PRO")
st.markdown("Herramienta avanzada para generar, evaluar y seleccionar modelos de pronóstico.")

# --- Sección 1: Carga y Preprocesamiento de Datos ---
st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader(
    "Suba su archivo (CSV o Excel)", 
    type=["csv", "xlsx", "xls"], 
    key="uploader_key", # Key única
    on_change=reset_on_file_change 
)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None:
            st.session_state.current_file_name = uploaded_file.name
        else:
            st.session_state.current_file_name = None


if st.session_state.get('df_loaded') is not None:
    df_input = st.session_state.df_loaded.copy()
    
    # Selectores de Columnas
    date_col_options = df_input.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']):
                dt_col_guess_idx = i; break
    
    # Inicializar selectores si no existen en el estado o si el df_loaded es nuevo
    if 'selected_date_col' not in st.session_state or st.session_state.selected_date_col is None:
        st.session_state.selected_date_col = date_col_options[dt_col_guess_idx] if date_col_options else None

    st.session_state.selected_date_col = st.sidebar.selectbox(
        "Columna de Fecha/Hora:", date_col_options, 
        index=date_col_options.index(st.session_state.selected_date_col) if st.session_state.selected_date_col and st.session_state.selected_date_col in date_col_options else dt_col_guess_idx, 
        key="date_selector_key"
    )

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()):
                val_col_guess_idx = i; break
    
    if 'selected_value_col' not in st.session_state or st.session_state.selected_value_col is None:
        st.session_state.selected_value_col = value_col_options[val_col_guess_idx] if value_col_options else None

    st.session_state.selected_value_col = st.sidebar.selectbox(
        "Columna a Pronosticar:", value_col_options, 
        index=value_col_options.index(st.session_state.selected_value_col) if st.session_state.selected_value_col and st.session_state.selected_value_col in value_col_options else val_col_guess_idx, 
        key="value_selector_key"
    )
    
    # Selectores de Frecuencia e Imputación
    freq_options_dict = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    selected_freq_label = st.sidebar.selectbox("Frecuencia (Remuestreo):", options=list(freq_options_dict.keys()), index=0, key="freq_selector_key", on_change=reset_model_related_state)
    desired_freq_code_actual = freq_options_dict[selected_freq_label]

    imputation_methods_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    selected_imputation_label = st.sidebar.selectbox("Imputación de Faltantes:", imputation_methods_list, index=1, key="imputation_selector_key", on_change=reset_model_related_state)
    imputation_method_code_actual = None if selected_imputation_label == "No imputar" else selected_imputation_label.split('(')[0].strip()

    # Botón de Preprocesamiento
    if st.sidebar.button("Aplicar Preprocesamiento", key="preprocess_button_key"):
        reset_model_related_state() # Siempre resetear modelos si se aplica preprocesamiento

        date_col_to_use_btn = st.session_state.get('selected_date_col')
        value_col_to_use_btn = st.session_state.get('selected_value_col')

        valid_preprocess_input = True
        if not date_col_to_use_btn or date_col_to_use_btn not in df_input.columns:
            st.sidebar.error("Por favor, seleccione una columna de fecha válida.")
            valid_preprocess_input = False
        if not value_col_to_use_btn or value_col_to_use_btn not in df_input.columns:
            st.sidebar.error("Por favor, seleccione una columna de valor válida.")
            valid_preprocess_input = False
        elif valid_preprocess_input and not pd.api.types.is_numeric_dtype(df_input[value_col_to_use_btn].dropna()):
             st.sidebar.error(f"La columna de valor '{value_col_to_use_btn}' no es numérica.")
             valid_preprocess_input = False
        
        if valid_preprocess_input:
            with st.spinner("Preprocesando datos..."):
                processed_df_result, msg_result = data_handler.preprocess_data(
                    df_input.copy(), date_col_to_use_btn, value_col_to_use_btn,
                    desired_freq=desired_freq_code_actual, imputation_method=imputation_method_code_actual
                )
            if processed_df_result is not None:
                st.session_state.df_processed = processed_df_result
                st.session_state.original_target_column_name = value_col_to_use_btn 
                st.success(f"Preprocesamiento OK. {msg_result}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(st.session_state.df_processed, value_col_to_use_btn)
                if not st.session_state.df_processed.empty:
                    series_for_acf_plot = st.session_state.df_processed[value_col_to_use_btn]
                    lags_for_acf_plot = min(len(series_for_acf_plot) // 2 -1, 60)
                    if lags_for_acf_plot > 5: st.session_state.acf_fig = data_handler.plot_acf_pacf(series_for_acf_plot, lags_for_acf_plot, value_col_to_use_btn)
                    else: st.session_state.acf_fig = None
                    _, auto_s_period_val = data_handler.get_series_frequency_and_period(st.session_state.df_processed.index)
                    st.session_state.auto_seasonal_period = auto_s_period_val
                    if st.session_state.user_seasonal_period == 1 or st.session_state.user_seasonal_period != auto_s_period_val:
                        st.session_state.user_seasonal_period = auto_s_period_val
            else: 
                st.error(f"Fallo en preprocesamiento: {msg_result}")
                st.session_state.df_processed = None # Asegurar que se marque como no procesado

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
# Esta sección se ejecuta si df_processed existe y es válido
if st.session_state.get('df_processed') is not None and st.session_state.get('original_target_column_name'):
    target_col_name_display = st.session_state.original_target_column_name
    
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_display, col2_acf_display = st.columns(2)
    with col1_diag_display:
        st.subheader("Diagnóstico de Datos")
        if st.session_state.data_diagnosis_report: st.markdown(st.session_state.data_diagnosis_report)
    with col2_acf_display:
        st.subheader("Análisis de Autocorrelación")
        if st.session_state.acf_fig: st.pyplot(st.session_state.acf_fig)
    
    st.subheader("Serie de Tiempo Preprocesada")
    fig_hist_processed = visualization.plot_historical_data(st.session_state.df_processed, target_col_name_display, title=f"Histórico de '{target_col_name_display}' (Preprocesado)")
    if fig_hist_processed: st.pyplot(fig_hist_processed)
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", min_value=1, value=st.session_state.forecast_horizon, step=1, key="horizon_cfg_key")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", min_value=1, value=st.session_state.user_seasonal_period, step=1, key="s_period_cfg_key", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    
    st.sidebar.subheader("Evaluación del Modelo")
    st.session_state.use_train_test_split = st.sidebar.checkbox("Usar Train/Test split", value=st.session_state.use_train_test_split, key="use_split_cfg_key")
    if st.session_state.use_train_test_split:
        min_train_size_for_split = max(5, 2 * st.session_state.user_seasonal_period + 1 if st.session_state.user_seasonal_period > 1 else 5)
        max_test_size_val = len(st.session_state.df_processed) - min_train_size_for_split
        max_test_size_val = max(1, max_test_size_val)
        default_test_size_val = min(max(1, st.session_state.forecast_horizon), max_test_size_val) # Default al horizonte, pero no más que max_test_size
        
        # Actualizar test_split_size en session_state si el valor actual es inválido o no está seteado
        if 'test_split_size' not in st.session_state or st.session_state.test_split_size > max_test_size_val or st.session_state.test_split_size <=0:
            st.session_state.test_split_size = default_test_size_val

        st.session_state.test_split_size = st.sidebar.number_input(
            "Tamaño Test Set (períodos):", min_value=1, 
            value=st.session_state.test_split_size, # Usar el valor del estado
            max_value=max_test_size_val, step=1, key="test_size_cfg_key",
            help=f"Máximo sugerido: {max_test_size_val}"
        )

    st.sidebar.subheader("Configuración de Modelos Específicos")
    st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", value=st.session_state.run_autoarima, key="run_arima_cfg_key")
    # ... (Expanders para AutoARIMA, Holt, Holt-Winters con sus keys ÚNICAS) ...
    with st.sidebar.expander("Parámetros AutoARIMA", expanded=False):
        # ... (number_inputs con keys como "arima_p_key", "arima_q_key", etc.)
        pass # Placeholder
    with st.sidebar.expander("Parámetros Holt y Holt-Winters", expanded=False):
        # ... (checkboxes y selectboxes con keys como "holt_damped_key", "hw_trend_key", etc.)
        pass # Placeholder


    if st.sidebar.button("📊 Generar y Evaluar Todos los Modelos", key="generate_models_button_key"):
        # Resetear resultados de modelos ANTES de correrlos de nuevo
        st.session_state.model_results = []
        st.session_state.best_model_name_auto = None
        st.session_state.selected_model_for_manual_explore = None
        
        df_proc_for_run = st.session_state.get('df_processed')
        target_col_for_run = st.session_state.get('original_target_column_name')

        if df_proc_for_run is None or target_col_for_run is None:
            st.error("🔴 Error: Datos no preprocesados correctamente. Por favor, vuelva a aplicar el preprocesamiento.")
        elif target_col_for_run not in df_proc_for_run.columns:
            st.error(f"🔴 Error: La columna '{target_col_for_run}' no se encuentra en los datos preprocesados.")
        else:
            series_full_for_run = df_proc_for_run[target_col_for_run].copy()
            h_for_run = st.session_state.forecast_horizon
            s_period_eff_for_run = st.session_state.user_seasonal_period
            
            train_series_for_run, test_series_for_run = series_full_for_run, pd.Series(dtype=series_full_for_run.dtype)
            if st.session_state.use_train_test_split:
                min_train_size_run = max(5, 2 * s_period_eff_for_run + 1 if s_period_eff_for_run > 1 else 5)
                current_test_size_run = st.session_state.get('test_split_size', 12) # Default si no está
                if current_test_size_run < (len(series_full_for_run) - min_train_size_run) and current_test_size_run > 0 :
                    train_series_for_run, test_series_for_run = forecasting_models.train_test_split_series(series_full_for_run, current_test_size_run)
                else:
                    st.warning(f"No es posible el split con test_size={current_test_size_run}. Usando toda la serie para entrenar/evaluar in-sample.")
                    st.session_state.use_train_test_split = False
            
            st.session_state.train_series_for_plot = train_series_for_run
            st.session_state.test_series_for_plot = test_series_for_run
            
            with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
                # --- EJECUCIÓN DE MODELOS ---
                # Reemplaza esta sección con tu lógica completa de ejecución de modelos
                # Asegúrate de que cada modelo añada un diccionario a st.session_state.model_results
                # con 'name', 'rmse', 'mae', 'forecast_future', 'conf_int_future', 'forecast_on_test'
                
                # Ejemplo Placeholder (DEBES REEMPLAZAR ESTO CON TU LÓGICA REAL):
                # Baselines
                holt_params_run = {'damped_trend': st.session_state.holt_damped}
                hw_params_run = {'trend': st.session_state.hw_trend, 'seasonal': st.session_state.hw_seasonal, 
                                 'damped_trend': st.session_state.hw_damped, 'use_boxcox': st.session_state.hw_boxcox}
                arima_params_run = {'max_p': st.session_state.arima_max_p, 'max_q': st.session_state.arima_max_q, 
                                    'max_d': st.session_state.arima_max_d, 'max_P': st.session_state.arima_max_P, 
                                    'max_Q': st.session_state.arima_max_Q, 'max_D': st.session_state.arima_max_D}

                all_model_functions = [
                    (forecasting_models.historical_average_forecast, [train_series_for_run, test_series_for_run, h_for_run]),
                    (forecasting_models.naive_forecast, [train_series_for_run, test_series_for_run, h_for_run]),
                ]
                if s_period_eff_for_run > 1:
                    all_model_functions.append(
                        (forecasting_models.seasonal_naive_forecast, [train_series_for_run, test_series_for_run, h_for_run, s_period_eff_for_run])
                    )
                
                statsmodels_specs = [("SES", {}), ("Holt", holt_params_run)]
                if s_period_eff_for_run > 1:
                    statsmodels_specs.append(("Holt-Winters", hw_params_run))

                for model_s_name, model_s_p_dict in statsmodels_specs:
                    all_model_functions.append(
                        (forecasting_models.forecast_with_statsmodels, 
                         [train_series_for_run, test_series_for_run, h_for_run, model_s_name, 
                          s_period_eff_for_run if model_s_name == "Holt-Winters" else None,
                          model_s_p_dict if model_s_name == "Holt" else None,
                          model_s_p_dict if model_s_name == "Holt-Winters" else None
                         ])
                    )
                
                if st.session_state.run_autoarima:
                    all_model_functions.append(
                        (forecasting_models.forecast_with_auto_arima, 
                         [train_series_for_run, test_series_for_run, h_for_run, s_period_eff_for_run, arima_params_run])
                    )

                for func, args in all_model_functions:
                    try:
                        fc, ci, rmse, mae, name = func(*args)
                        # Lógica para obtener forecast_on_test (necesitarás ajustar esto por modelo)
                        fc_on_test_val = None
                        # ... (deberías tener una forma de obtener esto para cada modelo si se usa test split)
                        st.session_state.model_results.append({
                           'name': name, 'rmse': rmse, 'mae': mae, 
                           'forecast_future': fc, 'conf_int_future': ci, 
                           'forecast_on_test': fc_on_test_val 
                        })
                    except Exception as e_model:
                        st.warning(f"Error al ejecutar {func.__name__}: {e_model}")
                        st.session_state.model_results.append({
                           'name': f"{func.__name__} (Error)", 'rmse': np.nan, 'mae': np.nan, 
                           'forecast_future': None, 'conf_int_future': None, 
                           'forecast_on_test': None
                        })

                if not st.session_state.model_results:
                    st.error("🔴 No se pudieron generar resultados para ningún modelo.")

            # Determinar el mejor modelo
            valid_results_for_sorting_run = [res for res in st.session_state.model_results if pd.notna(res.get('rmse')) and res.get('forecast_future') is not None and len(res['forecast_future']) == h_for_run]
            if valid_results_for_sorting_run:
                st.session_state.best_model_name_auto = min(valid_results_for_sorting_run, key=lambda x: x['rmse'])['name']
            else:
                if st.session_state.model_results: st.error("No se pudo determinar un modelo sugerido de los resultados.")
                st.session_state.best_model_name_auto = None


# --- Sección de Resultados y Pestañas ---
if st.session_state.get('df_processed') and st.session_state.get('original_target_column_name') and st.session_state.get('model_results'):
    target_col_for_tabs_display = st.session_state.original_target_column_name
    st.header("Resultados del Modelado y Pronóstico")

    tab_rec_display, tab_comp_display, tab_manual_display, tab_diag_guide_display = st.tabs([
        "⭐ Modelo Recomendado", "📊 Comparación General", 
        "⚙️ Explorar Manualmente", "💡 Diagnóstico y Guía"
    ])

    # --- Pestaña 1: Modelo Recomendado (Automático) ---
    with tab_rec_display:
        if st.session_state.best_model_name_auto and "Error" not in st.session_state.best_model_name_auto :
            st.subheader(f"Análisis del Modelo Recomendado: {st.session_state.best_model_name_auto}")
            model_data_auto_display = next((item for item in st.session_state.model_results if item["name"] == st.session_state.best_model_name_auto), None)
            if model_data_auto_display:
                # ... (resto de la lógica de la pestaña como la tenías, usando prepare_forecast_display_data y visualization)
                pass # Placeholder
        else:
            st.info("No se ha determinado un modelo recomendado o hubo un error.")

    # --- Pestaña 2: Comparación General ---
    with tab_comp_display:
        # ... (resto de la lógica de la pestaña como la tenías)
        pass # Placeholder

    # --- Pestaña 3: Explorar y Seleccionar Manualmente ---
    with tab_manual_display:
        # ... (resto de la lógica de la pestaña como la tenías)
        pass # Placeholder

    # --- Pestaña 4: Diagnóstico y Guía ---
    with tab_diag_guide_display:
        # ... (resto de la lógica de la pestaña como la tenías)
        pass # Placeholder

elif uploaded_file is None:
    st.info("👋 ¡Bienvenido! Por favor, cargue un archivo de datos para comenzar.")
elif st.session_state.get('df_loaded') is not None and st.session_state.get('df_processed') is None:
    st.warning("⚠️ Por favor, aplique el preprocesamiento a los datos cargados.")


# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.info("Asistente de Pronósticos PRO v3.4") # Nueva versión