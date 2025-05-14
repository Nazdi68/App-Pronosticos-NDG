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
        'df_loaded': None, 'current_file_name': None, # Para rastrear cambios de archivo
        'df_processed': None, 
        'selected_date_col': None, # Columna de fecha actualmente seleccionada por el usuario
        'selected_value_col': None, # Columna de valor actualmente seleccionada por el usuario
        'original_target_column_name': "Valor", # Nombre de la columna después de la selección válida
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'model_results': [],
        'best_model_name_auto': None,
        'selected_model_for_manual_explore': None,
        'use_train_test_split': True, 'test_split_size': 12, # Default a un número fijo de periodos
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

init_session_state()

# --- Funciones Auxiliares ---
def to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=True, sheet_name='Pronostico')
    processed_data = output.getvalue()
    return processed_data

def reset_on_file_change():
    """Llamado por on_change del file_uploader."""
    # Lista de claves que dependen del archivo cargado y deben resetearse
    keys_to_reset = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name', # Se re-seleccionará
        'data_diagnosis_report', 'acf_fig', 'model_results', 
        'best_model_name_auto', 'selected_model_for_manual_explore',
        'train_series_for_plot', 'test_series_for_plot',
        'auto_seasonal_period'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Forzar la reinicialización de algunas variables clave
    st.session_state.df_loaded = None # Forzar recarga del nuevo archivo
    st.session_state.df_processed = None
    st.session_state.model_results = []
    st.session_state.best_model_name_auto = None
    st.session_state.selected_model_for_manual_explore = None
    st.session_state.selected_date_col = None
    st.session_state.selected_value_col = None
    # init_session_state() # No es necesario si ya borramos y df_loaded = None fuerza el flujo

def reset_model_results_and_processed_data():
    """Llamado cuando cambian parámetros de preprocesamiento o modelos."""
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
    key="file_uploader_widget", # Key única
    on_change=reset_on_file_change # Llamar al reset cuando el archivo cambia
)

if uploaded_file:
    if st.session_state.df_loaded is None: # Solo cargar si no está cargado o fue reseteado
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None:
            st.session_state.current_file_name = uploaded_file.name # Guardar nombre del archivo actual
        else: # Si la carga falla
            st.session_state.current_file_name = None


if st.session_state.df_loaded is not None:
    df_input = st.session_state.df_loaded.copy()
    
    date_col_options = df_input.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']):
                dt_col_guess_idx = i; break
    
    # Selector de Columna de Fecha
    # Usar st.session_state.get para evitar errores si la clave no existe aún
    # y para manejar el reseteo
    current_selected_date_col = st.session_state.get('selected_date_col')
    date_col_idx = 0
    if current_selected_date_col and current_selected_date_col in date_col_options:
        date_col_idx = date_col_options.index(current_selected_date_col)
    elif date_col_options: # Si no hay selección previa, usar el guess
        date_col_idx = dt_col_guess_idx
        st.session_state.selected_date_col = date_col_options[date_col_idx] # Actualizar estado


    st.session_state.selected_date_col = st.sidebar.selectbox(
        "Columna de Fecha/Hora:", date_col_options, 
        index=date_col_idx, 
        key="date_col_selector_widget"
    )

    value_col_options = [col for col in df_input.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input[col].dropna()):
                val_col_guess_idx = i; break

    current_selected_value_col = st.session_state.get('selected_value_col')
    value_col_idx = 0
    if current_selected_value_col and current_selected_value_col in value_col_options:
        value_col_idx = value_col_options.index(current_selected_value_col)
    elif value_col_options: # Si no hay selección previa, usar el guess
        value_col_idx = val_col_guess_idx
        st.session_state.selected_value_col = value_col_options[value_col_idx]


    st.session_state.selected_value_col = st.sidebar.selectbox(
        "Columna a Pronosticar:", value_col_options, 
        index=value_col_idx, 
        key="value_col_selector_widget"
    )
    
    freq_options = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    selected_freq_key = st.sidebar.selectbox("Frecuencia (Remuestreo):", options=list(freq_options.keys()), index=0, key="desired_freq_selector_widget")
    desired_freq_code = freq_options[selected_freq_key]

    imputation_methods = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    selected_imputation_key = st.sidebar.selectbox("Imputación de Faltantes:", imputation_methods, index=1, key="imputation_selector_widget")
    imputation_method_code = None if selected_imputation_key == "No imputar" else selected_imputation_key.split('(')[0].strip()


    if st.sidebar.button("Aplicar Preprocesamiento", key="preprocess_btn_widget"):
        reset_model_results_and_processed_data() # Resetear antes de preprocesar

        # Leer las selecciones actuales de los widgets (o del estado si se actualizó)
        date_col_to_use = st.session_state.get('selected_date_col')
        value_col_to_use = st.session_state.get('selected_value_col')

        valid_input = True
        if not date_col_to_use or date_col_to_use not in df_input.columns:
            st.sidebar.error("Por favor, seleccione una columna de fecha válida.")
            valid_input = False
        if not value_col_to_use or value_col_to_use not in df_input.columns:
            st.sidebar.error("Por favor, seleccione una columna de valor válida.")
            valid_input = False
        elif valid_input and not pd.api.types.is_numeric_dtype(df_input[value_col_to_use].dropna()):
             st.sidebar.error(f"La columna de valor '{value_col_to_use}' no es numérica.")
             valid_input = False
        
        if valid_input:
            with st.spinner("Preprocesando datos..."):
                processed_df, msg = data_handler.preprocess_data(
                    df_input.copy(), date_col_to_use, value_col_to_use,
                    desired_freq=desired_freq_code, imputation_method=imputation_method_code
                )
            if processed_df is not None:
                st.session_state.df_processed = processed_df
                st.session_state.original_target_column_name = value_col_to_use # Guardar el nombre usado
                st.success(f"Preprocesamiento OK. {msg}")
                st.session_state.data_diagnosis_report = data_handler.diagnose_data(st.session_state.df_processed, value_col_to_use)
                if not st.session_state.df_processed.empty:
                    series_acf = st.session_state.df_processed[value_col_to_use]
                    lags_acf = min(len(series_acf) // 2 -1, 60)
                    if lags_acf > 5: st.session_state.acf_fig = data_handler.plot_acf_pacf(series_acf, lags_acf, value_col_to_use)
                    else: st.session_state.acf_fig = None
                    _, auto_s_period = data_handler.get_series_frequency_and_period(st.session_state.df_processed.index)
                    st.session_state.auto_seasonal_period = auto_s_period
                    if st.session_state.user_seasonal_period == 1 or st.session_state.user_seasonal_period != auto_s_period:
                        st.session_state.user_seasonal_period = auto_s_period
            else: 
                st.error(f"Fallo en preprocesamiento: {msg}")
                st.session_state.df_processed = None
    
# --- Mostrar Diagnóstico y Gráficos Iniciales ---
if st.session_state.get('df_processed') is not None and st.session_state.get('original_target_column_name'):
    target_col = st.session_state.original_target_column_name
    st.header("Resultados del Preprocesamiento y Diagnóstico")
    # ... (resto de la lógica para mostrar diagnóstico, ACF, serie histórica preprocesada,
    #      asegurándose de usar target_col o st.session_state.original_target_column_name) ...
    col1_diag, col2_acf = st.columns(2)
    with col1_diag:
        st.subheader("Diagnóstico de Datos")
        if st.session_state.data_diagnosis_report: st.markdown(st.session_state.data_diagnosis_report)
    with col2_acf:
        st.subheader("Análisis de Autocorrelación")
        if st.session_state.acf_fig: st.pyplot(st.session_state.acf_fig)
    
    st.subheader("Serie de Tiempo Preprocesada")
    fig_hist_proc = visualization.plot_historical_data(st.session_state.df_processed, target_col, title=f"Histórico de '{target_col}' (Preprocesado)")
    if fig_hist_proc: st.pyplot(fig_hist_proc)
    st.markdown("---")


    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    # ... (inputs para horizonte, periodo estacional, train/test split, AutoARIMA, etc. con KEYS ÚNICAS) ...
    # Por ejemplo:
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", min_value=1, value=st.session_state.forecast_horizon, step=1, key="horizon_cfg_widget")
    st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional:", min_value=1, value=st.session_state.user_seasonal_period, step=1, key="s_period_cfg_widget", help=f"Sugerido: {st.session_state.auto_seasonal_period}")
    # ... (Asegúrate que TODAS las keys de los widgets aquí y en los expanders sean únicas)


    if st.sidebar.button("📊 Generar y Evaluar Todos los Modelos", key="generate_models_btn_widget"):
        reset_model_results_and_processed_data() # Resetear resultados previos si se regeneran modelos
        
        # Re-leer df_processed y target_col por si acaso, aunque no deberían cambiar aquí
        df_proc_run = st.session_state.get('df_processed')
        target_col_run = st.session_state.get('original_target_column_name')

        if df_proc_run is None or target_col_run is None:
            st.error("Datos no preprocesados. Por favor, aplique preprocesamiento primero.")
        else:
            series_full_run = df_proc_run[target_col_run].copy()
            h_run = st.session_state.forecast_horizon
            s_period_eff_run = st.session_state.user_seasonal_period
            
            # ... (Lógica de train/test split como antes) ...
            train_series_run, test_series_run = series_full_run, pd.Series(dtype=series_full_run.dtype) # Default
            if st.session_state.use_train_test_split:
                # Validar test_split_size
                min_train_size = max(5, 2 * s_period_eff_run + 1 if s_period_eff_run > 1 else 5)
                if st.session_state.test_split_size < (len(series_full_run) - min_train_size) and st.session_state.test_split_size > 0:
                    train_series_run, test_series_run = forecasting_models.train_test_split_series(series_full_run, st.session_state.test_split_size)
                else:
                    st.warning(f"No es posible el split con test_size={st.session_state.test_split_size}. Se usará toda la serie para entrenar.")
                    st.session_state.use_train_test_split = False # Desactivar si no es viable
            
            st.session_state.train_series_for_plot = train_series_run
            st.session_state.test_series_for_plot = test_series_run
            
            # ... (Lógica para ejecutar todos los modelos como la tenías, asegurándote de pasar
            #      train_series_run, test_series_run, h_run, s_period_eff_run, y los parámetros
            #      de st.session_state para Holt, HW, ARIMA) ...
            
            # Ejemplo para un modelo:
            # fc_avg, _, rmse_avg, mae_avg, name_avg = forecasting_models.historical_average_forecast(train_series_run, test_series_run, h_run)
            # st.session_state.model_results.append({'name': name_avg, ...})

            # Después de correr todos los modelos:
            valid_results_sort = [res for res in st.session_state.model_results if pd.notna(res.get('rmse')) and res.get('forecast_future') is not None and len(res['forecast_future']) == h_run]
            if valid_results_sort:
                st.session_state.best_model_name_auto = min(valid_results_sort, key=lambda x: x['rmse'])['name']
            else:
                st.error("No se pudo determinar un modelo sugerido.")
                st.session_state.best_model_name_auto = None


# --- Sección de Resultados y Pestañas ---
if st.session_state.get('df_processed') is not None and st.session_state.get('model_results'):
    target_col_display = st.session_state.original_target_column_name
    st.header("Resultados del Modelado y Pronóstico")

    # ... (Lógica de las pestañas como la tenías, asegurándote de usar
    #      st.session_state.original_target_column_name o target_col_display
    #      y de que la función prepare_forecast_display_data use el índice correcto
    #      de st.session_state.df_processed[target_col_display].index) ...

    # Ejemplo dentro de una pestaña:
    # model_data_to_show = next((item for item in st.session_state.model_results if item["name"] == st.session_state.best_model_name_auto), None)
    # if model_data_to_show:
    #     final_df, fc_series, pi_df = prepare_forecast_display_data(
    #         model_data_to_show, 
    #         st.session_state.df_processed[target_col_display].index, # Pasar el índice correcto
    #         st.session_state.forecast_horizon
    #     )
    #     if final_df is not None:
    #          fig = visualization.plot_final_forecast(st.session_state.df_processed[target_col_display], fc_series, pi_df, ...)
    #          st.pyplot(fig)


else:
    if uploaded_file is None: # Solo mostrar si no se ha cargado nada aún
        st.info("👋 ¡Bienvenido! Por favor, cargue un archivo de datos para comenzar.")

# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.info("Asistente de Pronósticos PRO v3.2") # Incremento de versión