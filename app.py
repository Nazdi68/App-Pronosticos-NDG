# app.py (v4.0.2 - Flujo Simplificado Completo)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models 
import recommendations   

st.set_page_config(page_title="Asistente de Pronósticos Simplificado", layout="wide")

# --- Estado de la Sesión ---
def init_session_state_v4():
    defaults = {
        'df_loaded': None, 'current_file_name': None, 'df_processed': None, 
        'selected_date_col': None, 'selected_value_col': None, 
        'original_target_column_name': "Valor", 
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'sugerencia_modelo': None, 'sugerencia_explicacion': None,
        'modelo_ejecutado_info': None, 
        'forecast_df_final': None 
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state_v4¡Absolutamente! Pido disculpas, parece que la respuesta se cortó.

Aquí tienes el archivo `app.py` completo (v4.0.2 - Flujo Simplificado Completo), continuando desde donde se quedó y asegurando que toda la lógica esté presente.

```python
# app.py (v4.0.2 - Flujo Simplificado Completo)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models 
import recommendations   

st.set_page_config(page_title="Asistente de Pronósticos Simplificado", layout="wide")

# --- Estado de la Sesión ---
def init_session_state_v4():
    defaults = {
        'df_loaded': None, 'current_file_name': None, 'df_processed': None, 
        'selected_date_col': None, 'selected_value_col': None, 
        'original_target_column_name': "Valor", 
        'data_diagnosis_report': None, 'acf_fig': None,
        'forecast_horizon': 12, 'user_seasonal_period': 1, 'auto_seasonal_period': 1,
        'sugerencia_modelo': None, 'sugerencia_explicacion': None,
        'modelo_ejecutado_info': None, 
        'forecast_df_final': None 
        # Eliminamos moving_avg_window y parámetros de modelos complejos del estado por ahora
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state_v4()

# --- Funciones Auxiliares ---
def to_excel(df):
    output = BytesIO(); df.to_excel(output, index=True, sheet_name='Pronostico')
    return output.getvalue()

def reset_on_file_change_v4():
    keys_to_reset = [
        'df_processed', 'selected_date_col', 'selected_value_col', 
        'original_target_column_name', 'data_diagnosis_report', 'acf_fig', 
        'sugerencia_modelo', 'sugerencia_explicacion',
        'modelo_ejecutado_info', 'forecast_df_final',
        'auto_seasonal_period', 'user_seasonal_period'
    ]
    for key_to_del in keys_to_reset:
        if key_to_del in st.session_state: del st.session_state[key_to_del]
    st.session_state.df_loaded = None
    init_session_state_v4() 

def reset_after_preproc_params_change_v4(): # Llamado si cambia Frecuencia o Imputación
    st.session_state.df_processed = None # Forzar reprocesamiento y regeneración de sugerencias
    st.session_state.sugerencia_modelo = None; st.session_state.sugerencia_explicacion = None
    st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None

def sugerir_tipo_modelo_simple(serie_procesada, auto_seasonal_period=1):
    """Sugerencia de modelo muy básica basada en heurísticas simples."""
    if serie_procesada is None or serie_procesada.empty:
        return "N/A", "No hay datos suficientes para analizar y sugerir un modelo.", []
    
    sugerencias_log = []
    modelo_sugerido_tipo = "SES" 
    explicacion = "Para series cortas o sin patrones claros, SES es un buen punto de partida."

    if len(serie_procesada) < 15:
        explicacion = "Los datos son muy limitados (menos de 15 observaciones). Se sugiere un modelo baseline simple como Ingénuo o Promedio Histórico. La aplicación intentará con SES."
        sugerencias_log.append("Datos muy limitados.")
        return "SES", explicacion, sugerencias_log

    tiene_tendencia_visible = False
    tiene_estacionalidad_visible = False

    # Análisis simple de tendencia
    if len(serie_procesada) >= 20: 
        primera_mitad_mean = serie_procesada.iloc[:len(serie_procesada)//2].mean()
        segunda_mitad_mean = serie_procesada.iloc[len(serie_procesada)//2:].mean()
        # Evitar error si la media es cero o muy cercana
        denominador_tendencia = abs(primera_mitad_mean) if abs(primera_mitad_mean) > 1e-9 else 1.0
        diff_relativa_tendencia = abs(segunda_mitad_mean - primera_mitad_mean) / denominador_tendencia
        
        if diff_relativa_tendencia > 0.15: # Umbral para considerar tendencia
            sugerencias_log.append("Posible tendencia detectada.")
            tiene_tendencia_visible = True
    
    # Análisis simple de estacionalidad
    if auto_seasonal_period > 1 and len(serie_procesada) >= 2 * auto_seasonal_period:
        sugerencias_log.append(f"Posible estacionalidad con período {auto_seasonal_period} (inferida de la frecuencia y datos suficientes).")
        tiene_estacionalidad_visible = True

    # Lógica de decisión del modelo
    if tiene_tendencia_visible and tiene_estacionalidad_visible:
        modelo_sugerido_tipo = "Holt-Winters"
        explicacion = "Se detectaron posibles patrones de tendencia y estacionalidad. Se sugiere el modelo de Holt-Winters. Asegúrese de que el 'Período Estacional' sea el correcto."
    elif tiene_tendencia_visible:
        modelo_sugerido_tipo = "Holt"
        explicacion = "Se detectó una posible tendencia sin estacionalidad clara. Se sugiere el modelo de Holt."
    elif tiene_estacionalidad_visible:
        modelo_sugerido_tipo = "Holt-Winters (sin tendencia)"
        explicacion = "Se detectó posible estacionalidad sin una tendencia fuerte. Se sugiere Holt-Winters (sin componente de tendencia) o AutoARIMA con estacionalidad."
    elif len(serie_procesada) > 25: # Si no hay patrones claros pero hay suficientes datos
        modelo_sugerido_tipo = "AutoARIMA"
        explicacion = "No se detectaron patrones fuertes de tendencia o estacionalidad con el análisis simple. AutoARIMA intentará encontrar un modelo ARIMA/SARIMA adecuado automáticamente."
    
    sugerencias_log.append(f"Sugerencia final: {modelo_sugerido_tipo}. Razón: {explicacion}")
    return modelo_sugerido_tipo, explicacion, sugerencias_log


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
    forecast_values = np.array(forecast_values_raw) # Asegurar que sea un array
    
    min_len = len(forecast_dates)
    if len(forecast_values) != len(forecast_dates):
        min_len = min(len(forecast_values), len(forecast_dates))
        forecast_values = forecast_values[:min_len]
        forecast_dates = forecast_dates[:min_len]
    
    if min_len == 0: # Si no hay puntos de pronóstico después de alinear
        return pd.DataFrame(columns=['Fecha','Pronostico']).set_index('Fecha'), pd.Series(dtype='float64'), None

    conf_int_df_raw = model_data.get('conf_int_future')
    export_dict = {'Fecha': forecast_dates, 'Pronostico': forecast_values}; pi_display_df = None
    if conf_int_df_raw is not None and not conf_int_df_raw.empty:
        pi_indexed = conf_int_df_raw.copy()
        if len(pi_indexed) == len(forecast_dates): # Solo si las longitudes coinciden
            pi_indexed.index = forecast_dates
            export_dict['Limite Inferior PI'] = pi_indexed['lower'].values
            export_dict['Limite Superior PI'] = pi_indexed['upper'].values
            pi_display_df = pi_indexed[['lower', 'upper']]
            
    final_export_df = pd.DataFrame(export_dict)
    if not final_export_df.empty: 
        final_export_df = final_export_df.set_index('Fecha')
        forecast_series_for_plot = final_export_df['Pronostico']
    else: 
        forecast_series_for_plot = pd.Series(dtype='float64') # Serie vacía si el df está vacío
        final_export_df = pd.DataFrame(columns=['Fecha','Pronostico']).set_index('Fecha') # Asegurar df vacío con índice Fecha

    return final_export_df, forecast_series_for_plot, pi_display_df

# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos (Versión Simplificada)")
st.markdown("Herramienta guiada para generar pronósticos de series de tiempo.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo (CSV o Excel)", type=["csv", "xlsx", "xls"], key="uploader_key_v4_0_2", on_change=reset_on_file_change_v4)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb = st.session_state.get('df_loaded')

if df_input_sb is not None:
    date_col_options = df_input_sb.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options:
        for i, col in enumerate(date_col_options):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    sel_date_idx = date_col_options.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options else dt_col_guess_idx
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options, index=sel_date_idx, key="date_sel_key_v4_0_2")

    value_col_options = [col for col in df_input_sb.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options:
        for i, col in enumerate(value_col_options):
            if pd.api.types.is_numeric_dtype(df_input_sb[col].dropna()): val_col_guess_idx = i; break
    sel_val_idx = value_col_options.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options, index=sel_val_idx, key="val_sel_key_v4_0_2")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v4_0_2", on_change=reset_after_preproc_params_change_v4)
    desired_freq = freq_map[freq_label]
    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v4_0_2", on_change=reset_after_preproc_params_change_v4)
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v4_0_2"):
        st.session_state.df_processed = None; reset_after_preproc_params_change_v4() # Resetear antes
        date_col_btn = st.session_state.get('selected_date_col'); value_col_btn = st.session_state.get('selected_value_col'); valid_btn = True
        if not date_col_btn or date_col_btn not in df_input_sb.columns: st.sidebar.error("Seleccione fecha."); valid_btn=False
        if not value_col_btn or value_col_btn not in df_input_sb.columns: st.sidebar.error("Seleccione valor."); valid_btn=False
        elif valid_btn and not pd.api.types.is_numeric_dtype(df_input_sb[value_col_btn].dropna()): st.sidebar.error(f"'{value_col_btn}' no numérica."); valid_btn=False
        if valid_btn:
            with st.spinner("Preprocesando..."): proc_df,msg_raw = data_handler.preprocess_data(df_input_sb.copy(),date_col_btn,value_col_btn,desired_freq,imp_code)
            msg_disp = msg_raw
            if msg_raw: 
                if "MS" in msg_raw: msg_disp=msg_raw.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw: msg_disp=msg_raw.replace(" D."," D (Diario).") # Chequeo de espacio
                elif msg_raw.strip().endswith("D"): msg_disp=msg_raw.replace("D", "D (Diario)") # Chequeo sin punto

            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed=proc_df; st.session_state.original_target_column_name=value_col_btn; st.success(f"Preprocesamiento OK. {msg_disp}")
                st.session_state.data_diagnosis_report=data_handler.diagnose_data(proc_df,value_col_btn)
                if not proc_df.empty:
                    s_acf=proc_df[value_col_btn];l_acf=min(len(s_acf)//2-1,60)
                    if l_acf > 5: st.session_state.acf_fig=data_handler.plot_acf_pacf(s_acf,l_acf,value_col_btn)
                    else: st.session_state.acf_fig=None
                    _,auto_s_val=data_handler.get_series_frequency_and_period(proc_df.index)
                    st.session_state.auto_seasonal_period=auto_s_val
                    st.session_state.user_seasonal_period = auto_s_val 
                    st.session_state.sugerencia_modelo, st.session_state.sugerencia_explicacion, _ = sugerir_tipo_modelo_simple(s_acf, auto_seasonal_period=auto_s_val)
            else: st.error(f"Fallo preproc: {msg_raw or 'DataFrame vacío.'}"); st.session_state.df_processed=None

# --- Mostrar Diagnóstico y Gráficos Iniciales + Sugerencia ---
df_processed_main = st.session_state.get('df_processed')
target_col_main = st.session_state.get('original_target_column_name')

if df_processed_main is not None and not df_processed_main.empty and target_col_main:
    st.header("1. Resultados del Preprocesamiento y Diagnóstico")
    col1_diag, col2_acf = st.columns(2)
    with col1_diag: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf: 
        st.subheader("Autocorrelación (ACF/PACF)")
        acf_fig_plot = st.session_state.get('acf_fig')
        if acf_fig_plot is not None: 
            try: st.pyplot(acf_fig_plot)
            except Exception as e_acf: st.error(f"Error al mostrar ACF/PACF: {e_acf}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie de Tiempo Preprocesada")
    if target_col_main in df_processed_main.columns:
        fig_hist_plot = visualization.plot_historical_data(df_processed_main, target_col_main, f"Histórico de '{target_col_main}'")
        if fig_hist_plot: st.pyplot(fig_hist_plot)
    st.markdown("---")

    if st.session_state.get('sugerencia_modelo') and st.session_state.get('sugerencia_explicacion'):
        st.subheader("🧠 Sugerencia de Modelo Inicial")
        st.info(f"**Tipo de Modelo Sugerido:** {st.session_state.sugerencia_modelo}\n\n**Razón:** {st.session_state.sugerencia_explicacion}")
        st.markdown("Esta es una sugerencia basada en un análisis simple. La aplicación intentará ajustar este tipo de modelo o AutoARIMA si es apropiado.")
    st.markdown("---")

    # --- Sección 2: Configuración del Pronóstico (Sidebar) ---
    st.sidebar.header("2. Configuración del Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v4_simple")
    if st.session_state.get('sugerencia_modelo') and ("Holt-Winters" in st.session_state.sugerencia_modelo or "AutoARIMA" in st.session_state.sugerencia_modelo):
        st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional (si aplica):", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v4_simple", help=f"Detectado: {st.session_state.auto_seasonal_period}")
    
    if st.sidebar.button("🚀 Generar Pronóstico", key="gen_forecast_btn_v4_action"):
        st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
        modelo_a_usar = st.session_state.get('sugerencia_modelo', "AutoARIMA")
        serie_pron = df_processed_main[target_col_main].copy() # Usar variables definidas arriba
        h_pron_val = st.session_state.forecast_horizon
        s_period_pron_val = st.session_state.get('user_seasonal_period', 1)
        fc_vals, ci_df, fitted_vals, nombre_modelo = None, None, None, f"{modelo_a_usar} (No Ejecutado)"

        with st.spinner(f"Ajustando '{modelo_a_usar}' y generando pronóstico..."):
            try:
                if modelo_a_usar == "SES": fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.run_ses_simple(serie_pron, h_pron_val)
                elif modelo_a_usar == "Holt": fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.run_holt_simple(serie_pron, h_pron_val, damped=False) # Default damped a False
                elif "Holt-Winters" in modelo_a_usar:
                    trend_hw = 'add' if "(sin tendencia)" not in modelo_a_usar else None
                    fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.run_hw_simple(serie_pron,h_pron_val,s_period_pron_val,trend=trend_hw,seasonal='add')
                elif modelo_a_usar == "AutoARIMA":
                    arima_params_s = {'max_p':3,'max_q':3,'max_d':2,'max_P':1,'max_Q':1,'max_D':1}
                    fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.run_autoarima_simple(serie_pron,h_pron_val,s_period_pron_val,arima_params=arima_params_s)
                elif modelo_a_usar == "Promedio Histórico": fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.historical_average_simple(serie_pron,h_pron_val) # Asumiendo que existe en forecasting_models
                elif modelo_a_usar == "Ingénuo": fc_vals,ci_df,fitted_vals,nombre_modelo = forecasting_models.naive_simple(serie_pron,h_pron_val) # Asumiendo que existe

                if fc_vals is not None and fitted_vals is not None:
                    rmse_calc_val, mae_calc_val = np.nan, np.nan
                    if len(fitted_vals) == len(serie_pron): rmse_calc_val,mae_calc_val = forecasting_models.calculate_metrics(serie_pron,fitted_vals)
                    else:
                        offset_val = len(serie_pron) - len(fitted_vals)
                        if offset_val >= 0 and len(fitted_vals) > 0: rmse_calc_val,mae_calc_val = forecasting_models.calculate_metrics(serie_pron[offset_val:], fitted_vals)
                    st.session_state.modelo_ejecutado_info = {"name":nombre_modelo,"rmse":rmse_calc_val,"mae":mae_calc_val,"forecast_future":fc_vals,"conf_int_future":ci_df}
                    df_fc_final_btn_res,_,_ = prepare_forecast_display_data(st.session_state.modelo_ejecutado_info,serie_pron.index,h_pron_val)
                    st.session_state.forecast_df_final = df_fc_final_btn_res
                    st.success(f"Pronóstico generado con: {nombre_modelo}")
                else: st.error(f"Modelo '{nombre_modelo or modelo_a_usar}' no generó pronóstico."); st.session_state.modelo_ejecutado_info = {"name": (nombre_modelo or modelo_a_usar) + " (FALLÓ)", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None}
            except Exception as e_model_run: st.error(f"Error ejecutando '{modelo_a_usar}': {e_model_run}"); st.session_state.modelo_ejecutado_info = {"name": modelo_a_usar + f" (FALLÓ: {type(e_model_run).__name__})", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None}

# --- Mostrar Resultados del Pronóstico ---
if st.session_state.get('modelo_ejecutado_info') and st.session_state.get('forecast_df_final') is not None:
    st.header("2. Resultados del Pronóstico")
    info_modelo_final_disp = st.session_state.modelo_ejecutado_info
    df_pronostico_final_disp = st.session_state.forecast_df_final
    serie_historica_final_disp = st.session_state.df_processed[st.session_state.original_target_column_name]

    st.subheader(f"Modelo Utilizado: {info_modelo_final_disp['name']}")
    if pd.notna(info_modelo_final_disp.get('rmse')):
        st.markdown(f"**Métricas de Ajuste (In-Sample):** RMSE = {info_modelo_final_disp['rmse']:.2f}, MAE = {info_modelo_final_disp['mae']:.2f}")
    
    pi_df_plot_final = None
    if 'Limite Inferior PI' in df_pronostico_final_disp.columns and 'Limite Superior PI' in df_pronostico_final_disp.columns:
        pi_df_plot_final = df_pronostico_final_disp[['Limite Inferior PI', 'Limite Superior PI']]

    fig_pronostico_final_v4_disp = visualization.plot_final_forecast(serie_historica_final_disp, df_pronostico_final_disp['Pronostico'],pi_df_plot_final, model_name=info_modelo_final_disp['name'],value_col_name=st.session_state.original_target_column_name)
    if fig_pronostico_final_v4_disp: st.pyplot(fig_pronostico_final_v4_disp)
    else: st.warning("No se pudo generar el gráfico del pronóstico.")

    st.markdown("##### Valores del Pronóstico"); st.dataframe(df_pronostico_final_disp.style.format("{:.2f}"))
    excel_data_final_v4_btn = to_excel(df_pronostico_final_disp)
    dl_key_final_v4 = f"dl_fc_simple_{info_modelo_final_disp['name'][:10].replace(' ','_')}_v4"
    st.download_button(f"📥 Descargar ({info_modelo_final_disp['name']})", excel_data_final_v4_btn, f"pronostico_{st.session_state.original_target_column_name}.xlsx", key=dl_key_final_v4)
    st.markdown("---")
    st.subheader("💡 Recomendaciones y Próximos Pasos")
    st.markdown(recommendations.generate_recommendations_simple( 
        selected_model_name=info_modelo_final_disp['name'],
        data_diag_summary=st.session_state.data_diagnosis_report,
        has_pis=(pi_df_plot_final is not None and not pi_df_plot_final.empty),
        target_column_name=st.session_state.original_target_column_name,
        model_rmse=info_modelo_final_disp.get('rmse'), model_mae=info_modelo_final_disp.get('mae'),
        forecast_horizon=st.session_state.forecast_horizon
    ))

elif uploaded_file is None and st.session_state.get('df_loaded') is None : 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') is not None and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
elif st.session_state.get('df_loaded') is not None and \
     st.session_state.get('df_processed') is not None and \
     not st.session_state.get('df_processed').empty and \
     st.session_state.get('modelo_ejecutado_info') is None:
     if st.session_state.get('sugerencia_modelo'):
         st.info(f"Datos preprocesados. El modelo sugerido es **{st.session_state.sugerencia_modelo}**. Configure el horizonte y haga clic en 'Generar Pronóstico'.")
     else:
         st.info("Datos preprocesados. Configure el horizonte y haga clic en 'Generar Pronóstico'.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v4.0.2")    
 