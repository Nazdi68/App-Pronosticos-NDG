# app.py (v4.0.1 - Depuración Simplificada)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Asegúrate de que estos archivos .py estén en el mismo directorio que app.py
import data_handler
import visualization
import forecasting_models # Necesitará las funciones run_..._simple
import recommendations   # Necesitará generate_recommendations_simple

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

def reset_after_preproc_params_change_v4():
    st.session_state.df_processed = None 
    st.session_state.sugerencia_modelo = None; st.session_state.sugerencia_explicacion = None
    st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
    st.session_state.data_diagnosis_report = None; st.session_state.acf_fig = None

def sugerir_tipo_modelo_simple(serie, auto_seasonal_period=1):
    if serie is None or serie.empty:
        return "N/A", "No hay datos para analizar.", []
    sugerencias = []
    modelo_sugerido_tipo = "SES" 
    explicacion = "Patrones no claros; SES o Naive como base."

    if len(serie) < 15: # Muy pocos datos para algo complejo
        explicacion = "Datos muy limitados. Se sugiere un modelo baseline simple como Naive o Promedio Histórico (no implementado como sugerencia automática aún)."
        return "Baselines Simples", explicacion, ["Datos limitados."]

    tiene_tendencia_visible = False
    tiene_estacionalidad_visible = False

    if len(serie) >= 20: # Umbral mínimo para análisis de tendencia/estacionalidad simple
        primera_mitad_mean = serie.iloc[:len(serie)//2].mean()
        segunda_mitad_mean = serie.iloc[len(serie)//2:].mean()
        diff_relativa = abs(segunda_mitad_mean - primera_mitad_mean) / (abs(primera_mitad_mean) + 1e-9) # Evitar división por cero
        if diff_relativa > 0.15: # Umbral más alto para tendencia
            sugerencias.append("Posible tendencia detectada.")
            tiene_tendencia_visible = True
    
    if auto_seasonal_period > 1 and len(serie) >= 2 * auto_seasonal_period:
        sugerencias.append(f"Posible estacionalidad con período {auto_seasonal_period} inferida de la frecuencia.")
        tiene_estacionalidad_visible = True

    if tiene_tendencia_visible and tiene_estacionalidad_visible:
        modelo_sugerido_tipo = "Holt-Winters"
        explicacion = "Se detectó posible tendencia y estacionalidad. Holt-Winters podría ser adecuado. Verifique el 'Período Estacional'."
    elif tiene_tendencia_visible:
        modelo_sugerido_tipo = "Holt"
        explicacion = "Se detectó posible tendencia sin estacionalidad clara. El modelo de Holt podría ser adecuado."
    elif tiene_estacionalidad_visible: # Estacionalidad sin tendencia clara
        modelo_sugerido_tipo = "Holt-Winters (sin tendencia)"
        explicacion = "Se detectó posible estacionalidad. Holt-Winters (configurado sin tendencia) o AutoARIMA con estacionalidad podrían ser adecuados."
    elif len(serie) > 30: # Si no hay patrones claros pero hay suficientes datos
        modelo_sugerido_tipo = "AutoARIMA"
        explicacion = "No se detectaron patrones fuertes de tendencia o estacionalidad. AutoARIMA intentará encontrar un modelo ARIMA/SARIMA adecuado."
    
    return modelo_sugerido_tipo, explicacion, sugerencias

# --- Interfaz de Usuario ---
st.title("🔮 Asistente de Pronósticos (Versión Simplificada)")
st.markdown("Herramienta guiada para generar pronósticos de series de tiempo.")

st.sidebar.header("1. Carga y Preprocesamiento")
uploaded_file = st.sidebar.file_uploader("Suba su archivo (CSV o Excel)", type=["csv", "xlsx", "xls"], key="uploader_v4_debug", on_change=reset_on_file_change_v4)

if uploaded_file:
    if st.session_state.df_loaded is None: 
        st.session_state.df_loaded = data_handler.load_data(uploaded_file)
        if st.session_state.df_loaded is not None: st.session_state.current_file_name = uploaded_file.name
        else: st.session_state.current_file_name = None; st.sidebar.error("No se pudo cargar el archivo.")

df_input_sb = st.session_state.get('df_loaded')

if df_input_sb is not None:
    date_col_options_sb = df_input_sb.columns.tolist()
    dt_col_guess_idx = 0
    if date_col_options_sb:
        for i, col in enumerate(date_col_options_sb):
            if any(keyword in str(col).lower() for keyword in ['date', 'fecha', 'time', 'periodo']): dt_col_guess_idx = i; break
    sel_date_idx = date_col_options_sb.index(st.session_state.selected_date_col) if st.session_state.get('selected_date_col') and st.session_state.selected_date_col in date_col_options_sb else dt_col_guess_idx
    st.session_state.selected_date_col = st.sidebar.selectbox("Columna de Fecha/Hora:", date_col_options_sb, index=sel_date_idx, key="date_sel_key_v4_debug")

    value_col_options_sb = [col for col in df_input_sb.columns if col != st.session_state.get('selected_date_col')]
    val_col_guess_idx = 0
    if value_col_options_sb:
        for i, col in enumerate(value_col_options_sb):
            if pd.api.types.is_numeric_dtype(df_input_sb[col].dropna()): val_col_guess_idx = i; break
    sel_val_idx = value_col_options_sb.index(st.session_state.selected_value_col) if st.session_state.get('selected_value_col') and st.session_state.selected_value_col in value_col_options_sb else val_col_guess_idx
    st.session_state.selected_value_col = st.sidebar.selectbox("Columna a Pronosticar:", value_col_options_sb, index=sel_val_idx, key="val_sel_key_v4_debug")
    
    freq_map = {"Original/Inferida": None, "Diaria": "D", "Semanal": "W-MON", "Mensual": "MS", "Trimestral": "QS", "Anual": "AS"}
    freq_label = st.sidebar.selectbox("Frecuencia:", options=list(freq_map.keys()), key="freq_sel_key_v4_debug", on_change=reset_after_preproc_params_change_v4)
    desired_freq = freq_map[freq_label]
    imp_list = ["No imputar", "Interpolar Lineal", "Adelante (ffill)", "Atrás (bfill)", "Media", "Mediana"]
    imp_label = st.sidebar.selectbox("Imputación Faltantes:", imp_list, index=1, key="imp_sel_key_v4_debug", on_change=reset_after_preproc_params_change_v4)
    imp_code = None if imp_label == "No imputar" else imp_label.split('(')[0].strip()

    if st.sidebar.button("Aplicar Preprocesamiento", key="preproc_btn_key_v4_debug"):
        st.session_state.df_processed = None; reset_after_preproc_params_change_v4() 
        date_col_btn = st.session_state.get('selected_date_col'); value_col_btn = st.session_state.get('selected_value_col'); valid_btn = True
        if not date_col_btn or date_col_btn not in df_input_sb.columns: st.sidebar.error("Seleccione fecha."); valid_btn=False
        if not value_col_btn or value_col_btn not in df_input_sb.columns: st.sidebar.error("Seleccione valor."); valid_btn=False
        elif valid_btn and not pd.api.types.is_numeric_dtype(df_input_sb[value_col_btn].dropna()): st.sidebar.error(f"'{value_col_btn}' no numérica."); valid_btn=False
        if valid_btn:
            with st.spinner("Preprocesando..."): proc_df,msg_raw = data_handler.preprocess_data(df_input_sb.copy(),date_col_btn,value_col_btn,desired_freq,imp_code)
            msg_disp = msg_raw; 
            if msg_raw: 
                if "MS" in msg_raw: msg_disp=msg_raw.replace("MS","MS (Inicio de Mes - Mensual)")
                elif " D." in msg_raw: msg_disp=msg_raw.replace(" D."," D (Diario).")
                elif msg_raw.endswith("D"): msg_disp=msg_raw.replace("D", "D (Diario)")
            if proc_df is not None and not proc_df.empty:
                st.session_state.df_processed=proc_df; st.session_state.original_target_column_name=value_col_btn; st.success(f"Preproc. OK. {msg_disp}")
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

df_processed_main_v4_disp = st.session_state.get('df_processed')
target_col_main_v4_disp = st.session_state.get('original_target_column_name')

if df_processed_main_v4_disp is not None and not df_processed_main_v4_disp.empty and target_col_main_v4_disp:
    st.header("1. Resultados del Preprocesamiento y Diagnóstico")
    col1_diag_v4_disp, col2_acf_v4_disp = st.columns(2)
    with col1_diag_v4_disp: st.subheader("Diagnóstico"); st.markdown(st.session_state.data_diagnosis_report or "N/A")
    with col2_acf_v4_disp: 
        st.subheader("Autocorrelación (ACF/PACF)")
        acf_fig_v4_disp = st.session_state.get('acf_fig')
        if acf_fig_v4_disp: 
            try: st.pyplot(acf_fig_v4_disp)
            except Exception as e_acf_v4_disp: st.error(f"Error al mostrar ACF/PACF: {e_acf_v4_disp}")
        else: st.info("ACF/PACF no disponible.")
    st.subheader("Serie de Tiempo Preprocesada")
    if target_col_main_v4_disp in df_processed_main_v4_disp.columns:
        fig_hist_v4_disp = visualization.plot_historical_data(df_processed_main_v4_disp, target_col_main_v4_disp, f"Histórico de '{target_col_main_v4_disp}'")
        if fig_hist_v4_disp: st.pyplot(fig_hist_v4_disp)
    st.markdown("---")

    if st.session_state.get('sugerencia_modelo') and st.session_state.get('sugerencia_explicacion'):
        st.subheader("🧠 Sugerencia de Modelo Inicial")
        st.info(f"**Tipo de Modelo Sugerido:** {st.session_state.sugerencia_modelo}\n\n**Razón:** {st.session_state.sugerencia_explicacion}")
        st.markdown("Esta es una sugerencia basada en un análisis simple. La aplicación intentará ajustar este tipo de modelo o AutoARIMA.")
    st.markdown("---")

    st.sidebar.header("2. Configuración del Pronóstico")
    st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte de Pronóstico:", value=st.session_state.forecast_horizon, min_value=1, step=1, key="h_key_v4_simple_debug")
    if st.session_state.get('sugerencia_modelo') and ("Holt-Winters" in st.session_state.sugerencia_modelo or "AutoARIMA" in st.session_state.sugerencia_modelo):
        st.session_state.user_seasonal_period = st.sidebar.number_input("Período Estacional (si aplica):", value=st.session_state.user_seasonal_period, min_value=1, step=1, key="s_key_v4_simple_debug", help=f"Detectado: {st.session_state.auto_seasonal_period}")
    
    if st.sidebar.button("🚀 Generar Pronóstico", key="gen_forecast_btn_v4_debug_action"):
        st.session_state.modelo_ejecutado_info = None; st.session_state.forecast_df_final = None
        modelo_a_usar_btn = st.session_state.get('sugerencia_modelo', "AutoARIMA")
        serie_a_pronosticar_btn = df_processed_main_v4_disp[target_col_main_v4_disp].copy()
        h_pron_btn = st.session_state.forecast_horizon
        s_period_pron_btn = st.session_state.get('user_seasonal_period', 1)
        fc_vals_res, ci_df_res, fitted_vals_res, nombre_modelo_res = None, None, None, f"{modelo_a_usar_btn} (No Ejecutado)"

        with st.spinner(f"Ajustando '{modelo_a_usar_btn}' y generando pronóstico..."):
            try:
                if modelo_a_usar_btn == "SES":
                    fc_vals_res,ci_df_res,fitted_vals_res,nombre_modelo_res = forecasting_models.run_ses_simple(serie_a_pronosticar_btn, h_pron_btn)
                elif modelo_a_usar_btn == "Holt":
                    fc_vals_res,ci_df_res,fitted_vals_res,nombre_modelo_res = forecasting_models.run_holt_simple(serie_a_pronosticar_btn, h_pron_btn, damped=False)
                elif "Holt-Winters" in modelo_a_usar_btn:
                    trend_hw_run = 'add' if "(sin tendencia)" not in modelo_a_usar_btn else None
                    fc_vals_res,ci_df_res,fitted_vals_res,nombre_modelo_res = forecasting_models.run_hw_simple(serie_a_pronosticar_btn,h_pron_btn,s_period_pron_btn,trend=trend_hw_run,seasonal='add')
                elif modelo_a_usar_btn == "AutoARIMA":
                    arima_params_run_simple = {'max_p':3,'max_q':3,'max_d':2,'max_P':1,'max_Q':1,'max_D':1}
                    fc_vals_res,ci_df_res,fitted_vals_res,nombre_modelo_res = forecasting_models.run_autoarima_simple(serie_a_pronosticar_btn,h_pron_btn,s_period_pron_btn,arima_params=arima_params_run_simple)
                elif modelo_a_usar_btn == "Promedio Histórico":
                     fc_vals_res, ci_df_res, fitted_vals_res, nombre_modelo_res = forecasting_models.historical_average_simple(serie_a_pronosticar_btn, h_pron_btn)
                elif modelo_a_usar_btn == "Ingénuo":
                     fc_vals_res, ci_df_res, fitted_vals_res, nombre_modelo_res = forecasting_models.naive_simple(serie_a_pronosticar_btn, h_pron_btn)
                else: st.error(f"Modelo '{modelo_a_usar_btn}' no implementado."); nombre_modelo_res = f"{modelo_a_usar_btn} (No Imp.)"

                if fc_vals_res is not None and fitted_vals_res is not None:
                    rmse_calc, mae_calc = np.nan, np.nan
                    if len(fitted_vals_res) == len(serie_a_pronosticar_btn): rmse_calc,mae_calc = forecasting_models.calculate_metrics(serie_a_pronosticar_btn,fitted_vals_res)
                    else:
                        offset = len(serie_a_pronosticar_btn) - len(fitted_vals_res)
                        if offset >= 0 and len(fitted_vals_res) > 0: rmse_calc,mae_calc = forecasting_models.calculate_metrics(serie_a_pronosticar_btn[offset:], fitted_vals_res)
                    st.session_state.modelo_ejecutado_info = {"name":nombre_modelo_res,"rmse":rmse_calc,"mae":mae_calc,"forecast_future":fc_vals_res,"conf_int_future":ci_df_res}
                    idx_fc_final = serie_a_pronosticar_btn.index
                    df_fc_final_btn,_,_ = prepare_forecast_display_data(st.session_state.modelo_ejecutado_info,idx_fc_final,h_pron_btn)
                    st.session_state.forecast_df_final = df_fc_final_btn
                    st.success(f"Pronóstico generado con: {nombre_modelo_res}")
                else: st.error(f"Modelo '{nombre_modelo_res}' no generó pronóstico."); st.session_state.modelo_ejecutado_info = {"name": nombre_modelo_res + " (FALLÓ)", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None}
            except Exception as e_model_run_v4: st.error(f"Error ejecutando '{modelo_a_usar_btn}': {e_model_run_v4}"); st.session_state.modelo_ejecutado_info = {"name": modelo_a_usar_btn + f" (FALLÓ: {type(e_model_run_v4).__name__})", "rmse":np.nan, "mae":np.nan, "forecast_future": None, "conf_int_future": None}

# --- Mostrar Resultados del Pronóstico ---
if st.session_state.get('modelo_ejecutado_info') and st.session_state.get('forecast_df_final') is not None:
    st.header("2. Resultados del Pronóstico")
    info_modelo_disp = st.session_state.modelo_ejecutado_info
    df_pronostico_disp = st.session_state.forecast_df_final
    serie_historica_disp = st.session_state.df_processed[st.session_state.original_target_column_name]

    st.subheader(f"Modelo Utilizado: {info_modelo_disp['name']}")
    if pd.notna(info_modelo_disp.get('rmse')):
        st.markdown(f"**Métricas de Ajuste (In-Sample):** RMSE = {info_modelo_disp['rmse']:.2f}, MAE = {info_modelo_disp['mae']:.2f}")
    
    pi_df_para_plot = None
    if 'Limite Inferior PI' in df_pronostico_disp.columns and 'Limite Superior PI' in df_pronostico_disp.columns:
        pi_df_para_plot = df_pronostico_disp[['Limite Inferior PI', 'Limite Superior PI']]

    fig_pronostico_final_v4 = visualization.plot_final_forecast(
        serie_historica_disp, df_pronostico_disp['Pronostico'],
        pi_df_para_plot, model_name=info_modelo_disp['name'],
        value_col_name=st.session_state.original_target_column_name
    )
    if fig_pronostico_final_v4: st.pyplot(fig_pronostico_final_v4)
    else: st.warning("No se pudo generar el gráfico del pronóstico.")

    st.markdown("##### Valores del Pronóstico"); st.dataframe(df_pronostico_disp.style.format("{:.2f}"))
    excel_data_final_v4 = to_excel(df_pronostico_disp)
    st.download_button(f"📥 Descargar ({info_modelo['name']})",
        excel_data_final_v4,
        f"pronostico_{st.session_state.original_target_column_name}.xlsx",
        key=f"dl_forecast_simple_v4_debug_{info_modelo['name'].replace(' ','_')}" # Key única
    )
    st.markdown("---")
    st.subheader("💡 Recomendaciones y Próximos Pasos")
    # ... (código de la llamada a recommendations.generate_recommendations_simple)
    pi_df_para_rec = None
    if 'forecast_df_final' in st.session_state and \
       st.session_state.forecast_df_final is not None and \
       'Limite Inferior PI' in st.session_state.forecast_df_final.columns:
        pi_df_para_rec = st.session_state.forecast_df_final[['Limite Inferior PI', 'Limite Superior PI']]

    recomendaciones_texto = recommendations.generate_recommendations_simple(
        selected_model_name=info_modelo['name'], 
        data_diag_summary=st.session_state.data_diagnosis_report,
        has_pis=(pi_df_para_rec is not None and not pi_df_para_rec.empty),
        target_column_name=st.session_state.original_target_column_name,
        model_rmse=info_modelo.get('rmse'),
        model_mae=info_modelo.get('mae'),
        forecast_horizon=st.session_state.forecast_horizon
    )
    st.markdown(recomendaciones_texto)

# --- Mensajes si las pestañas no se muestran ---
# (El resto del código, como en la v4.0.1)
elif uploaded_file is None and st.session_state.get('df_loaded') is None : 
    st.info("👋 ¡Bienvenido! Cargue un archivo para comenzar.")
elif st.session_state.get('df_loaded') is not None and \
     (st.session_state.get('df_processed') is None or \
      (isinstance(st.session_state.get('df_processed'), pd.DataFrame) and st.session_state.get('df_processed').empty)):
    st.warning("⚠️ Por favor, aplique preprocesamiento a los datos cargados o verifique el resultado.")
elif st.session_state.get('df_loaded') is not None and \
     st.session_state.get('df_processed') is not None and \
     not st.session_state.get('df_processed').empty and \
     st.session_state.get('modelo_ejecutado_info') is None: # Si se preprocesó pero no se generó pronóstico
     if st.session_state.get('sugerencia_modelo'):
         st.info(f"Datos preprocesados. El modelo sugerido es **{st.session_state.sugerencia_modelo}**. Configure el horizonte y haga clic en 'Generar Pronóstico'.")
     else:
         st.info("Datos preprocesados. Configure el horizonte y haga clic en 'Generar Pronóstico'.")

st.sidebar.markdown("---"); st.sidebar.info("Asistente de Pronósticos PRO v4.0.2")      
 