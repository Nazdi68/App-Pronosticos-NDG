# data_handler.py
import pandas as pd
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Añadido plot_pacf
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Formato de archivo no soportado. Por favor, suba un CSV o Excel.")
                return None
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None
    return None

def preprocess_data(df, date_column_name, target_column_name, desired_freq=None, imputation_method=None):
    if df is None or df.empty:
        return None, "No hay datos para procesar."
    if not date_column_name or not target_column_name:
        return None, "Columnas de fecha u objetivo no especificadas."
    if date_column_name not in df.columns or target_column_name not in df.columns:
        return None, "Las columnas de fecha u objetivo especificadas no se encuentran en los datos."


    try:
        df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')
        if df[date_column_name].isnull().all(): # Si todas las fechas son NaT después de la conversión
            return None, f"La columna de fecha '{date_column_name}' no contiene fechas válidas."
        # Eliminar filas donde la fecha es NaT ANTES de establecer el índice
        df.dropna(subset=[date_column_name], inplace=True)
        if df.empty:
            return None, "No quedan datos después de eliminar fechas inválidas."

        df = df.set_index(date_column_name)
        df = df.sort_index()
    except Exception as e:
        return None, f"Error al procesar la columna de fecha '{date_column_name}': {e}"

    if target_column_name not in df.columns: # Doble check por si acaso
        return None, f"La columna objetivo '{target_column_name}' no se encontró después de indexar."
    
    series = df[[target_column_name]].copy() 
    
    try:
        series[target_column_name] = pd.to_numeric(series[target_column_name], errors='coerce')
    except Exception as e:
        return None, f"La columna '{target_column_name}' no pudo ser convertida a tipo numérico: {e}"

    if series[target_column_name].isnull().all():
        return None, f"La columna objetivo '{target_column_name}' está completamente vacía o no es numérica después de la conversión."

    initial_freq_inferred = pd.infer_freq(series.index)
    current_freq_msg = f"Frecuencia original inferida: {initial_freq_inferred if initial_freq_inferred else 'No regular'}."
    
    if desired_freq:
        try:
            # Lógica de agregación para downsampling
            if desired_freq in ['MS', 'QS', 'AS'] and initial_freq_inferred and initial_freq_inferred not in ['MS', 'QS', 'AS']: # Ej: D a M
                 # Asegurar que el índice sea DatetimeIndex para resample
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                series = series.resample(desired_freq).mean() # Usar media por defecto
            else: # Upsampling o cambio de frecuencia
                series = series.asfreq(desired_freq)
            current_freq_msg = f"Datos remuestreados a frecuencia '{desired_freq}'."
            st.success(current_freq_msg)
        except Exception as e:
            st.warning(f"No se pudo remuestrear a '{desired_freq}': {e}. Se usará frecuencia original/inferida.")
    elif initial_freq_inferred: # Si no se desea una frecuencia específica pero se infiere una, asegurar regularidad
        try:
            series = series.asfreq(initial_freq_inferred)
        except ValueError: # Si falla asfreq (ej. duplicados)
            series.index = series.index.drop_duplicates(keep='first')
            series = series.asfreq(initial_freq_inferred)


    missing_before_imputation = series[target_column_name].isnull().sum()
    if missing_before_imputation > 0 and imputation_method:
        imputation_applied_msg = f"Aplicando imputación: '{imputation_method}' a {missing_before_imputation} valores faltantes."
        if imputation_method == 'Interpolar Lineal':
            series[target_column_name] = series[target_column_name].interpolate(method='linear')
        elif imputation_method == 'Adelante': # Simplificado
            series[target_column_name] = series[target_column_name].fillna(method='ffill')
        elif imputation_method == 'Atrás': # Simplificado
            series[target_column_name] = series[target_column_name].fillna(method='bfill')
        elif imputation_method == 'Media':
            series[target_column_name] = series[target_column_name].fillna(series[target_column_name].mean())
        elif imputation_method == 'Mediana':
            series[target_column_name] = series[target_column_name].fillna(series[target_column_name].median())
        
        series[target_column_name] = series[target_column_name].fillna(method='bfill').fillna(method='ffill') # Relleno final
        
        missing_after_imputation = series[target_column_name].isnull().sum()
        if missing_after_imputation == 0 :
            st.success(f"{imputation_applied_msg} Imputación exitosa.")
        else:
            st.warning(f"{imputation_applied_msg} Aún quedan {missing_after_imputation} NaNs.")

    if series[target_column_name].isnull().any():
        st.error(f"La serie '{target_column_name}' todavía contiene NaNs después del preprocesamiento. No se puede continuar.")
        return None, "NaNs persistentes en la serie objetivo."
    
    if series.empty:
        return None, "La serie de datos quedó vacía después del preprocesamiento."

    return series, current_freq_msg


def diagnose_data(series_df, value_column_name):
    if series_df is None or value_column_name not in series_df.columns:
        return "No hay datos o columna de valor no encontrada para diagnosticar."
    series = series_df[value_column_name].copy()
    if series.empty:
        return "La serie para diagnosticar está vacía."
    
    report = [f"**Diagnóstico de Datos para '{value_column_name}' (después del preprocesamiento):**"]
    report.append(f"- Observaciones: {len(series)}")
    
    if isinstance(series.index, pd.DatetimeIndex) and series.index.name:
        report.append(f"- Rango de fechas: {series.index.min().strftime('%Y-%m-%d')} a {series.index.max().strftime('%Y-%m-%d')}")
        inferred_freq = pd.infer_freq(series.index)
        report.append(f"- Frecuencia de datos: {inferred_freq if inferred_freq else 'No regular / No inferida'}")
    
    missing_values = series.isnull().sum()
    report.append(f"- Valores faltantes: {missing_values} ({(missing_values / len(series) * 100) if len(series)>0 else 0:.2f}%)")

    Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
    lower_b = Q1 - 1.5 * IQR; upper_b = Q3 + 1.5 * IQR
    outliers = series[(series < lower_b) | (series > upper_b)]
    report.append(f"- Posibles atípicos (1.5*IQR): {len(outliers)} ({(len(outliers)/len(series)*100) if len(series)>0 else 0 :.2f}%)")
    report.append(f"  - Rango normal (IQR): {lower_b:.2f} - {upper_b:.2f}")
    report.append(f"  - Mín: {series.min():.2f}, Máx: {series.max():.2f}")
    if len(outliers) > 0: report.append("  - *Revisar atípicos; pueden influir en el pronóstico.*")

    return "\n".join(report)

def get_series_frequency_and_period(idx):
    inferred_freq = pd.infer_freq(idx)
    seasonal_period = 1
    if inferred_freq:
        if 'M' in inferred_freq: seasonal_period = 12
        elif 'Q' in inferred_freq: seasonal_period = 4
        elif 'W' in inferred_freq: seasonal_period = 52
        elif 'D' in inferred_freq: seasonal_period = 7
        elif 'B' in inferred_freq: seasonal_period = 5
        elif 'H' in inferred_freq: seasonal_period = 24
    return inferred_freq, seasonal_period

def plot_acf_pacf(series, lags=None, target_column_name=""):
    if series is None or series.empty or len(series) <=1 : # Necesita al menos 2 puntos para ACF/PACF
        # st.warning("Datos insuficientes para graficar ACF/PACF.")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # Más compacto
    try:
        plot_acf(series, ax=axes[0], lags=lags, title=f'ACF - {target_column_name}', zero=False) # zero=False para no mostrar lag 0
        axes[0].grid(True)
    except Exception as e_acf: # Capturar error si ACF falla
        axes[0].text(0.5, 0.5, f'Error al generar ACF:\n{str(e_acf)[:50]}...', ha='center', va='center')
        # print(f"Error ACF: {e_acf}")

    try:
        # Asegurar suficientes lags para PACF, y que no exceda len(series)/2 -1
        pacf_lags = lags
        if lags is not None and lags >= len(series) // 2:
            pacf_lags = len(series) // 2 - 1
        
        if pacf_lags is not None and pacf_lags < 1: # Si pacf_lags se vuelve < 1
            pacf_lags = None # Dejar que statsmodels decida o no graficar

        plot_pacf(series, ax=axes[1], lags=pacf_lags, method='ywm', title=f'PACF - {target_column_name}', zero=False)
        axes[1].grid(True)
    except Exception as e_pacf: # Capturar error si PACF falla
        axes[1].text(0.5, 0.5, f'Error al generar PACF:\n{str(e_pacf)[:50]}...', ha='center', va='center')
        # print(f"Error PACF: {e_pacf}")

    plt.tight_layout()
    return fig

