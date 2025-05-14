# visualization.py
import matplotlib.pyplot as plt
# import seaborn as sns # No se usa activamente, se puede comentar
import pandas as pd

def plot_historical_data(series_df, value_col_name, title="Serie de Tiempo Histórica"):
    if series_df is None or series_df.empty or value_col_name not in series_df.columns:
        return None
    
    series = series_df[value_col_name]
    if series.empty: return None

    fig, ax = plt.subplots(figsize=(10, 5)) # Un poco más compacto
    series.plot(ax=ax, label='Histórico', marker='.', linestyle='-')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel(value_col_name, fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_forecast_vs_actual(historical_series, test_series, forecast_on_test, model_name, value_col_name):
    if (historical_series is None or historical_series.empty) and \
       (test_series is None or test_series.empty): 
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    
    if historical_series is not None and not historical_series.empty:
        historical_series.plot(ax=ax, label=f'Entrenamiento ({value_col_name})', marker='.')
    
    if test_series is not None and not test_series.empty:
        test_series.plot(ax=ax, label=f'Prueba Real ({value_col_name})', marker='.', color='darkorange')

    if forecast_on_test is not None :
        # Si forecast_on_test es un array numpy y test_series tiene índice, crear Series
        if isinstance(forecast_on_test, np.ndarray) and test_series is not None and not test_series.empty:
            if len(forecast_on_test) == len(test_series.index):
                 forecast_on_test_series = pd.Series(forecast_on_test, index=test_series.index)
                 forecast_on_test_series.plot(ax=ax, label=f'Pronóstico en Prueba ({model_name})', linestyle='--', color='green')
            # else: # No plotear si las longitudes no coinciden
                # print("Longitud de pronóstico en prueba no coincide con datos de prueba.")
        elif isinstance(forecast_on_test, pd.Series): # Si ya es una Serie
            forecast_on_test.plot(ax=ax, label=f'Pronóstico en Prueba ({model_name})', linestyle='--', color='green')


    ax.set_title(f"Validación: {model_name} vs. Datos de Prueba", fontsize=14)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel(value_col_name, fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_final_forecast(full_historical_series, future_forecast_series, future_conf_int_df=None, model_name="", value_col_name=""):
    if full_historical_series is None or full_historical_series.empty or \
       future_forecast_series is None or future_forecast_series.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    
    full_historical_series.plot(ax=ax, label=f'Histórico ({value_col_name})', marker='.')
    future_forecast_series.plot(ax=ax, label=f'Pronóstico ({model_name})', marker='^', linestyle='--')
    
    ax.set_title(f"Pronóstico Futuro con {model_name}", fontsize=14)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel(value_col_name, fontsize=10)
    
    if future_conf_int_df is not None and not future_conf_int_df.empty:
        # Asegurar que las columnas se llamen 'lower' y 'upper'
        # Si vienen de pmdarima, ya deberían ser así. Si vienen de statsmodels, pueden ser diferentes.
        # La función prepare_forecast_display_data en app.py debería estandarizar esto.
        if 'lower' in future_conf_int_df.columns and 'upper' in future_conf_int_df.columns:
             # Asegurar que el índice de conf_int coincida con future_forecast_series
            if not future_conf_int_df.index.equals(future_forecast_series.index):
                 future_conf_int_df = future_conf_int_df.set_index(future_forecast_series.index)

            ax.fill_between(future_conf_int_df.index, 
                            future_conf_int_df['lower'], 
                            future_conf_int_df['upper'], 
                            color='gray', alpha=0.3, label='Intervalo de Predicción (95%)')
        # else:
            # print("Advertencia: Columnas 'lower'/'upper' no encontradas en conf_int_df para graficar.")

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
