import pandas as pd
import numpy as np

def generate_recommendations_simple(
    selected_model_name,
    data_diag_summary,
    has_pis=False,
    target_column_name="la variable",
    model_rmse=None,
    model_mae=None,
    forecast_horizon=12,
    model_params=None
):
    """
    Genera recomendaciones, interpretación y próximos pasos basados en el modelo de pronóstico.
    """
    recs = ["## 💡 Recomendaciones, Interpretación y Próximos Pasos"]

    # Sección: Modelo utilizado
    recs.append("\n### Sobre el Modelo Utilizado")
    if selected_model_name and "Error" not in selected_model_name and "FALLÓ" not in selected_model_name:
        recs.append(f"- Se utilizó el modelo: **'{selected_model_name}'** para pronosticar **'{target_column_name}'**.")

        # Explicaciones generales por tipo de modelo (copiar de v2.0 si aplica)
        # ... (aquí puedes agregar descripciones detalladas de cada modelo) ...

        # Parámetros estimados
        if isinstance(model_params, dict) and model_params:
            recs.append("  - **Parámetros Estimados por el Modelo:**")
            for param_name, param_value in model_params.items():
                if isinstance(param_value, (list, np.ndarray)) and len(param_value) > 4 and param_name == 'initial_seasons':
                    preview = [round(p, 4) for p in param_value[:4]] + ['...']
                    recs.append(f"    - `{param_name}`: `{preview}` (mostrando primeros 4)")
                elif isinstance(param_value, float):
                    recs.append(f"    - `{param_name}`: `{param_value:.4f}`")
                else:
                    recs.append(f"    - `{param_name}`: `{param_value}`")
            # Descripciones de parámetros comunes
            if 'smoothing_level' in model_params:
                recs.append("      - *`smoothing_level` (alfa): controla peso de observaciones recientes.*")
            if 'smoothing_trend' in model_params:
                recs.append("      - *`smoothing_trend` (beta): controla peso de tendencia.*")
            if 'smoothing_seasonal' in model_params:
                recs.append("      - *`smoothing_seasonal` (gamma): controla peso de estacionalidad.*")
            if 'order' in model_params:
                recs.append("      - *`(p,d,q)`: órdenes de ARIMA.*")
            if 'seasonal_order' in model_params:
                recs.append("      - *`(P,D,Q,m)`: órdenes estacionales de SARIMA.*")

        # Ajuste in-sample: RMSE y MAE
        if model_rmse is not None and not np.isnan(model_rmse):
            # Preparar texto para MAE
            if model_mae is not None and not np.isnan(model_mae):
                mae_text = f"{model_mae:.2f}"
            else:
                mae_text = "N/A"
            recs.append(
                f"  - **Ajuste a Históricos (In-Sample):** "
                f"RMSE = {model_rmse:.2f}, MAE = {mae_text}."
            )
            recs.append(
                "    - *Valores más bajos indican mejor ajuste a datos pasados. "
                "NO indican qué tan bien pronosticará datos futuros.*"
            )
        else:
            recs.append("- No se pudo calcular RMSE de ajuste in-sample.")

    else:
        recs.append("- No se pudo ejecutar un modelo válido o ocurrió un error.")

    # Sección: Interpretación del gráfico
    recs.append("\n### Interpretación de la Visualización")
    recs.append(data_diag_summary)

    # Sección: Calidad de datos
    recs.append("\n### Calidad de Datos")
    recs.append(
        "Revisa la consistencia de fechas, valores faltantes y posibles atípicos antes de basar decisiones críticas."
    )

    # Sección: Próximos pasos
    recs.append("\n### Próximos Pasos Sugeridos")
    recs.append(
        f"- Revisa el pronóstico para los siguientes {forecast_horizon} períodos y ajústalo según contexto de negocio."
    )
    recs.append(
        "- Monitorea nuevos datos y vuelve a entrenar el modelo periódicamente para mantener la precisión."
    )

    return "\n".join(recs)
