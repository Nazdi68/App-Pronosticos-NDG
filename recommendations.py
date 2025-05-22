# recommendations.py (v2.1)
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
    model_params=None # Nuevo parámetro
    ):
    recs = ["## 💡 Recomendaciones, Interpretación y Próximos Pasos"]
    
    recs.append("\n### Sobre el Modelo Utilizado")
    if selected_model_name and "Error" not in selected_model_name and "FALLÓ" not in selected_model_name:
        recs.append(f"- Se utilizó el modelo: **'{selected_model_name}'** para generar el pronóstico de **'{target_column_name}'**.")
        
        # Explicaciones generales por tipo de modelo (como antes)
        # ... (copia aquí las explicaciones de la v2.0 de recommendations.py)

        # Mostrar parámetros estimados si están disponibles
        if model_params and isinstance(model_params, dict) and model_params:
            recs.append("  - **Parámetros Estimados por el Modelo:**")
            for param_name, param_value in model_params.items():
                if isinstance(param_value, (list, np.ndarray)) and len(param_value) > 4 and param_name == 'initial_seasons':
                     recs.append(f"    - `{param_name}`: `{[round(p, 4) for p in param_value[:4]] + ['...']}` (mostrando primeros 4)")    
                elif isinstance(param_value, float):
                    recs.append(f"    - `{param_name}`: `{param_value:.4f}`")
                else:
                    recs.append(f"    - `{param_name}`: `{param_value}`")
            if "smoothing_level" in model_params: recs.append("      - *`smoothing_level` (alfa): Controla el peso dado a las observaciones recientes para el nivel.*")
            if "smoothing_trend" in model_params: recs.append("      - *`smoothing_trend` (beta): Controla el peso dado a las observaciones recientes para la tendencia.*")
            if "smoothing_seasonal" in model_params: recs.append("      - *`smoothing_seasonal` (gamma): Controla el peso dado a las observaciones recientes para la estacionalidad.*")
            if "order" in model_params: recs.append("      - *`(p,d,q)`: Órdenes del componente no estacional de ARIMA.*")
            if "seasonal_order" in model_params: recs.append("      - *`(P,D,Q,m)`: Órdenes del componente estacional de ARIMA.*")


        if model_rmse is not None and not np.isnan(model_rmse):
            recs.append(f"  - **Ajuste a Históricos (In-Sample):** RMSE = {model_rmse:.2f}, MAE = {model_mae:.2f if model_mae is not None and not np.isnan(model_mae) else 'N/A'}.")
            recs.append("    - *Valores más bajos indican mejor ajuste a datos pasados. NO indican qué tan bien pronosticará datos futuros.*")
    else:
        recs.append("- No se pudo ejecutar un modelo o tuvo problemas.")

    # ... (Resto de las secciones: Interpretación del Gráfico, Calidad de Datos, Próximos Pasos
    #      COPIA AQUÍ ESAS SECCIONES DE LA VERSIÓN v2.0 de recommendations.py)
    #      Asegúrate de que el texto sea genérico y no dependa de `model_results_list` o `eval_on_test_set`
    #      ya que esos ya no se pasan en este flujo simplificado.

    return "\n".join(recs)