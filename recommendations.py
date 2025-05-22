# recommendations.py
import pandas as pd # Necesario si se analizan los PIs como DataFrames
import numpy as np  # Necesario para np.isnan

def generate_recommendations_simple(
    selected_model_name, 
    data_diag_summary, # String con el resumen del diagnóstico de datos
    has_pis=False,       # Booleano: ¿El modelo generó intervalos de predicción?
    target_column_name="la variable", # String: Nombre de la columna pronosticada
    model_rmse=None,     # Float o None: RMSE del modelo (in-sample)
    model_mae=None,      # Float o None: MAE del modelo (in-sample)
    forecast_horizon=12  # Int: Horizonte del pronóstico
    ):
    """
    Genera texto de recomendación detallado para el flujo simplificado 
    (un solo modelo ejecutado).
    """
    
    recs = ["## 💡 Recomendaciones, Interpretación y Próximos Pasos"]

    # --- Sobre el Modelo Seleccionado/Utilizado ---
    recs.append("\n### Sobre el Modelo Utilizado")
    if selected_model_name and "Error" not in selected_model_name and "FALLÓ" not in selected_model_name and "Insuf" not in selected_model_name and "Inválido" not in selected_model_name:
        recs.append(f"- Se utilizó el modelo: **'{selected_model_name}'** para generar el pronóstico de **'{target_column_name}'**.")
        
        # Explicaciones generales por tipo de modelo
        if "AutoARIMA" in selected_model_name:
            recs.append("  - **AutoARIMA** es un modelo avanzado que intenta encontrar la mejor estructura ARIMA (o SARIMA si se detecta estacionalidad) para tus datos. Busca patrones de autocorrelación (cómo los valores pasados se relacionan con los futuros) y estacionalidad (patrones que se repiten). Es muy flexible pero puede ser sensible a la cantidad de datos y a la complejidad de los mismos.")
            recs.append("  - *Sugerencia:* Si AutoARIMA da un error o un pronóstico plano, podría ser útil tener más datos históricos o simplificar los parámetros de búsqueda si se expusieran (no es el caso en esta versión simplificada).")
        elif "Holt-Winters" in selected_model_name or "HW" in selected_model_name:
             recs.append("  - **Holt-Winters** es un modelo de suavización exponencial diseñado para series de tiempo que presentan **tendencia** (una dirección general creciente o decreciente) y **estacionalidad** (patrones que se repiten a intervalos fijos, ej., cada 12 meses).")
             recs.append("  - *Funcionamiento:* Estima un nivel base, una tendencia y un componente estacional, dando más peso a las observaciones recientes. En esta versión, los componentes de tendencia y estacionalidad se intentan ajustar automáticamente (usualmente como 'aditivos').")
        elif "Holt" in selected_model_name and "Winters" not in selected_model_name:
            recs.append("  - El modelo de **Holt** (Suavización Exponencial Doble) es adecuado para datos con **tendencia lineal** (crecen o decrecen de forma constante) pero sin un patrón estacional claro.")
            recs.append("  - *Funcionamiento:* Estima un nivel y una pendiente (tendencia), dando más peso a los datos recientes.")
        elif "SES" in selected_model_name or "Simple" in selected_model_name:
            recs.append("  - La **Suavización Exponencial Simple (SES)** es útil para series de tiempo que **no tienen tendencia ni estacionalidad** aparentes, es decir, fluctúan alrededor de un nivel medio constante.")
            recs.append("  - *Funcionamiento:* Calcula un promedio ponderado de las observaciones pasadas, donde los datos más recientes tienen mayor influencia.")
        elif "Promedio Móvil" in selected_model_name:
             recs.append(f"  - El **Promedio Móvil Simple** (con la ventana configurada) suaviza la serie calculando el promedio de un número fijo de observaciones pasadas. Es un modelo simple, bueno como línea base o para datos con cambios lentos y poca estructura compleja.")
        elif "Estacional Ingénuo" in selected_model_name:
            recs.append("  - El **Estacional Ingénuo** es un modelo baseline que asume que el valor de un período será el mismo que el del mismo período en la temporada anterior. Es útil si hay una **estacionalidad muy fuerte y repetitiva**.")
        elif "Ingénuo" in selected_model_name:
            recs.append("  - El modelo **Ingénuo (o de Último Valor)** simplemente usa el último valor observado como el pronóstico para todos los períodos futuros. Es un baseline simple y efectivo si la serie cambia muy lentamente o sigue un 'paseo aleatorio'.")
        elif "Promedio Histórico" in selected_model_name:
            recs.append("  - El **Promedio Histórico** calcula el promedio de todos los datos históricos y lo usa como pronóstico. Es el baseline más simple, útil si no se esperan cambios significativos respecto al comportamiento pasado general.")
        
        # Métricas de ajuste (In-Sample)
        if model_rmse is not None and model_mae is not None and not (np.isnan(model_rmse) or np.isnan(model_mae)):
            recs.append(f"  - **Ajuste a los Datos Históricos (In-Sample):** RMSE = {model_rmse:.2f}, MAE = {model_mae:.2f}.")
            recs.append("    - *RMSE (Error Cuadrático Medio Raíz) y MAE (Error Absoluto Medio) miden el error promedio del modelo al predecir los datos históricos que ya conoce. Valores más bajos indican un mejor ajuste a los datos pasados. Estas métricas NO indican qué tan bien pronosticará datos futuros no vistos.*")
        else:
            recs.append("  - *No se pudieron calcular métricas de ajuste (RMSE/MAE) para este modelo, posiblemente debido a errores durante el ajuste o datos insuficientes.*")
    else:
        recs.append("- No se pudo ejecutar un modelo o el modelo seleccionado tuvo problemas. Revise los mensajes de error o los datos de entrada.")

    # --- Interpretación del Gráfico de Pronóstico ---
    recs.append("\n### 🤔 Cómo Interpretar el Gráfico de Pronóstico")
    recs.append(f"- La **línea de datos históricos** (generalmente sólida) muestra los valores pasados de **'{target_column_name}'**.")
    recs.append(f"- La **línea de pronóstico** (generalmente discontinua o de otro color) muestra las predicciones del modelo para los próximos **{forecast_horizon}** períodos.")
    if has_pis:
        recs.append("- El **área sombreada** alrededor del pronóstico es el **intervalo de predicción** (usualmente al 95% de confianza).")
        recs.append("  - Esto significa que, si el modelo es correcto y los patrones pasados continúan, hay una alta probabilidad (95%) de que los valores reales futuros caigan dentro de esta banda sombreada.")
        recs.append("  - Una **banda más ancha** indica **mayor incertidumbre** en el pronóstico. Una banda más estrecha sugiere más confianza (pero no certeza absoluta).")
    else:
        recs.append("- *Para este pronóstico, no se generaron o no fue posible calcular los intervalos de predicción. Solo se muestra la predicción puntual (la estimación central del modelo).*")
    
    # --- Sobre la Calidad de los Datos (Reiterar del Diagnóstico) ---
    if data_diag_summary:
        recs.append("\n### 📊 Recordatorios Importantes sobre sus Datos")
        recs.append("La calidad de sus datos de entrada impacta directamente la fiabilidad del pronóstico:")
        if "Valores faltantes: 0 (" not in data_diag_summary and "faltantes: 0.0 (" not in data_diag_summary:
             recs.append("  - **Valores Faltantes:** Su diagnóstico inicial indicó la presencia de valores faltantes. Aunque se intentó una imputación, los datos faltantes pueden introducir imprecisiones. Lo ideal es tener series de datos completas.")
        if "Posibles atípicos (1.5*IQR): 0 (" not in data_diag_summary:
             recs.append("  - **Valores Atípicos:** Su diagnóstico inicial señaló posibles valores atípicos. Estos puntos extremos pueden distorsionar cómo el modelo aprende los patrones. Es bueno investigar si son errores o eventos reales y considerar cómo tratarlos.")
        if "Frecuencia de datos: No regular" in data_diag_summary:
            recs.append("  - **Frecuencia de Datos:** Si sus datos no tienen una frecuencia regular (ej. no son consistentemente diarios, mensuales, etc.), esto dificulta el modelado. La opción de remuestreo puede ayudar a estandarizar esto.")
        recs.append("  - *Recomendación General:* Revise su proceso de recolección de datos para asegurar la mayor calidad y consistencia posible.")
    
    # --- Próximos Pasos y Consideraciones Generales ---
    recs.append("\n### 🚀 Próximos Pasos y Qué Considerar")
    recs.append("- **Análisis Crítico:** Compare el pronóstico generado con su propio conocimiento del negocio, expectativas e información externa (ej. planes de marketing, acciones de la competencia, factores económicos generales). ¿El pronóstico parece razonable?")
    recs.append("- **Incertidumbre:** Recuerde que todo pronóstico tiene incertidumbre. Los intervalos de predicción (si se muestran) le dan una idea de esto. Planifique considerando un rango de posibles resultados, no solo la predicción puntual.")
    recs.append("- **Ajustes Manuales:** Ningún modelo estadístico es perfecto ni puede prever eventos únicos o cambios drásticos en el entorno. Es muy probable que necesite realizar **ajustes manuales** al pronóstico basándose en su juicio experto.")
    recs.append("- **Monitoreo Continuo:** Un pronóstico es una foto en un momento dado. A medida que obtiene nuevos datos reales, compare cómo se desempeñó el pronóstico (tracking de la señal de error) y considere reajustar o re-seleccionar el modelo periódicamente.")
    recs.append("- **Experimentación (si desea profundizar):** Si los resultados no son satisfactorios, y si la herramienta tuviera más opciones (como en versiones anteriores o futuras), podría experimentar con:")
    recs.append("    - Diferentes métodos de imputación de faltantes o frecuencias de remuestreo.")
    recs.append("    - Diferentes tipos de modelos o parámetros de modelos (no disponible en esta versión simplificada).")
    recs.append("- **Limitaciones de Modelos Univariados:** Esta aplicación (en su versión simplificada) utiliza principalmente modelos univariados, lo que significa que basan sus predicciones solo en los valores pasados de la misma variable que se está pronosticando ('{target_column_name}'). No consideran explícitamente otros factores externos (variables exógenas) que podrían influir en '{target_column_name}' (como promociones, precios, actividad económica, etc.). Si estos factores son muy importantes, modelos más complejos que los incorporen podrían ser necesarios.")

    return "\n".join(recs)