# recommendations.py
def generate_recommendations(best_model_name, data_diag_summary, forecast_generated=False, has_pis=False, eval_on_test_set=False):
    recs = ["**Recomendaciones y Conclusiones Clave:**"]
    
    if best_model_name and "Error" not in best_model_name and "insuficientes" not in best_model_name :
        eval_method = "en el conjunto de prueba (out-of-sample)" if eval_on_test_set else "en los datos históricos (in-sample)"
        recs.append(f"- El modelo **'{best_model_name}'** tuvo el mejor rendimiento {eval_method} (según RMSE).")
        if "AutoARIMA" in best_model_name:
            recs.append("  - *AutoARIMA intenta encontrar un modelo ARIMA/SARIMA que se ajuste a patrones complejos de autocorrelación y estacionalidad.*")
        elif "HW" in best_model_name or "Holt-Winters" in best_model_name:
             recs.append("  - *Holt-Winters es adecuado para datos con tendencia y estacionalidad.*")
        elif "Holt" in best_model_name and "Winters" not in best_model_name:
            recs.append("  - *El modelo de Holt es útil para datos con tendencia pero sin estacionalidad clara.*")
        elif "SES" in best_model_name or "Simple" in best_model_name:
            recs.append("  - *La Suavización Exponencial Simple es para datos sin tendencia ni estacionalidad.*")

    else:
        recs.append("- No se pudo determinar un mejor modelo o el modelo recomendado tuvo problemas.")

    if forecast_generated:
        recs.append("- **Interpretación del Pronóstico:**")
        recs.append("  - La **línea de pronóstico** es la predicción central del modelo.")
        if has_pis:
            recs.append("  - El **área sombreada** (intervalo de predicción) muestra el rango probable de los valores reales (usualmente con 95% de confianza). Una banda más ancha indica mayor incertidumbre.")
    
    recs.append("- **Importante:** Este pronóstico es una estimación. Úselo como una guía y combínelo con su conocimiento del negocio y factores externos no presentes en los datos históricos.")
    
    if data_diag_summary:
        recs.append("\n**Recordatorios sobre los Datos (del Diagnóstico):**")
        if "Valores faltantes: 0 (" not in data_diag_summary and "faltantes: 0.0 (" not in data_diag_summary: # Hacerlo más robusto
             recs.append("  - Se reportaron **valores faltantes** o la imputación no los cubrió todos. Esto afecta la calidad del pronóstico.")
        if "Posibles atípicos (1.5*IQR): 0 (" not in data_diag_summary:
             recs.append("  - Se detectaron **posibles valores atípicos**. Revíselos, ya que pueden distorsionar los modelos.")
        if "Frecuencia de datos: No regular" in data_diag_summary:
            recs.append("  - La **frecuencia de los datos no parece ser regular**. Esto puede complicar el modelado. Intente asegurar una frecuencia constante si es posible (ej. usando la opción de remuestreo).")
        recs.append("  - *Mejorar la calidad y consistencia de los datos de entrada es clave para pronósticos más fiables.*")
        
    recs.append("\n**Próximos Pasos Sugeridos:**")
    if eval_on_test_set:
        recs.append("- Revise el gráfico de 'Validación en Conjunto de Prueba' para ver qué tan bien el modelo predijo datos que no vio durante el entrenamiento.")
    recs.append("- Experimente con diferentes parámetros de preprocesamiento (frecuencia, imputación) o configuraciones de modelo si los resultados no son óptimos.")
    recs.append("- Si la estacionalidad es importante (ver ACF), asegúrese de que el período estacional sea correcto y que el modelo elegido la maneje (ej. Holt-Winters, AutoARIMA con componente estacional).")

    return "\n".join(recs)