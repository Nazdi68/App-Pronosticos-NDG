# recommendations.py

def generate_recommendations(
    selected_model_name, 
    data_diag_summary, 
    forecast_generated=False, 
    has_pis=False, 
    eval_on_test_set=False,
    model_results_list=None, # Lista completa de resultados de modelos
    target_column_name="la variable" # Nombre de la columna objetivo
    ):
    """Genera texto de recomendación basado en los resultados del análisis."""
    
    recs = ["### 💡 Recomendaciones y Conclusiones Clave"] # Usar Markdown para título
    
    # --- Recomendación del Modelo ---
    if selected_model_name and "Error" not in selected_model_name and "Insuf" not in selected_model_name and "Inválido" not in selected_model_name :
        eval_method = "en el conjunto de prueba (datos que el modelo no vio durante su ajuste inicial)" if eval_on_test_set else "en los datos históricos completos (in-sample)"
        recs.append(f"- El modelo **'{selected_model_name}'** fue seleccionado (o tuvo el mejor rendimiento {eval_method} si es la recomendación automática).")
        
        # Comentarios específicos por tipo de modelo
        if "AutoARIMA" in selected_model_name:
            recs.append("  - *AutoARIMA es un modelo sofisticado que intenta encontrar automáticamente una estructura ARIMA (o SARIMA para datos estacionales) que se ajuste bien a los patrones de autocorrelación y estacionalidad de sus datos. Puede ser muy potente para series complejas.*")
        elif "Holt-Winters" in selected_model_name or "HW" in selected_model_name:
             recs.append("  - *Holt-Winters es un modelo de suavización exponencial diseñado para datos con **tendencia y estacionalidad**. Verifique si los tipos de tendencia (aditiva/multiplicativa) y estacionalidad elegidos coinciden con el comportamiento visual de su serie.*")
        elif "Holt" in selected_model_name and "Winters" not in selected_model_name:
            recs.append("  - *El modelo de Holt es una suavización exponencial para datos con **tendencia** pero sin un patrón estacional claro. La opción de 'tendencia amortiguada' puede hacerlo más conservador a largo plazo.*")
        elif "SES" in selected_model_name or "Simple" in selected_model_name:
            recs.append("  - *La Suavización Exponencial Simple es adecuada para datos **sin tendencia ni estacionalidad** clara, donde el nivel de la serie fluctúa alrededor de una media.*")
        elif "Promedio Móvil" in selected_model_name:
             recs.append(f"  - *El Promedio Móvil Simple (con la ventana elegida) suaviza las fluctuaciones a corto plazo. Puede ser un buen baseline o útil para datos con cambios lentos y poca estructura compleja.*")
        elif "Estacional Ingénuo" in selected_model_name:
            recs.append("  - *El modelo Estacional Ingénuo es un baseline útil para datos con **estacionalidad fuerte y estable**, donde el valor de un período es similar al del mismo período en la temporada anterior.*")
        elif "Ingénuo" in selected_model_name:
            recs.append("  - *El modelo Ingénuo (último valor) es un baseline simple, efectivo si su serie se comporta como un 'paseo aleatorio' o cambia muy lentamente.*")
        elif "Promedio Histórico" in selected_model_name:
            recs.append("  - *El Promedio Histórico es el baseline más simple, útil si no hay patrones claros de tendencia o estacionalidad y se espera que el futuro sea similar al promedio del pasado.*")
    else:
        recs.append("- No se pudo determinar un modelo principal o el modelo seleccionado tuvo problemas en su ejecución.")

    # --- Interpretación del Pronóstico ---
    if forecast_generated:
        recs.append("\n### 🤔 Interpretando el Pronóstico")
        recs.append(f"- La **línea de pronóstico** en el gráfico muestra la predicción central del modelo para **'{target_column_name}'**.")
        if has_pis:
            recs.append("- El **área sombreada** representa el **intervalo de predicción** (usualmente al 95% de confianza). Indica el rango donde se espera que caigan los valores reales. Una banda más ancha significa mayor incertidumbre en el pronóstico.")
        else:
            recs.append("- *No se generaron intervalos de predicción para este pronóstico específico, por lo que solo se muestra la predicción puntual.*")
        recs.append("- **Recuerde:** Este es un pronóstico estadístico basado en patrones históricos. No puede prever eventos completamente inesperados o cambios drásticos en el entorno que no estén reflejados en los datos pasados.")
    
    # --- Sobre la Calidad de los Datos ---
    if data_diag_summary:
        recs.append("\n### 📊 Recordatorios sobre la Calidad de los Datos")
        if "Valores faltantes: 0 (" not in data_diag_summary and "faltantes: 0.0 (" not in data_diag_summary: # Ajustar si el texto exacto cambia
             recs.append("  - **Valores Faltantes:** Se detectaron (o no se pudieron imputar todos) los valores faltantes. La presencia de datos faltantes puede reducir significativamente la precisión del pronóstico. Considere revisar la fuente de sus datos o probar diferentes métodos de imputación.")
        if "Posibles atípicos (1.5*IQR): 0 (" not in data_diag_summary:
             recs.append("  - **Valores Atípicos:** Se identificaron posibles valores atípicos. Estos puntos extremos pueden distorsionar el ajuste de los modelos. Investigue si son errores de datos o eventos reales que necesiten un tratamiento especial (ej. ajuste manual o exclusión si son errores).")
        if "Frecuencia de datos: No regular" in data_diag_summary:
            recs.append("  - **Frecuencia de Datos:** La frecuencia de sus datos no parece ser regular. Los modelos de series de tiempo funcionan mejor con datos a intervalos constantes (ej. diarios, mensuales exactos). Considere usar la opción de remuestreo para estandarizar la frecuencia si es apropiado para sus datos.")
        recs.append("  - *En general, la premisa **'basura entra, basura sale' (garbage in, garbage out)** aplica a los pronósticos. Mejorar la calidad y consistencia de sus datos de entrada es el paso más importante para obtener pronósticos más fiables.*")
    
    # --- Próximos Pasos y Consideraciones Adicionales ---
    recs.append("\n### 🚀 Próximos Pasos y Consideraciones")
    if eval_on_test_set:
        recs.append("- **Validación:** Si usó un conjunto de prueba, analice el gráfico de 'Validación en Test'. ¿El modelo sigue bien los datos que no vio antes? Grandes desviaciones aquí pueden indicar sobreajuste al conjunto de entrenamiento.")
    
    # Sugerencia basada en si el mejor modelo fue un baseline
    if model_results_list and selected_model_name:
        best_model_entry = next((m for m in model_results_list if m['name'] == selected_model_name), None)
        if best_model_entry and best_model_entry.get('rmse') is not None:
            is_baseline = any(b_name in selected_model_name for b_name in ["Promedio Histórico", "Ingénuo", "Promedio Móvil"])
            if is_baseline:
                recs.append("- **Modelos Simples vs. Complejos:** Si un modelo baseline simple (como Promedio Histórico o Ingénuo) es el mejor o está entre los mejores, podría indicar que su serie de tiempo tiene poca estructura predecible con los modelos probados, o que es muy ruidosa. En estos casos, pronósticos más complejos no necesariamente ofrecen mejores resultados.")
            
            # Comparar con el segundo mejor si hay más de un modelo válido
            valid_models = [m for m in model_results_list if pd.notna(m.get('rmse')) and m.get('forecast_future') is not None]
            if len(valid_models) > 1:
                valid_models.sort(key=lambda x: x['rmse'])
                if selected_model_name == valid_models[0]['name'] and len(valid_models) > 1: # Si el seleccionado es el mejor
                    second_best_rmse = valid_models[1]['rmse']
                    best_rmse = valid_models[0]['rmse']
                    if best_rmse !=0 and (second_best_rmse - best_rmse) / best_rmse < 0.10: # Si el segundo mejor está dentro del 10%
                        recs.append(f"- **Alternativas Cercanas:** El modelo '{valid_models[1]['name']}' tuvo un rendimiento muy similar. Podría considerarlo si ofrece otras ventajas (ej. mayor simplicidad, mejor interpretabilidad).")

    recs.append("- **Experimentación:** No dude en probar diferentes parámetros para los modelos (ej. ventana del promedio móvil, tipos de tendencia/estacionalidad en Holt-Winters, límites en AutoARIMA), o diferentes métodos de preprocesamiento.")
    recs.append("- **Conocimiento del Dominio:** ¡Su experiencia es invaluable! ¿Hay eventos futuros (promociones, cambios de mercado, nuevos productos, factores económicos) que los datos históricos no capturan? Considere ajustar manualmente el pronóstico basado en esta información externa.")
    recs.append("- **Iteración:** El pronóstico es un proceso iterativo. Monitoree la precisión de sus pronósticos a lo largo del tiempo y reajuste los modelos o su enfoque según sea necesario.")
    recs.append("- **Limitaciones:** Recuerde que esta herramienta utiliza modelos univariados (excepto si en el futuro se añade funcionalidad para variables exógenas). Si otros factores externos influyen fuertemente en '{target_column_name}', estos modelos no los capturarán explícitamente.")

    return "\n".join(recs)