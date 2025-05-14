# app.py
# ... (init_session_state, to_excel, reset_on_file_change, 
#      reset_model_results_and_processed_data, prepare_forecast_display_data
#      y la sección 1 de Carga y Preprocesamiento deberían estar como en la última versión que te di) ...

# --- Mostrar Diagnóstico y Gráficos Iniciales ---
# Esta sección solo se muestra si el preprocesamiento fue exitoso
if st.session_state.get('df_processed') is not None and st.session_state.get('original_target_column_name'):
    target_col_for_display = st.session_state.original_target_column_name # Usar una variable local para claridad
    
    # ... (código para mostrar diagnóstico, ACF, serie histórica) ...
    # Asegúrate de pasar target_col_for_display a las funciones de visualización si es necesario

    # --- Sección 2: Configuración del Pronóstico y Modelos (Sidebar) ---
    st.sidebar.header("2. Configuración de Pronóstico")
    # ... (todos tus st.sidebar.number_input, st.sidebar.checkbox, st.sidebar.expander para parámetros)
    # Asegúrate de que todas las keys sean ÚNICAS. Por ejemplo:
    # st.session_state.forecast_horizon = st.sidebar.number_input("Horizonte:", ..., key="horizon_cfg_widget")
    # st.session_state.run_autoarima = st.sidebar.checkbox("Ejecutar AutoARIMA", ..., key="run_arima_widget")


    if st.sidebar.button("📊 Generar y Evaluar Todos los Modelos", key="generate_models_btn_widget_action"): # Key única
        # Resetear resultados de modelos ANTES de correrlos de nuevo
        st.session_state.model_results = []
        st.session_state.best_model_name_auto = None
        st.session_state.selected_model_for_manual_explore = None
        
        # Obtener los datos preprocesados y el nombre de la columna objetivo del estado de sesión
        df_proc_to_run = st.session_state.get('df_processed')
        target_col_to_run = st.session_state.get('original_target_column_name')

        if df_proc_to_run is None or target_col_to_run is None:
            st.error("🔴 Error: Datos no preprocesados correctamente. Por favor, vuelva a aplicar el preprocesamiento.")
        elif target_col_to_run not in df_proc_to_run.columns:
            st.error(f"🔴 Error: La columna objetivo '{target_col_to_run}' no se encuentra en los datos preprocesados.")
        else:
            series_full_to_run = df_proc_to_run[target_col_to_run].copy()
            h_to_run = st.session_state.forecast_horizon
            s_period_eff_to_run = st.session_state.user_seasonal_period
            
            train_series_to_run, test_series_to_run = series_full_to_run, pd.Series(dtype=series_full_to_run.dtype)
            if st.session_state.use_train_test_split:
                min_train_size = max(5, 2 * s_period_eff_to_run + 1 if s_period_eff_to_run > 1 else 5)
                # Ajustar test_split_size si es necesario y posible
                if 'test_split_size' in st.session_state and \
                   st.session_state.test_split_size < (len(series_full_to_run) - min_train_size) and \
                   st.session_state.test_split_size > 0:
                    train_series_to_run, test_series_to_run = forecasting_models.train_test_split_series(series_full_to_run, st.session_state.test_split_size)
                else:
                    st.warning(f"No es posible el split con test_size={st.session_state.get('test_split_size', 'N/A')}. Se usará toda la serie para entrenar/evaluar in-sample.")
                    st.session_state.use_train_test_split = False # Desactivar si no es viable
            
            st.session_state.train_series_for_plot = train_series_to_run
            st.session_state.test_series_for_plot = test_series_to_run
            
            with st.spinner("Calculando modelos... Esto puede tardar unos momentos."):
                # --- EJECUCIÓN DE MODELOS ---
                # (Aquí va toda tu lógica para llamar a historical_average_forecast, naive_forecast, 
                #  seasonal_naive_forecast, forecast_with_statsmodels para SES, Holt, HW, 
                #  y forecast_with_auto_arima si st.session_state.run_autoarima es True.
                #  Asegúrate de que cada llamada a estos modelos guarde sus resultados en 
                #  st.session_state.model_results como un diccionario con 'name', 'rmse', 'mae', 
                #  'forecast_future', 'conf_int_future', 'forecast_on_test')
                #
                # Ejemplo de cómo añadir un resultado:
                # fc, ci, rmse, mae, name = tu_funcion_de_modelo(...)
                # fc_on_test = tu_logica_para_obtener_predicciones_en_test(...)
                # st.session_state.model_results.append({
                #    'name': name, 'rmse': rmse, 'mae': mae, 
                #    'forecast_future': fc, 'conf_int_future': ci, 
                #    'forecast_on_test': fc_on_test
                # })
                # (Asegúrate de que esto se haga para CADA modelo que ejecutes)
                # ... (TU CÓDIGO DE EJECUCIÓN DE MODELOS AQUÍ) ...
                # Esto es solo un placeholder, debes tener la lógica real de tus mensajes anteriores
                if not st.session_state.model_results: # Si después de intentar correr modelos, la lista está vacía
                    st.error("🔴 No se pudieron generar resultados para ningún modelo. Verifique los parámetros y los datos.")


            # Determinar el mejor modelo después de que todos hayan corrido
            valid_results_for_sorting = [
                res for res in st.session_state.model_results 
                if pd.notna(res.get('rmse')) and 
                   res.get('forecast_future') is not None and 
                   len(res['forecast_future']) == h_to_run
            ]
            if valid_results_for_sorting:
                best_model_auto_entry_run = min(valid_results_for_sorting, key=lambda x: x['rmse'])
                st.session_state.best_model_name_auto = best_model_auto_entry_run['name']
            else:
                if st.session_state.model_results: # Si hubo intentos pero todos fallaron métricas/predicción
                     st.error("No se pudo determinar un modelo sugerido entre los resultados obtenidos.")
                st.session_state.best_model_name_auto = None


# --- Sección de Resultados y Pestañas ---
# Esta sección solo se muestra si el preprocesamiento fue OK Y los modelos se han generado
if st.session_state.get('df_processed') is not None and \
   st.session_state.get('original_target_column_name') and \
   st.session_state.get('model_results'): # Verificar que model_results tenga algo

    target_col_for_tabs = st.session_state.original_target_column_name
    st.header("Resultados del Modelado y Pronóstico")

    # Pestañas
    tab_rec, tab_comp, tab_manual, tab_diag_guide = st.tabs([
        "⭐ Modelo Recomendado", "📊 Comparación General", 
        "⚙️ Explorar Manualmente", "💡 Diagnóstico y Guía"
    ])

    # --- Pestaña 1: Modelo Recomendado (Automático) ---
    with tab_rec:
        if st.session_state.best_model_name_auto and "Error" not in st.session_state.best_model_name_auto :
            st.subheader(f"Análisis del Modelo Recomendado: {st.session_state.best_model_name_auto}")
            model_data_auto = next((item for item in st.session_state.model_results if item["name"] == st.session_state.best_model_name_auto), None)

            if model_data_auto:
                final_export_df_auto, fc_series_auto, pi_df_auto = prepare_forecast_display_data(
                    model_data_auto, 
                    st.session_state.df_processed[target_col_for_tabs].index,
                    st.session_state.forecast_horizon
                )
                if final_export_df_auto is not None:
                    # Gráfico de Validación
                    if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty and model_data_auto.get('forecast_on_test') is not None:
                        st.markdown("##### Validación en Conjunto de Prueba")
                        # ... (código para plot_forecast_vs_actual)
                        fig_val_auto = visualization.plot_forecast_vs_actual(st.session_state.train_series_for_plot, st.session_state.test_series_for_plot, pd.Series(model_data_auto['forecast_on_test'], index=st.session_state.test_series_for_plot.index), st.session_state.best_model_name_auto, target_col_for_tabs)
                        if fig_val_auto: st.pyplot(fig_val_auto)
                    
                    # Gráfico de Pronóstico Futuro
                    st.markdown(f"##### Pronóstico Futuro con {st.session_state.best_model_name_auto}")
                    # ... (código para plot_final_forecast)
                    fig_fc_auto = visualization.plot_final_forecast(st.session_state.df_processed[target_col_for_tabs], fc_series_auto, pi_df_auto, model_name=st.session_state.best_model_name_auto, value_col_name=target_col_for_tabs)
                    if fig_fc_auto: st.pyplot(fig_fc_auto)


                    # Tabla de Pronóstico y Descarga
                    st.markdown("##### Valores del Pronóstico")
                    st.dataframe(final_export_df_auto.style.format("{:.2f}"))
                    excel_data_auto = to_excel(final_export_df_auto)
                    st.download_button(f"📥 Descargar Pronóstico ({st.session_state.best_model_name_auto})", excel_data_auto, f"pronostico_recomendado_{target_col_for_tabs}.xlsx", key="dl_auto_tab_rec")
                    
                    # Recomendaciones
                    st.markdown("##### Recomendaciones para este Modelo")
                    rec_text_auto = recommendations.generate_recommendations(
                        st.session_state.best_model_name_auto, 
                        st.session_state.data_diagnosis_report, 
                        True, 
                        (pi_df_auto is not None), 
                        (st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty)
                    )
                    st.markdown(rec_text_auto)
                else: st.warning("No se pudo preparar la visualización del pronóstico para el modelo recomendado.")
            else: st.info("No se encontraron datos para el modelo recomendado. Ejecute los modelos.")
        else:
            st.info("No se ha determinado un modelo recomendado o hubo un error. Por favor, genere y evalúe los modelos.")

    # --- Pestaña 2: Comparación General ---
    with tab_comp:
        # ... (código como lo tenías, asegurando que st.session_state.best_model_name_auto se usa para resaltar)
        st.subheader("Métricas de Rendimiento de Todos los Modelos Probados")
        eval_type_comp_tab = "en Conjunto de Prueba" if st.session_state.use_train_test_split and st.session_state.test_series_for_plot is not None and not st.session_state.test_series_for_plot.empty else "In-Sample (toda la serie)"
        st.markdown(f"RMSE y MAE {eval_type_comp_tab}. Valores más bajos son mejores.")
        metrics_list_comp_tab = [{'Modelo': r['name'], 'RMSE': r['rmse'], 'MAE': r['mae']} for r in st.session_state.model_results if r.get('forecast_future') is not None and pd.notna(r['rmse'])] # Solo válidos
        if metrics_list_comp_tab:
            metrics_df_comp_tab = pd.DataFrame(metrics_list_comp_tab).sort_values(by='RMSE').reset_index(drop=True)
            def highlight_best_tab(row): return ['background-color: lightgreen' if row.Modelo == st.session_state.best_model_name_auto else ''] * len(row)
            st.dataframe(metrics_df_comp_tab.style.format({'RMSE': "{:.3f}", 'MAE': "{:.3f}"}).apply(highlight_best_tab, axis=1))
            if st.session_state.best_model_name_auto: st.info(f"🏆 Modelo Automáticamente Sugerido (menor RMSE): **{st.session_state.best_model_name_auto}**")
        else: st.warning("No hay resultados de modelos para mostrar o todos fallaron.")


    # --- Pestaña 3: Explorar y Seleccionar Manualmente ---
    with tab_manual:
        st.subheader("Explorar Resultados y Seleccionar un Modelo Manualmente")
        available_models_manual = [res['name'] for res in st.session_state.model_results if res.get('forecast_future') is not None and pd.notna(res['rmse'])]
        
        if not available_models_manual:
            st.warning("No hay modelos válidos disponibles para seleccionar.")
        else:
            # Lógica para el selector y mostrar detalles del modelo manual
            # ... (como lo tenías, asegurando que prepare_forecast_display_data se llama correctamente
            #      y que las funciones de visualización reciben los datos correctos, incluyendo el título del modelo)
            # Ejemplo:
            default_manual_idx = 0
            if st.session_state.selected_model_for_manual_explore in available_models_manual:
                default_manual_idx = available_models_manual.index(st.session_state.selected_model_for_manual_explore)
            elif st.session_state.best_model_name_auto in available_models_manual:
                default_manual_idx = available_models_manual.index(st.session_state.best_model_name_auto)
            
            st.session_state.selected_model_for_manual_explore = st.selectbox("Seleccione un modelo:", options=available_models_manual, index=default_manual_idx, key="manual_model_selector_tab")
            model_data_manual = next((item for item in st.session_state.model_results if item["name"] == st.session_state.selected_model_for_manual_explore), None)

            if model_data_manual:
                final_export_df_man, fc_series_man, pi_df_man = prepare_forecast_display_data(
                    model_data_manual, 
                    st.session_state.df_processed[target_col_for_tabs].index, 
                    st.session_state.forecast_horizon
                )
                if final_export_df_man is not None:
                    # ... (gráficos, tabla, descarga, recomendaciones para el modelo manual) ...
                    st.markdown(f"##### Pronóstico Futuro con {st.session_state.selected_model_for_manual_explore}")
                    fig_fc_man = visualization.plot_final_forecast(st.session_state.df_processed[target_col_for_tabs], fc_series_man, pi_df_man, model_name=st.session_state.selected_model_for_manual_explore, value_col_name=target_col_for_tabs)
                    if fig_fc_man: st.pyplot(fig_fc_man)
                    # ... (resto: tabla, descarga, recomendaciones) ...


    # --- Pestaña 4: Diagnóstico y Guía ---
    with tab_diag_guide:
        # ... (código como lo tenías) ...
        st.subheader("Diagnóstico de Datos (Post-Preprocesamiento)")
        if st.session_state.data_diagnosis_report: st.markdown(st.session_state.data_diagnosis_report)
        st.subheader("Guía General de Interpretación")
        st.markdown(""" ... (tu texto de guía) ... """)

elif uploaded_file is None: # Solo mostrar si no se ha cargado nada aún
    st.info("👋 ¡Bienvenido! Por favor, cargue un archivo de datos para comenzar.")
else: # Si se cargó un archivo pero df_processed es None (falló el preprocesamiento)
    st.warning("⚠️ Parece que hubo un problema con el preprocesamiento de los datos. Por favor, verifique las selecciones de columnas y vuelva a intentarlo.")


# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.info("Asistente de Pronósticos PRO v3.3")