# app.py
import streamlit as st
import pandas as pd # Necesario para la línea que da error en tu traceback anterior

st.title("Prueba de Importación")

try:
    import forecasting_models
    st.success("¡forecasting_models.py importado exitosamente!")
    forecasting_models.placeholder_function() # Llama a la función para asegurar que existe
    st.write("Función de placeholder llamada.")
except SyntaxError as se:
    st.error(f"SyntaxError al importar forecasting_models: {se}")
except ImportError as ie:
    st.error(f"ImportError para forecasting_models: {ie}")
except Exception as e:
    st.error(f"Otro error al importar forecasting_models: {e}")

st.write("Fin de la prueba de importación.")
