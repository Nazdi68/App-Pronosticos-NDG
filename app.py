# app.py
import streamlit as st

st.title("Prueba de Importación de Módulo")

try:
    import forecasting_models
    st.success("MÓDULO 'forecasting_models.py' IMPORTADO CORRECTAMENTE.")
    
    # Intentar llamar a la función de prueba
    mensaje = forecasting_models.test_function_from_models()
    st.write(f"Resultado de la función de prueba: {mensaje}")

except SyntaxError as se:
    st.error(f"ERROR DE SINTAXIS al intentar importar 'forecasting_models.py':")
    st.exception(se) # Muestra el traceback completo del SyntaxError
except ImportError as ie:
    st.error(f"ERROR DE IMPORTACIÓN para 'forecasting_models.py':")
    st.exception(ie)
except Exception as e:
    st.error(f"OTRO ERROR durante la importación de 'forecasting_models.py':")
    st.exception(e)

st.info("Fin de la prueba de importación en app.py.")
