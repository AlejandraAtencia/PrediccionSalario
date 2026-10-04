"""App Streamlit: ingreso laboral de magísteres en áreas de analítica de datos con empleo formal (GEIH 2025-2026, DANE)."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ingreso magíster en analítica", page_icon="📊", layout="centered")


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ingreso_stem.joblib")


art = cargar_modelo()
cats, base = art["categorias"], art["perfil_base"]
ETIQUETAS = {
    "EXPERIENCIA": "Experiencia: años cotizando a pensión", "EDAD": "Edad", "HORAS_SEMANA": "Horas de trabajo por semana",
    "SEXO": "Sexo", "DEPARTAMENTO": "Departamento", "AREA_FORMACION": "Área de formación de la maestría",
    "SECTOR": "Sector económico", "OCUPACION": "Ocupación", "POSICION": "Posición ocupacional",
    "CONTRATO": "Tipo de contrato", "MODALIDAD": "Lugar de trabajo",
}
RANGOS = {"EXPERIENCIA": (0, 45), "EDAD": (22, 75), "HORAS_SEMANA": (10, 84)}

st.title("Predicción del ingreso laboral")
st.write(
    "Estima el **ingreso laboral mensual** (pesos de 2026) de un graduado de **maestría en un área de analítica de datos** "
    "(matemáticas, estadística, bases de datos, sistemas, inteligencia artificial o ciencia de datos) con **empleo formal** "
    f"en Colombia. Modelo **{art['modelo']}** entrenado con {art['n_registros']} registros de la GEIH (DANE), "
    "enero de 2025 a julio de 2026. Los valores iniciales corresponden a un **magíster en ciencia de datos**."
)

entradas = {}
with st.form("perfil"):
    visibles = [v for v in art["num"] + art["cat"] if v != "REGION"]
    col = st.columns(2)
    for i, v in enumerate(visibles):
        with col[i % 2]:
            if v in art["num"]:
                lo, hi = RANGOS.get(v, (0, 100))
                entradas[v] = st.number_input(ETIQUETAS.get(v, v), lo, hi, int(base.get(v, lo)))
            else:
                opciones = cats[v]
                d = base.get(v)
                entradas[v] = st.selectbox(ETIQUETAS.get(v, v), opciones,
                                           index=opciones.index(d) if d in opciones else 0)
    enviar = st.form_submit_button("Predecir ingreso", type="primary")

if enviar:
    perfil = {**base, **entradas}
    if "DEPARTAMENTO" in perfil:
        perfil["REGION"] = art["dpto_region"].get(perfil["DEPARTAMENTO"], "Sin información")
    X = pd.DataFrame([perfil])[art["num"] + art["cat"]]
    lp = art["pipeline"].predict(X)[0]
    mediana, p10, p90 = np.exp(lp), np.exp(lp + art["q10"]), np.exp(lp + art["q90"])
    st.metric("Ingreso laboral mensual estimado (mediana)", f"${mediana:,.0f}")
    st.write(f"Rango probable (80 % de los casos similares): **${p10:,.0f}** a **${p90:,.0f}**")
    st.caption("La mediana es el ingreso típico: la mitad de las personas con este perfil gana más y la otra mitad menos. "
               "El rango se calculó con los errores de validación cruzada del modelo.")

with st.expander("Desempeño del modelo en el conjunto de prueba (30 %)"):
    m = art["metricas_test"]
    st.write(f"- MAE: ${m['MAE (pesos)']:,.0f}\n- RMSE (log): {m['RMSE (log)']:.3f}\n- R²: {m['R²']:.3f}")
    st.caption("Fuente: DANE – GEIH 2025 y 2026, publicada en Datos Abiertos Colombia (datos.gov.co). "
               "Ingresos de 2025 llevados a pesos de 2026 con el IPC (5,10 %).")
