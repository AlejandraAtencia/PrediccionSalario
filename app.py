"""App Streamlit: ingreso laboral de magísteres en áreas de analítica de datos con empleo formal en Colombia(GEIH 2025-2026, DANE)."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ingreso magíster en áreas de analítica con empleo formal en Colombia", page_icon="📊", layout="centered")

# ---------------------------------------------------------------- estilos
st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; max-width: 820px;}
.hero {
    background: linear-gradient(135deg, #4338CA 0%, #6D28D9 55%, #9333EA 100%);
    color: #FFFFFF; border-radius: 18px; padding: 1.8rem 2rem 1.6rem; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero .etiqueta {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .85; margin: 0;}
.hero h1 {color: #FFFFFF; font-size: 2rem; line-height: 1.2; margin: .35rem 0 .6rem; padding: 0;}
.hero p {color: rgba(255,255,255,.9); font-size: .98rem; line-height: 1.55; margin: 0;}
.chips {margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .45rem;}
.chip {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
       border-radius: 999px; padding: .22rem .75rem; font-size: .8rem; color: #FFFFFF;}
.seccion {font-weight: 600; font-size: 1.02rem; margin: 0 0 .2rem; color: #312E81;}
.resultado {
    background: #FFFFFF; border: 1px solid #E0E3F1; border-radius: 18px;
    padding: 1.6rem 1.8rem; margin-top: 1.2rem; box-shadow: 0 6px 22px rgba(30, 27, 75, .08);
}
.resultado .lbl {font-size: .85rem; color: #6B7280; text-transform: uppercase; letter-spacing: .06em;}
.resultado .valor {font-size: 2.6rem; font-weight: 700; color: #4338CA; line-height: 1.15; margin: .2rem 0;}
.resultado .sub {font-size: .9rem; color: #6B7280;}
.delta {display: inline-block; margin-top: .6rem; padding: .2rem .65rem; border-radius: 999px;
        font-size: .82rem; font-weight: 600;}
.delta.up {background: #DCFCE7; color: #166534;}
.delta.down {background: #FEE2E2; color: #991B1B;}
.delta.eq {background: #EEF0F8; color: #4B5563;}
.rango {margin-top: 1.4rem;}
.rango .titulo {font-size: .88rem; color: #374151; margin-bottom: .55rem;}
.barra {position: relative; height: 12px; border-radius: 999px;
        background: linear-gradient(90deg, #C7D2FE, #818CF8 50%, #C7D2FE);}
.marca {position: absolute; top: -6px; width: 4px; height: 24px; background: #312E81;
        border-radius: 2px; transform: translateX(-50%);}
.extremos {display: flex; justify-content: space-between; margin-top: .5rem; font-size: .85rem; color: #374151;}
.extremos span b {display: block; font-size: 1rem; color: #111827;}
.extremos .der {text-align: right;}
.pie {text-align: center; font-size: .8rem; color: #9CA3AF; margin-top: 2.5rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ingreso_stem.joblib")


art = cargar_modelo()
cats, base = art["categorias"], art["perfil_base"]

ETIQUETAS = {
    "EXPERIENCIA": "Experiencia (años cotizando a pensión)", "EDAD": "Edad", "HORAS_SEMANA": "Horas de trabajo por semana",
    "SEXO": "Sexo", "DEPARTAMENTO": "Departamento", "AREA_FORMACION": "Área de formación de la maestría",
    "SECTOR": "Sector económico", "OCUPACION": "Ocupación", "POSICION": "Posición ocupacional",
    "CONTRATO": "Tipo de contrato", "MODALIDAD": "Lugar de trabajo",
}
RANGOS = {"EXPERIENCIA": (0, 45), "EDAD": (22, 75), "HORAS_SEMANA": (10, 84)}
SECCIONES = [
    ("👤  Perfil personal", ["EDAD", "SEXO", "EXPERIENCIA"]),
    ("🎓  Formación y ubicación", ["AREA_FORMACION", "DEPARTAMENTO"]),
    ("💼  Empleo", ["SECTOR", "OCUPACION", "POSICION", "CONTRATO", "MODALIDAD", "HORAS_SEMANA"]),
]


def pesos(x):
    """Formato colombiano: $ 8.500.000"""
    return "$ " + f"{x:,.0f}".replace(",", ".")


def predecir(perfil):
    perfil = dict(perfil)
    if "DEPARTAMENTO" in perfil:
        perfil["REGION"] = art["dpto_region"].get(perfil["DEPARTAMENTO"], "Sin información")
    X = pd.DataFrame([perfil])[art["num"] + art["cat"]]
    return art["pipeline"].predict(X)[0]


def campo(v):
    if v in art["num"]:
        lo, hi = RANGOS.get(v, (0, 100))
        valor = int(min(max(base.get(v, lo), lo), hi))
        return st.slider(ETIQUETAS.get(v, v), lo, hi, valor)
    opciones = cats[v]
    d = base.get(v)
    return st.selectbox(ETIQUETAS.get(v, v), opciones, index=opciones.index(d) if d in opciones else 0)


# ---------------------------------------------------------------- encabezado
n_reg = f"{int(art['n_registros']):,}".replace(",", ".")
st.markdown(
    f"""
<div class="hero">
  <p class="etiqueta">Maestría en analítica de datos · Colombia</p>
  <h1>¿Cuánto podría ganar un magíster en áreas de analítica de datos con empleo formal en Colombia?</h1>
  <p>Estima el <b>ingreso laboral mensual</b> (pesos de 2026) de una persona con maestría en matemáticas,
  estadística, bases de datos, sistemas, inteligencia artificial o ciencia de datos y <b>empleo formal</b>.
  Los valores iniciales corresponden a un magíster en ciencia de datos: ajústalos a tu perfil.</p>
  <div class="chips">
    <span class="chip">Modelo: {art['modelo']}</span>
    <span class="chip">{n_reg} registros</span>
    <span class="chip">GEIH · DANE · ene 2025 – jul 2026</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- formulario
visibles = [v for v in art["num"] + art["cat"] if v != "REGION"]
agrupadas = {v for _, vs in SECCIONES for v in vs}
secciones = [(t, [v for v in vs if v in visibles]) for t, vs in SECCIONES]
otras = [v for v in visibles if v not in agrupadas]
if otras:
    secciones.append(("⚙️  Otras variables", otras))

entradas = {}
with st.form("perfil", border=False):
    for titulo, variables in secciones:
        if not variables:
            continue
        with st.container(border=True):
            st.markdown(f'<p class="seccion">{titulo}</p>', unsafe_allow_html=True)
            col = st.columns(2)
            for i, v in enumerate(variables):
                with col[i % 2]:
                    entradas[v] = campo(v)
    enviar = st.form_submit_button("Estimar ingreso  →", type="primary", use_container_width=True)

# ---------------------------------------------------------------- resultado
if enviar:
    lp = predecir({**base, **entradas})
    mediana, p10, p90 = np.exp(lp), np.exp(lp + art["q10"]), np.exp(lp + art["q90"])
    ref = np.exp(predecir(base))
    dif = (mediana / ref - 1) * 100
    if abs(dif) < 0.5:
        delta = '<span class="delta eq">Igual al perfil de referencia</span>'
    elif dif > 0:
        delta = f'<span class="delta up">▲ {dif:.0f} % sobre el perfil de referencia</span>'
    else:
        delta = f'<span class="delta down">▼ {abs(dif):.0f} % bajo el perfil de referencia</span>'
    pos = 0 if p90 == p10 else (mediana - p10) / (p90 - p10) * 100

    st.markdown(
        f"""
<div class="resultado">
  <div class="lbl">Ingreso laboral mensual estimado</div>
  <div class="valor">{pesos(mediana)}</div>
  <div class="sub">Mediana · pesos de 2026</div>
  {delta}
  <div class="rango">
    <div class="titulo">Rango probable: 8 de cada 10 personas con un perfil similar ganan entre</div>
    <div class="barra"><div class="marca" style="left:{pos:.1f}%"></div></div>
    <div class="extremos">
      <span>Percentil 10<b>{pesos(p10)}</b></span>
      <span class="der">Percentil 90<b>{pesos(p90)}</b></span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        "La mediana es el ingreso típico: la mitad de las personas con este perfil gana más y la otra mitad menos. "
        "El rango se calculó con los errores de validación cruzada del modelo. El perfil de referencia es un "
        "magíster en ciencia de datos con los valores iniciales del formulario."
    )

# ---------------------------------------------------------------- desempeño
st.write("")
with st.expander("📈  Desempeño del modelo en el conjunto de prueba (30 %)"):
    m = art["metricas_test"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", pesos(m["MAE (pesos)"]), help="Error absoluto medio en pesos.")
    c2.metric("RMSE (log)", f"{m['RMSE (log)']:.3f}".replace(".", ","), help="Raíz del error cuadrático medio en escala logarítmica.")
    c3.metric("R²", f"{m['R²']:.3f}".replace(".", ","), help="Proporción de la variación del (log) ingreso explicada por el modelo.")
    st.caption(
        "Fuente: DANE – GEIH 2025 y 2026, publicada en Datos Abiertos Colombia (datos.gov.co). "
        "Ingresos de 2025 llevados a pesos de 2026 con el IPC (5,10 %)."
    )

st.markdown('<div class="pie">Datos: DANE – Gran Encuesta Integrada de Hogares · Estimaciones con fines académicos</div>',
            unsafe_allow_html=True)
