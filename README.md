# Predicción del ingreso laboral de magísteres en analítica de datos (GEIH 2025-2026)

Aplicación web en Streamlit que estima el **ingreso laboral mensual** de una persona graduada de una **maestría en un área de analítica de datos** (CINE‑F 0541, 0542, 0612, 0613, 0619, 0714) con **empleo formal** en Colombia. El caso de uso es el perfil de un **magíster en ciencia de datos** (0714).

Proyecto integrador desarrollado con la metodología **CRISP‑DM**.

## Datos
- Fuente: DANE, Gran Encuesta Integrada de Hogares (GEIH), enero de 2025 a julio de 2026 (19 meses). Ingresos de 2025 llevados a pesos de 2026 con el IPC (5,10 %).
- Datos Abiertos Colombia: https://www.datos.gov.co/dataset/Gran-Encuesta-Integrada-de-Hogares-GEIH-2025/rpws-7zjb (2025) y https://www.datos.gov.co/dataset/Gran-Encuesta-Integrada-de-Hogares-GEIH-2026/nzxb-qax7 (2026)
- Población: ocupados con título de maestría (P3042 = 12, P3043 = 9) en áreas de analítica de datos que cotizan a pensión (P6920 = 1).
- Variable objetivo: `log(INGLABO)`. La app muestra el resultado en pesos.

## Modelo
- Se compararon 5 modelos clásicos (Ridge, KNN, Árbol de decisión, SVR, Red neuronal MLP) y 3 ensambles (Votación, Bagging, Gradient Boosting), con validación cruzada de 5 particiones sobre el 70 % de entrenamiento y evaluación en el 30 % de prueba.
- `modelo_ingreso_stem.joblib` contiene el pipeline completo (imputación, codificación, escalado y modelo), las categorías válidas y los parámetros del intervalo de predicción.

## Archivos
| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit |
| `modelo_ingreso_stem.joblib` | Pipeline serializado |
| `requirements.txt` | Dependencias con versiones fijas |
| `README.md` | Este documento |

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Enlaces
- Cuaderno Colab: <pegar enlace>
- App publicada: <pegar URL de Streamlit>
