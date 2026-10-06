# Predicción del ingreso laboral de profesionales con posgrado en Colombia (GEIH 2025-2026)

Aplicación web en Streamlit que estima el **ingreso laboral mensual** (pesos de 2026) de una persona con **título de posgrado** (especialización, maestría o doctorado) y **empleo formal** en Colombia, según su nivel y campo de formación, su experiencia, su ubicación y las condiciones del empleo. El perfil de referencia es un **magíster en ciencia de datos**.

Proyecto integrador desarrollado con la metodología **CRISP‑DM**.

## Datos
- Fuente: DANE, Gran Encuesta Integrada de Hogares (GEIH), enero de 2025 a julio de 2026 (19 meses). Ingresos de 2025 llevados a pesos de 2026 con el IPC (5,10 %).
- Datos Abiertos Colombia: https://www.datos.gov.co/dataset/Gran-Encuesta-Integrada-de-Hogares-GEIH-2025/rpws-7zjb (2025) y https://www.datos.gov.co/dataset/Gran-Encuesta-Integrada-de-Hogares-GEIH-2026/nzxb-qax7 (2026)
- Población: ocupados con título de posgrado (P3043 = 8, 9 o 10) en cualquier campo de formación, que cotizan a pensión (P6920 = 1) y tienen ingreso laboral > 0.
- Campo de formación: las 6 áreas de analítica (CINE‑F 0541, 0542, 0612, 0613, 0619, 0714) como categorías propias; el resto agrupado en el campo específico CINE‑F.
- Variable objetivo: `log(INGLABO)`. La app muestra el resultado en pesos.

## Modelo
- Se compararon 5 modelos clásicos (Ridge, KNN, Árbol de decisión, SVR, Red neuronal MLP) y 3 ensambles (Votación, Bagging, HistGradientBoosting), con validación cruzada de 5 particiones sobre el 70 % de entrenamiento y evaluación en el 30 % de prueba.
- `modelo_ingreso_posgrado.joblib` contiene el pipeline completo (imputación, codificación, escalado y modelo), las categorías válidas y los parámetros del intervalo de predicción.

## Archivos
| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit |
| `modelo_ingreso_posgrado.joblib` | Pipeline serializado (se genera en la Fase 6 del cuaderno) |
| `requirements.txt` | Dependencias con versiones fijas (las mismas de Colab) |
| `README.md` | Este documento |

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Enlaces
- Cuaderno Colab: <pegar enlace>
- App publicada: <pegar URL de Streamlit>
