# Diagnóstico de fallas en rodamientos mediante TDA y Machine Learning

Este proyecto desarrolla un flujo de mantenimiento predictivo para diagnosticar fallas en rodamientos usando señales de vibración del dataset CWRU. El procedimiento combina análisis físico de señales, ventaneo, reconstrucción del espacio de fase mediante Takens, homología persistente y clasificación supervisada.

## Objetivo

Evaluar si la dinámica vibracional de un rodamiento contiene información suficiente para distinguir entre:

- condición normal,
- falla en bola (`Ball`),
- falla en pista interna (`Inner_Race`),
- falla en pista externa (`Outer_Race`).

Se comparan tres representaciones de las señales:

- features estadísticas clásicas,
- features topológicas TDA,
- features híbridas: clásicas + TDA.

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_mantenimiento_predictivo_tda.ipynb
│   └── 02_clasificacion_features_tda.ipynb
├── src/
│   ├── 01_mantenimiento_predictivo_tda.py
│   └── 02_clasificacion_features_tda.py
└── results/
    ├── comparacion_modelos.csv
    ├── resumen_features_TDA_por_clase.csv
    ├── reportes y matrices de confusion
    └── figuras_finales/
```

Los datos crudos y archivos generados pesados no se suben al repositorio. Ver `data/README.md`.

## Dataset

Se utiliza el **Case Western Reserve University Bearing Dataset (CWRU)**.

Se trabaja con señales del acelerómetro del lado motriz:

```text
DE_time
```

Clases usadas:

| Clase | Archivos |
|---|---|
| Normal | `Time_Normal_1_098` |
| Ball | `B007_1_123`, `B014_1_190`, `B021_1_227` |
| Inner_Race | `IR007_1_110`, `IR014_1_175`, `IR021_1_214` |
| Outer_Race | `OR007_6_1_136`, `OR014_6_1_202`, `OR021_6_1_239` |

## Instalación

Crear un entorno de Python e instalar dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principales:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `ripser`
- `persim`
- `jupyter`

## Cómo regenerar los datos

1. Descargar los archivos `.mat` del dataset CWRU.
2. Colocarlos en:

```text
archive/raw/
```

o:

```text
raw/
```

3. Ejecutar:

```text
notebooks/01_mantenimiento_predictivo_tda.ipynb
```

Ese notebook realiza:

1. Detección de rutas local/Colab.
2. Extracción de señales `DE_time`.
3. Guardado de señales `.npy`.
4. Análisis exploratorio físico:
   - señales temporales,
   - estadísticos clásicos,
   - FFT,
   - autocorrelación,
   - espacio de fases 2D,
   - comparación por severidad.
5. Ventaneo:
   - `window_size = 2048`,
   - `overlap = 0.75`,
   - `step = 512`.
6. Cálculo de `tau` por autocorrelación en ventanas.
7. Reconstrucción de Takens:
   - `d = 3`.
8. Homología persistente:
   - `H0`, `H1`, `H2`,
   - submuestreo uniforme de puntos por ventana.
9. Extracción de features TDA.
10. Análisis de sensibilidad del submuestreo.

## Features generadas

### Features clásicas

Calculadas por ventana:

- `mean`
- `std`
- `rms`
- `skewness`
- `kurtosis`
- `peak_to_peak`
- `crest_factor`
- `max`
- `min`

### Features TDA

Para cada dimensión homológica `H0`, `H1`, `H2`:

- `count`
- `total_persistence`
- `max_persistence`
- `mean_persistence`
- `persistence_entropy`

Total:

```text
15 features TDA
```

### Features híbridas

Concatenación de:

```text
9 features clásicas + 15 features TDA = 24 features
```

## Clasificación

La clasificación se realiza en:

```text
notebooks/02_clasificacion_features_tda.ipynb
```

Modelo usado:

```text
SVM con kernel RBF
```

Justificación:

- las clases no necesariamente son separables linealmente,
- las features TDA provienen de una reconstrucción no lineal de la dinámica,
- SVM-RBF funciona bien en datasets tabulares de tamaño medio,
- se usa el mismo modelo para comparar representaciones de forma justa.

Se evalúan:

1. SVM-RBF con features clásicas.
2. SVM-RBF con features TDA.
3. SVM-RBF con features híbridas.

## Resultados

Resultados obtenidos con:

- `random_state = 42`,
- `test_size = 0.20`,
- partición estratificada por clase,
- búsqueda de hiperparámetros con `GridSearchCV`,
- métrica de selección: `f1_macro`.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Features |
|---|---:|---:|---:|---:|---:|
| Clásico SVM-RBF | 0.9568 | 0.9650 | 0.9635 | 0.9640 | 9 |
| TDA SVM-RBF | 0.8780 | 0.8995 | 0.8988 | 0.8981 | 15 |
| Híbrido SVM-RBF | 0.9428 | 0.9534 | 0.9519 | 0.9526 | 24 |

En esta corrida, el modelo con features clásicas obtiene el mejor desempeño. El modelo híbrido mejora al modelo TDA, pero no supera al clásico. Esto sugiere que, para este dataset, los cambios de energía, amplitud e impulsividad son muy discriminativos.

## Figuras finales

Las matrices de confusión de cada clasificador están en:

```text
results/figuras_finales/
```

Incluye:

- matriz de confusión del modelo clásico,
- matriz de confusión del modelo TDA,
- matriz de confusión del modelo híbrido,
- comparación general de métricas.

## Consideraciones metodológicas

Las ventanas tienen traslape, por lo que no son completamente independientes. La evaluación actual debe interpretarse como una comparación inicial de representaciones. Una validación industrial más estricta requeriría más señales por clase y separación por archivo, máquina o condición experimental.

## Trabajo futuro

- Evaluar validación por archivo o por máquina.
- Probar submuestreo topológico más robusto, como farthest point sampling o witness complexes.
- Explorar modelos adicionales: Random Forest, XGBoost, redes neuronales ligeras.
- Integrar el pipeline en un sistema edge/industrial conectado a PLC, SCADA u OPC-UA.
- Extender el enfoque hacia predicción de vida útil remanente (`RUL`) con datos de degradación temporal.

## Nota sobre datos y referencias

Los datos crudos y PDFs de referencia no se incluyen en el repositorio para evitar archivos pesados y posibles restricciones de distribución. El README de `data/` describe cómo organizar los archivos necesarios para reproducir el proyecto.
