# -*- coding: utf-8 -*-
# Generated from 02_clasificacion_features_tda.ipynb
# This script mirrors the notebook cells for reproducible execution.

# %% [markdown] Cell 1
# # Clasificación de fallas: features clásicas vs features TDA
#
# Este notebook evalua si las fallas en rodamientos pueden clasificarse a partir de dos representaciones distintas de las ventanas de vibracion:
#
# 1. **Features estadísticas clásicas**: describen energia, dispersion, impulsividad y amplitud de cada ventana.
# 2. **Features topológicas TDA**: describen la geometria de la reconstruccion de Takens mediante homologia persistente.
#
# El mdelo hibrido, que combinaria features clasicas y TDA, se deja como trabajo futuro para mantener el alcance del proyecto dentro del tiempo disponible.

# %% [markdown] Cell 2
# ## Justificación del modelo de clasificación
#
# Se utiliza **Support Vector Machine con kernel RBF** (`SVC(kernel="rbf")`) para ambos conjuntos de features. Esta elección permite comparar de forma justa las representaciones, porque el clasificador se mantiene fijo y solo cambia la informacón de entrada.
#
# El kernel RBF es adecuado porque las clases no necesariamente son separables linealmente. En partícular, las features TDA provienen de una reconstrucción no lineal de la dinámica vibracional, por lo que es razonable usar un modelo con fronteras de decisión no lineales.
#
# Tambien se usa `StandardScaler`, ya que SVM es sensible a la escala de las variables.

# %% [markdown] Cell 3
# ## Nota metodológica importante
#
# La particion train/test se realiza a nivel de ventana con estratificación por clase. Esto permite evaluar los modelos con todas las clases presentes en entrenamiento y prueba.
#
# Sin embargo, las ventanas provienen de señales largas y muchas tienen traslape, por lo que no son completamente independientes. Por esa razón, las métricas deben interpretarse como una evaluación inicial del poder discriminativo de las features, no como una validación industrial definitiva. Una validacion maá estricta requeriría más archivos por clase para separar por archivo, máquina o condición experimental.

# %% [markdown] Cell 4
# # 1 Librerias y rutas

# %% Cell 5
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE = Path.cwd()
VENTANAS_DIR = BASE / "ventanas_DE"
FEATURES_TDA_DIR = BASE / "features_TDA"
OUTPUT_FIGURE = BASE / "Figuras_TDA"
OUTPUT_MODELOS = BASE / "clasificacion_modelos"

OUTPUT_FIGURE.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELOS.mkdir(parents=True, exist_ok=True)

print("BASE:", BASE)
print("Ventanas:", VENTANAS_DIR)
print("Features TDA:", FEATURES_TDA_DIR)
print("Salida modelos:", OUTPUT_MODELOS)

# %% [markdown] Cell 6
# # 2 Carga de metadata y features TDA

# %% Cell 7
metadata_ventanas_path = VENTANAS_DIR / "metadata_ventanas_tau.csv"
features_tda_path = FEATURES_TDA_DIR / "features_topologicas_ventanas.csv"

if not metadata_ventanas_path.exists():
    raise FileNotFoundError(f"No se encontro metadata de ventanas: {metadata_ventanas_path}")

if not features_tda_path.exists():
    raise FileNotFoundError(f"No se encontraron features TDA: {features_tda_path}")

metadata_ventanas = pd.read_csv(
    metadata_ventanas_path,
    dtype={"ventana_id": str, "archivo_origen": str, "ruta_ventana": str, "clase": str, "severidad": str, "carga": str}
)

features_tda = pd.read_csv(
    features_tda_path,
    dtype={"ventana_id": str, "archivo_origen": str, "clase": str, "severidad": str, "carga": str}
)

print("Metadata ventanas:", metadata_ventanas.shape)
print("Features TDA:", features_tda.shape)
display(features_tda.head())
display(features_tda["clase"].value_counts().rename_axis("clase").reset_index(name="n_ventanas"))

# %% [markdown] Cell 8
# # 3 Features estadísticas clásicas por ventana
#
# Las features clásicas se calculan directamente sobre cada ventana temporal. Estas variables resumen amplitud, energía e impulsividad. Si ya existe el archivo `features_clasicas_ventanas.csv`, se carga para evitar recalcular.

# %% Cell 9
def resolver_ruta_ventana(fila):
    """Resuelve la ruta de una ventana tanto si viene como ruta absoluta como si se reconstruye localmente."""
    ruta_guardada = Path(str(fila["ruta_ventana"]))
    if ruta_guardada.exists():
        return ruta_guardada

    ruta_reconstruida = VENTANAS_DIR / fila["clase"] / f"{fila['ventana_id']}.npy"
    if ruta_reconstruida.exists():
        return ruta_reconstruida

    raise FileNotFoundError(f"No se encontro la ventana: {fila['ventana_id']}")


def calcular_features_clasicas(signal):
    """Calcula features estadisticas clasicas para una ventana 1D."""
    x = np.asarray(signal).ravel()
    media = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x ** 2))
    centered = x - media

    if std == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = np.mean(centered ** 3) / (std ** 3)
        kurtosis = np.mean(centered ** 4) / (std ** 4)

    peak_to_peak = np.max(x) - np.min(x)
    crest_factor = np.max(np.abs(x)) / rms if rms != 0 else np.nan

    return {
        "mean": media,
        "std": std,
        "rms": rms,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "max": np.max(x),
        "min": np.min(x),
    }

# %% Cell 10
features_clasicas_path = OUTPUT_MODELOS / "features_clasicas_ventanas.csv"

if features_clasicas_path.exists():
    features_clasicas = pd.read_csv(
        features_clasicas_path,
        dtype={"ventana_id": str, "archivo_origen": str, "clase": str, "severidad": str, "carga": str}
    )
else:
    registros = []

    for i, fila in metadata_ventanas.iterrows():
        ventana = np.load(resolver_ruta_ventana(fila))
        feats = calcular_features_clasicas(ventana)

        registros.append({
            "ventana_id": fila["ventana_id"],
            "archivo_origen": fila["archivo_origen"],
            "clase": fila["clase"],
            "severidad": fila["severidad"],
            "carga": fila["carga"],
            **feats
        })

        if (i + 1) % 1000 == 0:
            print(f"Ventanas procesadas: {i + 1}/{len(metadata_ventanas)}")

    features_clasicas = pd.DataFrame(registros)
    features_clasicas.to_csv(features_clasicas_path, index=False)

print("Features clasicas:", features_clasicas.shape)
print("Guardadas en:", features_clasicas_path)
display(features_clasicas.head())

# %% [markdown] Cell 11
# # 4 Preparacion de datasets y split train/test

# %% Cell 12
# Mantener las mismas ventanas y el mismo split para ambos modelos.
common_ids = set(features_clasicas["ventana_id"]).intersection(features_tda["ventana_id"])

features_clasicas_modelo = (
    features_clasicas[features_clasicas["ventana_id"].isin(common_ids)]
    .sort_values("ventana_id")
    .reset_index(drop=True)
)

features_tda_modelo = (
    features_tda[features_tda["ventana_id"].isin(common_ids)]
    .sort_values("ventana_id")
    .reset_index(drop=True)
)

assert (features_clasicas_modelo["ventana_id"].values == features_tda_modelo["ventana_id"].values).all()

y = features_clasicas_modelo["clase"]
indices = np.arange(len(y))

idx_train, idx_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

split = pd.DataFrame({
    "ventana_id": features_clasicas_modelo["ventana_id"],
    "split": "train"
})
split.loc[idx_test, "split"] = "test"
split.to_csv(OUTPUT_MODELOS / "split_train_test.csv", index=False)

print("Train:", len(idx_train))
print("Test:", len(idx_test))
display(split["split"].value_counts().rename_axis("split").reset_index(name="n"))

# %% Cell 13
features_clasicas_cols = [
    "mean", "std", "rms", "skewness", "kurtosis",
    "peak_to_peak", "crest_factor", "max", "min"
]

features_tda_cols = [col for col in features_tda_modelo.columns if col.startswith("H")]

print("Features clasicas:", len(features_clasicas_cols), features_clasicas_cols)
print("Features TDA:", len(features_tda_cols), features_tda_cols)

# %% [markdown] Cell 14
# # 5 Entrenamiento con SVM-RBF
#
# Se usa `GridSearchCV` sobre el conjunto de entrenamiento para seleccionar `C` y `gamma` maximizando `f1_macro`. La métrica `f1_macro` es adecuada porque promedia el desempeño por clase sin favorecer automáticamente a clases con más ventanas.

# %% Cell 15
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def entrenar_evaluar_svm(nombre, dataframe, feature_cols):
    """Entrena SVM-RBF, evalua en test y guarda metricas/figuras."""
    X = dataframe[feature_cols].astype(float)
    y = dataframe["clase"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE))
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid.fit(X.iloc[idx_train], y.iloc[idx_train])

    pred = grid.predict(X.iloc[idx_test])
    y_test = y.iloc[idx_test]

    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )

    reporte = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).transpose()
    ruta_reporte = OUTPUT_MODELOS / f"reporte_clasificacion_{nombre}.csv"
    reporte.to_csv(ruta_reporte)

    clases = sorted(y.unique())
    matriz = confusion_matrix(y_test, pred, labels=clases)
    matriz_df = pd.DataFrame(matriz, index=clases, columns=clases)
    ruta_matriz = OUTPUT_MODELOS / f"matriz_confusion_{nombre}.csv"
    matriz_df.to_csv(ruta_matriz)

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=clases)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Matriz de confusión - {nombre}")
    plt.tight_layout()
    ruta_figura = OUTPUT_FIGURE / f"matriz_confusion_{nombre}.png"
    plt.savefig(ruta_figura, dpi=150, bbox_inches="tight")
    plt.show()

    resultado = {
        "modelo": nombre,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "best_params": json.dumps(grid.best_params_),
        "cv_best_f1_macro": grid.best_score_,
        "n_features": len(feature_cols),
        "reporte": str(ruta_reporte),
        "matriz_confusion": str(ruta_matriz),
        "figura": str(ruta_figura)
    }

    return resultado, grid

# %% Cell 16
resultado_clasico, modelo_clasico = entrenar_evaluar_svm(
    "clasico_svm_rbf",
    features_clasicas_modelo,
    features_clasicas_cols
)

resultado_tda, modelo_tda = entrenar_evaluar_svm(
    "tda_svm_rbf",
    features_tda_modelo,
    features_tda_cols
)

comparacion_modelos = pd.DataFrame([resultado_clasico, resultado_tda])
ruta_comparacion = OUTPUT_MODELOS / "comparacion_modelos.csv"
comparacion_modelos.to_csv(ruta_comparacion, index=False)

display(comparacion_modelos[[
    "modelo", "accuracy", "precision_macro", "recall_macro",
    "f1_macro", "cv_best_f1_macro", "best_params", "n_features"
]])
print("Comparacion guardada en:", ruta_comparacion)

# %% [markdown] Cell 17
# # 6 Comparación de resultados

# %% Cell 18
metricas = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparacion_modelos))
width = 0.2

for j, metrica in enumerate(metricas):
    ax.bar(
        x + (j - 1.5) * width,
        comparacion_modelos[metrica],
        width,
        label=metrica
    )

ax.set_xticks(x)
ax.set_xticklabels(comparacion_modelos["modelo"], rotation=10)
ax.set_ylim(0, 1.05)
ax.set_title("Comparación de modelos SVM-RBF")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()

ruta_figura_comparacion = OUTPUT_FIGURE / "comparacion_modelos_svm_rbf.png"
plt.savefig(ruta_figura_comparacion, dpi=150, bbox_inches="tight")
plt.show()

print("Figura guardada en:", ruta_figura_comparacion)

# %% [markdown] Cell 19
# ## Discusión esperada
#
# El modelo clásico evalua si la energía, amplitud e impulsividad de las ventanas son suficientes para distinguir las fallas. El modelo TDA evalua si la geometria de la dinámica reconstruida aporta información discriminativa.
#
# Si el modelo clásico supera al TDA, no invalida el enfoque topológico: indica que, bajo esta configuración de submuestreo y features, las medidas estadísticas capturan de forma mas directa diferencias fuertes de amplitud e impulsividad.
#
# Si el modelo TDA logra desempeño competitivo, entonces se apoya la hipótesis de que la geometría del atractor reconstruido contiene información diagnóstica. El modelo híbrido se evalua al final para medir si combinar sensibilidad física y geometría topológica mejora la clasificación.

# %% [markdown] Cell 20
# # 7 Modelo híbrido: features clásicas + TDA
#
# Aunque inicialmente se compararon solo dos representaciones, ahora se agrega un tercer experimento híbrido. La idea es evaluar si las features estadísticas clásicas y las features topológicas contienen información complementaria.
#
# Se mantiene exactamente el mismo clasificador, el mismo split train/test y la misma busqueda de hiperparámetros. Así, cualquier cambio en desempeño se debe al conjunto de features y no al modelo.

# %% [markdown] Cell 21
# ## 7.1 Features consideradas en el modelo híbrido
#
# El modelo híbrido concatena dos bloques de variables:
#
# **Features clásicas por ventana:**
#
# - `mean`
# - `std`
# - `rms`
# - `skewness`
# - `kurtosis`
# - `peak_to_peak`
# - `crest_factor`
# - `max`
# - `min`
#
# **Features TDA:**
#
# - Para `H0`, `H1` y `H2`: `count`, `total_persistence`, `max_persistence`, `mean_persistence` y `persistence_entropy`.
#
# En total se usan `9 + 15 = 24` variables.

# %% Cell 22
# Construir dataset hibrido alineando las mismas ventanas
features_hibridas = pd.concat([
    features_clasicas_modelo[["ventana_id", "archivo_origen", "clase", "severidad", "carga"]],
    features_clasicas_modelo[features_clasicas_cols],
    features_tda_modelo[features_tda_cols]
], axis=1)

features_hibridas_cols = features_clasicas_cols + features_tda_cols
features_hibridas_path = OUTPUT_MODELOS / "features_hibridas_ventanas.csv"
features_hibridas.to_csv(features_hibridas_path, index=False)

print("Features hibridas:", features_hibridas.shape)
print("Numero de variables predictoras:", len(features_hibridas_cols))
print("Dataset hibrido guardado en:", features_hibridas_path)
display(features_hibridas.head())

# %% [markdown] Cell 23
# ## 7.2 Entrenamiento del modelo híbrido
#
# El entrenamiento se realiza paso a paso igual que en los modelos anteriores:
#
# 1. Se concatenan las features clasicas y TDA por `ventana_id`.
# 2. Se usa el mismo `train/test split` estratificado.
# 3. Se estandarizan todas las variables con `StandardScaler`.
# 4. Se entrena `SVC(kernel="rbf")`.
# 5. Se seleccionan `C` y `gamma` con `GridSearchCV` usando `f1_macro`.
# 6. Se evalua en el conjunto de prueba con accuracy, precision macro, recall macro, F1 macro y matriz de confusión.

# %% Cell 24
resultado_hibrido, modelo_hibrido = entrenar_evaluar_svm(
    "hibrido_svm_rbf",
    features_hibridas,
    features_hibridas_cols
)

comparacion_modelos = comparacion_modelos[comparacion_modelos["modelo"] != "hibrido_svm_rbf"]
comparacion_modelos = pd.concat([
    comparacion_modelos,
    pd.DataFrame([resultado_hibrido])
], ignore_index=True)

comparacion_modelos.to_csv(OUTPUT_MODELOS / "comparacion_modelos.csv", index=False)
display(comparacion_modelos[[
    "modelo", "accuracy", "precision_macro", "recall_macro",
    "f1_macro", "cv_best_f1_macro", "best_params", "n_features"
]])

# %% Cell 25
# Actualizar grafica comparativa incluyendo el modelo hibrido
metricas = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(comparacion_modelos))
width = 0.18

for j, metrica in enumerate(metricas):
    ax.bar(
        x + (j - 1.5) * width,
        comparacion_modelos[metrica],
        width,
        label=metrica
    )

ax.set_xticks(x)
ax.set_xticklabels(comparacion_modelos["modelo"], rotation=10)
ax.set_ylim(0, 1.05)
ax.set_title("Comparación de modelos SVM-RBF")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()

ruta_figura_comparacion = OUTPUT_FIGURE / "comparacion_modelos_svm_rbf.png"
plt.savefig(ruta_figura_comparacion, dpi=150, bbox_inches="tight")
plt.show()

print("Figura actualizada en:", ruta_figura_comparacion)

# %% [markdown] Cell 26
# # 8 Discusion final de clasificación
#
# Con `random_state = 42`, `test_size = 0.20`, SVM-RBF y busqueda de hiperparámetros, se obtuvo en la ejecucion local:
#
# | Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Variables |
# |---|---:|---:|---:|---:|---:|
# | Clasico SVM-RBF | 0.9568 | 0.9650 | 0.9635 | 0.9640 | 9 |
# | TDA SVM-RBF | 0.8780 | 0.8995 | 0.8988 | 0.8981 | 15 |
# | Hibrido SVM-RBF | 0.9428 | 0.9534 | 0.9519 | 0.9526 | 24 |
#
# El modelo clásico obtuvo el mejor desempeño. El modelo híbrido mejora claramente al modelo TDA, lo que indica que las features clásicas aportan información muy fuerte para este dataset. Sin embargo, el híbrido no supera al clásico, lo cual sugiere que las features TDA, bajo la configuración actual de submuestreo y resumen topológico, no agregan suficiente información adicional para mejorar el clasificador.
#
# Esto es físicamente razonable: las fallas del dataset CWRU producen cambios marcados de amplitud, energía e impulsividad, capturados directamente por `RMS`, `kurtosis`, `peak_to_peak` y `crest_factor`. Las features TDA siguen siendo relevantes porque logran clasificar con buen desempeño usando información geométrica de la dinámica, pero en esta configuración no dominan a las features clásicas.
#
# Como trabajo futuro se podría evaluar TDA con mayor número de puntos, estrategias de submuestreo más robustas, selección de features, otros kernels o validación por archivo/condicion experimental.

