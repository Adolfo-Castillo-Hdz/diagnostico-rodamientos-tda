# -*- coding: utf-8 -*-
# Generated from 01_mantenimiento_predictivo_tda.ipynb
# This script mirrors the notebook cells for reproducible execution.

# %% [markdown] Cell 1
# # Mantenimiento predictivo por medio de TDA
#
# Este notebook desarrolla un flujo de analisis para diagnosticar fallas en rodamientos a partir de senales de vibracion. El objetivo es estudiar si la forma geometrica y topologica de las senales permite distinguir entre una condicion normal y fallas en bola, pista interna y pista externa.
#
# El proyecto combina procesamiento de senales, reconstruccion del espacio de fase mediante el teorema de Takens y Topological Data Analysis (TDA), especialmente homologia persistente. La idea central es transformar una senal temporal en una nube de puntos y despues extraer descriptores topologicos que puedan alimentar un modelo de clasificacion.

# %% [markdown] Cell 2
# ## Como se interpretan los documentos y datos
#
# Los articulos PDF incluidos en la carpeta se usan como referencia metodologica: ayudan a justificar el uso de TDA en diagnostico de fallas, la reconstruccion de atractores, el uso de diagramas de persistencia y la comparacion contra caracteristicas clasicas de vibracion.
#
# Los datos provienen del Case Western Reserve University Bearing Dataset (CWRU). En este notebook se trabaja principalmente con las senales del acelerometro del lado motriz, conocidas como DE (Drive End), porque suelen tener mayor relacion con el defecto mecanico y son comunes en la literatura. Cada archivo `.mat` representa una condicion de operacion: normal o con una falla especifica.
#
# Las clases consideradas son: Normal, Ball fault, Inner Race fault y Outer Race fault, con severidades de 0.007, 0.014 y 0.021 pulgadas para los casos de falla.

# %% [markdown] Cell 3
# ## Tecnicas que se aplicaran
#
# 1. Extraccion de senales: se leen los archivos `.mat` y se extrae la variable `DE_time` correspondiente a cada medicion.
# 2. Analisis exploratorio: se grafican fragmentos de las senales y se calculan estadisticas basicas como media, desviacion estandar, minimo y maximo.
# 3. Ventaneo: cada senal se divide en ventanas temporales para generar multiples muestras de analisis.
# 4. Reconstruccion de Takens: cada ventana temporal se transforma en una nube de puntos mediante retardos temporales.
# 5. Homologia persistente: sobre cada nube de puntos se calculan caracteristicas topologicas en dimensiones `H0`, `H1` y, si es viable computacionalmente, `H2`.
# 6. Extraccion de features: se resumen los diagramas de persistencia con medidas como conteo, persistencia total, persistencia maxima y entropia de persistencia.
# 7. Clasificacion: se entrena un modelo, por ejemplo SVM con kernel RBF, para evaluar si las caracteristicas topologicas distinguen correctamente los tipos de falla.

# %% [markdown] Cell 4
# # 0 Librerias a utilizar y reproductibilidad

# %% [markdown] Cell 5
# ## Dependencias
#
# Este notebook requiere `numpy`, `pandas`, `scipy`, `matplotlib` y, para la etapa de TDA, `ripser`. Si `ripser` no esta instalado, en Google Colab puedes ejecutar una celda aparte con `%pip install ripser`. En local, instala las dependencias desde tu entorno de Python antes de correr el notebook.

# %% Cell 6
import numpy as np
import pandas as pd
import re
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # requerido por matplotlib para proyeccion 3D

try:
    from ripser import ripser
except ImportError:
    ripser = None
    print("Aviso: ripser no esta instalado. Instalalo antes de calcular homologia persistente.")

SEMILLA = np.random.seed(42)

# %% [markdown] Cell 7
# # 1 Rutas

# %% Cell 8
# Rutas compatibles con Google Colab y ejecucion local
try:
    from google.colab import drive
    EN_COLAB = True
except ImportError:
    drive = None
    EN_COLAB = False


def localizar_carpeta_datos(base_dir):
    candidatos = [
        base_dir / 'archive' / 'raw',
        base_dir / 'raw',
    ]

    for candidato in candidatos:
        if candidato.exists():
            return candidato

    for parent in [base_dir, *base_dir.parents]:
        for candidato in candidatos:
            if candidato.exists():
                return candidato

    return None

if EN_COLAB:
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/proyecto TDA')
else:
    candidatos_base = [Path.cwd(), *Path.cwd().parents]
    BASE = next(
        (c for c in candidatos_base if (c / 'archive' / 'raw').exists() or (c / 'raw').exists()), Path.cwd()
    )

DATA = localizar_carpeta_datos(BASE)
if DATA is None:
    raise FileNotFoundError(
        f"No se encontró la carpeta de datos esperada. Busqué en {BASE} y sus padres.\n"
        "Coloca los archivos .mat en 'raw' o 'archive/raw' dentro de la carpeta del proyecto."
    )

OUTPUT_FIGURE = BASE / 'Figuras_TDA'
SIGNALS = BASE / 'DE_signals'

OUTPUT_FIGURE.mkdir(parents=True, exist_ok=True)
SIGNALS.mkdir(parents=True, exist_ok=True)

print("Rutas creadas correctamente.")
print('Entorno:', 'Google Colab' if EN_COLAB else 'Local')
print('Carpeta base:', BASE)
print('Datos a trabajar:', DATA)
print('Resultados de figuras:', OUTPUT_FIGURE)
print('Senales de DE:', SIGNALS)

# %% [markdown] Cell 9
# # 2 Extracción y guardado de señales necesarias

# %% Cell 10
# ============================
# 1. Ruta a la carpeta raw
# ============================

RAW_DIR = DATA   # cambia esto si tu carpeta tiene otra ruta


# ============================
# 2. Función para leer metadatos del nombre
# ============================

def extraer_info_nombre(nombre_archivo):
    """
    Extrae tipo de falla, severidad, carga e ID desde el nombre del archivo.

    Ejemplos:
    B007_1_123        -> Ball, 007, 1, 123
    IR014_1_175       -> Inner Race, 014, 1, 175
    OR021_6_1_239     -> Outer Race, 021, 1, 239
    Time_Normal_1_098 -> Normal, None, 1, 098
    """

    nombre = os.path.splitext(nombre_archivo)[0]

    if nombre.startswith("Time_Normal"):
        partes = nombre.split("_")
        return {
            "clase": "Normal",
            "tipo_falla": "Normal",
            "severidad": "000",
            "carga": partes[-2],
            "id_archivo": partes[-1],
        }

    # Casos B, IR, OR
    patron = r"^(B|IR|OR)(\d{3})_(?:\d+_)?(\d+)_(\d+)$"
    match = re.match(patron, nombre)

    if match is None:
        raise ValueError(f"No pude interpretar el nombre: {nombre_archivo}")

    codigo_falla = match.group(1)
    severidad = match.group(2)
    carga = match.group(3)
    id_archivo = match.group(4)

    mapa_fallas = {
        "B": "Ball",
        "IR": "Inner_Race",
        "OR": "Outer_Race"
    }

    return {
        "clase": mapa_fallas[codigo_falla],
        "tipo_falla": mapa_fallas[codigo_falla],
        "severidad": severidad,
        "carga": carga,
        "id_archivo": id_archivo,
    }


# ============================
# 3. Función para encontrar la variable DE
# ============================

def obtener_variable_de(mat_dict, id_archivo):
    """
    Busca la variable DE_time correspondiente al ID del archivo.
    Ejemplo:
    id_archivo = '175' -> X175_DE_time
    """

    variable_esperada = f"X{id_archivo}_DE_time"

    if variable_esperada in mat_dict:
        return variable_esperada

    claves_de = [k for k in mat_dict.keys() if k.endswith("_DE_time")]

    if len(claves_de) == 0:
        raise ValueError("No se encontró ninguna variable DE_time.")

    raise ValueError(
        f"No se encontró {variable_esperada}. Variables DE disponibles: {claves_de}"
    )


# ============================
# 4. Extraer señales DE
# ============================

RAW_DIR = Path(DATA)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"RAW_DIR no existe: {RAW_DIR}")

mat_files = sorted([f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.mat'])
print(f"RAW_DIR usado: {RAW_DIR}")
print(f"Archivos .mat encontrados: {len(mat_files)}")
for f in mat_files:
    print(' -', f.name)

senales_de = {}

for ruta in mat_files:
    archivo = ruta.name

    try:
        info = extraer_info_nombre(archivo)

        mat = loadmat(ruta)

        variable_de = obtener_variable_de(mat, info["id_archivo"])

        x_de = mat[variable_de].flatten()

        clave = os.path.splitext(archivo)[0]

        senales_de[clave] = {
            "signal": x_de,
            "variable_de": variable_de,
            "clase": info["clase"],
            "tipo_falla": info["tipo_falla"],
            "severidad": info["severidad"],
            "carga": info["carga"],
            "id_archivo": info["id_archivo"],
            "n_muestras": len(x_de),
        }

        print(f"Archivo procesado: {archivo}")
        print(f"  Variable DE: {variable_de}")
        print(f"  Clase: {info['clase']}")
        print(f"  Severidad: {info['severidad']}")
        print(f"  Carga: {info['carga']}")
        print(f"  Muestras: {len(x_de)}")
        print("-" * 50)

    except Exception as e:
        print(f"No se pudo procesar {archivo}")
        print("Error:", e)
        print("-" * 50)


print("\nTotal de señales DE extraídas:", len(senales_de))

# %% Cell 11
os.makedirs(SIGNALS, exist_ok=True)

metadata = []

for nombre, datos in senales_de.items():

    señal = datos["signal"]

    np.save(
        os.path.join(SIGNALS, f"{nombre}.npy"),
        señal
    )

    metadata.append({
        "archivo": nombre,
        "clase": datos["clase"],
        "severidad": datos["severidad"],
        "carga": datos["carga"],
        "muestras": datos["n_muestras"]
    })

metadata = pd.DataFrame(metadata)

metadata.to_csv(
    os.path.join(SIGNALS, "metadata.csv"),
    index=False
)

print("Señales guardadas correctamente.")

# %% [markdown] Cell 12
# # 3 Primeras 10,000 muestras de la señal normal del motor

# %% Cell 13
nombre = "Time_Normal_1_098.npy"

# Corrected: Load the signal directly from the .npy file
x = np.load(SIGNALS / nombre)

plt.figure(figsize=(12, 4))
plt.plot(x[:5000])
plt.title(f"Señal DE - {nombre}")
plt.xlabel("Muestra")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# %% Cell 14
x = np.load(SIGNALS / 'Time_Normal_1_098.npy')

print(f"Número de registros:", x.shape)
print(f"Mínimo:", x.min())
print(f"Máximo", x.max())
print(f"Promedio", x.mean())
print(f"Desviación estandar", x.std())

# %% Cell 15
archivos = [
    "Time_Normal_1_098.npy",
    "B007_1_123.npy",
    "IR007_1_110.npy",
    "OR007_6_1_136.npy"
]

for archivo in archivos:

    # Corrected: Load the numpy array from the file path
    x = np.load(SIGNALS / archivo)

    plt.figure(figsize=(14,4))
    plt.plot(x[:10000], linewidth=0.8)

    plt.title(f"{archivo} - Primeras 10000 muestras")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Amplitud")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% Cell 16
for archivo in archivos:

    x = np.load(SIGNALS / archivo)

    print("\n", archivo)
    print("Muestras:", len(x))
    print("Min:", np.min(x))
    print("Max:", np.max(x))
    print("Media:", np.mean(x))
    print("Std:", np.std(x))

# %% [markdown] Cell 17
# # 4 Analisis exploratorio fisico de las señales
#
# Antes de dividir las señales en ventanas conviene estudiar su comportamiento físico global. Esta etapa permite observar si las fallas modifican la amplitud, la impulsividad, el contenido frecuencial, la memoria temporal y la geometría dinámica de la vibración.
#
# Este análisis no reemplaza el ventaneo ni la clasificación. Su función es justificar la motivación del enfoque TDA: si las señales defectuosas cambian su dinámica temporal y espectral, entonces tiene sentido estudiar también la geometría de sus atractores reconstruidos.

# %% [markdown] Cell 18
# El sistema estudiado corresponde a un rodamiento de elementos rodantes compuesto por una pista interna unida al eje rotatorio, una pista externa fija a la carcasa y un conjunto de bolas que transmiten la carga entre ambas superficies. Bajo condiciones normales, el contacto entre los elementos produce vibraciones de baja amplitud. La aparición de defectos localizados en las bolas, la pista interna o la pista externa genera impactos mecánicos periódicos que excitan las resonancias naturales del sistema. Estas vibraciones son registradas por un acelerómetro instalado sobre la carcasa, permitiendo inferir el estado del rodamiento mediante el análisis temporal y frecuencial de las señales adquiridas.

# %% [markdown] Cell 19
# ## 4.1 Carga ordenada de señales
#
# Se cargan las diez señales DE previamente extraidas. El orden se fija manualmente para comparar de forma consistente la condición normal contra fallas en bola, pista interna y pista externa.

# %% Cell 20
# Orden de analisis de las señales originales
ARCHIVOS_ANALISIS = [
    'Time_Normal_1_098',
    'B007_1_123', 'B014_1_190', 'B021_1_227',
    'IR007_1_110', 'IR014_1_175', 'IR021_1_214',
    'OR007_6_1_136', 'OR014_6_1_202', 'OR021_6_1_239'
]

metadata_senales = pd.read_csv(
    SIGNALS / 'metadata.csv',
    dtype={'archivo': str, 'clase': str, 'severidad': str, 'carga': str}
)
metadata_senales = metadata_senales.set_index('archivo').loc[ARCHIVOS_ANALISIS].reset_index()

senales_originales = {}
for archivo in ARCHIVOS_ANALISIS:
    ruta = SIGNALS / f'{archivo}.npy'
    if not ruta.exists():
        raise FileNotFoundError(f'No se encontro la senal extraida: {ruta}')
    senales_originales[archivo] = np.load(ruta)

display(metadata_senales)
print('Senales cargadas:', len(senales_originales))

# %% [markdown] Cell 21
# ## 4.2 Inspeccion temporal: x(t)
#
# Se grafican las primeras `10000` muestras de cada señal. Esta visualización permite revisar amplitudes, impactos, picos aislados y posibles patrones periodicos. Una falla localizada suele introducir impulsos repetitivos o eventos de amplitud más extrema que una condición sana.

# %% Cell 22
# Graficar primeras 10000 muestras de todas las señales
N_MUESTRAS_VISUAL = 10000

fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
axes = axes.ravel()

for ax, archivo in zip(axes, ARCHIVOS_ANALISIS):
    x = senales_originales[archivo]
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]

    ax.plot(x[:N_MUESTRAS_VISUAL], linewidth=0.7)
    ax.set_title(f"{archivo} | {info['clase']} | severidad {info['severidad']}")
    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.3)

for ax in axes[-2:]:
    ax.set_xlabel('Muestra')

plt.suptitle('Primeras 10000 muestras de cada señal DE', fontsize=16, y=1.01)
plt.tight_layout()
ruta_xt = OUTPUT_FIGURE / 'exploratorio_xt_10000_muestras.png'
plt.savefig(ruta_xt, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_xt)

# %% [markdown] Cell 23
# ## 4.3 Estadísticas clásicos
#
# Los estadísticas clásicos resumen propiedades fisicas de la vibración. El `RMS` mide energia vibratoria; `peak-to-peak` mide rango dinámico; `crest factor` relaciona picos con energia; `skewness` mide asimetría; y `kurtosis` es especialmente útil para detectar impulsividad. En diagnostico de fallas, una kurtosis elevada suele asociarse con impactos o eventos extremos.

# %% Cell 24
def calcular_estadisticos_clasicos(signal):
    """Calcula estadisticos clasicos para una senal de vibracion."""
    x = np.asarray(signal).ravel()
    mu = np.mean(x)
    sigma = np.std(x)
    rms = np.sqrt(np.mean(x ** 2))
    centered = x - mu

    if sigma == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = np.mean(centered ** 3) / (sigma ** 3)
        kurtosis = np.mean(centered ** 4) / (sigma ** 4)

    peak_to_peak = np.max(x) - np.min(x)
    crest_factor = np.max(np.abs(x)) / rms if rms != 0 else np.nan

    return {
        'media': mu,
        'std': sigma,
        'rms': rms,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'peak_to_peak': peak_to_peak,
        'crest_factor': crest_factor
    }

registros_stats = []
for archivo in ARCHIVOS_ANALISIS:
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0].to_dict()
    stats = calcular_estadisticos_clasicos(senales_originales[archivo])
    registros_stats.append({
        'archivo': archivo,
        'clase': info['clase'],
        'severidad': info['severidad'],
        'carga': info['carga'],
        **stats
    })

estadisticos_clasicos = pd.DataFrame(registros_stats)
ruta_stats = BASE / 'estadisticos_clasicos_senales.csv'
estadisticos_clasicos.to_csv(ruta_stats, index=False)

display(estadisticos_clasicos)
print('Estadisticos guardados en:', ruta_stats)

# %% Cell 25
# Comparacion visual de estadisticos sensibles a fallas
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
metricas = ['rms', 'kurtosis', 'crest_factor']

for ax, metrica in zip(axes, metricas):
    ax.bar(estadisticos_clasicos['archivo'], estadisticos_clasicos[metrica])
    ax.set_title(metrica)
    ax.tick_params(axis='x', rotation=75)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
ruta_stats_fig = OUTPUT_FIGURE / 'exploratorio_estadisticos_clasicos.png'
plt.savefig(ruta_stats_fig, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_stats_fig)

# %% [markdown] Cell 26
# ## 4.4 Análisis frecuencial mediante FFT
#
# La transformada rápida de Fourier permite observar como se distribuye la energía de la vibración en frecuencia. En rodamientos defectuosos pueden aparecer armónicos, bandas laterales o componentes adicionales asociadas a impactos y modulación. Para este dataset se asume una frecuencia de muestreo de `48000 Hz`, consistente con las señales CWRU de 48 kHz.

# %% Cell 27
# Parametros de frecuencia
FS = 48000  # Hz, señales CWRU de 48 kHz
FREQ_MAX_PLOT = 10000  # Hz

def calcular_fft(signal, fs=48000):
    """Calcula espectro de magnitud usando ventana Hann y eliminando la media."""
    x = np.asarray(signal).ravel()
    x = x - np.mean(x)
    ventana = np.hanning(len(x))
    xw = x * ventana
    frecuencias = np.fft.rfftfreq(len(xw), d=1 / fs)
    magnitud = np.abs(np.fft.rfft(xw)) / len(xw)
    return frecuencias, magnitud

fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
axes = axes.ravel()

for ax, archivo in zip(axes, ARCHIVOS_ANALISIS):
    f, mag = calcular_fft(senales_originales[archivo], fs=FS)
    mask = f <= FREQ_MAX_PLOT
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]

    ax.plot(f[mask], mag[mask], linewidth=0.7)
    ax.set_title(f"{archivo} | {info['clase']}")
    ax.set_ylabel('|X(f)|')
    ax.grid(True, alpha=0.3)

for ax in axes[-2:]:
    ax.set_xlabel('Frecuencia [Hz]')

plt.suptitle('FFT de señales DE originales', fontsize=16, y=1.01)
plt.tight_layout()
ruta_fft = OUTPUT_FIGURE / 'exploratorio_fft_señales.png'
plt.savefig(ruta_fft, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_fft)

# %% [markdown] Cell 28
# ## 4.5 Autocorrelación y selección preliminar de tau
#
# La autocorrelación `ACF(tau)` mide que tanto se parece la señal a una versión retrasada de si misma. En este proyecto se usa con dos objetivos: interpretar memoria temporal o periodicidad, y proponer un retardo `tau` para la reconstrucción de Takens. Un criterio común es elegir el primer mínimo local de la ACF.

# %% Cell 29
MAX_LAG_ACF = 1000
N_MUESTRAS_ACF = 100000  # usar un fragmento largo acelera el calculo y conserva la estructura dinamica

def autocorrelacion_normalizada(signal, max_lag=1000):
    """Calcula ACF normalizada para retardos entre 0 y max_lag usando FFT."""
    x = np.asarray(signal).ravel()
    x = x - np.mean(x)

    if np.allclose(x, 0):
        return np.ones(max_lag + 1)

    # Autocorrelacion eficiente: transformada de Fourier del producto espectral.
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    espectro = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(espectro * np.conjugate(espectro), n=nfft)[:max_lag + 1]
    acf = acf / acf[0]

    return acf


def primer_minimo_local(acf):
    """Devuelve el primer lag que cumple condicion de minimo local."""
    for lag in range(1, len(acf) - 1):
        if acf[lag] < acf[lag - 1] and acf[lag] < acf[lag + 1]:
            return lag
    return int(np.argmin(acf[1:]) + 1)

registros_tau = []
acf_por_archivo = {}

for archivo in ARCHIVOS_ANALISIS:
    x_acf = senales_originales[archivo][:N_MUESTRAS_ACF]
    acf = autocorrelacion_normalizada(x_acf, max_lag=MAX_LAG_ACF)
    tau = primer_minimo_local(acf)
    acf_por_archivo[archivo] = acf
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0].to_dict()
    registros_tau.append({
        'archivo': archivo,
        'clase': info['clase'],
        'severidad': info['severidad'],
        'tau_primer_minimo_local': tau,
        'acf_tau': acf[tau]
    })

taus_acf = pd.DataFrame(registros_tau)
ruta_tau = BASE / 'taus_acf_señales.csv'
taus_acf.to_csv(ruta_tau, index=False)

display(taus_acf)
print('Taus preliminares guardados en:', ruta_tau)

# %% Cell 30
# Graficar autocorrelacion de todas las señales
fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
axes = axes.ravel()
lags = np.arange(MAX_LAG_ACF + 1)

for ax, archivo in zip(axes, ARCHIVOS_ANALISIS):
    acf = acf_por_archivo[archivo]
    tau = int(taus_acf.loc[taus_acf['archivo'] == archivo, 'tau_primer_minimo_local'].iloc[0])
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]

    ax.plot(lags, acf, linewidth=0.8)
    ax.axvline(tau, color='red', linestyle='--', linewidth=1, label=f'tau={tau}')
    ax.set_title(f"{archivo} | {info['clase']}")
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

for ax in axes[-2:]:
    ax.set_xlabel('Retardo tau [muestras]')

plt.suptitle('Autocorrelación y tau preliminar por señal', fontsize=16, y=1.01)
plt.tight_layout()
ruta_acf = OUTPUT_FIGURE / 'exploratorio_autocorrelacion_tau.png'
plt.savefig(ruta_acf, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_acf)

# %% [markdown] Cell 31
# ## 4.6 Espacio de fases 2D
#
# Antes de construir el embedding de Takens en 3D, se visualiza una proyección simple `(x_t, x_{t+tau})`. Si la falla cambia la dinámica, pueden aparecer nubes mas dispersas, deformadas o con estructuras distintas. Esta visualización conecta directamente la señal temporal con la idea de geometría del atractor.

# %% Cell 32
ARCHIVOS_FASE_2D = ['Time_Normal_1_098', 'B007_1_123', 'IR007_1_110', 'OR007_6_1_136']
N_PUNTOS_FASE = 8000

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, archivo in zip(axes, ARCHIVOS_FASE_2D):
    x = senales_originales[archivo]
    tau = int(taus_acf.loc[taus_acf['archivo'] == archivo, 'tau_primer_minimo_local'].iloc[0])
    n = min(N_PUNTOS_FASE, len(x) - tau)
    info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]

    ax.scatter(x[:n], x[tau:tau + n], s=2, alpha=0.35)
    ax.set_title(f"{archivo} | {info['clase']} | tau={tau}")
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('$x_{t+tau}$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Espacio de fases 2D para señales representativas', fontsize=16, y=1.01)
plt.tight_layout()
ruta_fase = OUTPUT_FIGURE / 'exploratorio_espacio_fases_2d.png'
plt.savefig(ruta_fase, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_fase)

# %% [markdown] Cell 33
# ## 4.7 Comparación por severidad
#
# Se análiza si la dinámica de la vibración cambia de forma gradual conforme aumenta la severidad del defecto. Para ello se comparan las tres severidades disponibles en cada familia de falla:
#
# - `Ball`: `B007`, `B014`, `B021`
# - `Inner_Race`: `IR007`, `IR014`, `IR021`
# - `Outer_Race`: `OR007`, `OR014`, `OR021`
#
# La señal normal se conserva como referencia física, pero no entra en la progresión de severidad porque no tiene defecto. Las preguntas principales son: si aumenta el `RMS`, si aumenta la `kurtosis`, si aparecen más armónicos en la FFT y si el atractor 2D se vuelve mas disperso o complejo.

# %% Cell 34
# Agrupacion de archivos por familia de falla y severidad
SEVERIDAD_GRUPOS = {
    'Ball': ['B007_1_123', 'B014_1_190', 'B021_1_227'],
    'Inner_Race': ['IR007_1_110', 'IR014_1_175', 'IR021_1_214'],
    'Outer_Race': ['OR007_6_1_136', 'OR014_6_1_202', 'OR021_6_1_239'],
}

orden_severidad = {'007': 0.007, '014': 0.014, '021': 0.021}

comparacion_severidad = estadisticos_clasicos[
    estadisticos_clasicos['archivo'].isin(sum(SEVERIDAD_GRUPOS.values(), []))
].copy()
comparacion_severidad['severidad_in'] = comparacion_severidad['severidad'].map(orden_severidad)
comparacion_severidad = comparacion_severidad.sort_values(['clase', 'severidad_in'])

ruta_comparacion_severidad = BASE / 'comparacion_severidad_estadisticos.csv'
comparacion_severidad.to_csv(ruta_comparacion_severidad, index=False)

display(comparacion_severidad[['archivo', 'clase', 'severidad', 'severidad_in', 'rms', 'kurtosis', 'peak_to_peak', 'crest_factor']])
print('Comparacion por severidad guardada en:', ruta_comparacion_severidad)

# %% Cell 35
# Tendencia de RMS y kurtosis conforme aumenta la severidad
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
metricas_severidad = ['rms', 'kurtosis']

for ax, metrica in zip(axes, metricas_severidad):
    for clase, grupo in comparacion_severidad.groupby('clase'):
        grupo = grupo.sort_values('severidad_in')
        ax.plot(
            grupo['severidad_in'],
            grupo[metrica],
            marker='o',
            linewidth=2,
            label=clase
        )

    ax.set_title(f'{metrica} vs severidad')
    ax.set_xlabel('Severidad del defecto [in]')
    ax.set_ylabel(metrica)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
ruta_severidad_stats = OUTPUT_FIGURE / 'severidad_rms_kurtosis.png'
plt.savefig(ruta_severidad_stats, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_severidad_stats)

# %% [markdown] Cell 36
# ### FFT por severidad
#
# La siguiente comparación superpone los espectros de las tres severidades dentro de cada familia de falla. Si el defecto modifica progresivamente la dinámica, se esperaria observar cambios en la energía espectral, aparición de componentes adicionales, arm+onicos o mayor riqueza frecuencial.

# %% Cell 37
# FFT comparativa por severidad y familia de falla
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

for ax, (familia, archivos_familia) in zip(axes, SEVERIDAD_GRUPOS.items()):
    for archivo in archivos_familia:
        info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]
        f, mag = calcular_fft(senales_originales[archivo], fs=FS)
        mask = f <= FREQ_MAX_PLOT
        ax.plot(f[mask], mag[mask], linewidth=0.8, label=f"{info['severidad']} in")

    ax.set_title(f'FFT por severidad - {familia}')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel('|X(f)|')
plt.tight_layout()
ruta_severidad_fft = OUTPUT_FIGURE / 'severidad_fft_por_familia.png'
plt.savefig(ruta_severidad_fft, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_severidad_fft)

# %% [markdown] Cell 38
# ### Atractor 2D por severidad
#
# Finalmente se compara el espacio de fases 2D por severidad. La pregunta física es si el atractor se vuelve más disperso, deformado o complejo conforme el defecto crece. Esto no es todavia TDA formal, pero anticipa la idea de que la geometría de la dinámica contiene información diagnóstica.

# %% Cell 39
# Espacio de fases 2D por severidad
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
N_PUNTOS_SEVERIDAD_FASE = 6000

for fila, (familia, archivos_familia) in enumerate(SEVERIDAD_GRUPOS.items()):
    for col, archivo in enumerate(archivos_familia):
        ax = axes[fila, col]
        x = senales_originales[archivo]
        info = metadata_senales[metadata_senales['archivo'] == archivo].iloc[0]
        tau = int(taus_acf.loc[taus_acf['archivo'] == archivo, 'tau_primer_minimo_local'].iloc[0])
        n = min(N_PUNTOS_SEVERIDAD_FASE, len(x) - tau)

        ax.scatter(x[:n], x[tau:tau + n], s=2, alpha=0.3)
        ax.set_title(f"{familia} | {info['severidad']} in | tau={tau}")
        ax.set_xlabel('$x_t$')
        ax.set_ylabel('$x_{t+tau}$')
        ax.grid(True, alpha=0.3)

plt.suptitle('Espacio de fases 2D por severidad', fontsize=16, y=1.01)
plt.tight_layout()
ruta_severidad_fase = OUTPUT_FIGURE / 'severidad_espacio_fases_2d.png'
plt.savefig(ruta_severidad_fase, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_severidad_fase)

# %% [markdown] Cell 40
# ### Lectura fisica de la severidad
#
# Si `RMS` aumenta con la severidad, se interpreta como mayor energía vibratoria conforme crece el defecto. Si `kurtosis` aumenta, sugiere mayor impulsividad o eventos extremos. Si la FFT muestra más componentes, armónicos o energía distribuida, entonces la falla esta enriqueciendo el contenido frecuencial. Si el atractor 2D se vuelve más disperso o irregular, se refuerza la hipótesis de que la geometría de la dinámica cambia con el daño.
#
# No es obligatorio que todas las metrícas crezcan de forma monótona. Una respuesta no monótona también es físicamente relevante, porque indica que el tipo de falla, su ubicación, la carga y la modulación mecánica influyen en la señal. En ese caso, TDA puede aportar información adicional al capturar cambios geométricos que no se resumen completamente con un solo estadístico.

# %% [markdown] Cell 41
# # 5 Ventaneo de las señales DE
#
# En esta etapa cada señal completa se divide en segmentos más pequeños llamados ventanas. Esto permite convertir una medición larga en muchas muestras de análisis, todas con la misma etiqueta de la señal original.
#
# Se usara un tamano de ventana de `2048` muestras y un traslape de `75%`. Con esos parámetros, el desplazamiento entre ventanas es de `512` muestras. Este formato es adecuado para las siguientes etapas del proyecto: seleccion del retardo temporal, reconstrucción de Takens y extraccion de características topológicas.

# %% Cell 42
# Parametros generales del ventaneo
WINDOW_SIZE = 2048
OVERLAP = 0.75
STEP = int(WINDOW_SIZE * (1 - OVERLAP))

# Carpeta donde se guardaran las ventanas generadas
VENTANAS_DIR = BASE / 'ventanas_DE'
VENTANAS_DIR.mkdir(parents=True, exist_ok=True)

print('Tamano de ventana:', WINDOW_SIZE)
print('Traslape:', OVERLAP)
print('Paso entre ventanas:', STEP)
print('Carpeta de ventanas:', VENTANAS_DIR)

# %% Cell 43
def crear_ventanas(signal, window_size=2048, overlap=0.75):
    """
    Divide una senal 1D en ventanas temporales con traslape.

    Parametros
    ----------
    signal : array-like
        Senal temporal de entrada. Debe ser un arreglo unidimensional.
    window_size : int
        Numero de muestras por ventana.
    overlap : float
        Fraccion de traslape entre ventanas consecutivas. Debe cumplir 0 <= overlap < 1.

    Retorna
    -------
    ventanas : np.ndarray
        Matriz de forma (n_ventanas, window_size).
    indices : list[tuple[int, int]]
        Lista con pares (inicio, fin) para ubicar cada ventana dentro de la senal original.
    """

    signal = np.asarray(signal).ravel()

    if window_size <= 0:
        raise ValueError('window_size debe ser mayor que 0.')

    if not 0 <= overlap < 1:
        raise ValueError('overlap debe estar en el intervalo [0, 1).')

    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError('El paso entre ventanas debe ser mayor que 0. Reduce el overlap.')

    if len(signal) < window_size:
        return np.empty((0, window_size)), []

    ventanas = []
    indices = []

    # Se conservan solo ventanas completas para mantener todas las muestras con igual longitud.
    for inicio in range(0, len(signal) - window_size + 1, step):
        fin = inicio + window_size
        ventanas.append(signal[inicio:fin])
        indices.append((inicio, fin))

    return np.asarray(ventanas), indices

# %% Cell 44
# Aplicar el ventaneo a todas las senales .npy guardadas en DE_signals
metadata_senales_path = SIGNALS / 'metadata.csv'

if not metadata_senales_path.exists():
    raise FileNotFoundError(f'No se encontro el archivo de metadata de senales: {metadata_senales_path}')

metadata_senales = pd.read_csv(metadata_senales_path, dtype={'archivo': str, 'clase': str, 'severidad': str, 'carga': str})
registros_ventanas = []

for _, fila in metadata_senales.sort_values(['clase', 'archivo']).iterrows():
    archivo_origen = fila['archivo']
    clase = fila['clase']
    severidad = fila['severidad']
    carga = fila['carga']

    ruta_senal = SIGNALS / f'{archivo_origen}.npy'
    if not ruta_senal.exists():
        print(f'Aviso: no se encontro {ruta_senal}. Se omite esta senal.')
        continue

    signal = np.load(ruta_senal)
    ventanas, indices = crear_ventanas(signal, window_size=WINDOW_SIZE, overlap=OVERLAP)

    # Guardamos cada clase en una subcarpeta para facilitar inspeccion manual posterior.
    clase_dir = VENTANAS_DIR / clase
    clase_dir.mkdir(parents=True, exist_ok=True)

    for i, (ventana, (inicio, fin)) in enumerate(zip(ventanas, indices)):
        ventana_id = f'{archivo_origen}_w{i:04d}'
        ruta_ventana = clase_dir / f'{ventana_id}.npy'
        np.save(ruta_ventana, ventana)

        registros_ventanas.append({
            'ventana_id': ventana_id,
            'archivo_origen': archivo_origen,
            'ruta_ventana': str(ruta_ventana),
            'clase': clase,
            'severidad': severidad,
            'carga': carga,
            'indice_inicio': inicio,
            'indice_fin': fin,
            'window_size': WINDOW_SIZE,
            'overlap': OVERLAP,
            'step': STEP,
            'muestras_senal_original': len(signal)
        })

    print(f'{archivo_origen}: {len(ventanas)} ventanas generadas')

metadata_ventanas = pd.DataFrame(registros_ventanas)
metadata_ventanas_path = VENTANAS_DIR / 'metadata_ventanas.csv'
metadata_ventanas.to_csv(metadata_ventanas_path, index=False)

print('\nTotal de ventanas generadas:', len(metadata_ventanas))
print('Metadata de ventanas guardada en:', metadata_ventanas_path)
display(metadata_ventanas.head())

# %% Cell 45
# Resumen del dataset de ventanas
resumen_ventanas = (
    metadata_ventanas
    .groupby(['clase', 'archivo_origen'])
    .size()
    .reset_index(name='n_ventanas')
)

print('Ventanas por clase:')
display(metadata_ventanas['clase'].value_counts().rename_axis('clase').reset_index(name='n_ventanas'))

print('Ventanas por archivo de origen:')
display(resumen_ventanas)

# %% [markdown] Cell 46
# ## Visualizacion de ventanas por clase
#
# La siguiente grafica permite inspeccionar una ventana representativa de cada clase. Esta revision visual no sustituye el analisis cuantitativo, pero ayuda a verificar que el ventaneo conserva la estructura temporal de las senales y que las clases se estan leyendo correctamente.

# %% Cell 47
# Visualizar una ventana de ejemplo por clase
clases = sorted(metadata_ventanas['clase'].unique())
fig, axes = plt.subplots(len(clases), 1, figsize=(14, 3 * len(clases)), sharex=True)

if len(clases) == 1:
    axes = [axes]

for ax, clase in zip(axes, clases):
    ejemplo = metadata_ventanas[metadata_ventanas['clase'] == clase].iloc[0]
    ventana = np.load(ejemplo['ruta_ventana'])

    ax.plot(ventana, linewidth=0.8)
    ax.set_title(f"Clase: {clase} | Archivo: {ejemplo['archivo_origen']} | Ventana: {ejemplo['ventana_id']}")
    ax.set_ylabel('Amplitud')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Indice dentro de la ventana')
plt.tight_layout()

ruta_figura_ventanas_png = OUTPUT_FIGURE / 'ventanas_por_clase.png'
ruta_figura_ventanas_svg = OUTPUT_FIGURE / 'ventanas_por_clase.svg'
plt.savefig(ruta_figura_ventanas_png, dpi=150, bbox_inches='tight')
plt.savefig(ruta_figura_ventanas_svg, bbox_inches='tight')
plt.show()

print('Figura PNG guardada en:', ruta_figura_ventanas_png)
print('Figura SVG guardada en:', ruta_figura_ventanas_svg)

# %% [markdown] Cell 48
# # 6 Retardo temporal tau en ventanas y reconstruccion de Takens
#
# Una vez construido el dataset de ventanas, el siguiente paso es estimar el retardo temporal `tau` que se usara en la reconstruccion del espacio de fases. Este parametro controla que tan separadas quedan las coordenadas retrasadas de Takens.
#
# Si `tau` es demasiado pequeno, las coordenadas quedan casi repetidas y la nube de puntos se aplasta cerca de la diagonal. Si `tau` es demasiado grande, se puede perder relacion dinamica entre las coordenadas. Por eso se estima `tau` mediante autocorrelacion y se usa el primer minimo local como criterio inicial.

# %% [markdown] Cell 49
# ## 6.1 Autocorrelacion de todas las ventanas
#
# Para cada ventana se calcula la autocorrelacion normalizada y se obtiene un `tau` candidato. Despues se resumen los resultados por clase, archivo y severidad. Esta tabla permite decidir si conviene usar un `tau` global o un `tau` adaptativo.

# %% Cell 50
# Parametros para estimar tau en ventanas
MAX_LAG_VENTANA = 200
TAU_MIN = 1

metadata_ventanas_path = VENTANAS_DIR / 'metadata_ventanas.csv'
if not metadata_ventanas_path.exists():
    raise FileNotFoundError(f'No se encontro metadata de ventanas: {metadata_ventanas_path}')

metadata_ventanas = pd.read_csv(
    metadata_ventanas_path,
    dtype={'ventana_id': str, 'archivo_origen': str, 'ruta_ventana': str, 'clase': str, 'severidad': str, 'carga': str}
)

print('Ventanas a analizar:', len(metadata_ventanas))
display(metadata_ventanas.head())

# %% Cell 51
def resolver_ruta_ventana(fila):
    """Resuelve la ruta de una ventana tanto en local como en Colab."""
    ruta_guardada = Path(str(fila['ruta_ventana']))
    if ruta_guardada.exists():
        return ruta_guardada

    ruta_reconstruida = VENTANAS_DIR / fila['clase'] / f"{fila['ventana_id']}.npy"
    if ruta_reconstruida.exists():
        return ruta_reconstruida

    raise FileNotFoundError(f"No se encontro la ventana {fila['ventana_id']}")


def autocorrelacion_ventana(signal, max_lag=200):
    """Calcula ACF normalizada de una ventana 1D hasta max_lag."""
    x = np.asarray(signal).ravel()
    x = x - np.mean(x)

    if np.allclose(x, 0):
        return np.ones(max_lag + 1)

    max_lag = min(max_lag, len(x) - 2)
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    espectro = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(espectro * np.conjugate(espectro), n=nfft)[:max_lag + 1]
    acf = acf / acf[0]

    return acf


def tau_primer_minimo_local(acf, tau_min=1):
    """Obtiene el primer minimo local de la ACF."""
    for tau in range(max(1, tau_min), len(acf) - 1):
        if acf[tau] < acf[tau - 1] and acf[tau] < acf[tau + 1]:
            return tau

    return int(np.argmin(acf[max(1, tau_min):]) + max(1, tau_min))

# %% Cell 52
# Calcular tau para todas las ventanas
registros_tau_ventanas = []

for i, fila in metadata_ventanas.iterrows():
    ruta_ventana = resolver_ruta_ventana(fila)
    ventana = np.load(ruta_ventana)
    acf = autocorrelacion_ventana(ventana, max_lag=MAX_LAG_VENTANA)
    tau = tau_primer_minimo_local(acf, tau_min=TAU_MIN)

    registros_tau_ventanas.append({
        'ventana_id': fila['ventana_id'],
        'archivo_origen': fila['archivo_origen'],
        'clase': fila['clase'],
        'severidad': fila['severidad'],
        'carga': fila['carga'],
        'tau': tau,
        'acf_tau': acf[tau],
        'max_lag': MAX_LAG_VENTANA
    })

    if (i + 1) % 1000 == 0:
        print(f'Ventanas procesadas: {i + 1}/{len(metadata_ventanas)}')

tau_ventanas = pd.DataFrame(registros_tau_ventanas)
metadata_ventanas_tau = metadata_ventanas.merge(tau_ventanas[['ventana_id', 'tau', 'acf_tau', 'max_lag']], on='ventana_id', how='left')

ruta_tau_ventanas = VENTANAS_DIR / 'metadata_ventanas_tau.csv'
metadata_ventanas_tau.to_csv(ruta_tau_ventanas, index=False)

print('Metadata con tau guardada en:', ruta_tau_ventanas)
display(metadata_ventanas_tau.head())

# %% Cell 53
# Resumen de tau por clase, archivo y severidad
resumen_tau_clase = (
    metadata_ventanas_tau
    .groupby('clase')['tau']
    .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    .reset_index()
)

resumen_tau_archivo = (
    metadata_ventanas_tau
    .groupby(['clase', 'archivo_origen', 'severidad'])['tau']
    .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    .reset_index()
)

TAU_GLOBAL = int(round(metadata_ventanas_tau['tau'].median()))
TAU_POR_CLASE = metadata_ventanas_tau.groupby('clase')['tau'].median().round().astype(int).to_dict()

ruta_resumen_tau_clase = VENTANAS_DIR / 'resumen_tau_por_clase.csv'
ruta_resumen_tau_archivo = VENTANAS_DIR / 'resumen_tau_por_archivo.csv'
resumen_tau_clase.to_csv(ruta_resumen_tau_clase, index=False)
resumen_tau_archivo.to_csv(ruta_resumen_tau_archivo, index=False)

print('Tau global recomendado por mediana:', TAU_GLOBAL)
print('Tau por clase:', TAU_POR_CLASE)
print('Resumen por clase guardado en:', ruta_resumen_tau_clase)
print('Resumen por archivo guardado en:', ruta_resumen_tau_archivo)
display(resumen_tau_clase)
display(resumen_tau_archivo)

# %% Cell 54
# Distribucion de tau por clase
plt.figure(figsize=(10, 5))
clases_tau = sorted(metadata_ventanas_tau['clase'].unique())
datos_tau = [metadata_ventanas_tau.loc[metadata_ventanas_tau['clase'] == clase, 'tau'] for clase in clases_tau]

plt.boxplot(datos_tau, labels=clases_tau, showmeans=True)
plt.axhline(TAU_GLOBAL, color='red', linestyle='--', linewidth=1.5, label=f'TAU_GLOBAL={TAU_GLOBAL}')
plt.title('Distribucion del retardo tau por clase')
plt.ylabel('tau [muestras]')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
ruta_tau_boxplot = OUTPUT_FIGURE / 'tau_ventanas_boxplot_por_clase.png'
plt.savefig(ruta_tau_boxplot, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_tau_boxplot)

# %% [markdown] Cell 55
# ## 6.2 Embedding 2D y reconstruccion de Takens con d=3
#
# Con `tau` estimado, se construyen coordenadas retrasadas. Primero se visualiza el embedding 2D `(x_t, x_{t+tau})` porque es facil de interpretar. Despues se realiza la reconstruccion de Takens con dimension `d=3`: `(x_t, x_{t+tau}, x_{t+2tau})`.
#
# En esta etapa se usan ventanas representativas por clase para verificar que la geometria reconstruida sea razonable antes de calcular homologia persistente sobre todo el dataset.

# %% Cell 56
def takens_embedding(signal, dimension=3, tau=1):
    """
    Reconstruye el espacio de fases mediante coordenadas retrasadas de Takens.

    Para dimension=3 retorna puntos de la forma:
    (x_t, x_{t+tau}, x_{t+2tau})
    """
    x = np.asarray(signal).ravel()

    if dimension < 2:
        raise ValueError('dimension debe ser al menos 2.')
    if tau < 1:
        raise ValueError('tau debe ser mayor o igual a 1.')

    n_puntos = len(x) - (dimension - 1) * tau
    if n_puntos <= 0:
        raise ValueError('La senal es demasiado corta para la dimension y tau indicados.')

    return np.column_stack([x[i * tau:i * tau + n_puntos] for i in range(dimension)])


def seleccionar_ventanas_representativas(metadata_tau):
    """Selecciona una ventana por clase, cercana a la mediana de tau de esa clase."""
    seleccion = []

    for clase, grupo in metadata_tau.groupby('clase'):
        tau_mediana = grupo['tau'].median()
        idx = (grupo['tau'] - tau_mediana).abs().idxmin()
        seleccion.append(metadata_tau.loc[idx])

    return pd.DataFrame(seleccion).sort_values('clase').reset_index(drop=True)


ventanas_representativas = seleccionar_ventanas_representativas(metadata_ventanas_tau)
display(ventanas_representativas[['ventana_id', 'archivo_origen', 'clase', 'severidad', 'tau']])

# %% Cell 57
# Visualizacion del embedding 2D para ventanas representativas
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, (_, fila) in zip(axes, ventanas_representativas.iterrows()):
    ventana = np.load(resolver_ruta_ventana(fila))
    tau = int(fila['tau'])
    emb2 = takens_embedding(ventana, dimension=2, tau=tau)

    ax.scatter(emb2[:, 0], emb2[:, 1], s=3, alpha=0.35)
    ax.set_title(f"{fila['clase']} | {fila['ventana_id']} | tau={tau}")
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('$x_{t+tau}$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Embedding 2D de ventanas representativas', fontsize=16, y=1.01)
plt.tight_layout()
ruta_embedding_2d = OUTPUT_FIGURE / 'embedding_2d_ventanas_representativas.png'
plt.savefig(ruta_embedding_2d, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_embedding_2d)

# %% Cell 58
# Reconstruccion de Takens con d=3 para ventanas representativas

fig = plt.figure(figsize=(14, 10))

for i, (_, fila) in enumerate(ventanas_representativas.iterrows(), start=1):
    ventana = np.load(resolver_ruta_ventana(fila))
    tau = int(fila['tau'])
    emb3 = takens_embedding(ventana, dimension=3, tau=tau)

    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], s=2, alpha=0.35)
    ax.set_title(f"{fila['clase']} | tau={tau} | d=3")
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('$x_{t+tau}$')
    ax.set_zlabel('$x_{t+2tau}$')

plt.suptitle('Reconstruccion de Takens d=3 en ventanas representativas', fontsize=16, y=0.98)
plt.tight_layout()
ruta_takens_3d = OUTPUT_FIGURE / 'takens_3d_ventanas_representativas.png'
plt.savefig(ruta_takens_3d, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_takens_3d)

# %% [markdown] Cell 59
# ## 6.3 Interpretacion de tau y Takens
#
# El resumen de `tau` permite decidir si se usara un valor global o valores adaptativos por clase o por ventana. Para una primera implementacion de TDA, el valor global por mediana es una opcion estable porque mantiene todas las reconstrucciones bajo el mismo criterio.
#
# El embedding 2D sirve como verificacion visual: si la condicion normal y las fallas producen nubes con dispersion o forma distinta, esto sugiere que la dinamica reconstruida contiene informacion diagnostica. La reconstruccion de Takens con `d=3` sera la entrada natural para calcular homologia persistente en la siguiente etapa.

# %% [markdown] Cell 60
# # 7 Homología persistente y features topológicas
#
# En esta etapa se cuantifica la geometría de las nubes de puntos obtenidas con la reconstrucción de Takens. Para cada ventana se construye una nube de puntos en dimensión `d=3` y se calcula homología persistente mediante un complejo Vietoris-Rips usando `ripser`.
#
# La estrategia se divide en dos pasos: primero se prueban ventanas representativas para verificar que los diagramas tengan estructura interpretable; después se escala el calculo a todo el dataset de ventanas para obtener un dataset tabular de features topológicas.

# %% [markdown] Cell 61
# ## 7.1 Configuración de homología persistente
#
# Para controlar el costo computacional, cada nube de Takens se submuestrea de forma uniforme a `100` puntos. Esto permite calcular `H0`, `H1` y `H2` para las 9257 ventanas en tiempo razonable. No se guardan todos los diagramas crudos, solo las features resumidas.

# %% Cell 62
FEATURES_TDA_DIR = BASE / 'features_TDA'
FEATURES_TDA_DIR.mkdir(parents=True, exist_ok=True)

TAKENS_DIMENSION = 3
MAXDIM_HOMOLOGIA = 2
MAX_POINTS_PH = 100

print('Dimension de Takens:', TAKENS_DIMENSION)
print('Homologia hasta H:', MAXDIM_HOMOLOGIA)
print('Puntos por nube para ripser:', MAX_POINTS_PH)
print('Salida features TDA:', FEATURES_TDA_DIR)

# %% Cell 63
def submuestrear_nube_puntos(point_cloud, max_points=100):
    """Submuestrea uniformemente una nube de puntos para reducir costo de Vietoris-Rips."""
    point_cloud = np.asarray(point_cloud)

    if len(point_cloud) <= max_points:
        return point_cloud

    indices = np.linspace(0, len(point_cloud) - 1, max_points).astype(int)
    return point_cloud[indices]


def features_persistencia(diagrama):
    """Resume un diagrama de persistencia en features numericas."""
    diagrama = np.asarray(diagrama)

    if diagrama.size == 0 or diagrama.ndim != 2 or diagrama.shape[1] != 2:
        return {
            'count': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'persistence_entropy': 0.0
        }

    finitos = diagrama[np.isfinite(diagrama[:, 1])]
    if len(finitos) == 0:
        return {
            'count': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'persistence_entropy': 0.0
        }

    persistencias = finitos[:, 1] - finitos[:, 0]
    persistencias = persistencias[persistencias > 0]

    if len(persistencias) == 0:
        return {
            'count': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'persistence_entropy': 0.0
        }

    total = float(np.sum(persistencias))
    probabilidades = persistencias / total

    return {
        'count': int(len(persistencias)),
        'total_persistence': total,
        'max_persistence': float(np.max(persistencias)),
        'mean_persistence': float(np.mean(persistencias)),
        'persistence_entropy': float(-np.sum(probabilidades * np.log(probabilidades + 1e-12)))
    }


def features_diagramas(diagramas, maxdim=2):
    """Extrae features para H0, H1 y H2."""
    registro = {}

    for dim in range(maxdim + 1):
        diagrama = diagramas[dim] if dim < len(diagramas) else np.empty((0, 2))
        feats = features_persistencia(diagrama)

        for nombre, valor in feats.items():
            registro[f'H{dim}_{nombre}'] = valor

    return registro


def calcular_persistencia_ventana(fila, dimension=3, max_points=100, maxdim=2):
    """Calcula Takens, ripser y features topologicas para una ventana."""
    ventana = np.load(resolver_ruta_ventana(fila))
    tau = int(fila['tau'])
    nube = takens_embedding(ventana, dimension=dimension, tau=tau)
    nube = submuestrear_nube_puntos(nube, max_points=max_points)
    diagramas = ripser(nube, maxdim=maxdim)['dgms']
    feats = features_diagramas(diagramas, maxdim=maxdim)

    return nube, diagramas, feats

# %% [markdown] Cell 64
# ## 7.2 Prueba con ventanas representativas
#
# Primero se calculan diagramas de persistencia para una ventana representativa por clase. Si los diagramas muestran persistencias no triviales y las features difieren entre clases, entonces es razonable escalar el calculo al dataset completo.

# %% Cell 65
features_representativas = []
diagramas_representativos = {}

for _, fila in ventanas_representativas.iterrows():
    nube, diagramas, feats = calcular_persistencia_ventana(
        fila,
        dimension=TAKENS_DIMENSION,
        max_points=MAX_POINTS_PH,
        maxdim=MAXDIM_HOMOLOGIA
    )

    diagramas_representativos[fila['clase']] = diagramas
    features_representativas.append({
        'ventana_id': fila['ventana_id'],
        'archivo_origen': fila['archivo_origen'],
        'clase': fila['clase'],
        'severidad': fila['severidad'],
        'tau': int(fila['tau']),
        **feats
    })

features_TDA_representativas = pd.DataFrame(features_representativas)
ruta_features_representativas = FEATURES_TDA_DIR / 'features_TDA_representativas.csv'
features_TDA_representativas.to_csv(ruta_features_representativas, index=False)

display(features_TDA_representativas)
print('Features representativas guardadas en:', ruta_features_representativas)

# %% Cell 66
# Diagramas de persistencia para ventanas representativas
fig, axes = plt.subplots(len(ventanas_representativas), MAXDIM_HOMOLOGIA + 1, figsize=(13, 3.2 * len(ventanas_representativas)))

if len(ventanas_representativas) == 1:
    axes = np.array([axes])

for fila_idx, (_, fila) in enumerate(ventanas_representativas.iterrows()):
    diagramas = diagramas_representativos[fila['clase']]

    for dim in range(MAXDIM_HOMOLOGIA + 1):
        ax = axes[fila_idx, dim]
        diagrama = diagramas[dim]
        finitos = diagrama[np.isfinite(diagrama[:, 1])] if len(diagrama) else diagrama

        if len(finitos):
            ax.scatter(finitos[:, 0], finitos[:, 1], s=18, alpha=0.75)
            minimo = min(finitos[:, 0].min(), finitos[:, 1].min())
            maximo = max(finitos[:, 0].max(), finitos[:, 1].max())
            margen = (maximo - minimo) * 0.08 if maximo > minimo else 1
            ax.plot([minimo - margen, maximo + margen], [minimo - margen, maximo + margen], 'k--', linewidth=1)
            ax.set_xlim(minimo - margen, maximo + margen)
            ax.set_ylim(minimo - margen, maximo + margen)

        ax.set_title(f"{fila['clase']} | H{dim}")
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
ruta_diagramas_representativos = OUTPUT_FIGURE / 'persistencia_diagramas_representativos.png'
plt.savefig(ruta_diagramas_representativos, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_diagramas_representativos)

# %% Cell 67
# Criterio simple para escalar: debe haber persistencia no nula en H1 o H2
persistencia_no_trivial = (
    features_TDA_representativas['H1_total_persistence'].sum() > 0
    or features_TDA_representativas['H2_total_persistence'].sum() > 0
)

diferencias_entre_clases = features_TDA_representativas[[
    'H0_total_persistence', 'H1_total_persistence', 'H2_total_persistence'
]].std().sum() > 0

ESCALAR_TDA_COMPLETO = bool(persistencia_no_trivial and diferencias_entre_clases)

print('Persistencia no trivial:', persistencia_no_trivial)
print('Diferencias entre clases:', diferencias_entre_clases)
print('Escalar a todo el dataset:', ESCALAR_TDA_COMPLETO)

# %% [markdown] Cell 68
# ## 7.3 Extracción completa de features topológicas
#
# Una vez validada la prueba representativa, se calcula el conjunto completo de features topológicas para todas las ventanas. Cada fila del dataset final corresponde a una ventana y contiene features de `H0`, `H1` y `H2`.

# %% Cell 69
if ESCALAR_TDA_COMPLETO:
    registros_features_tda = []

    for i, fila in metadata_ventanas_tau.iterrows():
        _, _, feats = calcular_persistencia_ventana(
            fila,
            dimension=TAKENS_DIMENSION,
            max_points=MAX_POINTS_PH,
            maxdim=MAXDIM_HOMOLOGIA
        )

        registros_features_tda.append({
            'ventana_id': fila['ventana_id'],
            'archivo_origen': fila['archivo_origen'],
            'clase': fila['clase'],
            'severidad': fila['severidad'],
            'carga': fila['carga'],
            'tau': int(fila['tau']),
            'embedding_dimension': TAKENS_DIMENSION,
            'max_points': MAX_POINTS_PH,
            'maxdim': MAXDIM_HOMOLOGIA,
            **feats
        })

        if (i + 1) % 500 == 0:
            print(f'Ventanas procesadas: {i + 1}/{len(metadata_ventanas_tau)}')

    features_topologicas = pd.DataFrame(registros_features_tda)
    ruta_features_topologicas = FEATURES_TDA_DIR / 'features_topologicas_ventanas.csv'
    features_topologicas.to_csv(ruta_features_topologicas, index=False)

    print('Features topologicas completas guardadas en:', ruta_features_topologicas)
    display(features_topologicas.head())
else:
    print('No se escala el calculo porque la prueba representativa no mostro persistencia util.')

# %% Cell 70
if ESCALAR_TDA_COMPLETO:
    resumen_features_TDA_clase = (
        features_topologicas
        .groupby('clase')
        .agg({
            'ventana_id': 'count',
            'H0_total_persistence': 'mean',
            'H1_total_persistence': 'mean',
            'H2_total_persistence': 'mean',
            'H0_persistence_entropy': 'mean',
            'H1_persistence_entropy': 'mean',
            'H2_persistence_entropy': 'mean'
        })
        .reset_index()
        .rename(columns={'ventana_id': 'n_ventanas'})
    )

    ruta_resumen_features_TDA = FEATURES_TDA_DIR / 'resumen_features_TDA_por_clase.csv'
    resumen_features_TDA_clase.to_csv(ruta_resumen_features_TDA, index=False)

    display(resumen_features_TDA_clase)
    print('Resumen de features TDA guardado en:', ruta_resumen_features_TDA)

# %% Cell 71
if ESCALAR_TDA_COMPLETO:
    metricas_tda = [
        'H0_total_persistence', 'H1_total_persistence', 'H2_total_persistence',
        'H0_persistence_entropy', 'H1_persistence_entropy', 'H2_persistence_entropy'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    clases = sorted(features_topologicas['clase'].unique())

    for ax, metrica in zip(axes, metricas_tda):
        datos = [features_topologicas.loc[features_topologicas['clase'] == clase, metrica] for clase in clases]
        ax.boxplot(datos, tick_labels=clases, showfliers=False, showmeans=True)
        ax.set_title(metrica)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    ruta_boxplots_tda = OUTPUT_FIGURE / 'features_TDA_boxplots_por_clase.png'
    plt.savefig(ruta_boxplots_tda, dpi=150, bbox_inches='tight')
    plt.show()

    print('Figura guardada en:', ruta_boxplots_tda)

# %% [markdown] Cell 72
# ## 7.4 Interpretación de las features TDA
#
# Las features de `H0` resumen conectividad de la nube de puntos; las de `H1` capturan ciclos o bucles; y las de `H2` capturan cavidades tridimensionales. Si las medias o distribuciones cambian entre clases, esto indica que la geometría reconstruída por Takens contiene información util para clasificación.
#

# %% [markdown] Cell 73
# # 8 Análisis de sensibilidad del submuestreo
#
# En la sección anterior se uso un submuestreo de `100` puntos por nube de Takens para calcular homología persistente en todo el dataset. Esta decision reduce el costo computacional, pero puede afectar las features topológicas.
#
# Por eso se realiza un análisis de sensibilidad usando ventanas representativas por clase y comparando `max_points = 100`, `200`, `300`, `400` y `500`. El objetivo es observar si las tendencias topológicas se mantienen al aumentar la resolución de la nube de puntos antes de decidir que configuración usar para todo el dataset.

# %% [markdown] Cell 74
# ## 8.1 Calculo de sensibilidad
#
# Se recalculan las features TDA de las ventanas representativas con tres niveles de submuestreo. `500` puntos se toma como referencia local para medir cuanto cambian las features al usar menos puntos.

# %% Cell 75
PUNTOS_SENSIBILIDAD = [100, 200, 300, 400, 500]

registros_sensibilidad = []

for _, fila in ventanas_representativas.iterrows():
    ventana = np.load(resolver_ruta_ventana(fila))
    tau = int(fila['tau'])
    nube_completa = takens_embedding(ventana, dimension=TAKENS_DIMENSION, tau=tau)

    for max_points in PUNTOS_SENSIBILIDAD:
        nube = submuestrear_nube_puntos(nube_completa, max_points=max_points)
        diagramas = ripser(nube, maxdim=MAXDIM_HOMOLOGIA)['dgms']
        feats = features_diagramas(diagramas, maxdim=MAXDIM_HOMOLOGIA)

        registros_sensibilidad.append({
            'ventana_id': fila['ventana_id'],
            'archivo_origen': fila['archivo_origen'],
            'clase': fila['clase'],
            'severidad': fila['severidad'],
            'tau': tau,
            'max_points': max_points,
            'n_puntos_embedding': len(nube),
            **feats
        })

sensibilidad_submuestreo = pd.DataFrame(registros_sensibilidad)
ruta_sensibilidad = FEATURES_TDA_DIR / 'sensibilidad_submuestreo_representativas.csv'
sensibilidad_submuestreo.to_csv(ruta_sensibilidad, index=False)

display(sensibilidad_submuestreo)
print('Analisis de sensibilidad guardado en:', ruta_sensibilidad)

# %% Cell 76
# Comparar cada configuracion contra 300 puntos
metricas_sensibilidad = [
    'H0_total_persistence', 'H1_total_persistence', 'H2_total_persistence',
    'H0_persistence_entropy', 'H1_persistence_entropy', 'H2_persistence_entropy'
]

registros_diferencias = []

for clase, grupo in sensibilidad_submuestreo.groupby('clase'):
    referencia = grupo[grupo['max_points'] == 500].iloc[0]

    for _, fila in grupo.iterrows():
        registro = {
            'clase': clase,
            'max_points': int(fila['max_points'])
        }

        for metrica in metricas_sensibilidad:
            denominador = abs(referencia[metrica])
            if denominador > 1e-12:
                diferencia = abs(fila[metrica] - referencia[metrica]) / denominador * 100
            else:
                diferencia = np.nan
            registro[f'{metrica}_diff_pct_vs_500'] = diferencia

        registros_diferencias.append(registro)

diferencias_sensibilidad = pd.DataFrame(registros_diferencias)
ruta_diferencias_sensibilidad = FEATURES_TDA_DIR / 'sensibilidad_submuestreo_diferencias_vs_500.csv'
diferencias_sensibilidad.to_csv(ruta_diferencias_sensibilidad, index=False)

display(diferencias_sensibilidad)
print('Diferencias relativas guardadas en:', ruta_diferencias_sensibilidad)

# %% Cell 77
# Visualizar sensibilidad de las features principales
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

for ax, metrica in zip(axes, metricas_sensibilidad):
    for clase, grupo in sensibilidad_submuestreo.groupby('clase'):
        grupo = grupo.sort_values('max_points')
        ax.plot(grupo['max_points'], grupo[metrica], marker='o', linewidth=2, label=clase)

    ax.set_title(metrica)
    ax.set_xlabel('max_points')
    ax.grid(True, alpha=0.3)

axes[0].legend()
plt.tight_layout()
ruta_fig_sensibilidad = OUTPUT_FIGURE / 'sensibilidad_submuestreo_features_TDA.png'
plt.savefig(ruta_fig_sensibilidad, dpi=150, bbox_inches='tight')
plt.show()

print('Figura guardada en:', ruta_fig_sensibilidad)

# %% Cell 78
# Resumen numerico para decidir configuracion
metricas_principales = [
    'H0_total_persistence_diff_pct_vs_500',
    'H1_total_persistence_diff_pct_vs_500',
    'H2_total_persistence_diff_pct_vs_500'
]

resumen_decision_submuestreo = (
    diferencias_sensibilidad[diferencias_sensibilidad['max_points'].isin([100, 200, 300, 400, 500])]
    .groupby('max_points')[metricas_principales]
    .mean()
    .reset_index()
)
resumen_decision_submuestreo['promedio_diff_pct_vs_500'] = resumen_decision_submuestreo[metricas_principales].mean(axis=1)

display(resumen_decision_submuestreo)

diff_200 = resumen_decision_submuestreo.loc[
    resumen_decision_submuestreo['max_points'] == 200,
    'promedio_diff_pct_vs_500'
].iloc[0]

if diff_200 <= 15:
    MAX_POINTS_RECOMENDADO = 200
    criterio = '200 puntos es suficientemente cercano a 500 puntos.'
else:
    MAX_POINTS_RECOMENDADO = 500
    criterio = '200 puntos todavia cambia de forma apreciable respecto a 500 puntos.'

print('MAX_POINTS_RECOMENDADO:', MAX_POINTS_RECOMENDADO)
print('Criterio:', criterio)

