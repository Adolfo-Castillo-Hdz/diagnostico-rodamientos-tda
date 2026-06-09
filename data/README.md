# Datos

Este repositorio no incluye los archivos crudos ni las ventanas generadas porque son pesados.

## Dataset utilizado

El proyecto usa el **Case Western Reserve University Bearing Dataset (CWRU)** con señales de vibración del acelerómetro del lado motriz:

- `DE_time` = Drive End acceleration signal.

Las clases consideradas son:

- `Normal`: `Time_Normal_1_098`
- `Ball`: `B007_1_123`, `B014_1_190`, `B021_1_227`
- `Inner_Race`: `IR007_1_110`, `IR014_1_175`, `IR021_1_214`
- `Outer_Race`: `OR007_6_1_136`, `OR014_6_1_202`, `OR021_6_1_239`

## Estructura esperada para regenerar el proyecto

Coloca los archivos `.mat` en una de estas rutas:

```text
archive/raw/
```

o

```text
raw/
```

El notebook `notebooks/01_mantenimiento_predictivo_tda.ipynb` detecta ambas rutas.

## Archivos generados localmente

Al ejecutar el pipeline se generan carpetas como:

```text
DE_signals/
ventanas_DE/
features_TDA/
clasificacion_modelos/
Figuras_TDA/
```

Estas carpetas se excluyen de Git con `.gitignore` para evitar subir datos pesados o derivados.
