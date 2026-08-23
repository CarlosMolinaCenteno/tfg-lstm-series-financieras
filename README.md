# Redes neuronales recurrentes para la predicción de series temporales financieras

Código del Trabajo de Fin de Grado en Matemáticas de la Universidad Complutense de Madrid.

El trabajo estudia las redes recurrentes desde su derivación matemática —una red recurrente es el método de Euler regresivo aplicado a una ecuación diferencial ordinaria con retardos— hasta su aplicación a rendimientos bursátiles. Este repositorio contiene únicamente el **código del experimento**; la memoria se entrega aparte.

## El experimento en una frase

Un modelo global recurrente entrenado sobre los rendimientos logarítmicos de los constituyentes históricos del S&P 500, evaluado con MASE frente a la predicción ingenua, **en dos tareas**: predecir el rendimiento y predecir su magnitud. El contraste entre ambas es el resultado.

La razón de las dos tareas es que la hipótesis de eficiencia de mercados restringe la esperanza condicionada del **nivel** de los precios y no dice nada sobre los momentos de orden superior, mientras que la dependencia temporal documentada en los datos está en la **magnitud** de los rendimientos. El experimento pone a prueba esa distinción en lugar de intentar una predicción genérica.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt    # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

Requiere Python 3.11 o superior. Desarrollado con 3.14.

## Uso

```python
from tfg import datos

# Valores que pertenecieron al S&P 500 de forma continua en el intervalo.
tickers = datos.miembros_durante("2012-01-03", "2026-01-02")

# Precios ajustados por desdoblamientos y dividendos, con cache en disco.
precios = datos.descarga_precios(tickers, "2012-01-03", "2026-01-02")
rendimientos = datos.rendimientos_log(precios)
```

Los datos **no se incluyen en el repositorio**: se reconstruyen ejecutando el módulo, y quedan cacheados en `data/`.

## Decisiones de diseño que conviene conocer

**Universo sin sesgo de supervivencia.** Se usa la lista de constituyentes *históricos* del S&P 500, no la actual. Tomar los miembros de hoy y descargar su historia completa seleccionaría las empresas que no quebraron ni fueron excluidas, lo que sesga la muestra y —lo que más importa aquí— sesga la distribución de la volatilidad, que es justo lo que la segunda tarea pretende predecir. Las fechas de alta y baja proceden de [`fja05680/sp500`](https://github.com/fja05680/sp500).

**Precios ajustados, siempre.** Sin ajustar por desdoblamientos y dividendos, un desdoblamiento 2:1 aparece como un rendimiento del −50 % en un día.

**Caché en disco.** El backend de Yahoo Finance ha cambiado varias veces y `yfinance` sufre roturas periódicas. La versión está fijada en `requirements.txt` y los datos descargados se guardan en parquet, de modo que el análisis pueda rehacerse sin red.

**Partición cronológica con margen.** Una ventana se asigna a un tramo por la fecha de su último paso de entrada, y se descartan además las que alcanzarían con su salida el tramo siguiente. Sin esa segunda condición hay fuga: el modelo se entrenaría con objetivos que caen dentro del periodo de validación. `tests/test_ventanas.py` lo comprueba explícitamente.

**Escala estimada solo con entrenamiento, y constante por serie.** Estimarla sobre la muestra completa filtraría información del futuro. Y se usa una escala por serie en lugar de una por ventana porque normalizar cada ventana por su propia volatilidad destruiría justo la señal que la segunda tarea pretende predecir.

**El horizonte de predicción se reporta, no se ajusta.** Cambiar el horizonte no cambia el modelo: cambia el problema. Probar varios y comunicar el mejor sería elegir la pregunta que mejor se responde. Se fija una rejilla de antemano y se publican todos los resultados.

## Estado

| Etapa | |
|---|---|
| Datos | ✅ |
| Ventanas y partición | ✅ |
| Líneas base y métrica | |
| Modelo | |
| Evaluación | |

## Licencia

MIT para el código. La bibliografía consultada no se distribuye aquí.
