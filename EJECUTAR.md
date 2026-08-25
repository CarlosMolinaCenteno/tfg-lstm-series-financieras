# Ejecutar el experimento

Instrucciones para lanzar la ejecución definitiva en un equipo distinto del de desarrollo.
Este fichero está pensado para leerse solo: no hace falta contexto del resto del proyecto.

## Qué se va a ejecutar

Un modelo recurrente global sobre los rendimientos logarítmicos de 275 valores del
S&P 500 entre 2012 y 2026, evaluado con origen móvil sobre una rejilla de horizontes,
en dos tareas: predecir el rendimiento y predecir su magnitud.

Rejilla: 12 orígenes × 4 horizontes × 2 tareas × 2 semillas = **192 entrenamientos**.

## Puesta en marcha

Requiere Python 3.11 o superior y conexión a internet la primera vez.

```bash
git clone https://github.com/CarlosMolinaCenteno/tfg-lstm-series-financieras
cd tfg-lstm-series-financieras

python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # Windows
# python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Linux/macOS
```

## Paso 1 — comprobaciones (medio minuto)

Las cuatro deben terminar con «Todas las comprobaciones pasan».

```bash
.venv/Scripts/python tests/test_ventanas.py
.venv/Scripts/python tests/test_metricas.py
.venv/Scripts/python tests/test_modelo.py
.venv/Scripts/python tests/test_evaluacion.py
```

## Paso 2 — prueba de humo (unos 6 minutos)

Descarga los datos, ejecuta una versión reducida y deja un CSV. Sirve para
comprobar que todo funciona antes de comprometer varias horas.

```bash
.venv/Scripts/python -u src/ejecuta_definitivo.py --origenes 2 --paso 20 --semillas 1 --salida humo.csv
```

La primera línea que imprime es el resumen del equipo: hilos y memoria. **Anótala.**

## Paso 3 — elegir el submuestreo según la memoria disponible

`--paso N` toma una de cada N ventanas de entrenamiento. Dos ventanas consecutivas
comparten casi todos sus valores, así que el submuestreo quita redundancia más que
información; pero cuanto menor sea el paso, más datos y más tiempo.

| RAM libre | Orden | Muestras | Duración estimada (4 hilos) |
|---|---|---|---|
| más de 12 GB | `--paso 1` | 913 000 | unas 28 h |
| 8 – 12 GB | `--paso 3` | 304 000 | unas 9 h |
| 4 – 8 GB | `--paso 5` *(por defecto)* | 183 000 | unas 6 h |
| menos de 4 GB | `--paso 10` | 91 000 | unas 3 h |

Con más de 4 hilos el tiempo baja aproximadamente en proporción.

## Paso 4 — la ejecución

```bash
.venv/Scripts/python -u src/ejecuta_definitivo.py --paso 5 > ejecucion.log 2>&1
```

- Escribe `resultados/rejilla_definitiva.csv` **al terminar cada origen**, no solo al
  final: si se interrumpe, lo hecho hasta ese punto está guardado.
- Cada origen imprime cuántos minutos lleva y cuántos quedan.
- No necesita supervisión. Se puede cerrar la sesión si el proceso no depende de ella
  (`nohup` en Linux, o dejar la ventana abierta en Windows).

## Qué hay que devolver

Dos ficheros, unos 150 KB en total:

- `resultados/rejilla_definitiva.csv`
- `ejecucion.log`

## Si algo falla

| Síntoma | Causa y solución |
|---|---|
| `Recursos insuficientes` al importar torch | Falta memoria. Cerrar aplicaciones o subir `--paso` |
| Violación de segmento | Lo mismo: subir `--paso` |
| `YFTzMissingError` en algún ticker | Normal. Se reintenta solo; si queda vacío, la serie se descarta |
| La descarga falla del todo | Yahoo se rompe periódicamente. Reintentar más tarde |
| Tarda mucho más de lo estimado | Comprobar los hilos en la primera línea. Con 2 hilos, el doble |

## Qué NO hay que hacer

- No modificar los valores por defecto salvo `--paso`. La rejilla de horizontes, el
  número de orígenes y el diseño de la evaluación son decisiones metodológicas
  registradas, no parámetros de ajuste.
- No ejecutar varias veces y quedarse con el mejor resultado. Se ejecuta una vez y se
  informa de lo que salga.
