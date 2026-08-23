"""Etapa 2: segmentacion en ventanas, particion cronologica y normalizacion.

Convierte el panel de rendimientos (fechas x tickers) en los tensores que
consume la red, respetando el orden temporal en todos los pasos.

Correspondencia con la memoria:

- La segmentacion en ventanas de longitud K es la forma practica del
  desenrollado del capitulo 3, y su justificacion es la Proposicion 1 de
  Sherstinsky: si la secuencia se puede tratar como coleccion de segmentos
  independientes, estos se pueden procesar en cualquier orden. Es lo que
  permite formar minilotes sin destruir la estructura temporal.
- La normalizacion no es cosmetica: mantiene las entradas fuera de la zona
  de saturacion de la no linealidad, que es donde la derivada se anula y el
  gradiente muere (capitulo 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TAREAS = ("nivel", "magnitud")


# ---------------------------------------------------------------------------
# Particion cronologica
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Particion:
    """Fechas de corte. Todo lo anterior a `fin_entrena` es entrenamiento.

    `fin_prueba` acota el tramo de prueba por la derecha. Es opcional porque
    con una particion unica la prueba llega hasta el final de los datos, pero
    hace falta con origen movil, donde cada origen evalua sobre una ventana
    de longitud fija.
    """

    fin_entrena: pd.Timestamp
    fin_valida: pd.Timestamp
    fin_prueba: pd.Timestamp | None = None

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        fin = "" if self.fin_prueba is None else f" < {self.fin_prueba.date()}"
        return (f"Particion(entrena < {self.fin_entrena.date()} "
                f"<= valida < {self.fin_valida.date()} <= prueba{fin})")


def particion_cronologica(fechas: pd.DatetimeIndex,
                          prop_entrena: float = 0.70,
                          prop_valida: float = 0.15) -> Particion:
    """Corta el eje temporal en tres tramos consecutivos.

    **Cronologica y no aleatoria.** Barajar permitiria entrenar con el futuro
    y evaluar sobre el pasado, lo que produce resultados excelentes y
    completamente falsos.
    """
    if not fechas.is_monotonic_increasing:
        raise ValueError("las fechas deben venir ordenadas")
    n = len(fechas)
    return Particion(fin_entrena=fechas[int(n * prop_entrena)],
                     fin_valida=fechas[int(n * (prop_entrena + prop_valida))])


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------
def escala_por_serie(rendimientos: pd.DataFrame,
                     particion: Particion) -> pd.Series:
    """Desviacion tipica de cada serie, calculada **solo con entrenamiento**.

    Dos decisiones, y las dos importan:

    1. **Solo con el tramo de entrenamiento.** Estimar la escala sobre la
       muestra completa filtraria informacion del futuro a la fase de
       entrenamiento, que es una fuga sutil y facil de cometer.

    2. **Una escala por serie, constante en el tiempo** -- y no una escala por
       ventana. Normalizar cada ventana por su propia volatilidad pondria a
       todas las ventanas en el mismo rango, pero **destruiria exactamente la
       senal que la tarea de magnitud quiere predecir**: el nivel de
       volatilidad y su variacion en el tiempo. Lo que se busca es poner los
       distintos valores en una escala comparable entre si, que es lo que el
       modelo global necesita, sin tocar la variacion temporal dentro de cada
       uno.
    """
    tramo = rendimientos.loc[rendimientos.index < particion.fin_entrena]
    escala = tramo.std()
    if (escala <= 0).any() or escala.isna().any():
        malas = escala[(escala <= 0) | escala.isna()].index.tolist()
        raise ValueError(f"escala nula o indefinida en: {malas}")
    return escala


def normaliza(rendimientos: pd.DataFrame, escala: pd.Series) -> pd.DataFrame:
    """Divide cada serie por su escala de entrenamiento."""
    return rendimientos.div(escala, axis=1)


# ---------------------------------------------------------------------------
# Ventana movil
# ---------------------------------------------------------------------------
@dataclass
class Muestras:
    """Tensores listos para la red, con su trazabilidad."""

    X: np.ndarray          # (n, K)  ventana de entrada
    y: np.ndarray          # (n, H)  ventana de salida
    ticker: np.ndarray     # (n,)    a que serie pertenece cada muestra
    origen: np.ndarray     # (n,)    fecha del ultimo paso de la entrada

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return (f"Muestras(n={len(self):,}, K={self.X.shape[1]}, "
                f"H={self.y.shape[1]}, series={len(np.unique(self.ticker))})")

    def subconjunto(self, mascara: np.ndarray) -> "Muestras":
        return Muestras(self.X[mascara], self.y[mascara],
                        self.ticker[mascara], self.origen[mascara])


def _objetivo(bloque: np.ndarray, tarea: str) -> np.ndarray:
    """Transforma la ventana de salida segun la tarea.

    - `nivel`: el rendimiento tal cual. Es lo que restringe la hipotesis de
      eficiencia, y donde el capitulo 2 predice que no hay senal explotable.
    - `magnitud`: su valor absoluto. Es donde esta documentada la dependencia
      de largo alcance, y donde el capitulo 2 predice que si la hay.
    """
    if tarea == "nivel":
        return bloque
    if tarea == "magnitud":
        return np.abs(bloque)
    raise ValueError(f"tarea desconocida: {tarea!r}; use una de {TAREAS}")


def construye_ventanas(rendimientos: pd.DataFrame, K: int, H: int,
                       tarea: str = "nivel",
                       paso: int = 1) -> Muestras:
    """Trocea cada serie en pares (ventana de entrada, ventana de salida).

    La entrada es siempre el rendimiento; lo que cambia con la tarea es el
    objetivo. Asi las dos tareas ven exactamente la misma informacion y la
    comparacion entre ellas es limpia.

    Cada desplazamiento de la ventana genera una muestra nueva, lo que actua
    ademas como mecanismo de aumento de datos.
    """
    if K < 1 or H < 1:
        raise ValueError("K y H deben ser positivos")

    fechas = rendimientos.index
    Xs, ys, tks, ors = [], [], [], []

    for tk in rendimientos.columns:
        serie = rendimientos[tk]
        valido = serie.notna().to_numpy()
        v = serie.to_numpy(dtype=float)

        for ini in range(0, len(v) - K - H + 1, paso):
            corte = slice(ini, ini + K + H)
            if not valido[corte].all():
                continue          # se descarta la ventana con huecos
            Xs.append(v[ini:ini + K])
            ys.append(_objetivo(v[ini + K:ini + K + H], tarea))
            tks.append(tk)
            ors.append(fechas[ini + K - 1])

    if not Xs:
        raise ValueError("ninguna ventana valida; revise K, H y los huecos")

    return Muestras(X=np.asarray(Xs, dtype=np.float32),
                    y=np.asarray(ys, dtype=np.float32),
                    ticker=np.asarray(tks),
                    origen=np.asarray(ors, dtype="datetime64[ns]"))


def reparte(muestras: Muestras, particion: Particion, H: int
            ) -> dict[str, Muestras]:
    """Separa las muestras en entrenamiento, validacion y prueba.

    **Sin solapamiento entre tramos.** Una ventana se asigna por la fecha de
    su ultimo paso de entrada, y ademas se descartan las que, aun teniendo su
    origen en un tramo, alcanzarian con su salida el tramo siguiente. Sin esa
    segunda condicion habria fuga: el modelo se entrenaria con objetivos que
    caen dentro del periodo de validacion.
    """
    origen = pd.DatetimeIndex(muestras.origen)
    fin_e, fin_v = particion.fin_entrena, particion.fin_valida

    # Margen de H dias habiles para que la salida no invada el tramo siguiente.
    margen = pd.Timedelta(days=int(np.ceil(H * 7 / 5)) + 3)

    ent = origen < (fin_e - margen)
    val = (origen >= fin_e) & (origen < (fin_v - margen))
    pru = origen >= fin_v
    if particion.fin_prueba is not None:
        pru &= origen < (particion.fin_prueba - margen)

    return {"entrena": muestras.subconjunto(ent),
            "valida": muestras.subconjunto(val),
            "prueba": muestras.subconjunto(pru)}


def baraja_segmentos(muestras: Muestras, semilla: int) -> Muestras:
    """Permuta las muestras.

    Es lo que autoriza el corolario de la Proposicion 1: **no se barajan las
    observaciones, se barajan los segmentos**. Dentro de cada ventana el orden
    temporal se respeta; entre ventanas es irrelevante, y eso devuelve la
    independencia que el gradiente estocastico necesita.
    """
    rng = np.random.default_rng(semilla)
    return muestras.subconjunto(rng.permutation(len(muestras)))
