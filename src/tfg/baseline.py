"""Etapa 3b: predicciones de referencia.

Van **antes** del modelo, y a proposito: son baratas de implementar y sirven
para verificar la tuberia de evaluacion. En particular, el MASE de la
prediccion de referencia frente a si misma tiene que dar exactamente 1; si no
lo da, el error esta en la metrica y no en el modelo.

Cada predictor recibe la matriz de ventanas de entrada `X` de forma (n, K) y
devuelve una prediccion de forma (n, H).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tarea de nivel
# ---------------------------------------------------------------------------
def cero(X: np.ndarray, H: int) -> np.ndarray:
    """Predice rendimiento nulo.

    **Es la referencia teorica del trabajo.** Sobre precios equivale a comprar
    y mantener, que es lo que la submartingala identifica como imbatible por
    ninguna regla mecanica. Batirla de forma significativa es el resultado que
    el capitulo 6 pone a prueba en la tarea de nivel.
    """
    return np.zeros((len(X), H), dtype=np.float32)


def deriva(X: np.ndarray, H: int) -> np.ndarray:
    """Predice la media de la ventana de entrada.

    Es el modelo de **rendimientos esperados constantes**, que es el que Fama
    (1991) declara rechazado por los contrastes modernos. Incluirlo permite
    comprobar ese rechazo con nuestros datos.
    """
    return np.repeat(X.mean(axis=1, keepdims=True), H, axis=1)


def ingenua(X: np.ndarray, H: int) -> np.ndarray:
    """Repite el ultimo valor observado.

    Es la ingenua **de manual**. Sobre rendimientos no es comprar y mantener
    -- ver la nota de `metricas` -- pero se incluye porque es la referencia
    con la que la literatura de prediccion compara, y su MASE clasico vale 1
    por construccion.
    """
    return np.repeat(X[:, -1:], H, axis=1)


# ---------------------------------------------------------------------------
# Tarea de magnitud
# ---------------------------------------------------------------------------
def volatilidad_reciente(X: np.ndarray, H: int, ventana: int | None = None
                         ) -> np.ndarray:
    """Predice la media de los valores absolutos recientes.

    **Es la referencia de la tarea de magnitud, y tiene que ser esta y no una
    constante.** El tramo de prueba puede pertenecer a un regimen de
    volatilidad distinto del de entrenamiento -- en el piloto lo es, y por
    bastante --, de modo que una referencia estimada en entrenamiento estaria
    sistematicamente descentrada y cualquier modelo la superaria sin merito.
    Esta se adapta al regimen porque solo mira la propia ventana de entrada.

    Con `ventana=None` usa la ventana de entrada completa.
    """
    corte = X if ventana is None else X[:, -ventana:]
    media = np.abs(corte).mean(axis=1, keepdims=True)
    return np.repeat(media, H, axis=1)


def persistencia_magnitud(X: np.ndarray, H: int) -> np.ndarray:
    """Repite el ultimo valor absoluto observado.

    Referencia mas exigente que la anterior en presencia de agrupamiento de
    volatilidad, y mas ruidosa. Se incluye para no elegir la referencia mas
    comoda.
    """
    return np.repeat(np.abs(X[:, -1:]), H, axis=1)


# ---------------------------------------------------------------------------
# Linea base intermedia
# ---------------------------------------------------------------------------
class RegresionAgrupada:
    """Regresion lineal global: un autorregresivo comun a todas las series.

    Es la linea base intermedia del diseno de Hewamalage et al., y su papel es
    **separar dos ganancias que de otro modo se confunden**: comparar esta con
    una regresion por serie aisla la ganancia de ser global; comparar la red
    con esta aisla la ganancia de ser recurrente. Sin ella, un buen resultado
    de la red no dice cual de las dos cosas lo produjo.

    Se resuelve por minimos cuadrados sobre la matriz de ventanas, que es un
    problema convexo con solucion cerrada -- el contraste con la optimizacion
    no convexa del Anexo A.
    """

    def __init__(self, absoluto: bool = False) -> None:
        """`absoluto=True` alimenta el modelo con |X| en vez de con X.

        Hace falta para que la comparacion en la tarea de magnitud sea justa.
        Una funcion lineal de los rendimientos con signo **no puede
        representar el valor absoluto**: si la entrada tipica es simetrica en
        torno a cero, el mejor ajuste lineal de |y| es practicamente una
        constante. Comparar la red con un modelo lineal asi handicapado no
        demostraria que hace falta no linealidad, solo que el rival estaba
        atado. Con |X| de entrada, el rival puede capturar la persistencia de
        la volatilidad y la comparacion mide lo que dice medir.
        """
        self.coef: np.ndarray | None = None
        self.absoluto = absoluto

    def _diseno(self, X: np.ndarray) -> np.ndarray:
        Z = np.abs(X) if self.absoluto else X
        return np.hstack([Z, np.ones((len(Z), 1), dtype=Z.dtype)])

    def ajusta(self, X: np.ndarray, y: np.ndarray) -> "RegresionAgrupada":
        self.coef, *_ = np.linalg.lstsq(self._diseno(X), y, rcond=None)
        return self

    def predice(self, X: np.ndarray, H: int | None = None) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("hay que llamar a ajusta() primero")
        return (self._diseno(X) @ self.coef).astype(np.float32)


PREDICTORES_NIVEL = {"cero": cero, "deriva": deriva, "ingenua": ingenua}
PREDICTORES_MAGNITUD = {"volatilidad reciente": volatilidad_reciente,
                        "persistencia": persistencia_magnitud}
