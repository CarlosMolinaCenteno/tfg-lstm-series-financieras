"""Experimentos de control de la tarea de magnitud.

## Por que existe este modulo

La referencia de la tarea de magnitud es la **media** de los valores
absolutos de la ventana, mientras que la perdida de entrenamiento y la
metrica de evaluacion son el **error absoluto**. El predictor optimo bajo
error absoluto es la **mediana** condicionada, no la media; y como los
rendimientos tienen colas pesadas, la distribucion de |rho| esta sesgada a la
derecha y su mediana queda por debajo de su media.

La consecuencia es que la referencia esta sistematicamente descentrada
respecto del optimo de la perdida con la que se la juzga, y **cualquier**
metodo que se limite a corregir ese sesgo de localizacion gana, sin
necesidad de memoria ni de no linealidad. Mientras no se descarte, la mejora
de la red no esta atribuida.

Este modulo implementa los controles que lo zanjan:

1. `mediana_reciente` -- la mediana de la ventana en lugar de la media.
2. `calibra_l1` -- el factor multiplicativo optimo bajo error absoluto,
   ajustado **solo en entrenamiento**. Un unico parametro.
3. `ewma` -- promedio ponderado exponencialmente. Distingue "ponderar" de
   "tener memoria larga": si un EWMA bate a la media plana, la ganancia no
   exige dependencia de largo alcance.
4. `har` -- el modelo heterogeneo autorregresivo de Corsi (2009), la
   referencia del area para volatilidad a varios horizontes.
5. `Garch11` -- GARCH(1,1) por serie con cuasi-maxima verosimilitud
   gaussiana, la referencia canonica del agrupamiento de volatilidad.
6. `RegresionAgrupadaL1` -- la regresion agrupada ajustada con la **misma**
   perdida que la red. La version por minimos cuadrados estima la media
   condicionada, que es justo el funcional de la referencia, de modo que su
   empate con ella no informa de nada.

Todos los predictores respetan la interfaz de `baseline`: reciben `X` de
forma (n, K) y devuelven una prediccion de forma (n, H).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

# ---------------------------------------------------------------------------
# 1. Localizacion: mediana frente a media
# ---------------------------------------------------------------------------


def constante_optima_l1(y_entrena: np.ndarray) -> float:
    """Constante que minimiza el error absoluto en entrenamiento.

    Es la mediana de los objetivos, y es el analogo para la tarea de
    **nivel** de lo que `escala_optima_l1` es para la de magnitud: si la
    referencia de la tarea de nivel —predecir cero— estuviera descentrada
    respecto del optimo de la perdida, esta constante lo revelaria. Se
    incluye por simetria, para que la misma objecion se aplique a las dos
    tareas y no solo a una.
    """
    y = np.asarray(y_entrena, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    return float(np.median(y)) if len(y) else 0.0


def constante(X: np.ndarray, H: int, valor: float) -> np.ndarray:
    """Predice siempre el mismo valor."""
    return np.full((len(X), H), valor, dtype=np.float32)


def mediana_reciente(X: np.ndarray, H: int, ventana: int | None = None
                     ) -> np.ndarray:
    """Predice la **mediana** de los valores absolutos recientes.

    Es la referencia del capitulo 6 con el unico cambio de sustituir la media
    por la mediana. Si buena parte de la ganancia de la red desaparece frente
    a este predictor, la ganancia era de localizacion y no de estructura.
    """
    corte = np.abs(X if ventana is None else X[:, -ventana:])
    return np.repeat(np.median(corte, axis=1, keepdims=True), H, axis=1)


# ---------------------------------------------------------------------------
# 2. Calibracion de escala bajo error absoluto
# ---------------------------------------------------------------------------


def escala_optima_l1(y: np.ndarray, pred: np.ndarray) -> float:
    """Factor `c` que minimiza  sum |y - c*pred|.

    El minimo de esa funcion convexa y lineal a trozos se alcanza en la
    **mediana ponderada** de los cocientes y/pred, con pesos |pred|. Se
    calcula exactamente y no por busqueda, que ademas es mas barato.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    m = np.isfinite(y) & np.isfinite(p) & (np.abs(p) > 0)
    if not m.any():
        return 1.0
    r, w = y[m] / p[m], np.abs(p[m])
    orden = np.argsort(r)
    r, w = r[orden], w[orden]
    acumulado = np.cumsum(w)
    corte = np.searchsorted(acumulado, acumulado[-1] / 2.0)
    return float(r[min(corte, len(r) - 1)])


def escalas_por_serie_l1(y: np.ndarray, pred: np.ndarray,
                         ticker: np.ndarray) -> pd.Series:
    """Un factor de escala por serie, ajustado en entrenamiento."""
    idx = pd.Series(ticker)
    return pd.Series(
        {tk: escala_optima_l1(y[idx.to_numpy() == tk],
                              pred[idx.to_numpy() == tk])
         for tk in pd.unique(idx)})


def aplica_escala(pred: np.ndarray, factor: float | pd.Series,
                  ticker: np.ndarray | None = None) -> np.ndarray:
    """Multiplica la prediccion por un factor global o por uno de cada serie."""
    if isinstance(factor, pd.Series):
        if ticker is None:
            raise ValueError("con escala por serie hace falta el ticker")
        c = factor.reindex(pd.Series(ticker)).to_numpy(dtype=np.float64)
        c = np.where(np.isfinite(c), c, 1.0)
        return (pred * c[:, None]).astype(np.float32)
    return (pred * float(factor)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Ponderacion exponencial
# ---------------------------------------------------------------------------


def ewma(X: np.ndarray, H: int, lam: float = 0.94) -> np.ndarray:
    """Promedio de |X| ponderado exponencialmente, con decaimiento `lam`.

    `lam=0.94` es el valor de RiskMetrics para datos diarios. La prediccion
    es constante a lo largo del horizonte, como la de la referencia, de modo
    que la unica diferencia con ella es **como se pondera la ventana**.
    """
    K = X.shape[1]
    pesos = lam ** np.arange(K - 1, -1, -1, dtype=np.float64)
    pesos /= pesos.sum()
    v = np.abs(X.astype(np.float64)) @ pesos
    return np.repeat(v[:, None], H, axis=1).astype(np.float32)


def ajusta_lambda(X: np.ndarray, y: np.ndarray,
                  malla: np.ndarray | None = None) -> float:
    """Elige `lam` minimizando el error absoluto **en entrenamiento**.

    Es el segundo parametro del control: con dos —escala y decaimiento— un
    predictor sin memoria larga ni no linealidad ya deberia acercarse a la
    red si la ganancia de esta fuese solo de ponderacion.
    """
    if malla is None:
        malla = np.arange(0.80, 0.995, 0.01)
    mejor, mejor_err = 0.94, np.inf
    for lam in malla:
        p = ewma(X, y.shape[1], float(lam))
        c = escala_optima_l1(y, p)
        err = float(np.abs(y - c * p).mean())
        if err < mejor_err:
            mejor, mejor_err = float(lam), err
    return mejor


# ---------------------------------------------------------------------------
# 4. HAR (Corsi 2009)
# ---------------------------------------------------------------------------


class HAR:
    """Modelo heterogeneo autorregresivo sobre |rho|.

    Tres regresores construidos con la propia ventana de entrada: la media de
    los valores absolutos del ultimo dia, de la ultima semana (5 sesiones) y
    del ultimo mes (22 sesiones). Es la forma estandar de reproducir el
    decaimiento lento de la autocorrelacion de la magnitud sin recurrencia.

    Se ajusta agrupando todas las series, igual que la red: es un modelo
    global, y asi la comparacion no confunde "ser global" con "ser recurrente".
    """

    RETARDOS = (1, 5, 22)

    def __init__(self, l1: bool = False) -> None:
        self.coef: np.ndarray | None = None
        self.l1 = l1

    def _diseno(self, X: np.ndarray) -> np.ndarray:
        A = np.abs(X.astype(np.float64))
        cols = [A[:, -k:].mean(axis=1) for k in self.RETARDOS if k <= A.shape[1]]
        return np.column_stack(cols + [np.ones(len(A))])

    def ajusta(self, X: np.ndarray, y: np.ndarray) -> "HAR":
        Z = self._diseno(X)
        if self.l1:
            self.coef = _minimos_absolutos(Z, y.astype(np.float64))
        else:
            self.coef, *_ = np.linalg.lstsq(Z, y.astype(np.float64), rcond=None)
        return self

    def predice(self, X: np.ndarray, H: int | None = None) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("hay que llamar a ajusta() primero")
        return (self._diseno(X) @ self.coef).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. GARCH(1,1)
# ---------------------------------------------------------------------------


@dataclass
class ParametrosGarch:
    omega: float
    alfa: float
    beta: float

    @property
    def persistencia(self) -> float:
        return self.alfa + self.beta

    @property
    def varianza_larga(self) -> float:
        p = self.persistencia
        return self.omega / (1.0 - p) if p < 1.0 else np.nan


def _verosimilitud_garch(par: np.ndarray, R: np.ndarray) -> float:
    """Cuasi-log-verosimilitud gaussiana negativa de un GARCH(1,1).

    `R` es una matriz (T, S): la recursion se lleva a la vez sobre las `S`
    series, con los **mismos** parametros para todas. La alternativa —una
    terna por serie— multiplicaria el coste por `S` sin cambiar la
    conclusion, y ademas seria comparar un modelo con cientos de parametros
    con una red que tiene uno solo por todas las series.
    """
    omega, alfa, beta = np.exp(par)          # positividad por reparametrizacion
    if alfa + beta >= 0.999 or omega <= 0:
        return 1e12
    T = R.shape[0]
    s2 = np.nanvar(R, axis=0)
    s2 = np.where(np.isfinite(s2) & (s2 > 0), s2, 1.0)
    total = 0.0
    for t in range(T):
        x2 = R[t] ** 2
        valido = np.isfinite(x2)
        total += float(np.sum(np.log(s2[valido]) + x2[valido] / s2[valido]))
        s2 = omega + alfa * np.where(valido, x2, s2) + beta * s2
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e12
    return 0.5 * total


def ajusta_garch(R: np.ndarray) -> ParametrosGarch:
    """QMLE gaussiana de un GARCH(1,1) agrupado sobre un panel.

    Acepta un vector (una serie) o una matriz (T, S). Se parte de los valores
    tipicos en datos diarios de renta variable —alfa cerca de 0,05 y beta
    cerca de 0,90— y se optimiza sin restricciones sobre el logaritmo de los
    parametros, lo que garantiza positividad sin necesidad de restricciones.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    var0 = float(np.nanvar(R))
    if not np.isfinite(var0) or var0 <= 0 or len(R) < 100:
        return ParametrosGarch(max(var0, 1e-8) * 0.05, 0.05, 0.90)
    inicio = np.log([max(var0 * 0.05, 1e-8), 0.05, 0.90])
    sol = optimize.minimize(_verosimilitud_garch, inicio, args=(R,),
                            method="Nelder-Mead",
                            options={"maxiter": 400, "xatol": 1e-4,
                                     "fatol": 1e-2})
    omega, alfa, beta = np.exp(sol.x)
    if alfa + beta >= 0.999 or not np.isfinite(omega):
        return ParametrosGarch(max(var0 * 0.05, 1e-8), 0.05, 0.90)
    return ParametrosGarch(float(omega), float(alfa), float(beta))


def predice_garch(X: np.ndarray, H: int, par: ParametrosGarch) -> np.ndarray:
    """Filtra la varianza a lo largo de la ventana y predice H pasos.

    Se arranca el filtro en la varianza de la propia ventana, se recorre esta
    hacia adelante y se proyecta la varianza condicionada con
    `E[s2_{t+h}] = s2_larga + (alfa+beta)^{h-1} (s2_{t+1} - s2_larga)`.
    La prediccion de |rho| es proporcional a la desviacion tipica; la
    constante de proporcionalidad depende de la distribucion de las
    innovaciones y **no se fija a mano**: se calibra en entrenamiento con
    `escala_optima_l1`, igual que a los demas controles.
    """
    A = X.astype(np.float64)
    n, K = A.shape
    s2 = np.var(A, axis=1)
    s2 = np.where(s2 > 0, s2, 1.0)
    for k in range(K):
        s2 = par.omega + par.alfa * A[:, k] ** 2 + par.beta * s2
    p = par.persistencia
    larga = par.varianza_larga
    if not np.isfinite(larga):
        larga = float(np.mean(s2))
    h = np.arange(H)
    proyectada = larga + (p ** h)[None, :] * (s2[:, None] - larga)
    proyectada = np.clip(proyectada, 1e-12, None)
    return np.sqrt(proyectada).astype(np.float32)


class GarchAgrupado:
    """GARCH(1,1) con parametros comunes a todas las series del panel.

    Se ajusta agrupando, como la red y como la regresion agrupada: el panel
    ya esta normalizado serie a serie, de modo que compartir `(omega, alfa,
    beta)` es razonable y hace la comparacion limpia. Un GARCH por serie
    tendria cientos de parametros frente a los ~5000 de la red repartidos
    entre 271 series, y ademas dispararia el coste sin cambiar la lectura.
    """

    def __init__(self) -> None:
        self.par: ParametrosGarch | None = None

    def ajusta(self, panel_entrena: pd.DataFrame) -> "GarchAgrupado":
        self.par = ajusta_garch(panel_entrena.to_numpy(dtype=np.float64))
        return self

    def predice(self, X: np.ndarray, H: int) -> np.ndarray:
        if self.par is None:
            raise RuntimeError("hay que llamar a ajusta() primero")
        return predice_garch(X, H, self.par)


# ---------------------------------------------------------------------------
# 6. Regresion agrupada con perdida absoluta
# ---------------------------------------------------------------------------


def _minimos_absolutos(Z: np.ndarray, y: np.ndarray, iteraciones: int = 60,
                       epsilon: float = 1e-6) -> np.ndarray:
    """Minimiza  sum |y - Z b|  por minimos cuadrados reponderados.

    El algoritmo es el clasico de Weiszfeld: en cada paso se resuelve un
    problema de minimos cuadrados con pesos `1/|residuo|`. Converge al minimo
    de la norma L1 y no necesita mas dependencias que las que ya hay.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    plano = y.ndim == 1
    Y = y[:, None] if plano else y
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    for _ in range(iteraciones):
        res = np.abs(Y - Z @ b).mean(axis=1)
        raiz = np.sqrt(1.0 / np.maximum(res, epsilon))[:, None]
        nuevo, *_ = np.linalg.lstsq(Z * raiz, Y * raiz, rcond=None)
        convergido = np.max(np.abs(nuevo - b)) < 1e-9
        b = nuevo
        if convergido:
            break
    return b[:, 0] if plano else b


class RegresionAgrupadaL1:
    """Regresion agrupada ajustada con **error absoluto**, no cuadratico.

    Es el control que responde a la objecion mas directa: por minimos
    cuadrados la regresion estima la media condicionada, que es exactamente
    el funcional que estima la referencia, de modo que su empate con ella era
    previsible y no informaba sobre no linealidad ni sobre memoria. Ajustada
    con la misma perdida que la red, la comparacion si mide lo que dice medir.
    """

    def __init__(self, absoluto: bool = True) -> None:
        self.coef: np.ndarray | None = None
        self.absoluto = absoluto

    def _diseno(self, X: np.ndarray) -> np.ndarray:
        Z = np.abs(X) if self.absoluto else X
        Z = Z.astype(np.float64)
        return np.hstack([Z, np.ones((len(Z), 1))])

    def ajusta(self, X: np.ndarray, y: np.ndarray) -> "RegresionAgrupadaL1":
        self.coef = _minimos_absolutos(self._diseno(X), y.astype(np.float64))
        return self

    def predice(self, X: np.ndarray, H: int | None = None) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("hay que llamar a ajusta() primero")
        return (self._diseno(X) @ self.coef).astype(np.float32)
