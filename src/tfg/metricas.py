"""Etapa 3a: el error absoluto escalado (MASE) y su agregacion.

El MASE compara el error de un modelo con el de una prediccion de referencia
sobre la misma serie, lo que lo hace independiente de escala -- imprescindible
al comparar valores con volatilidades muy distintas.

## Una precision que no es menor

La definicion de manual escala por el error de la prediccion **ingenua**
(repetir el ultimo valor observado) calculada dentro de la muestra. Eso es lo
correcto cuando la serie modelada es un **precio**: repetir el ultimo precio
es comprar y mantener, que es justo la referencia que la submartingala
identifica como imbatible.

Aqui la serie modelada es el **rendimiento**, y repetir el ultimo rendimiento
no es comprar y mantener: es una prediccion mala, porque los rendimientos
apenas estan autocorrelados. Escalar por su error inflaria el denominador y
haria parecer bueno a cualquier modelo.

La referencia coherente con el capitulo 2 es la prediccion **nula** sobre
rendimientos, que equivale exactamente a comprar y mantener sobre precios.
Por eso `escala_referencia` recibe el predictor de referencia como argumento
en lugar de fijarlo, y el capitulo 6 reporta las dos versiones: la escalada
por la referencia teorica y la escalada por la ingenua clasica, que es la
comparable con la literatura de prediccion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def escala_referencia(y_entrena: np.ndarray, pred_entrena: np.ndarray,
                      ticker_entrena: np.ndarray) -> pd.Series:
    """Error absoluto medio de la referencia, por serie y **dentro de muestra**.

    Es el denominador del MASE. Se calcula sobre el tramo de entrenamiento
    para que no dependa del periodo evaluado.
    """
    err = np.abs(y_entrena - pred_entrena).mean(axis=1)
    escala = pd.Series(err).groupby(pd.Series(ticker_entrena)).mean()
    if (escala <= 0).any():
        malas = escala[escala <= 0].index.tolist()
        raise ValueError(f"escala de referencia nula en: {malas}")
    return escala


def mase(y: np.ndarray, pred: np.ndarray, ticker: np.ndarray,
         escala: pd.Series) -> pd.Series:
    """MASE de cada serie: error del modelo dividido por el de la referencia.

    Un valor mayor o igual que 1 significa que el modelo no aporta nada sobre
    la referencia.
    """
    err = np.abs(y - pred).mean(axis=1)
    por_serie = pd.Series(err).groupby(pd.Series(ticker)).mean()
    faltan = set(por_serie.index) - set(escala.index)
    if faltan:
        raise ValueError(f"sin escala de referencia para: {sorted(faltan)}")
    return por_serie / escala.reindex(por_serie.index)


def resumen(valores: pd.Series) -> dict[str, float]:
    """Media y mediana, con su dispersion.

    **Las dos, siempre.** Hewamalage et al. reportan ambas porque difieren:
    las metricas medias quedan dominadas por errores atipicos de algunas
    series, hasta el punto de que en uno de sus conjuntos las redes ganaban
    en mediana y perdian en media. Dar solo una de las dos oculta eso.
    """
    v = valores.dropna()
    n = len(v)
    return {"media": float(v.mean()),
            "mediana": float(v.median()),
            "desv": float(v.std(ddof=1)) if n > 1 else float("nan"),
            "ic95_semiancho": float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n_series": n}


def tabla(resultados: dict[str, pd.Series]) -> pd.DataFrame:
    """Compone la tabla de comparacion entre modelos."""
    return pd.DataFrame({k: resumen(v) for k, v in resultados.items()}).T


def mase_relativo(valores: pd.Series, valores_referencia: pd.Series) -> pd.Series:
    """MASE del modelo dividido por el MASE de la referencia, serie a serie.

    ## Por que hace falta esto

    El denominador del MASE se calcula **dentro de muestra**, sobre el tramo de
    entrenamiento. Eso es lo correcto -- asi no depende del periodo evaluado --
    pero tiene una consecuencia que conviene no pasar por alto: si el tramo de
    prueba pertenece a un regimen de volatilidad distinto, **el MASE de la
    propia referencia deja de valer 1**.

    Ocurre en estos datos: el periodo de prueba es menos volatil que el de
    entrenamiento, de modo que todos los MASE bajan a la vez y un valor de
    0,75 no significa batir a la referencia en un 25 %, sino simplemente que
    el periodo era mas tranquilo.

    El cociente entre ambos reancla la comparacion en 1 por construccion, a
    costa de perder comparabilidad con otros estudios. **Se reportan los dos**:
    el MASE para comparar con la literatura, y este para decidir si el modelo
    aporta algo.
    """
    comun = valores.index.intersection(valores_referencia.index)
    return valores.loc[comun] / valores_referencia.loc[comun]
