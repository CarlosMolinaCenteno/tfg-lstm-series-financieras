"""Orquestacion: ejecuta las lineas base sobre un panel y devuelve la tabla.

Se separa de los modulos anteriores para que el capitulo 6 y los cuadernos
llamen a una sola funcion y no repitan el cableado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baseline as B, metricas as M, ventanas as V

REFERENCIA = {"nivel": "cero", "magnitud": "volatilidad reciente"}


def lineas_base(rendimientos: pd.DataFrame, K: int, H: int, tarea: str,
                particion: V.Particion | None = None,
                tramo: str = "prueba") -> pd.DataFrame:
    """Evalua todas las referencias de una tarea y devuelve la tabla resumen.

    Columnas: MASE (media, mediana, intervalo) y MASE relativo a la
    referencia, que es el que esta anclado en 1.
    """
    if particion is None:
        particion = V.particion_cronologica(rendimientos.index)
    panel = rendimientos[V.series_utilizables(rendimientos, particion)]
    norm = V.normaliza(panel, V.escala_por_serie(panel, particion))

    muestras = V.construye_ventanas(norm, K, H, tarea)
    trozos = V.reparte(muestras, particion, H)
    ent, eva = trozos["entrena"], trozos[tramo]

    predictores = (B.PREDICTORES_NIVEL if tarea == "nivel"
                   else B.PREDICTORES_MAGNITUD)
    ref = REFERENCIA[tarea]

    escala = M.escala_referencia(ent.y, predictores[ref](ent.X, H), ent.ticker)
    resultados = {n: M.mase(eva.y, f(eva.X, H), eva.ticker, escala)
                  for n, f in predictores.items()}

    # Regresion agrupada. En la tarea de magnitud se le dan los valores
    # absolutos como entrada: ver la nota en baseline.RegresionAgrupada.
    for etiqueta, absoluto in [("regresion agrupada", tarea == "magnitud")]:
        reg = B.RegresionAgrupada(absoluto=absoluto).ajusta(ent.X, ent.y)
        resultados[etiqueta] = M.mase(eva.y, reg.predice(eva.X), eva.ticker, escala)
    if tarea == "magnitud":
        reg2 = B.RegresionAgrupada(absoluto=False).ajusta(ent.X, ent.y)
        resultados["regresion agrupada (con signo)"] = M.mase(
            eva.y, reg2.predice(eva.X), eva.ticker, escala)

    tabla = M.tabla(resultados)[["media", "mediana", "ic95_semiancho", "n_series"]]
    tabla.columns = ["MASE medio", "MASE mediano", "IC95", "series"]
    rel = {n: M.resumen(M.mase_relativo(v, resultados[ref]))["media"]
           for n, v in resultados.items()}
    tabla.insert(2, "MASE rel.", pd.Series(rel))
    tabla.attrs["referencia"] = ref
    tabla.attrs["n_entrena"] = len(ent)
    tabla.attrs["n_evalua"] = len(eva)
    return tabla
