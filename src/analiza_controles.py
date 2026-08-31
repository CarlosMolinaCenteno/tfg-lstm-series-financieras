"""Compara la red con los controles, pareando por origen.

    python src/analiza_controles.py

Lee resultados/rejilla_definitiva.csv (la red y las lineas base originales)
y resultados/controles.csv (los controles de atribucion), y produce los
cuadros del capitulo 6: la tabla de controles por horizonte y el contraste
de la red frente a cada control, pareado por origen.

El pareado por origen es el mismo criterio que el resto del capitulo: cada
origen aporta una observacion de cada metodo, promediada sobre semillas. No
se parea por serie, porque las series de un mismo indice estan
correlacionadas entre si y tratarlas como independientes regalaria
significacion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg.evaluacion import wilcoxon_pareado

RAIZ = Path(__file__).resolve().parents[1]
REJILLA = RAIZ / "resultados" / "rejilla_definitiva.csv"
# Los controles se ejecutaron en dos pasadas -- ver la cabecera de
# ejecuta_controles.py -- y se leen todos los ficheros que haya.
CONTROLES = sorted(RAIZ.glob("resultados/controles*.csv"))

# Numero de parametros que cada metodo estima en entrenamiento. Es la
# columna que da sentido a la comparacion: lo que se pregunta no es si la
# red gana, sino cuanto de su ganancia necesita cinco mil parametros.
PARAMETROS = {
    "volatilidad reciente": "0",
    "LSTM": "~5000",
    "mediana de la ventana": "0",
    "media reescalada (c global)": "1",
    "media reescalada (c por serie)": "271",
    "EWMA RiskMetrics": "1",
    "EWMA ajustado": "2",
    "GARCH(1,1) agrupado": "4",
    "HAR": "4",
    "regresion agrupada (L2)": "61",
    "regresion agrupada (L1)": "61",
    "cero": "0",
    "constante optima": "1",
    "deriva": "0",
}

ORDEN = ["volatilidad reciente", "LSTM", "mediana de la ventana",
         "media reescalada (c global)", "media reescalada (c por serie)",
         "EWMA RiskMetrics", "EWMA ajustado", "GARCH(1,1) agrupado", "HAR",
         "regresion agrupada (L2)", "regresion agrupada (L1)"]

ORDEN_NIVEL = ["cero", "LSTM", "constante optima", "deriva",
               "regresion agrupada (L2)", "regresion agrupada (L1)"]


def carga() -> pd.DataFrame:
    if not CONTROLES:
        raise SystemExit("no hay resultados/controles*.csv; "
                         "ejecuta primero src/ejecuta_controles.py")
    rej = pd.read_csv(REJILLA)
    ctl = pd.concat([pd.read_csv(f) for f in CONTROLES], ignore_index=True)
    # De la rejilla solo interesa la red: las lineas base ya estan en los
    # controles, recalculadas con el mismo denominador.
    rej = rej[rej["metodo"] == "LSTM"]
    return pd.concat([rej, ctl], ignore_index=True)


def por_origen(d: pd.DataFrame, tarea: str, H: int) -> pd.DataFrame:
    s = d[(d["tarea"] == tarea) & (d["H"] == H)]
    return s.pivot_table(index="origen", columns="metodo",
                         values="mase_rel_medio", aggfunc="mean")


def main() -> int:
    d = carga()
    horizontes = sorted(d["H"].unique())

    for tarea, orden in (("magnitud", ORDEN), ("nivel", ORDEN_NIVEL)):
        sub = d[d["tarea"] == tarea]
        if sub.empty:
            continue
        print("=" * 72)
        print(f"TAREA DE {tarea.upper()}: media sobre orígenes, por horizonte")
        print("=" * 72)
        piv = sub.pivot_table(index="metodo", columns="H",
                              values="mase_rel_medio", aggfunc="mean")
        piv = piv.reindex([m for m in orden if m in piv.index])
        piv.insert(0, "par.", [PARAMETROS.get(m, "?") for m in piv.index])
        print(piv.round(4).to_string())
        print()

        print(f"Contraste de la red frente a cada control, pareado por origen")
        print("-" * 72)
        for H in horizontes:
            p = por_origen(d, tarea, H)
            if "LSTM" not in p:
                continue
            print(f"  H = {H}")
            for m in orden:
                if m == "LSTM" or m not in p:
                    continue
                r = wilcoxon_pareado(p["LSTM"], p[m])
                if not r:
                    continue
                dif = float(p[m].mean() - p["LSTM"].mean())
                print(f"    red vs {m:32s} "
                      f"gana {r['gana_en']:2d}/{r['n']:2d}  "
                      f"p = {r['p']:.4f}  "
                      f"ventaja media {dif:+.4f}")
            print()

    # --- cuanto de la ganancia explica un solo parametro -------------------
    print("=" * 72)
    print("QUÉ FRACCIÓN DE LA GANANCIA REPRODUCE UN SOLO PARÁMETRO")
    print("=" * 72)
    for H in horizontes:
        p = por_origen(d, "magnitud", H)
        if "LSTM" not in p or "media reescalada (c global)" not in p:
            continue
        g_red = 1.0 - float(p["LSTM"].mean())
        g_esc = 1.0 - float(p["media reescalada (c global)"].mean())
        print(f"  H = {H:2d}   red {100 * g_red:.2f} %   "
              f"escala {100 * g_esc:.2f} %   "
              f"fracción reproducida {100 * g_esc / g_red:.1f} %   "
              f"residuo {100 * (g_red - g_esc):.2f} puntos")

    # --- el factor de escala -----------------------------------------------
    ctl = pd.concat([pd.read_csv(f) for f in CONTROLES], ignore_index=True)
    if "parametro" in ctl:
        c = ctl[ctl["metodo"] == "media reescalada (c global)"]
        if not c.empty:
            print()
            print("Factor de escala óptimo bajo error absoluto, por horizonte:")
            print(c.groupby("H")["parametro"].agg(["mean", "min", "max"])
                  .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
