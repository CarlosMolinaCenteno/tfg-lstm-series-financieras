"""Mide el coste real de un entrenamiento sobre el panel completo y extrapola.

No se estima por regla de tres desde el piloto: se cronometra un entrenamiento
de verdad en el origen mas caro -- el ultimo, que es el que mas datos de
entrenamiento tiene -- y se multiplica por el tamano de la rejilla.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import datos, evaluacion as E, modelo as Mo, ventanas as V

INICIO, FIN = "2012-01-03", "2026-08-01"
K, H = 60, 5
N_ORIGENES, N_HORIZONTES, N_TAREAS, N_SEMILLAS = 12, 4, 2, 2


def mide(norm, part, paso):
    m = V.construye_ventanas(norm, K, H, "magnitud", paso=paso)
    t = V.reparte(m, part, H)
    ent, val = t["entrena"], t["valida"]
    cfg = Mo.Config(unidades=32, epocas_max=2, paciencia=99, semilla=0)
    t0 = time.time()
    _, hist = Mo.entrena(ent.X, ent.y, val.X, val.y, cfg)
    return len(ent), (time.time() - t0) / len(hist.perdida_entrena)


if __name__ == "__main__":
    tk = datos.miembros_durante(INICIO, FIN)
    r = datos.rendimientos_log(datos.descarga_precios(tk, INICIO, FIN, lote=50))
    print(f"panel completo: {r.shape[0]} fechas x {r.shape[1]} series")

    partes = E.origenes_moviles(r.index, n_origenes=N_ORIGENES, dias_prueba=126)
    part = partes[-1]
    print(f"{len(partes)} origenes; el ultimo es el mas caro:")
    print(f"  {part}")

    utiles = V.series_utilizables(r, part)
    print(f"series utilizables en este origen: {len(utiles)} de {r.shape[1]}")
    panel = r[utiles]
    norm = V.normaliza(panel, V.escala_por_serie(panel, part))

    rejilla = Path(__file__).resolve().parents[1] / "resultados" / "rejilla_piloto.csv"
    epocas = float(pd.read_csv(rejilla).query("metodo == 'LSTM'")["epocas"].mean())
    print(f"epocas medias hasta la parada temprana en el piloto: {epocas:.1f}")
    print("")

    n_entren = N_ORIGENES * N_HORIZONTES * N_TAREAS * N_SEMILLAS
    print(f"{'paso':>5} {'muestras':>12} {'s/epoca':>10} {'min/entren.':>12} {'h rejilla':>11}")
    print("-" * 55)
    for paso in (3, 5, 10):
        n, s = mide(norm, part, paso)
        minutos = s * epocas / 60
        print(f"{paso:>5} {n:>12,} {s:>10.1f} {minutos:>12.1f} "
              f"{minutos * n_entren / 60:>11.1f}")

    print("")
    print(f"rejilla = {N_ORIGENES} origenes x {N_HORIZONTES} horizontes x "
          f"{N_TAREAS} tareas x {N_SEMILLAS} semillas = {n_entren} entrenamientos")
    print("cota superior: el ultimo origen es el que mas datos tiene, "
          "los anteriores son mas baratos")
