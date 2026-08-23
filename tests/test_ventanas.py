"""Comprobaciones de la etapa 2.

La mas importante es la de fuga temporal: es el fallo que produce resultados
excelentes y falsos, y no se detecta mirando las metricas.

Ejecutar:  .venv/Scripts/python tests/test_ventanas.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfg import ventanas as V  # noqa: E402

K, H = 20, 5
FALLOS = []


def comprueba(nombre, condicion, detalle=""):
    estado = "ok  " if condicion else "FALLA"
    print(f"  [{estado}] {nombre}" + (f"   {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def panel_sintetico(n_fechas=600, n_series=4, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2015-01-01", periods=n_fechas)
    datos = rng.normal(0, 0.01, size=(n_fechas, n_series))
    df = pd.DataFrame(datos, index=fechas,
                      columns=[f"S{i}" for i in range(n_series)])
    df.iloc[100:110, 1] = np.nan          # hueco deliberado
    return df


def main():
    print("ETAPA 2 — comprobaciones\n")
    rend = panel_sintetico()
    part = V.particion_cronologica(rend.index)
    print(f"  {part}\n")

    # --- normalizacion -----------------------------------------------------
    esc = V.escala_por_serie(rend, part)
    esc_solo_train = rend.loc[rend.index < part.fin_entrena].std()
    comprueba("la escala usa solo el tramo de entrenamiento",
              np.allclose(esc.to_numpy(), esc_solo_train.to_numpy()))
    norm = V.normaliza(rend, esc)
    de = norm.loc[norm.index < part.fin_entrena].std()
    comprueba("tras normalizar, la desv. tipica de entrenamiento es 1",
              np.allclose(de.to_numpy(), 1.0, atol=1e-9),
              f"max desviacion {abs(de - 1).max():.2e}")

    # --- ventanas ----------------------------------------------------------
    m = V.construye_ventanas(norm, K, H, "nivel")
    comprueba("las formas son (n,K) y (n,H)",
              m.X.shape[1] == K and m.y.shape[1] == H, repr(m))
    comprueba("ninguna ventana contiene huecos",
              not np.isnan(m.X).any() and not np.isnan(m.y).any())

    mm = V.construye_ventanas(norm, K, H, "magnitud")
    comprueba("magnitud es el valor absoluto de nivel, misma entrada",
              np.array_equal(mm.X, m.X) and np.allclose(mm.y, np.abs(m.y)))

    # --- FUGA TEMPORAL: la comprobacion que de verdad importa --------------
    trozos = V.reparte(m, part, H)
    for k, t in trozos.items():
        print(f"    {k:9s} {len(t):>6,} muestras")

    fechas = norm.index
    pos = {f: i for i, f in enumerate(fechas)}

    def ultima_fecha_objetivo(t):
        """Fecha del ultimo paso de la ventana de salida de cada muestra."""
        idx = np.array([pos[pd.Timestamp(o)] for o in t.origen])
        return fechas[np.minimum(idx + H, len(fechas) - 1)]

    obj_ent = ultima_fecha_objetivo(trozos["entrena"])
    comprueba("NINGUN objetivo de entrenamiento cae en validacion o despues",
              (obj_ent < part.fin_entrena).all(),
              f"max objetivo {obj_ent.max().date()} < corte {part.fin_entrena.date()}")

    obj_val = ultima_fecha_objetivo(trozos["valida"])
    comprueba("NINGUN objetivo de validacion cae en prueba",
              (obj_val < part.fin_valida).all(),
              f"max objetivo {obj_val.max().date()} < corte {part.fin_valida.date()}")

    o_e = pd.DatetimeIndex(trozos["entrena"].origen)
    o_p = pd.DatetimeIndex(trozos["prueba"].origen)
    comprueba("los tramos van en orden y no se solapan",
              o_e.max() < o_p.min())

    # --- barajado ----------------------------------------------------------
    b = V.baraja_segmentos(trozos["entrena"], semilla=1)
    comprueba("barajar conserva el conjunto de muestras",
              len(b) == len(trozos["entrena"]) and
              np.allclose(np.sort(b.X.sum(axis=1)),
                          np.sort(trozos["entrena"].X.sum(axis=1))))
    comprueba("barajar con la misma semilla da el mismo orden",
              np.array_equal(b.X, V.baraja_segmentos(trozos["entrena"], 1).X))

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
