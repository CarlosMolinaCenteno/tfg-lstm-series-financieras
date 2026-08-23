"""Comprobaciones de la etapa 3.

La central es la de autoconsistencia: el MASE de la prediccion de referencia
frente a si misma tiene que valer **exactamente 1**. Si no lo vale, el error
esta en la metrica, y conviene descubrirlo ahora y no cuando haya un modelo
de por medio que pueda esconderlo.

Ejecutar:  .venv/Scripts/python tests/test_metricas.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfg import baseline as B, metricas as M  # noqa: E402

FALLOS = []


def comprueba(nombre, condicion, detalle=""):
    print(f"  [{'ok  ' if condicion else 'FALLA'}] {nombre}"
          + (f"   {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def datos(n=800, K=20, H=5, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(0, 1, (n, K)).astype(np.float32)
    y = rng.normal(0, 1, (n, H)).astype(np.float32)
    tk = np.array([f"S{i % 5}" for i in range(n)])
    return X, y, tk, H


def main():
    print("ETAPA 3 — comprobaciones\n")
    X, y, tk, H = datos()

    # --- autoconsistencia: la prueba que justifica hacer esto antes ---------
    for nombre, pred in B.PREDICTORES_NIVEL.items():
        p = pred(X, H)
        esc = M.escala_referencia(y, p, tk)
        v = M.mase(y, p, tk, esc)
        comprueba(f"MASE de «{nombre}» contra si misma vale 1",
                  np.allclose(v.to_numpy(), 1.0, atol=1e-6),
                  f"max desviacion {abs(v - 1).max():.2e}")

    # --- un modelo perfecto y uno pesimo -----------------------------------
    esc = M.escala_referencia(y, B.cero(X, H), tk)
    comprueba("MASE de un predictor perfecto vale 0",
              np.allclose(M.mase(y, y, tk, esc).to_numpy(), 0.0))
    peor = M.mase(y, B.cero(X, H) + 10.0, tk, esc)
    comprueba("MASE de un predictor pesimo es mucho mayor que 1",
              (peor > 5).all(), f"minimo {peor.min():.1f}")

    # --- independencia de escala -------------------------------------------
    f = 137.0
    esc_f = M.escala_referencia(y * f, B.cero(X * f, H), tk)
    v1 = M.mase(y, B.deriva(X, H), tk, M.escala_referencia(y, B.cero(X, H), tk))
    v2 = M.mase(y * f, B.deriva(X * f, H), tk, esc_f)
    comprueba("el MASE no cambia al reescalar los datos",
              np.allclose(v1.to_numpy(), v2.to_numpy(), atol=1e-5))

    # --- formas y agregacion ------------------------------------------------
    comprueba("cada predictor devuelve forma (n, H)",
              all(p(X, H).shape == (len(X), H)
                  for p in list(B.PREDICTORES_NIVEL.values())
                  + list(B.PREDICTORES_MAGNITUD.values())))
    r = M.resumen(v1)
    comprueba("el resumen da media, mediana e intervalo",
              {"media", "mediana", "ic95_semiancho", "n_series"} <= set(r)
              and r["n_series"] == 5)

    # --- regresion agrupada -------------------------------------------------
    rng = np.random.default_rng(1)
    Xr = rng.normal(0, 1, (2000, 6)).astype(np.float32)
    w = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.4], dtype=np.float32)
    yr = (Xr @ w)[:, None] + rng.normal(0, 0.01, (2000, 1)).astype(np.float32)
    reg = B.RegresionAgrupada().ajusta(Xr[:1500], yr[:1500])
    err = float(np.abs(reg.predice(Xr[1500:]) - yr[1500:]).mean())
    comprueba("la regresion agrupada recupera una relacion lineal",
              err < 0.02, f"error absoluto medio {err:.4f}")

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
