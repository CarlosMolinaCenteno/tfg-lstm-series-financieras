"""Recalcula la persistencia del GARCH(1,1) agrupado, origen por origen.

    python src/verifica_garch.py

El Anexo C afirma que «la persistencia estimada queda entre 0,90 y 0,96 segun
el origen». Ese numero se imprimia por pantalla al ejecutar
`src/ejecuta_controles.py`, pero **no se guarda en el CSV** -- la columna
`parametro` de las filas del GARCH guarda el factor de calibracion L1, que es
otra cosa -- y el registro de aquella pasada no se conservo. Es decir, la
cifra publicada no tenia artefacto detras.

Este guion la reconstruye con el mismo montaje que el original: mismo
universo, mismos origenes moviles, misma normalizacion por serie y mismo
ajuste agrupado por cuasi-maxima verosimilitud sobre el tramo de
entrenamiento. No entrena ninguna red, de modo que cuesta minutos y no horas.

Cambio asociado: `ejecuta_controles.py` guarda ya la persistencia en el CSV,
para que la proxima ejecucion no dependa de un registro por pantalla.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import controles as C        # noqa: E402
from tfg import datos                 # noqa: E402
from tfg import evaluacion as E       # noqa: E402
from tfg import ventanas as V         # noqa: E402

INICIO, FIN = "2012-01-03", "2026-08-01"
ORIGENES = 12
PRUEBA = 126


def main() -> int:
    t0 = time.time()
    print(f"universo {INICIO} a {FIN}", flush=True)
    tk = datos.miembros_durante(INICIO, FIN)
    r = datos.rendimientos_log(datos.descarga_precios(tk, INICIO, FIN, lote=50))
    print(f"panel: {r.shape[0]} fechas x {r.shape[1]} series\n", flush=True)

    persistencias = []
    for i, part in enumerate(E.origenes_moviles(r.index, ORIGENES, PRUEBA)):
        utiles = V.series_utilizables(r, part)
        panel = r[utiles]
        norm = V.normaliza(panel, V.escala_por_serie(panel, part))
        entrena = norm.loc[norm.index < part.fin_entrena]

        t = time.time()
        g = C.GarchAgrupado().ajusta(entrena)
        p = g.par.persistencia
        persistencias.append(p)
        print(f"  origen {i:2d}  omega={g.par.omega:.6f}  alfa={g.par.alfa:.4f}"
              f"  beta={g.par.beta:.4f}  persistencia={p:.4f}"
              f"   ({(time.time() - t) / 60:.1f} min)", flush=True)

    lo, hi = min(persistencias), max(persistencias)
    print(f"\npersistencia: minimo {lo:.4f}, maximo {hi:.4f}")
    print(f"redondeada a dos cifras: entre {lo:.2f} y {hi:.2f}")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    print("\nEl Anexo C dice «entre 0,90 y 0,96». Comparese con lo de arriba y"
          "\ncorrijase el texto si no coincide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
