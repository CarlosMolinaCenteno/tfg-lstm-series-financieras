"""Ejecucion definitiva: panel completo, 12 origenes moviles, rejilla de horizontes.

Un solo comando. Descarga los datos si no estan cacheados, ejecuta la rejilla
y guarda los resultados en resultados/rejilla_definitiva.csv.

    python src/ejecuta_definitivo.py

Parametros por defecto medidos en `estima_coste.py`: unas 6 horas en una
maquina de 4 hilos. Con mas nucleos, proporcionalmente menos.

Se puede reducir el alcance para una prueba rapida:

    python src/ejecuta_definitivo.py --origenes 3 --paso 10
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import datos, evaluacion as E, modelo as Mo

RAIZ = Path(__file__).resolve().parents[1]
INICIO, FIN = "2012-01-03", "2026-08-01"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origenes", type=int, default=12)
    p.add_argument("--paso", type=int, default=5,
                   help="submuestreo de ventanas de entrenamiento")
    p.add_argument("--semillas", type=int, default=2)
    p.add_argument("--K", type=int, default=60, help="ventana de entrada")
    p.add_argument("--unidades", type=int, default=32)
    p.add_argument("--salida", type=str, default="rejilla_definitiva.csv")
    a = p.parse_args()

    t0 = time.time()
    print(f"universo {INICIO} a {FIN}", flush=True)
    tk = datos.miembros_durante(INICIO, FIN)
    r = datos.rendimientos_log(datos.descarga_precios(tk, INICIO, FIN, lote=50))
    print(f"panel: {r.shape[0]} fechas x {r.shape[1]} series", flush=True)

    cfg = Mo.Config(unidades=a.unidades, epocas_max=40, paciencia=6)
    tabla = E.ejecuta_rejilla(
        r, horizontes=(1, 5, 10, 20), K=a.K,
        semillas=tuple(range(a.semillas)),
        n_origenes=a.origenes, dias_prueba=126,
        cfg=cfg, paso_entrena=a.paso)

    destino = RAIZ / "resultados" / a.salida
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    print(f"\nguardado en {destino}  ({len(tabla)} filas)")
    print(f"tiempo total {(time.time() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
