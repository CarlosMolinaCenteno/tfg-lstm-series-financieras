"""Ejecuta la rejilla completa sobre el piloto y guarda los resultados."""
import sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import datos, evaluacion as E, modelo as Mo

RAIZ = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    t0 = time.time()
    r = datos.rendimientos_log(datos.descarga_precios(
        datos.miembros_durante("2012-01-03", "2026-01-02")[:30],
        "2019-01-01", "2024-01-01", lote=30))
    print(f"panel {r.shape}", flush=True)

    cfg = Mo.Config(unidades=32, epocas_max=40, paciencia=6)
    tabla = E.ejecuta_rejilla(r, horizontes=(1, 5, 10, 20), K=60,
                              semillas=(0, 1, 2), n_origenes=4,
                              dias_prueba=126, cfg=cfg)
    destino = RAIZ / "resultados" / "rejilla_piloto.csv"
    tabla.to_csv(destino, index=False)
    print(f"\nguardado en {destino}  ({len(tabla)} filas)")
    print(f"tiempo total {(time.time()-t0)/60:.1f} min")
