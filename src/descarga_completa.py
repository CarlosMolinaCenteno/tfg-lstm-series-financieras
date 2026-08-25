"""Descarga el panel completo del universo historico. Solo hay que ejecutarlo una vez."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import datos

INICIO, FIN = "2012-01-03", "2026-08-01"

if __name__ == "__main__":
    t0 = time.time()
    tk = datos.miembros_durante(INICIO, FIN)
    print(f"{len(tk)} tickers con pertenencia continua", flush=True)
    p = datos.descarga_precios(tk, INICIO, FIN, lote=50, pausa=1.5)
    vacios = [c for c in p.columns if p[c].isna().all()]
    print(f"precios {p.shape} | rango {p.index.min().date()} a {p.index.max().date()}")
    print(f"tickers vacios: {vacios if vacios else 'ninguno'}")
    print(f"huecos totales: {int(p.isna().sum().sum()):,} de {p.size:,} "
          f"({100*p.isna().sum().sum()/p.size:.2f} %)")
    print(f"tiempo {(time.time()-t0)/60:.1f} min")
