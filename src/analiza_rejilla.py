"""Reconstruye los cuadros del capitulo 6 a partir de la rejilla definitiva.

    python src/analiza_rejilla.py

Lee resultados/rejilla_definitiva.csv y reproduce, en el mismo orden en que
aparecen en la memoria:

  1. Cuadro 1  -- error escalado relativo por horizonte y tarea.
  2. Cuadro 2  -- contrastes de Wilcoxon y de Diebold-Mariano.
  3. Cuadro 3  -- resultado origen por origen (Anexo C).
  4. Cuadro 6  -- norma del gradiente al principio y al final (Anexo C).
  5. Cuadro 7  -- dispersion entre origenes frente a entre semillas (Anexo C).

Existia hasta ahora un hueco de reproducibilidad: `tfg/evaluacion.py` tenia
las funciones que calculan todo esto -- `tabla_por_horizonte`,
`tabla_por_origen`, `contrasta_lstm`, `diebold_mariano` -- pero ningun guion
las llamaba, de modo que las cifras de la memoria no se podian regenerar con
una orden. Este fichero cierra ese hueco: lo que imprime es exactamente lo
que esta impreso en la memoria.

Los controles de atribucion van aparte, en `src/analiza_controles.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import evaluacion as ev          # noqa: E402
from tfg.experimento import REFERENCIA    # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
REJILLA = RAIZ / "resultados" / "rejilla_definitiva.csv"

HORIZONTES = (1, 5, 10, 20)
TAREAS = ("nivel", "magnitud")


def carga() -> pd.DataFrame:
    if not REJILLA.exists():
        raise SystemExit(f"no encuentro {REJILLA}; ejecuta antes "
                         "src/ejecuta_definitivo.py")
    return pd.read_csv(REJILLA)


def _por_origen(rej: pd.DataFrame, tarea: str, H: int,
                columna: str = "mase_rel_medio") -> pd.DataFrame:
    """Una fila por origen, promediando las semillas."""
    d = rej[(rej["tarea"] == tarea) & (rej["H"] == H)]
    return d.pivot_table(index="origen", columns="metodo",
                         values=columna, aggfunc="mean")


# ------------------------------------------------------- cuadro 1
def cuadro_resultado(rej: pd.DataFrame) -> None:
    print("\nCUADRO 1 -- error escalado relativo por horizonte y tarea")
    print("  (media y mediana son ENTRE SERIES; la desviacion, ENTRE ORIGENES)")
    print(f"  {'H':>3}  {'nivel: media':>13} {'mediana':>9} {'desv.':>7}"
          f"   {'magnitud: media':>16} {'mediana':>9} {'desv.':>7}")
    for H in HORIZONTES:
        fila = [f"  {H:>3}"]
        for tarea in TAREAS:
            d = rej[(rej["tarea"] == tarea) & (rej["H"] == H)
                    & (rej["metodo"] == "LSTM")]
            por_org = d.groupby("origen")["mase_rel_medio"].mean()
            fila.append(f"  {d['mase_rel_medio'].mean():>13.4f}"
                        f" {d['mase_rel_mediano'].mean():>9.4f}"
                        f" {por_org.std(ddof=1):>7.4f}")
        print("".join(fila))


# ------------------------------------------------------- cuadro 2
def cuadro_contrastes(rej: pd.DataFrame) -> None:
    n_comp = len(TAREAS) * len(HORIZONTES)      # correccion de Bonferroni
    print(f"\nCUADRO 2 -- contrastes (Bonferroni sobre {n_comp} comparaciones)")
    print(f"  {'tarea':>9} {'H':>3} {'p (W)':>9} {'p corr.':>9} {'p (DM)':>9}"
          f" {'gana':>6} {'mejora med.':>12}")
    for tarea in TAREAS:
        ref = REFERENCIA[tarea]
        for H in HORIZONTES:
            piv = _por_origen(rej, tarea, H)
            if "LSTM" not in piv or ref not in piv:
                continue
            w = ev.wilcoxon_pareado(piv["LSTM"], piv[ref], n_comp)
            dm = ev.diebold_mariano(piv["LSTM"], piv[ref], h=H)
            gana = int((piv["LSTM"] < piv[ref]).sum())
            mejora = float((piv[ref] - piv["LSTM"]).median())
            print(f"  {tarea:>9} {H:>3} {w['p']:>9.5f} {w['p_corregido']:>9.4f}"
                  f" {dm['p']:>9.4f} {gana:>4}/{len(piv):<2}"
                  f" {mejora:>+12.4f}")


# ------------------------------------------------------- cuadro 3
def cuadro_origenes(rej: pd.DataFrame, H: int = 5) -> None:
    print(f"\nCUADRO 3 -- resultado origen por origen (H={H})")
    print(f"  {'origen':>6} {'prueba desde':>13} {'nivel':>8} {'magnitud':>9}")
    fechas = (rej.groupby("origen")["prueba_desde"].first())
    columnas = {}
    for tarea in TAREAS:
        piv = _por_origen(rej, tarea, H)
        columnas[tarea] = piv["LSTM"]
    gana = {t: 0 for t in TAREAS}
    for org in sorted(fechas.index):
        fila = f"  {org:>6} {fechas[org]:>13}"
        for tarea in TAREAS:
            v = columnas[tarea][org]
            fila += f" {v:>8.4f}" if tarea == "nivel" else f" {v:>9.4f}"
            if v < 1.0:
                gana[tarea] += 1
        print(fila)
    print(f"  {'bate a su referencia':>20}   "
          f"{gana['nivel']:>2} de {len(fechas)}    {gana['magnitud']:>2} de {len(fechas)}")


# ------------------------------------------------------- cuadro 6
def cuadro_gradiente(rej: pd.DataFrame) -> None:
    print("\nCUADRO 6 -- norma del gradiente al principio y al final")
    print(f"  {'tarea':>9} {'primera epoca':>14} {'ultima epoca':>13}"
          f" {'crece en':>10} {'epocas':>7} {'mejor epoca':>12}")
    for tarea in TAREAS:
        d = rej[(rej["tarea"] == tarea) & (rej["metodo"] == "LSTM")].dropna(
            subset=["grad_inicio", "grad_final"])
        crece = int((d["grad_final"] > d["grad_inicio"]).sum())
        print(f"  {tarea:>9} {d['grad_inicio'].mean():>14.4f}"
              f" {d['grad_final'].mean():>13.4f}"
              f" {crece:>6} de {len(d):<3}"
              f" {d['epocas'].mean():>7.1f} {d['mejor_epoca'].mean():>12.1f}")


# ------------------------------------------------------- cuadro 7
def cuadro_varianza(rej: pd.DataFrame) -> None:
    print("\nCUADRO 7 -- dispersion entre origenes frente a entre semillas")
    print(f"  {'H':>3}  {'nivel: desv.':>13} {'razon':>7}"
          f"   {'magnitud: desv.':>16} {'razon':>7}")
    for H in HORIZONTES:
        fila = [f"  {H:>3}"]
        for tarea in TAREAS:
            d = rej[(rej["tarea"] == tarea) & (rej["H"] == H)
                    & (rej["metodo"] == "LSTM")]
            g = d.groupby("origen")["mase_rel_medio"]
            # Entre origenes: desviacion de la media sobre semillas.
            entre_org = g.mean().std(ddof=1)
            # Entre semillas: desviacion COMBINADA dentro de los origenes, esto
            # es, la raiz de la varianza media. No es lo mismo que la media de
            # las desviaciones -- por Jensen esta queda por debajo, y usarla
            # inflaria la razon en torno a un 30 % en estos datos.
            entre_sem = float(np.sqrt(g.var(ddof=1).mean()))
            razon = entre_org / entre_sem if entre_sem else np.nan
            fila.append(f"  {entre_org:>13.4f} {razon:>7.1f}")
        print("".join(fila))


def main() -> int:
    rej = carga()
    print(f"{REJILLA.name}: {len(rej)} filas, "
          f"{rej['origen'].nunique()} origenes, "
          f"{sorted(rej['metodo'].unique())}")
    cuadro_resultado(rej)
    cuadro_contrastes(rej)
    cuadro_origenes(rej)
    cuadro_gradiente(rej)
    cuadro_varianza(rej)
    print("\nEstos cinco cuadros son los que la memoria imprime. Los controles"
          "\nde atribucion se reconstruyen con src/analiza_controles.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
