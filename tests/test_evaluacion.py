"""Comprobaciones de la etapa 5.

Ejecutar:  .venv/Scripts/python tests/test_evaluacion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfg import evaluacion as E  # noqa: E402

FALLOS = []


def comprueba(nombre, condicion, detalle=""):
    print(f"  [{'ok  ' if condicion else 'FALLA'}] {nombre}"
          + (f"   {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def main():
    print("ETAPA 5 — comprobaciones\n")
    fechas = pd.bdate_range("2015-01-01", periods=1500)

    ps = E.origenes_moviles(fechas, n_origenes=4, dias_prueba=126)
    comprueba("se generan los origenes pedidos", len(ps) == 4)
    comprueba("los origenes van de mas antiguo a mas reciente",
              all(ps[i].fin_valida < ps[i + 1].fin_valida
                  for i in range(len(ps) - 1)))
    comprueba("en cada origen: entrena < valida < prueba",
              all(p.fin_entrena < p.fin_valida < p.fin_prueba for p in ps))
    comprueba("el entrenamiento crece con el origen",
              all(ps[i].fin_entrena < ps[i + 1].fin_entrena
                  for i in range(len(ps) - 1)))
    comprueba("las ventanas de prueba no se solapan",
              all(ps[i].fin_prueba <= ps[i + 1].fin_valida
                  for i in range(len(ps) - 1)))
    comprueba("no se generan mas origenes de los que caben",
              len(E.origenes_moviles(fechas[:400], n_origenes=10,
                                     dias_prueba=126)) < 10)

    # --- contrastes ---------------------------------------------------------
    rng = np.random.default_rng(0)
    idx = [f"S{i}" for i in range(30)]
    ref = pd.Series(rng.normal(1.0, 0.1, 30), index=idx)

    igual = E.wilcoxon_pareado(ref.copy(), ref)
    comprueba("sin diferencia, el contraste no rechaza", igual["p"] >= 0.99)

    mejor = ref - 0.15
    r = E.wilcoxon_pareado(mejor, ref)
    comprueba("con mejora sistematica, rechaza",
              r["p"] < 1e-4 and r["gana_en"] == 30,
              f"p={r['p']:.1e}, mejora mediana {r['mejora_mediana']:.3f}")

    ruido = ref + pd.Series(rng.normal(0, 0.1, 30), index=idx)
    comprueba("con solo ruido, no rechaza",
              E.wilcoxon_pareado(ruido, ref)["p"] > 0.05)

    corr = E.wilcoxon_pareado(mejor, ref, n_comparaciones=20)
    comprueba("Bonferroni multiplica el p por el numero de comparaciones",
              np.isclose(corr["p_corregido"], min(1.0, r["p"] * 20)))

    f = E.friedman({"a": ref, "b": mejor, "c": ruido})
    comprueba("Friedman detecta que los tres metodos difieren",
              f["p"] < 0.01 and f["k_metodos"] == 3, f"p={f['p']:.1e}")

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
