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


def prueba_rejilla_completa():
    """Ejecuta `ejecuta_rejilla` de verdad, con dos origenes y verboso.

    Existe por un fallo concreto: el bloque que imprime el progreso al
    terminar cada origen usaba `_t` y `_inicio` sin haberlos definido, de modo
    que **cualquier ejecucion real moria al completar el primer origen**. No
    lo detecto ninguna comprobacion porque todas ejercitaban las piezas por
    separado; ninguna llamaba a la funcion que las orquesta.

    La leccion es que las piezas verificadas una a una no garantizan que el
    conjunto funcione. Esta prueba es lenta -- entrena de verdad -- pero es
    barata comparada con descubrirlo en una ejecucion de cuatro horas.
    """
    import pandas as pd
    from tfg import evaluacion as Ev

    rng = np.random.default_rng(0)
    fechas = pd.bdate_range("2018-01-01", periods=900)
    panel = pd.DataFrame(rng.normal(0, 0.01, (900, 4)), index=fechas,
                         columns=[f"S{i}" for i in range(4)])

    from tfg import modelo as Mo
    cfg = Mo.Config(unidades=4, epocas_max=2, paciencia=1)
    tabla = Ev.ejecuta_rejilla(panel, horizontes=(2,), K=10, semillas=(0,),
                               n_origenes=2, dias_prueba=60, cfg=cfg,
                               paso_entrena=10, verboso=True)

    comprueba("ejecuta_rejilla completa varios origenes en modo verboso",
              len(tabla) > 0 and tabla["origen"].nunique() == 2,
              f"{len(tabla)} filas, {tabla['origen'].nunique()} origenes")
    comprueba("la rejilla no deja huecos",
              not tabla["mase_rel_medio"].isna().any())
    return tabla


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
    print("  --- prueba de integracion (entrena de verdad, tarda ~1 min) ---")
    prueba_rejilla_completa()

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


