"""Experimentos de control de la tarea de magnitud.

    python src/ejecuta_controles.py                  # todo
    python src/ejecuta_controles.py --sin-ablacion   # solo lo barato
    python src/ejecuta_controles.py --origenes 3 --paso 20   # prueba rapida

## Que pregunta responden

El capitulo 6 concluye que la ganancia de la red en la tarea de magnitud
procede de la no linealidad y de la memoria. Esa atribucion descansa en un
unico rival, la regresion agrupada por minimos cuadrados, y hay al menos tres
explicaciones alternativas que ese rival no descarta:

1. **Localizacion.** La referencia es la *media* de |rho| en la ventana, pero
   se entrena y se evalua con *error absoluto*, cuyo optimo es la *mediana*.
   Con colas pesadas, mediana << media, de modo que la referencia esta
   descentrada y cualquier correccion de escala gana sin estructura alguna.
2. **Ponderacion, no memoria.** Batir una media plana de 60 sesiones no exige
   dependencia de largo alcance: basta ponderar.
3. **Apalancamiento.** La red ve rendimientos *con signo*; la referencia y la
   regresion solo ven |rho|. Que los rendimientos negativos anticipen mayor
   volatilidad esta bien documentado, y podria explicar la ventaja entera.

Ademas, ajustar la regresion agrupada por minimos cuadrados la hace estimar
la media condicionada, que es el mismo funcional que estima la referencia:
su empate con ella era previsible y no informa de nada.

Este guion ejecuta los cuatro controles que zanjan el asunto, sobre el mismo
panel, los mismos origenes y la misma metrica que la rejilla definitiva:

  A. Referencias de localizacion: mediana de la ventana, y media reescalada
     por un factor `c` ajustado en entrenamiento (uno global y uno por serie).
  B. Referencias del dominio: EWMA (RiskMetrics y lambda ajustado),
     GARCH(1,1) por serie y HAR de Corsi.
  C. Regresion agrupada ajustada con la **misma** perdida absoluta que la red.
  D. Ablacion de signo: la misma red entrenada con |rho| de entrada. Si la
     ganancia sobrevive, no es apalancamiento.

Salida: un CSV con la misma estructura de columnas que
resultados/rejilla_definitiva.csv, para que las dos tablas se puedan
concatenar. La columna `mase_rel_medio` esta anclada en 1 sobre la MISMA
referencia del capitulo 6, de modo que los numeros son directamente
comparables con los cuadros de ese capitulo.

## Coste, y como se ejecuto en la practica

Las partes A-C no entrenan ninguna red: media hora para las dos tareas. La
ablacion D entrena 96 redes (12 origenes x 4 horizontes x 2 semillas) y
cuesta unas dos horas.

Los resultados publicados se obtuvieron en dos pasadas, por eso hay dos
ficheros:

    python src/ejecuta_controles.py --tareas magnitud --sin-ablacion
        -> resultados/controles_magnitud.csv
    python src/ejecuta_controles.py --tareas nivel --sin-ablacion --sin-garch
        -> resultados/controles_nivel.csv

`src/analiza_controles.py` los lee juntos con la rejilla definitiva. Una sola
pasada sin argumentos produce lo mismo mas la ablacion, en un unico fichero.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg import (baseline as B, controles as C, datos, evaluacion as E,
                 metricas as M, modelo as Mo, ventanas as V)
from tfg.experimento import REFERENCIA

RAIZ = Path(__file__).resolve().parents[1]
INICIO, FIN = "2012-01-03", "2026-08-01"


def _fila(i_org, part, tarea, H, metodo, semilla, valores, mase_ref, n_ent,
          n_pru, extra: dict | None = None) -> dict:
    rel = M.mase_relativo(valores, mase_ref)
    r, rr = M.resumen(valores), M.resumen(rel)
    fila = {"origen": i_org, "prueba_desde": part.fin_valida.date(),
            "tarea": tarea, "H": H, "metodo": metodo, "semilla": semilla,
            "mase_medio": r["media"], "mase_mediano": r["mediana"],
            "mase_rel_medio": rr["media"], "mase_rel_mediano": rr["mediana"],
            "ic95": rr["ic95_semiancho"], "n_series": r["n_series"],
            "n_entrena": n_ent, "n_prueba": n_pru,
            "epocas": None, "mejor_epoca": None,
            "grad_inicio": None, "grad_final": None}
    fila.update(extra or {})
    return fila


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origenes", type=int, default=12)
    p.add_argument("--paso", type=int, default=5)
    p.add_argument("--semillas", type=int, default=2)
    p.add_argument("--K", type=int, default=60)
    p.add_argument("--unidades", type=int, default=32)
    p.add_argument("--horizontes", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--tareas", type=str, nargs="+",
                   default=["magnitud", "nivel"],
                   choices=["magnitud", "nivel"])
    p.add_argument("--sin-ablacion", action="store_true",
                   help="omite la parte D, que es la unica que entrena redes")
    p.add_argument("--sin-garch", action="store_true",
                   help="omite el GARCH, que es lo mas lento de la parte B")
    p.add_argument("--solo-ablacion", action="store_true",
                   help="ejecuta unicamente la parte D, la ablacion de signo, "
                        "que es lo unico que entrena redes")
    p.add_argument("--salida", type=str, default="controles.csv")
    a = p.parse_args()

    destino = RAIZ / "resultados" / a.salida
    destino.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"universo {INICIO} a {FIN}", flush=True)
    tk = datos.miembros_durante(INICIO, FIN)
    r = datos.rendimientos_log(datos.descarga_precios(tk, INICIO, FIN, lote=50))
    print(f"panel: {r.shape[0]} fechas x {r.shape[1]} series", flush=True)

    cfg_base = Mo.Config(unidades=a.unidades, epocas_max=40, paciencia=6)
    particiones = E.origenes_moviles(r.index, a.origenes, 126)
    filas: list[dict] = []

    for i_org, part in enumerate(particiones):
        utiles = V.series_utilizables(r, part)
        panel = r[utiles]
        norm = V.normaliza(panel, V.escala_por_serie(panel, part))
        entrena_panel = norm.loc[norm.index < part.fin_entrena]

        garch = None
        if not a.sin_garch and not a.solo_ablacion:
            t_g = time.time()
            garch = C.GarchAgrupado().ajusta(entrena_panel)
            print(f"  origen {i_org}: GARCH agrupado {garch.par}, "
                  f"persistencia {garch.par.persistencia:.4f}, "
                  f"{(time.time() - t_g) / 60:.1f} min", flush=True)

        for tarea in a.tareas:
            if a.solo_ablacion and tarea != "magnitud":
                continue   # la ablacion de signo solo aplica a la magnitud
            ref_nombre = REFERENCIA[tarea]
            pred_ref = (B.cero if tarea == "nivel" else B.volatilidad_reciente)

            for H in a.horizontes:
                muestras = V.construye_ventanas(norm, a.K, H, tarea,
                                                paso=a.paso)
                tr = V.reparte(muestras, part, H)
                ent, pru = tr["entrena"], tr["prueba"]
                if len(ent) == 0 or len(pru) == 0:
                    continue

                # Denominador: exactamente el mismo que en el capitulo 6.
                escala = M.escala_referencia(ent.y, pred_ref(ent.X, H),
                                             ent.ticker)
                mase_ref = M.mase(pru.y, pred_ref(pru.X, H), pru.ticker,
                                  escala)

                def anota(nombre, pred_pru, semilla=None, extra=None,
                          _t=tarea, _H=H, _p=pru, _e=escala, _r=mase_ref,
                          _n=len(ent)):
                    if a.solo_ablacion and nombre != "LSTM sin signo":
                        return
                    filas.append(_fila(
                        i_org, part, _t, _H, nombre, semilla,
                        M.mase(_p.y, pred_pru, _p.ticker, _e),
                        _r, _n, len(_p), extra))

                anota(ref_nombre, pred_ref(pru.X, H))

                if tarea == "nivel":
                    # Control de localizacion analogo al de magnitud: si la
                    # prediccion nula estuviera descentrada respecto del
                    # optimo de la perdida absoluta, esta constante lo
                    # revelaria. Por simetria, la misma objecion ha de
                    # aplicarse a las dos tareas.
                    k = C.constante_optima_l1(ent.y)
                    anota("constante optima", C.constante(pru.X, H, k),
                          extra={"parametro": float(k)})
                    anota("deriva", B.deriva(pru.X, H))
                    reg2 = B.RegresionAgrupada(absoluto=False).ajusta(
                        ent.X, ent.y)
                    anota("regresion agrupada (L2)", reg2.predice(pru.X, H))
                    reg1 = C.RegresionAgrupadaL1(absoluto=False).ajusta(
                        ent.X, ent.y)
                    anota("regresion agrupada (L1)", reg1.predice(pru.X, H))
                else:
                    pred_ref_ent = pred_ref(ent.X, H)

                    # --- controles de localizacion (A) --------------------
                    anota("mediana de la ventana", C.mediana_reciente(pru.X, H))

                    c_glob = C.escala_optima_l1(ent.y, pred_ref_ent)
                    anota("media reescalada (c global)",
                          C.aplica_escala(pred_ref(pru.X, H), c_glob),
                          extra={"parametro": round(float(c_glob), 4)})

                    c_serie = C.escalas_por_serie_l1(ent.y, pred_ref_ent,
                                                     ent.ticker)
                    anota("media reescalada (c por serie)",
                          C.aplica_escala(pred_ref(pru.X, H), c_serie,
                                          pru.ticker))

                    # --- referencias del dominio (B) ----------------------
                    c_ewma = C.escala_optima_l1(ent.y, C.ewma(ent.X, H, 0.94))
                    anota("EWMA RiskMetrics",
                          C.aplica_escala(C.ewma(pru.X, H, 0.94), c_ewma),
                          extra={"parametro": round(float(c_ewma), 4)})

                    lam = C.ajusta_lambda(ent.X, ent.y)
                    c_lam = C.escala_optima_l1(ent.y, C.ewma(ent.X, H, lam))
                    anota("EWMA ajustado",
                          C.aplica_escala(C.ewma(pru.X, H, lam), c_lam),
                          extra={"parametro": round(float(lam), 4)})

                    har = C.HAR(l1=True).ajusta(ent.X, ent.y)
                    anota("HAR", har.predice(pru.X, H))

                    if garch is not None:
                        c_g = C.escala_optima_l1(ent.y,
                                                 garch.predice(ent.X, H))
                        # `parametro` guarda el factor de calibracion L1, y
                        # `persistencia` la del propio GARCH, que es lo que
                        # el Anexo C cita. Antes solo se imprimia.
                        anota("GARCH(1,1) agrupado",
                              C.aplica_escala(garch.predice(pru.X, H), c_g),
                              extra={"parametro": round(float(c_g), 4),
                                     "persistencia": round(
                                         float(garch.par.persistencia), 4)})

                    # --- regresion agrupada, las dos perdidas (C) ---------
                    reg2 = B.RegresionAgrupada(absoluto=True).ajusta(
                        ent.X, ent.y)
                    anota("regresion agrupada (L2)", reg2.predice(pru.X, H))
                    reg1 = C.RegresionAgrupadaL1(absoluto=True).ajusta(
                        ent.X, ent.y)
                    anota("regresion agrupada (L1)", reg1.predice(pru.X, H))

                    # --- ablacion de signo (D) ----------------------------
                    if not a.sin_ablacion and len(tr["valida"]):
                        val = tr["valida"]
                        for s in range(a.semillas):
                            cfg = Mo.Config(
                                **{**cfg_base.__dict__, "semilla": s})
                            red, hist = Mo.entrena(np.abs(ent.X), ent.y,
                                                   np.abs(val.X), val.y, cfg)
                            anota("LSTM sin signo",
                                  Mo.predice(red, np.abs(pru.X)), semilla=s,
                                  extra={
                                      "epocas": len(hist.perdida_entrena),
                                      "mejor_epoca": hist.mejor_epoca,
                                      "grad_inicio": hist.norma_gradiente[0],
                                      "grad_final": hist.norma_gradiente[-1]})

                ultimo = filas[-1]
                print(f"  origen {i_org} | {tarea:8s} | H={H:2d} | "
                      f"{ultimo['metodo']:26s} "
                      f"rel {ultimo['mase_rel_medio']:.4f}", flush=True)

        pd.DataFrame(filas).to_csv(destino, index=False)
        hechos, total = i_org + 1, len(particiones)
        transcurrido = (time.time() - t0) / 60
        print(f"  -- origen {hechos}/{total} completo | "
              f"{transcurrido:.0f} min | "
              f"quedan ~{transcurrido / hechos * (total - hechos):.0f} min --",
              flush=True)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(destino, index=False)
    print(f"\nguardado en {destino}  ({len(tabla)} filas)")
    print(f"tiempo total {(time.time() - t0) / 3600:.2f} h")

    for tarea in a.tareas:
        sub = tabla[tabla["tarea"] == tarea]
        if sub.empty:
            continue
        print(f"\nresumen de {tarea} (media sobre origenes, por horizonte):")
        print(sub.pivot_table(index="metodo", columns="H",
                              values="mase_rel_medio",
                              aggfunc="mean").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
