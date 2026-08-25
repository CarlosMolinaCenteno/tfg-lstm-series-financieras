"""Etapa 5: origen movil, rejilla de horizontes y contrastes.

Tres piezas, y cada una responde a una advertencia concreta de la
bibliografia:

- **Origen movil.** Fama (1991) atribuye buena parte de la autocorrelacion
  que encontro a largo plazo a un unico episodio historico, la Gran
  Depresion. Evaluar sobre un solo tramo de prueba deja el resultado a merced
  del regimen de ese tramo -- y en estos datos el regimen cambia, como se vio
  en la etapa 2. Promediar sobre varios origenes es lo que lo evita.

- **Rejilla de horizontes.** El horizonte **no se ajusta, se reporta**.
  Cambiar el horizonte no cambia el modelo: cambia el problema. Probar varios
  y comunicar el mejor seria elegir la pregunta que mejor se responde, que es
  el rastreo de datos que Fama describe. Se fija la rejilla de antemano y se
  publican todos los valores.

- **Contrastes.** Sin ellos, comparar dos medias no significa nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .ventanas import Particion


def origenes_moviles(fechas: pd.DatetimeIndex, n_origenes: int = 4,
                     dias_prueba: int = 126, dias_valida: int = 126
                     ) -> list[Particion]:
    """Genera particiones con ventana de entrenamiento creciente.

    Cada origen reserva `dias_prueba` habiles para evaluar y los
    `dias_valida` inmediatamente anteriores para la parada temprana; el
    entrenamiento es todo lo anterior, y crece a medida que el origen avanza.

    Se devuelven ordenadas de mas antigua a mas reciente.
    """
    n = len(fechas)
    particiones = []
    for k in range(n_origenes):
        fin_p = n - k * dias_prueba
        ini_p = fin_p - dias_prueba
        ini_v = ini_p - dias_valida
        if ini_v <= dias_valida:
            break
        particiones.append(Particion(
            fin_entrena=fechas[ini_v],
            fin_valida=fechas[ini_p],
            fin_prueba=fechas[min(fin_p, n - 1)]))
    return list(reversed(particiones))


# ---------------------------------------------------------------------------
# Contrastes
# ---------------------------------------------------------------------------
def wilcoxon_pareado(modelo: pd.Series, referencia: pd.Series,
                     n_comparaciones: int = 1) -> dict[str, float]:
    """Contraste de rangos con signo, pareado serie a serie.

    Es el que corresponde al comparar **dos** metodos sobre el mismo conjunto
    de series: no parametrico, que es lo prudente con distribuciones de colas
    pesadas, y pareado, porque cada serie da una observacion de cada metodo.

    `n_comparaciones` aplica la correccion de Bonferroni: al hacer muchas
    comparaciones, el nivel de significacion se divide entre su numero.
    """
    comun = modelo.index.intersection(referencia.index)
    a, b = modelo.loc[comun].to_numpy(), referencia.loc[comun].to_numpy()
    dif = a - b
    if np.allclose(dif, 0):
        return {"p": 1.0, "p_corregido": 1.0, "n": len(comun),
                "mejora_mediana": 0.0, "gana_en": 0}
    est, p = stats.wilcoxon(a, b, alternative="less")
    return {"p": float(p),
            "p_corregido": float(min(1.0, p * n_comparaciones)),
            "n": int(len(comun)),
            "mejora_mediana": float(np.median(b - a)),
            "gana_en": int((a < b).sum())}


def friedman(resultados: dict[str, pd.Series]) -> dict[str, float]:
    """Contraste de Friedman de rangos, para comparar varios metodos a la vez.

    Precede a cualquier comparacion por pares: si no rechaza, no hay
    diferencias que explorar.
    """
    nombres = list(resultados)
    comun = resultados[nombres[0]].index
    for n in nombres[1:]:
        comun = comun.intersection(resultados[n].index)
    matrices = [resultados[n].loc[comun].to_numpy() for n in nombres]
    est, p = stats.friedmanchisquare(*matrices)
    return {"estadistico": float(est), "p": float(p),
            "k_metodos": len(nombres), "n_series": int(len(comun))}


# ---------------------------------------------------------------------------
# Ejecucion completa
# ---------------------------------------------------------------------------
def ejecuta_rejilla(rendimientos: pd.DataFrame, horizontes: tuple[int, ...],
                    K: int, tareas: tuple[str, ...] = ("nivel", "magnitud"),
                    semillas: tuple[int, ...] = (0, 1, 2),
                    n_origenes: int = 4, dias_prueba: int = 126,
                    cfg=None, paso_entrena: int = 1,
                    verboso: bool = True) -> pd.DataFrame:
    """Entrena y evalua en toda la rejilla origen x horizonte x tarea x semilla.

    Devuelve una fila por combinacion, con el MASE de cada metodo y el
    relativo a la referencia de la tarea. **Se devuelve todo**: la seleccion
    posterior de que mostrar es una decision de redaccion, no de calculo, y
    debe poder auditarse.
    """
    from . import baseline as B, metricas as M, modelo as Mo, ventanas as V
    from .experimento import REFERENCIA

    cfg_base = cfg or Mo.Config()
    particiones = origenes_moviles(rendimientos.index, n_origenes, dias_prueba)
    filas = []

    for i_org, part in enumerate(particiones):
        utiles = V.series_utilizables(rendimientos, part)
        panel = rendimientos[utiles]
        norm = V.normaliza(panel, V.escala_por_serie(panel, part))
        for tarea in tareas:
            preds = (B.PREDICTORES_NIVEL if tarea == "nivel"
                     else B.PREDICTORES_MAGNITUD)
            ref = REFERENCIA[tarea]
            for H in horizontes:
                # El submuestreo se aplica a todo por coherencia: entrenar
                # con una de cada `paso` ventanas y evaluar con todas mezclaria
                # dos regimenes de muestreo distintos en la misma tabla.
                muestras = V.construye_ventanas(norm, K, H, tarea,
                                                paso=paso_entrena)
                tr = V.reparte(muestras, part, H)
                ent, val, pru = tr["entrena"], tr["valida"], tr["prueba"]
                if len(ent) == 0 or len(val) == 0 or len(pru) == 0:
                    continue

                escala = M.escala_referencia(ent.y, preds[ref](ent.X, H),
                                             ent.ticker)
                mase_ref = M.mase(pru.y, preds[ref](pru.X, H), pru.ticker,
                                  escala)

                # Referencias: no dependen de la semilla.
                base = {n: M.mase(pru.y, f(pru.X, H), pru.ticker, escala)
                        for n, f in preds.items()}
                reg = B.RegresionAgrupada(absoluto=(tarea == "magnitud"))
                reg.ajusta(ent.X, ent.y)
                base["regresion agrupada"] = M.mase(
                    pru.y, reg.predice(pru.X), pru.ticker, escala)

                for nombre, v in base.items():
                    filas.append(_fila(i_org, part, tarea, H, nombre, None,
                                       v, mase_ref, len(ent), len(pru)))

                for s in semillas:
                    c = Mo.Config(**{**cfg_base.__dict__, "semilla": s})
                    red, hist = Mo.entrena(ent.X, ent.y, val.X, val.y, c)
                    v = M.mase(pru.y, Mo.predice(red, pru.X), pru.ticker,
                               escala)
                    f = _fila(i_org, part, tarea, H, "LSTM", s, v, mase_ref,
                              len(ent), len(pru))
                    f["epocas"] = len(hist.perdida_entrena)
                    f["mejor_epoca"] = hist.mejor_epoca
                    f["grad_inicio"] = hist.norma_gradiente[0]
                    f["grad_final"] = hist.norma_gradiente[-1]
                    filas.append(f)

                if verboso:
                    ult = filas[-1]
                    print(f"  origen {i_org} | {tarea:9s} | H={H:2d} | "
                          f"LSTM rel {ult['mase_rel_medio']:.4f}")

    return pd.DataFrame(filas)


def _fila(i_org, part, tarea, H, metodo, semilla, valores, mase_ref,
          n_ent, n_pru) -> dict:
    from . import metricas as M
    rel = M.mase_relativo(valores, mase_ref)
    r, rr = M.resumen(valores), M.resumen(rel)
    return {"origen": i_org, "prueba_desde": part.fin_valida.date(),
            "tarea": tarea, "H": H, "metodo": metodo, "semilla": semilla,
            "mase_medio": r["media"], "mase_mediano": r["mediana"],
            "mase_rel_medio": rr["media"], "mase_rel_mediano": rr["mediana"],
            "ic95": rr["ic95_semiancho"], "n_series": r["n_series"],
            "n_entrena": n_ent, "n_prueba": n_pru}


def tabla_por_horizonte(rejilla: pd.DataFrame, tarea: str,
                        metodo: str = "LSTM") -> pd.DataFrame:
    """Resume la rejilla: una fila por horizonte, promediando origenes y semillas.

    Es la tabla que contrasta con nuestros datos la afirmacion de Fama de que
    la fraccion de varianza explicada crece con el horizonte.
    """
    d = rejilla[(rejilla["tarea"] == tarea) & (rejilla["metodo"] == metodo)]
    g = d.groupby("H")["mase_rel_medio"]
    out = pd.DataFrame({
        "MASE rel. medio": g.mean(),
        "desv. entre ejecuciones": g.std(ddof=1),
        "min": g.min(), "max": g.max(),
        "n ejecuciones": g.size(),
    })
    return out.round(4)


def tabla_por_origen(rejilla: pd.DataFrame, tarea: str, H: int) -> pd.DataFrame:
    """Una fila por origen: comprueba que el resultado no depende del periodo.

    Es la respuesta directa a la advertencia de Fama sobre atribuir a un
    hallazgo lo que en realidad produce un unico episodio historico.
    """
    d = rejilla[(rejilla["tarea"] == tarea) & (rejilla["H"] == H)]
    piv = d.pivot_table(index=["origen", "prueba_desde"], columns="metodo",
                        values="mase_rel_medio", aggfunc="mean")
    return piv.round(4)


def contrasta_lstm(rejilla: pd.DataFrame, tarea: str, H: int,
                   n_comparaciones: int = 1) -> dict[str, float]:
    """Contrasta la LSTM contra la referencia, pareando por origen.

    Cada origen aporta una observacion de cada metodo, promediada sobre
    semillas. Es el contraste conservador: no trata cada serie como
    independiente, que seria inflar la muestra.
    """
    from .experimento import REFERENCIA
    ref = REFERENCIA[tarea]
    d = rejilla[(rejilla["tarea"] == tarea) & (rejilla["H"] == H)]
    piv = d.pivot_table(index="origen", columns="metodo",
                        values="mase_rel_medio", aggfunc="mean")
    if "LSTM" not in piv or ref not in piv:
        return {}
    return wilcoxon_pareado(piv["LSTM"], piv[ref], n_comparaciones)
