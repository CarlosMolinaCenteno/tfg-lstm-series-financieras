"""Comprueba que cada cifra de la memoria coincide con los datos.

    python src/revisa_cifras.py

## Por que existe

Los cuadros se regeneran con `analiza_rejilla.py` y `analiza_controles.py`,
pero **la mayoria de las cifras de la memoria no estan en un cuadro**: van
dentro de la prosa. «Entre el 94 y el 97 %», «ocho de doce origenes», «3,82
horas», «105 962 ventanas», «un 42 % peor». Esas son las que nadie revisa y
las que un lector atento caza: el recuento de ejecuciones ya estuvo mal una
vez, 48 donde eran 96, y sobrevivio a varias lecturas.

Este guion cierra ese hueco. Cada afirmacion declara tres cosas: el texto
literal que tiene que aparecer en la memoria, el fichero donde debe estar, y
como se calcula el valor a partir de los datos publicados. Falla si el texto
no aparece -- porque entonces alguien lo reescribio y la comprobacion ya no
vigila nada -- y falla si el numero no cuadra.

No cubre las demostraciones ni el razonamiento: eso no lo puede comprobar
ningun guion. Cubre las cifras, que es donde el error es silencioso.
"""

from __future__ import annotations

import pathlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CAPS = RAIZ / "memoria" / "capitulos"
RES = RAIZ / "resultados"

HORIZONTES = (1, 5, 10, 20)


# --------------------------------------------------------------- datos
def carga():
    rej = pd.read_csv(RES / "rejilla_definitiva.csv")
    ctl = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("controles*.csv"))],
                    ignore_index=True)
    return rej, ctl


def _piv(d, tarea, H, col="mase_rel_medio"):
    x = d[(d["tarea"] == tarea) & (d["H"] == H)]
    return x.pivot_table(index="origen", columns="metodo", values=col,
                         aggfunc="mean")


# ------------------------------------------------- calculos, uno por cifra
def c_estrella(rej, ctl):
    """Rango del factor de escala optimo, promediado por horizonte."""
    x = ctl[ctl["metodo"] == "media reescalada (c global)"]
    m = x.groupby("H")["parametro"].mean()
    return float(m.min()), float(m.max())


def fraccion_reproducida(rej, ctl):
    """Que fraccion de la ganancia de la red reproduce el escalar."""
    fr = []
    for H in HORIZONTES:
        red = _piv(rej, "magnitud", H)["LSTM"].mean()
        esc = _piv(ctl, "magnitud", H)["media reescalada (c global)"].mean()
        fr.append((1 - esc) / (1 - red))
    return min(fr) * 100, max(fr) * 100


def residuo_pp(rej, ctl):
    """Lo que queda por explicar, en puntos porcentuales."""
    d = []
    for H in HORIZONTES:
        red = _piv(rej, "magnitud", H)["LSTM"].mean()
        esc = _piv(ctl, "magnitud", H)["media reescalada (c global)"].mean()
        d.append((esc - red) * 100)
    return min(d), max(d)


def gana_nivel_vs_constante(rej, ctl):
    """Origenes en que la red bate a la constante optima, por horizonte."""
    out = {}
    for H in HORIZONTES:
        a = _piv(rej, "nivel", H)["LSTM"]
        b = _piv(ctl, "nivel", H)["constante optima"]
        comun = a.index.intersection(b.index)
        out[H] = int((a.loc[comun] < b.loc[comun]).sum())
    return out


def crece_gradiente(rej, ctl):
    out = {}
    for tarea in ("nivel", "magnitud"):
        d = rej[(rej["tarea"] == tarea) & (rej["metodo"] == "LSTM")].dropna(
            subset=["grad_inicio", "grad_final"])
        out[tarea] = (int((d["grad_final"] > d["grad_inicio"]).sum()), len(d))
    return out


def epocas(rej, ctl):
    out = {}
    for tarea in ("nivel", "magnitud"):
        d = rej[(rej["tarea"] == tarea) & (rej["metodo"] == "LSTM")]
        out[tarea] = (float(d["epocas"].mean()), float(d["mejor_epoca"].mean()))
    return out


def ingenua_peor(rej, ctl):
    """Cuanto peor que la referencia sale la prediccion ingenua clasica."""
    v = rej[(rej["tarea"] == "nivel") & (rej["metodo"] == "ingenua")]
    return (v["mase_rel_medio"].mean() - 1) * 100


def ventanas_primer_origen(rej, ctl):
    """Ventanas de entrenamiento del primer origen con H=5.

    Depende del horizonte -- 106 233 con H=1 y 105 420 con H=20 -- asi que
    hay que fijar uno; la memoria cita el de H=5.
    """
    d = rej[(rej["origen"] == 0) & (rej["H"] == 5)]
    return int(d["n_entrena"].max())


def persistencia_garch(rej, ctl):
    """Rango de la persistencia del GARCH(1,1) agrupado, origen por origen.

    Se lee de resultados/verifica_garch.log, que produce
    `src/verifica_garch.py`. La pasada original solo la imprimia por
    pantalla y ese registro no se conservo; de ahi el guion aparte.
    """
    log = RES / "verifica_garch.log"
    if not log.exists():
        raise SystemExit("falta resultados/verifica_garch.log; ejecuta antes "
                         "src/verifica_garch.py")
    vals = [float(m) for m in re.findall(r"persistencia=([0-9.]+)",
                                        log.read_text(encoding="utf-8",
                                                      errors="replace"))]
    if not vals:
        raise SystemExit("no encuentro persistencias en verifica_garch.log")
    return min(vals), max(vals)


def gana_vs_referencia(rej, ctl, tarea):
    """Origenes en que la red bate a la referencia de su tarea, por horizonte.

    La referencia es el denominador del MASE relativo, asi que batirla es
    exactamente que la metrica quede por debajo de uno.
    """
    out = {}
    for H in HORIZONTES:
        s = _piv(rej, tarea, H)["LSTM"]
        out[H] = int((s < 1).sum())
    return out


def gana_nivel(rej, ctl):
    return gana_vs_referencia(rej, ctl, "nivel")


def gana_magnitud(rej, ctl):
    return gana_vs_referencia(rej, ctl, "magnitud")


def mejora_magnitud(rej, ctl):
    """Rango de la mejora media de la tarea de magnitud, en porcentaje.

    Es 1 menos el error escalado relativo medio de la red, horizonte por
    horizonte. El minimo (3,95 % con H=1) no llega a cuatro, asi que la cota
    inferior que la memoria declara tiene que quedar por debajo de el.
    """
    v = []
    for H in HORIZONTES:
        v.append((1 - _piv(rej, "magnitud", H)["LSTM"].mean()) * 100)
    return min(v), max(v)


def razon_dispersion(rej, ctl):
    """Rango de la razon entre dispersiones sobre las ocho celdas tarea x H.

    Numerador: desviacion de las medias por origen. Denominador: desviacion
    COMBINADA dentro de los origenes, la raiz de la varianza media -- no la
    media de las desviaciones, que la subestima. Misma convencion que
    `analiza_rejilla.cuadro_varianza`.
    """
    rs = []
    for tarea in ("nivel", "magnitud"):
        for H in HORIZONTES:
            d = rej[(rej["tarea"] == tarea) & (rej["H"] == H)
                    & (rej["metodo"] == "LSTM")]
            g = d.groupby("origen")["mase_rel_medio"]
            entre_sem = float(np.sqrt(g.var(ddof=1).mean()))
            if entre_sem:
                rs.append(g.mean().std(ddof=1) / entre_sem)
    return min(rs), max(rs)


def series_filtro(rej, ctl):
    return sorted(int(x) for x in rej["n_series"].unique())


def comprobaciones(rej, ctl):
    """Ejecuta los ficheros de tests/ y cuenta las comprobaciones que imprimen.

    No vale contar las llamadas a `comprueba()` con una expresion regular:
    dos estan dentro de bucles, y el recuento estatico sale 54 donde en
    ejecucion son 56. Lo que la memoria afirma es lo segundo.
    """
    import subprocess
    exe = RAIZ / ".venv" / "Scripts" / "python.exe"
    if not exe.exists():
        exe = pathlib.Path(sys.executable)
    n = 0
    for f in sorted((RAIZ / "tests").glob("test_*.py")):
        r = subprocess.run([str(exe), str(f)], capture_output=True, timeout=900)
        salida = r.stdout.decode("utf-8", "replace")
        n += salida.count("[ok  ]") + salida.count("[FALLA]")
    return n


# ------------------------------------------------------------ afirmaciones
# (fichero, texto que debe aparecer, calculo, comprobacion, descripcion)
AFIRMACIONES = [
    ("cap6-experimento.tex",
     "El factor vale $c^{*}$ entre 0,73 y 0,75",
     c_estrella,
     lambda v: round(v[0], 2) == 0.73 and round(v[1], 2) == 0.75,
     "rango del factor de escala"),

    ("cap6-experimento.tex",
     "entre el 94 y el\n97\\,\\%",
     fraccion_reproducida,
     lambda v: 93.5 <= v[0] < 94.5 and 96.5 <= v[1] < 97.5,
     "fraccion de la ganancia que reproduce el escalar"),

    ("cap6-experimento.tex",
     "de 0,12 a 0,30 puntos porcentuales",
     residuo_pp,
     lambda v: abs(v[0] - 0.12) < 0.01 and abs(v[1] - 0.30) < 0.01,
     "residuo sin explicar"),

    ("cap6-experimento.tex",
     "en ocho de doce orígenes con\n$H=1$, en nueve con $H=20$ y en diez con los otros dos",
     gana_nivel_vs_constante,
     lambda v: (v[1], v[5], v[10], v[20]) == (4, 2, 2, 3),
     "derrotas de la red frente a la constante optima"),

    ("anexoC-protocolo.tex",
     "Nivel & $0{,}0375$ & $0{,}0988$ & \\textbf{96 de 96}",
     crece_gradiente,
     lambda v: v["nivel"] == (96, 96),
     "el gradiente crece en las 96 ejecuciones de nivel"),

    ("anexoC-protocolo.tex",
     "Magnitud & $0{,}1184$ & $0{,}1292$ & 67 de 96",
     crece_gradiente,
     lambda v: v["magnitud"] == (67, 96),
     "el gradiente crece en 67 de 96 de magnitud"),

    ("anexoC-protocolo.tex",
     "duran de media 13,6 épocas frente a 8,9",
     epocas,
     lambda v: round(v["magnitud"][0], 1) == 13.6
     and round(v["nivel"][0], 1) == 8.9,
     "duracion media del entrenamiento"),

    ("anexoC-protocolo.tex",
     "época $6{,}6$",
     epocas,
     lambda v: round(v["magnitud"][1], 1) == 6.6,
     "mejor epoca en magnitud"),

    ("anexoC-protocolo.tex",
     "sale un 42,5\\,\\% peor que la referencia",
     ingenua_peor,
     lambda v: 42.45 <= v < 42.55,
     "la ingenua clasica frente a la referencia"),

    ("cap6-experimento.tex",
     "las 105\\,962 del\nprimer origen con $H=5$",
     ventanas_primer_origen,
     lambda v: v == 105962,
     "ventanas de entrenamiento del primer origen"),

    ("cap6-experimento.tex",
     "Entre 271 y 272 series\nsuperan el filtro",
     series_filtro,
     lambda v: v == [271, 272],
     "series que superan el filtro"),

    ("anexoC-protocolo.tex",
     "La persistencia estimada queda entre 0,90 y 0,96",
     persistencia_garch,
     lambda v: round(v[0], 2) == 0.90 and round(v[1], 2) == 0.96,
     "persistencia del GARCH agrupado"),

    # --- los resumenes: primeras cifras que se leen, y hasta la revision
    # --- del 2026-09-01 no las vigilaba nadie.
    ("resumen.tex",
     "gana en cinco o seis de doce",
     gana_nivel,
     lambda v: set(v.values()) <= {5, 6},
     "resumen: origenes ganados en la tarea de nivel"),

    ("resumen.tex",
     "en once o doce de los doce",
     gana_magnitud,
     lambda v: set(v.values()) <= {11, 12},
     "resumen: origenes ganados en la tarea de magnitud"),

    ("cap6-experimento.tex",
     "aporta entre el 3,9 y el 4,9 por ciento",
     mejora_magnitud,
     lambda v: v[0] >= 3.9 and v[0] < 4.0 and v[1] <= 4.9 and v[1] > 4.8,
     "capitulo 6: mejora de la tarea de magnitud"),

    ("cap7-conclusiones.tex",
     "y aporta entre el\n3,9 y el 4,9 por ciento",
     mejora_magnitud,
     lambda v: v[0] >= 3.9 and v[0] < 4.0 and v[1] <= 4.9 and v[1] > 4.8,
     "conclusiones: mejora de la tarea de magnitud"),

    ("resumen.tex",
     "mejora entre\nel 3,9 y el 4,9 por ciento",
     mejora_magnitud,
     lambda v: v[0] >= 3.9 and v[0] < 4.0 and v[1] <= 4.9 and v[1] > 4.8,
     "resumen castellano: mejora de la tarea de magnitud"),

    ("resumen.tex",
     "benchmark by 3.9 to 4.9 per cent",
     mejora_magnitud,
     lambda v: v[0] >= 3.9 and v[0] < 4.0 and v[1] <= 4.9 and v[1] > 4.8,
     "resumen ingles: mejora de la tarea de magnitud"),

    ("resumen.tex",
     "resulta de 1,8 a 22,3 veces mayor",
     razon_dispersion,
     lambda v: round(v[0], 1) == 1.8 and round(v[1], 1) == 22.3,
     "resumen castellano: razon entre dispersiones"),

    ("resumen.tex",
     "proves 1.8 to\n22.3 times larger",
     razon_dispersion,
     lambda v: round(v[0], 1) == 1.8 and round(v[1], 1) == 22.3,
     "resumen ingles: razon entre dispersiones"),

    ("cap6-experimento.tex",
     "\\textbf{1,8 a 22,3 veces mayor}",
     razon_dispersion,
     lambda v: round(v[0], 1) == 1.8 and round(v[1], 1) == 22.3,
     "capitulo 6: razon entre dispersiones"),

    ("cap7-conclusiones.tex",
     "es de 1,8 a\n22,3 veces mayor",
     razon_dispersion,
     lambda v: round(v[0], 1) == 1.8 and round(v[1], 1) == 22.3,
     "conclusiones: razon entre dispersiones"),

    ("anexoC-protocolo.tex",
     "El código incluye cincuenta y seis comprobaciones automáticas",
     comprobaciones,
     lambda v: v == 56,
     "numero de comprobaciones automaticas"),
]


def main() -> int:
    rej, ctl = carga()
    fallos = []
    for fichero, texto, calculo, comprueba, desc in AFIRMACIONES:
        p = CAPS / fichero
        cuerpo = p.read_text(encoding="utf-8")
        if texto not in cuerpo:
            print(f"  [TEXTO]  {fichero}: no encuentro {texto!r}")
            fallos.append(desc)
            continue
        valor = calculo(rej, ctl)
        ok = comprueba(valor)
        marca = "ok   " if ok else "FALLA"
        print(f"  [{marca}] {desc}: {valor}")
        if not ok:
            fallos.append(desc)

    print()
    if fallos:
        print(f"{len(fallos)} de {len(AFIRMACIONES)} afirmaciones no cuadran:")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print(f"Las {len(AFIRMACIONES)} afirmaciones numericas comprobadas cuadran "
          "con los datos publicados.")
    print("\nOjo con lo que esto NO comprueba: las demostraciones, los "
          "enunciados\nde los teoremas y el razonamiento. Eso hay que leerlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
