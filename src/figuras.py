"""Genera las figuras de la memoria en PDF vectorial.

    python src/figuras.py

Salida en `memoria/figuras/`. Cuatro figuras:

  fig-acf.pdf        autocorrelacion del rendimiento frente a la de su
                     magnitud, sobre el panel del S&P 500 (datos propios).
                     Es el hecho empirico sobre el que descansa todo el
                     trabajo, y hasta ahora solo estaba contado en palabras.
  fig-tanh.pdf       la tangente hiperbolica y su derivada, con la zona de
                     saturacion marcada.
  fig-gradiente.pdf  fraccion de gradiente que sobrevive con la distancia
                     temporal: red recurrente frente a celda con puertas.
  fig-origenes.pdf   error escalado relativo por origen, las dos tareas.

Criterios: sin color como portador de informacion (se distingue en blanco y
negro), tipografia serif para que case con el documento, y tamano pensado
para una caja de texto de 15 cm.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "memoria" / "figuras"
CACHE = RAIZ / "data" / "cache" / "precios_2012-01-03_2026-08-01_275t.parquet"
REJILLA = RAIZ / "resultados" / "rejilla_definitiva.csv"

ANCHO = 5.9          # pulgadas, algo menos que los 15 cm de caja

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def guarda(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / nombre
    fig.savefig(destino)
    plt.close(fig)
    print("  " + nombre + "  " + str(destino.stat().st_size // 1024) + " kB")


# ---------------------------------------------------------------------
def autocorrelacion(x, retardos):
    """Autocorrelacion muestral hasta `retardos`, ignorando los huecos."""
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    den = (x * x).sum()
    return np.array([(x[k:] * x[:n - k]).sum() / den for k in retardos])


def banda_permutacion(matriz, retardos, repeticiones=400, semilla=0):
    """Banda del 95 % de la ACF **promediada sobre el panel**, por remuestreo.

    La banda de manual, 1,96/raiz(T), es la de UNA serie. Lo que la figura
    dibuja es el promedio de varios cientos de series, cuya banda es mucho
    mas estrecha; pero no tanto como 1,96/raiz(T*S), porque las acciones de
    un mismo indice estan fuertemente correlacionadas por el factor de
    mercado y no aportan informacion independiente.

    La hipotesis nula que interesa es «independencia en el tiempo», no
    «independencia entre series». Se simula aplicando a TODAS las series la
    MISMA permutacion del eje temporal: eso destruye la dependencia serial y
    conserva intacta la matriz de covarianzas transversal, que es
    exactamente el contraste que la figura necesita.

    Devuelve los cuantiles 2,5 % y 97,5 % de la ACF media, retardo a retardo.
    """
    rng = np.random.default_rng(semilla)
    T = matriz.shape[0]
    muestras = np.empty((repeticiones, len(retardos)))
    for b in range(repeticiones):
        Z = matriz[rng.permutation(T)]
        Z = Z - Z.mean(axis=0, keepdims=True)
        den = (Z * Z).sum(axis=0)
        for i, k in enumerate(retardos):
            muestras[b, i] = float(np.mean((Z[k:] * Z[:T - k]).sum(axis=0) / den))
    return (np.quantile(muestras, 0.025, axis=0),
            np.quantile(muestras, 0.975, axis=0))


def fig_acf():
    """Nivel frente a magnitud: el hecho estilizado, con nuestros datos."""
    precios = pd.read_parquet(CACHE)
    rend = np.log(precios).diff().iloc[1:]

    retardos = np.arange(1, 61)
    acf_nivel, acf_magnitud, columnas = [], [], []
    for col in rend.columns:
        serie = rend[col].to_numpy(dtype=float)
        if np.isnan(serie).mean() > 0.5:
            continue
        acf_nivel.append(autocorrelacion(serie, retardos))
        acf_magnitud.append(autocorrelacion(np.abs(serie), retardos))
        columnas.append(col)
    acf_nivel = np.nanmean(np.vstack(acf_nivel), axis=0)
    acf_magnitud = np.nanmean(np.vstack(acf_magnitud), axis=0)
    n_series = len(columnas)
    n_obs = len(rend)

    # Banda correcta para el PROMEDIO, no para una serie suelta. La ingenua,
    # 1,96/raiz(T), es unas quince veces mas ancha y hace invisible el
    # efecto que la propia memoria admite que existe (autocorrelacion de
    # primer orden del orden de 0,03, segun Fama).
    panel = rend[columnas].to_numpy(dtype=float)
    panel = panel[~np.isnan(panel).any(axis=1)]
    inf, sup = banda_permutacion(panel, retardos)
    banda_serie = 1.96 / np.sqrt(n_obs)

    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.fill_between(retardos, inf, sup, color="0.82", zorder=0, linewidth=0)
    ax.plot(retardos, acf_magnitud, color="black", linewidth=1.4,
            label=r"magnitud, $\left|\rho_t\right|$")
    ax.plot(retardos, acf_nivel, color="0.45", linewidth=1.0,
            linestyle="--", label=r"nivel, $\rho_t$")
    ax.set_xlabel("retardo (sesiones)")
    ax.set_ylabel("autocorrelación media")
    ax.set_xlim(1, 60)
    ax.set_ylim(-0.075, 0.25)
    ax.legend(loc="upper right")
    ax.annotate("banda del 95 % de la ACF media, por permutación",
                xy=(40, sup[39]), xytext=(40, -0.065),
                fontsize=7.5, color="0.35", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.5))
    guarda(fig, "fig-acf.pdf")
    print("     (promedio sobre " + str(n_series) + " series, "
          + str(n_obs) + " sesiones; banda del promedio +-"
          + format(float(np.mean(sup)), ".4f")
          + ", banda de una serie +-" + format(banda_serie, ".4f") + ")")
    # El modulo medio de los retardos que salen de la banda es la cifra que
    # el capitulo 2 compara con el 0,03 de Fama; conviene imprimirla aqui
    # para que la afirmacion tenga de donde salir.
    fuera_nivel = (acf_nivel < inf) | (acf_nivel > sup)
    print("     (nivel: retardo 1 = " + format(acf_nivel[0], ".4f")
          + "; fuera de banda en " + str(int(fuera_nivel.sum()))
          + " de " + str(len(retardos)) + " retardos, modulo medio "
          + format(float(np.abs(acf_nivel[fuera_nivel]).mean()), ".4f") + ")")
    print("     (magnitud: fuera de banda en "
          + str(int(((acf_magnitud < inf) | (acf_magnitud > sup)).sum()))
          + " de " + str(len(retardos)) + " retardos)")
    return acf_nivel, acf_magnitud, (inf, sup)


# ---------------------------------------------------------------------
def fig_tanh():
    """La saturacion, que es de donde sale el problema del gradiente."""
    z = np.linspace(-5, 5, 800)
    t = np.tanh(z)
    dt = 1 - t ** 2

    fig, ax = plt.subplots(figsize=(ANCHO, 2.5))
    for corte in (-2, 2):
        ax.axvline(corte, color="0.7", linewidth=0.6, linestyle=":")
    ax.axvspan(-5, -2, color="0.92", zorder=0)
    ax.axvspan(2, 5, color="0.92", zorder=0)
    ax.plot(z, t, color="black", linewidth=1.3, label=r"$\tanh(z)$")
    ax.plot(z, dt, color="0.45", linewidth=1.3, linestyle="--",
            label=r"$\mathrm{d}\tanh(z)/\mathrm{d}z$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.15, 1.35)
    ax.set_xlabel("$z$")
    ax.legend(loc="lower right")
    ax.text(3.5, 1.16, "saturación", fontsize=7.5, color="0.35",
            ha="center")
    ax.text(-3.5, 1.16, "saturación", fontsize=7.5, color="0.35",
            ha="center")
    ax.text(0, 1.16, "región cuasi-lineal", fontsize=7.5, color="0.35",
            ha="center")
    guarda(fig, "fig-tanh.pdf")


# ---------------------------------------------------------------------
def fig_gradiente():
    """Las dos cotas del capitulo 4 y del capitulo 5, una al lado de otra."""
    q = np.arange(0, 61)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(ANCHO, 2.6), sharey=True)

    # Red recurrente: producto de dos factores menores que uno.
    for norma, gris, estilo in ((0.9, "black", "-"),
                                (0.7, "0.4", "--"),
                                (0.5, "0.65", ":")):
        izq.semilogy(q, (norma * 0.9) ** q, color=gris, linewidth=1.2,
                     linestyle=estilo,
                     label=r"$\left\|W_r\right\|=" + format(norma, ".1f") + "$")
    izq.set_title("red recurrente estándar", fontsize=9)
    izq.set_xlabel(r"distancia temporal $l-t$")
    izq.set_ylabel("cota de la fracción")
    izq.legend(loc="lower left", fontsize=7.5)

    # Celda con puertas: producto de puertas de olvido.
    for puerta, gris, estilo in ((1.0, "black", "-"),
                                 (0.99, "0.4", "--"),
                                 (0.95, "0.65", ":")):
        der.semilogy(q, puerta ** q, color=gris, linewidth=1.2,
                     linestyle=estilo,
                     label=r"$f=" + format(puerta, ".2f") + "$")
    der.set_title("celda con puerta de olvido", fontsize=9)
    der.set_xlabel(r"distancia temporal $l-t$")
    der.legend(loc="lower left", fontsize=7.5)

    for ax in (izq, der):
        ax.set_xlim(0, 60)
        ax.set_ylim(1e-12, 2)
    guarda(fig, "fig-gradiente.pdf")


# ---------------------------------------------------------------------
def fig_origenes():
    """El resultado del capitulo 6, origen por origen."""
    r = pd.read_csv(REJILLA)
    r = r[(r["H"] == 5) & (r["metodo"] == "LSTM")]
    piv = r.pivot_table(index="origen", columns="tarea",
                        values="mase_rel_medio", aggfunc="mean")
    fechas = (pd.read_csv(REJILLA)
              .drop_duplicates("origen").set_index("origen")["prueba_desde"])
    etiquetas = [pd.Timestamp(fechas[o]).strftime("%m/%y") for o in piv.index]
    x = np.arange(len(piv))

    fig, ax = plt.subplots(figsize=(ANCHO, 2.7))
    ax.axhline(1.0, color="black", linewidth=0.9)
    ancho = 0.38
    ax.bar(x - ancho / 2, piv["nivel"] - 1, ancho, bottom=1,
           color="0.75", edgecolor="black", linewidth=0.5, label="nivel")
    ax.bar(x + ancho / 2, piv["magnitud"] - 1, ancho, bottom=1,
           color="0.35", edgecolor="black", linewidth=0.5, label="magnitud")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7.5)
    ax.set_xlabel("origen (comienzo del tramo de prueba)")
    ax.set_ylabel("error escalado relativo")
    ax.set_ylim(0.89, 1.02)
    ax.legend(loc="lower left", ncol=2)
    ax.text(11.4, 1.006, "peor que la referencia", fontsize=7.5,
            color="0.35", ha="right", va="bottom")
    guarda(fig, "fig-origenes.pdf")
    print("     (nivel bate en " + str(int((piv["nivel"] < 1).sum()))
          + "/12, magnitud en " + str(int((piv["magnitud"] < 1).sum())) + "/12)")


if __name__ == "__main__":
    print("Generando figuras en " + str(SALIDA))
    fig_acf()
    fig_tanh()
    fig_gradiente()
    fig_origenes()
    print("Listo.")
