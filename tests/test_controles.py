"""Comprobaciones de los experimentos de control de la tarea de magnitud.

Las dos centrales son de identificabilidad: si se generan datos en los que se
conoce la respuesta —un factor de escala exacto, un GARCH con parametros
conocidos— el control tiene que recuperarla. Sin eso, un resultado negativo
del control no distinguiria «no hay efecto» de «el control esta mal».

Ejecutar:  .venv/Scripts/python tests/test_controles.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfg import controles as C  # noqa: E402

FALLOS = []


def comprueba(nombre, condicion, detalle=""):
    print(f"  [{'ok  ' if condicion else 'FALLA'}] {nombre}"
          + (f"   {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def main() -> int:
    rng = np.random.default_rng(0)
    print("Controles de la tarea de magnitud\n")

    # --- escala optima bajo error absoluto ---------------------------------
    p = np.abs(rng.normal(size=(2000, 3))) + 0.1
    comprueba("el factor de escala exacto se recupera",
              abs(C.escala_optima_l1(0.6 * p, p) - 0.6) < 1e-9)

    y = 0.6 * p + rng.normal(scale=0.02, size=p.shape)
    c = C.escala_optima_l1(y, p)
    comprueba("y tambien con ruido simetrico", abs(c - 0.6) < 0.02,
              f"c* = {c:.4f}")

    # El factor optimo tiene que batir a c=1 en el propio criterio.
    err1 = float(np.abs(y - p).mean())
    errc = float(np.abs(y - c * p).mean())
    comprueba("el factor optimo reduce el error absoluto", errc < err1,
              f"{err1:.4f} -> {errc:.4f}")

    # --- mediana frente a media --------------------------------------------
    # Con colas pesadas la mediana de |x| queda claramente por debajo de la
    # media: es el sesgo de localizacion que motiva todo el modulo.
    X = rng.standard_t(4, size=(500, 60))
    med = float(np.median(np.abs(X)))
    mea = float(np.abs(X).mean())
    comprueba("con colas pesadas, mediana(|x|) < media(|x|)", med < 0.85 * mea,
              f"{med:.3f} frente a {mea:.3f}")
    comprueba("mediana_reciente devuelve la forma correcta",
              C.mediana_reciente(X, 5).shape == (500, 5))

    # --- EWMA ---------------------------------------------------------------
    pesos_uno = C.ewma(X, 1, lam=1.0)[:, 0]
    comprueba("con lambda=1 el EWMA coincide con la media plana",
              np.allclose(pesos_uno, np.abs(X).mean(axis=1), atol=1e-5))
    reciente = C.ewma(X, 1, lam=0.5)[:, 0]
    comprueba("con lambda pequeno pesa lo reciente",
              float(np.corrcoef(reciente, np.abs(X[:, -5:]).mean(axis=1))[0, 1])
              > float(np.corrcoef(reciente, np.abs(X[:, :5]).mean(axis=1))[0, 1]))

    # --- minimos absolutos --------------------------------------------------
    # Con contaminacion asimetrica en una fraccion pequena de las filas, el
    # ajuste L1 tiene que quedar mas cerca del coeficiente verdadero que el L2.
    Z = rng.normal(size=(3000, 3))
    beta = np.array([1.0, -2.0, 0.5])
    yy = Z @ beta + rng.normal(scale=0.1, size=3000)
    contaminadas = rng.choice(3000, 150, replace=False)
    yy[contaminadas] += 50.0
    b_l2, *_ = np.linalg.lstsq(Z, yy, rcond=None)
    b_l1 = C._minimos_absolutos(Z, yy)
    e2 = float(np.abs(b_l2 - beta).max())
    e1 = float(np.abs(b_l1 - beta).max())
    comprueba("el ajuste L1 resiste la contaminacion mejor que el L2",
              e1 < e2, f"L1 {e1:.3f} frente a L2 {e2:.3f}")

    # --- HAR ---------------------------------------------------------------
    har = C.HAR().ajusta(X, np.abs(rng.standard_t(4, size=(500, 5))))
    comprueba("HAR usa tres regresores y una constante",
              har.coef.shape[0] == 4)
    comprueba("HAR predice con la forma correcta",
              har.predice(X[:10], 5).shape == (10, 5))

    # --- GARCH -------------------------------------------------------------
    # Serie simulada con parametros conocidos: el ajuste tiene que acercarse.
    T = 6000
    om, al, be = 0.05, 0.10, 0.85
    s2 = om / (1 - al - be)
    r = np.empty(T)
    z = rng.normal(size=T)
    for t in range(T):
        r[t] = np.sqrt(s2) * z[t]
        s2 = om + al * r[t] ** 2 + be * s2
    par = C.ajusta_garch(r)
    comprueba("el GARCH recupera la persistencia",
              abs(par.persistencia - (al + be)) < 0.06,
              f"{par.persistencia:.4f} frente a {al + be:.2f}")
    comprueba("la varianza a largo plazo es del orden correcto",
              0.5 < par.varianza_larga / (om / (1 - al - be)) < 2.0,
              f"{par.varianza_larga:.3f} frente a {om / (1 - al - be):.3f}")

    # Sobre datos sin efecto ARCH, alfa tiene que salir casi nulo.
    par_iid = C.ajusta_garch(rng.normal(size=4000))
    comprueba("sin agrupamiento de volatilidad, alfa es despreciable",
              par_iid.alfa < 0.05, f"alfa = {par_iid.alfa:.4f}")

    # --- prediccion del GARCH ----------------------------------------------
    Xg = r[:5940].reshape(99, 60)
    pred = C.predice_garch(Xg, 10, par)
    comprueba("la prediccion del GARCH es positiva y con la forma correcta",
              pred.shape == (99, 10) and np.all(pred > 0))
    # A horizonte largo, la prediccion converge a la volatilidad de largo plazo.
    lejos = float(np.abs(pred[:, -1] - np.sqrt(par.varianza_larga)).mean())
    cerca = float(np.abs(pred[:, 0] - np.sqrt(par.varianza_larga)).mean())
    comprueba("a mayor horizonte, mas cerca de la volatilidad incondicional",
              lejos < cerca, f"{cerca:.4f} -> {lejos:.4f}")

    # --- GARCH agrupado ----------------------------------------------------
    panel = pd.DataFrame({f"S{i}": r[i::4][:1400] for i in range(4)})
    g = C.GarchAgrupado().ajusta(panel)
    comprueba("el GARCH agrupado se ajusta sobre un panel",
              g.par is not None and 0 < g.par.persistencia < 1)

    # --- aplicacion de escalas ---------------------------------------------
    tickers = np.array(["A"] * 50 + ["B"] * 50)
    base = np.ones((100, 2), dtype=np.float32)
    factores = pd.Series({"A": 2.0, "B": 3.0})
    esc = C.aplica_escala(base, factores, tickers)
    comprueba("la escala por serie se aplica a la serie correcta",
              np.allclose(esc[:50], 2.0) and np.allclose(esc[50:], 3.0))

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
