"""Comprobaciones de la etapa 4.

La central es la verificacion del gradiente por diferencias finitas, que es
la que el Anexo A recomienda y el Anexo C recoge.

Ejecutar:  .venv/Scripts/python tests/test_modelo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfg import modelo as Mo  # noqa: E402

FALLOS = []


def comprueba(nombre, condicion, detalle=""):
    print(f"  [{'ok  ' if condicion else 'FALLA'}] {nombre}"
          + (f"   {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def main():
    print("ETAPA 4 — comprobaciones\n")

    err = Mo.comprueba_gradiente()
    comprueba("la retropropagacion coincide con las diferencias finitas",
              err < 1e-5, f"error relativo maximo {err:.2e}")

    rng = np.random.default_rng(0)
    K, H, n = 30, 3, 900
    X = rng.normal(0, 1, (n, K)).astype(np.float32)
    # Objetivo deliberadamente facil: el ultimo valor de la ventana. La
    # comprobacion es que el entrenamiento reduce la perdida de forma clara,
    # no que la arquitectura sea potente.
    y = np.repeat(X[:, -1:], H, axis=1).astype(np.float32)
    cfg = Mo.Config(unidades=16, epocas_max=80, paciencia=12, semilla=0)
    red, hist = Mo.entrena(X[:700], y[:700], X[700:], y[700:], cfg)

    comprueba("la salida tiene forma (n, H)",
              Mo.predice(red, X).shape == (n, H))
    comprueba("el entrenamiento reduce claramente la perdida",
              hist.perdida_valida[hist.mejor_epoca] < 0.3 * hist.perdida_valida[0],
              f"{hist.perdida_valida[0]:.4f} -> {hist.perdida_valida[hist.mejor_epoca]:.4f}")
    comprueba("la parada temprana devuelve los pesos de la mejor epoca",
              hist.mejor_epoca == int(np.argmin(hist.perdida_valida)))
    comprueba("se registra la norma del gradiente en cada epoca",
              len(hist.norma_gradiente) == len(hist.perdida_entrena))

    red2, _ = Mo.entrena(X[:700], y[:700], X[700:], y[700:], cfg)
    comprueba("misma semilla, mismo resultado",
              np.allclose(Mo.predice(red, X), Mo.predice(red2, X), atol=1e-6))

    cfg3 = Mo.Config(unidades=16, epocas_max=80, paciencia=12, semilla=7)
    red3, _ = Mo.entrena(X[:700], y[:700], X[700:], y[700:], cfg3)
    comprueba("semilla distinta, resultado distinto",
              not np.allclose(Mo.predice(red, X), Mo.predice(red3, X), atol=1e-6))

    print()
    if FALLOS:
        print(f"FALLAN {len(FALLOS)}: " + ", ".join(FALLOS))
        return 1
    print("Todas las comprobaciones pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
