"""Etapa 4: la LSTM global.

Una sola celda recurrente, compartida por todas las series, que recibe una
ventana de K rendimientos y produce los H siguientes valores del objetivo.

Correspondencia con la memoria:

- **Global** quiere decir que los parametros se estiman con todas las series a
  la vez, no que la prediccion de una serie use informacion de otra: en
  inferencia el modelo opera serie a serie.
- El **desenrollado** durante K pasos es lo que convierte el grafo con ciclo
  en uno aciclico y permite entrenar con retropropagacion.
- Se registra la **norma del gradiente** en cada epoca. No es telemetria
  ociosa: el capitulo 6 sostiene que el indicador ingenuo de convergencia es
  falso, porque el entrenamiento no se detiene en un punto critico y la norma
  del gradiente puede crecer durante un entrenamiento que va bien. Guardarla
  permite comprobarlo con nuestros datos en vez de citarlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    """Hiperparametros. Los que se exploran van sobre validacion."""

    unidades: int = 64
    capas: int = 1
    dropout: float = 0.0
    tasa: float = 1e-3
    minilote: int = 256
    epocas_max: int = 100
    paciencia: int = 10          # parada temprana
    recorte: float = 1.0         # recorte de la norma del gradiente
    semilla: int = 0


@dataclass
class Historial:
    """Lo que se guarda de cada entrenamiento."""

    perdida_entrena: list[float] = field(default_factory=list)
    perdida_valida: list[float] = field(default_factory=list)
    norma_gradiente: list[float] = field(default_factory=list)
    mejor_epoca: int = -1

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Historial(epocas={len(self.perdida_entrena)}, "
                f"mejor={self.mejor_epoca}, "
                f"val={self.perdida_valida[self.mejor_epoca]:.5f})"
                if self.perdida_valida else "Historial(vacio)")


class LSTMGlobal(nn.Module):
    """Celda LSTM seguida de una capa densa que produce el horizonte completo.

    La salida es un vector de H componentes producido de una vez (estrategia
    de salida multiple), no una cadena de predicciones realimentadas: asi se
    evita la acumulacion de error a lo largo del horizonte.
    """

    def __init__(self, H: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(input_size=1, hidden_size=cfg.unidades,
                            num_layers=cfg.capas, batch_first=True,
                            dropout=cfg.dropout if cfg.capas > 1 else 0.0)
        self.salida = nn.Linear(cfg.unidades, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:                    # (B, K) -> (B, K, 1)
            x = x.unsqueeze(-1)
        h, _ = self.lstm(x)
        return self.salida(h[:, -1, :])     # solo el ultimo paso

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _semillas(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


def entrena(X_ent: np.ndarray, y_ent: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            cfg: Config | None = None,
            verboso: bool = False) -> tuple[LSTMGlobal, Historial]:
    """Entrena con parada temprana sobre validacion.

    La perdida es el error absoluto medio y no el cuadratico. El motivo esta
    en el capitulo 2: las colas de los rendimientos son pesadas, lo que excluye
    la normal, y el error cuadratico es la log-verosimilitud negativa de un
    modelo gaussiano. Ademas coincide con el numerador del MASE, de modo que
    se optimiza lo mismo que se evalua.
    """
    cfg = cfg or Config()
    _semillas(cfg.semilla)

    H = y_ent.shape[1]
    modelo = LSTMGlobal(H, cfg)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg.tasa)
    perdida = nn.L1Loss()

    Xe = torch.from_numpy(np.asarray(X_ent, dtype=np.float32))
    ye = torch.from_numpy(np.asarray(y_ent, dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))

    hist = Historial()
    mejor, mejor_estado, sin_mejorar = float("inf"), None, 0
    n = len(Xe)

    for epoca in range(cfg.epocas_max):
        modelo.train()
        # Se baraja el ORDEN DE LAS VENTANAS, no el de las observaciones
        # dentro de ellas: es lo que autoriza el corolario de la Proposicion 1.
        orden = torch.randperm(n)
        suma, normas = 0.0, []

        for i in range(0, n, cfg.minilote):
            idx = orden[i:i + cfg.minilote]
            opt.zero_grad()
            p = perdida(modelo(Xe[idx]), ye[idx])
            p.backward()
            normas.append(float(torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), cfg.recorte)))
            opt.step()
            suma += p.detach().item() * len(idx)

        modelo.eval()
        with torch.no_grad():
            v = float(perdida(modelo(Xv), yv))

        hist.perdida_entrena.append(suma / n)
        hist.perdida_valida.append(v)
        # La norma se registra ANTES del recorte: es la que informa sobre la
        # dinamica del entrenamiento, no la que se aplica.
        hist.norma_gradiente.append(float(np.mean(normas)))

        if verboso:
            print(f"    epoca {epoca:3d}  entrena {suma/n:.5f}  "
                  f"valida {v:.5f}  |grad| {np.mean(normas):.3f}")

        if v < mejor - 1e-6:
            mejor, hist.mejor_epoca, sin_mejorar = v, epoca, 0
            mejor_estado = {k: t.clone() for k, t in modelo.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= cfg.paciencia:
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, hist


@torch.no_grad()
def predice(modelo: LSTMGlobal, X: np.ndarray, lote: int = 2048) -> np.ndarray:
    modelo.eval()
    salidas = []
    for i in range(0, len(X), lote):
        t = torch.from_numpy(np.asarray(X[i:i + lote], dtype=np.float32))
        salidas.append(modelo(t).numpy())
    return np.concatenate(salidas).astype(np.float32)


def comprueba_gradiente(H: int = 3, K: int = 8, n: int = 6,
                        eps: float = 1e-4, semilla: int = 0) -> float:
    """Contrasta el gradiente de la retropropagacion con diferencias finitas.

    Es la comprobacion que el Anexo A recomienda y que el Anexo C recoge: se
    entrena con retropropagacion, pero se verifica en algunos casos de prueba
    contra la aproximacion por diferencias centradas, cuyo error es de orden
    eps al cuadrado. Devuelve el error relativo maximo.
    """
    _semillas(semilla)
    cfg = Config(unidades=4, semilla=semilla)
    modelo = LSTMGlobal(H, cfg).double()
    X = torch.randn(n, K, dtype=torch.float64)
    y = torch.randn(n, H, dtype=torch.float64)
    perdida = nn.L1Loss()

    modelo.zero_grad()
    perdida(modelo(X), y).backward()

    peor = 0.0
    for par in modelo.parameters():
        plano, grad = par.data.view(-1), par.grad.view(-1)
        for j in range(0, plano.numel(), max(1, plano.numel() // 4)):
            v = plano[j].item()
            with torch.no_grad():
                plano[j] = v + eps
                mas = float(perdida(modelo(X), y))
                plano[j] = v - eps
                menos = float(perdida(modelo(X), y))
                plano[j] = v
            numerico = (mas - menos) / (2 * eps)
            analitico = float(grad[j])
            denom = max(1e-8, abs(numerico) + abs(analitico))
            peor = max(peor, abs(numerico - analitico) / denom)
    return peor
