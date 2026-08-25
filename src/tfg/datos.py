"""Descarga y cache de precios diarios del S&P 500.

Decisiones que implementa (docs/decisiones.md, 2026-08-23):

- Constituyentes **historicos**, no los actuales: usar la lista de hoy
  seleccionaria las empresas que no quebraron ni fueron excluidas, lo que
  sesga la muestra y, sobre todo, sesga la distribucion de la volatilidad
  --que es justo lo que la tarea de magnitud pretende predecir.
- Precios **ajustados** por splits y dividendos. Sin ajustar, un
  desdoblamiento 2:1 aparece como un rendimiento del -50 % en un dia.
- **Cache en disco** en formato parquet. El backend de Yahoo se rompe
  periodicamente; una vez descargados los datos, el resto del trabajo debe
  poder rehacerse sin red.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_CACHE = RAIZ / "data" / "cache"
DIR_RAW = RAIZ / "data" / "raw"

# Tabla de constituyentes actuales y de cambios historicos.
URL_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# Wikipedia rechaza el agente de usuario por defecto de urllib con un 403,
# asi que la descarga se hace con requests y cabecera explicita.
CABECERA = {"User-Agent": "TFG-UCM/1.0 (uso academico; contacto via repositorio)"}


def _asegura_directorios() -> None:
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    DIR_RAW.mkdir(parents=True, exist_ok=True)


def _tablas_wiki() -> list[pd.DataFrame]:
    """Descarga la pagina y devuelve sus tablas."""
    import requests
    respuesta = requests.get(URL_WIKI, headers=CABECERA, timeout=30)
    respuesta.raise_for_status()
    return pd.read_html(io.StringIO(respuesta.text))


def constituyentes_actuales(refrescar: bool = False) -> pd.DataFrame:
    """Tabla de constituyentes **actuales** del S&P 500, con sector.

    Util solo para describir la muestra y agrupar por sector. **No usar para
    definir el universo**: seleccionaria los supervivientes. Para eso esta
    `universo_historico`.
    """
    _asegura_directorios()
    destino = DIR_RAW / "sp500_actuales.parquet"
    if destino.exists() and not refrescar:
        return pd.read_parquet(destino)

    tablas = _tablas_wiki()
    df = tablas[0].rename(columns={"Symbol": "ticker", "Security": "nombre",
                                   "GICS Sector": "sector",
                                   "Date added": "alta"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df.to_parquet(destino)
    return df


def descarga_precios(tickers: list[str], inicio: str, fin: str,
                     lote: int = 50, pausa: float = 1.0,
                     refrescar: bool = False) -> pd.DataFrame:
    """Precios de cierre ajustados, en formato ancho (fechas x tickers).

    Descarga por lotes con pausa entre ellos, para no chocar con las cuotas
    que Yahoo introdujo en 2025. Cachea el resultado.
    """
    import yfinance as yf

    _asegura_directorios()
    clave = f"precios_{inicio}_{fin}_{len(tickers)}t"
    destino = DIR_CACHE / f"{clave}.parquet"
    if destino.exists() and not refrescar:
        return pd.read_parquet(destino)

    trozos = []
    for i in range(0, len(tickers), lote):
        bloque = tickers[i:i + lote]
        datos = yf.download(bloque, start=inicio, end=fin,
                            auto_adjust=True, progress=False,
                            group_by="column", threads=True)
        if datos.empty:
            continue
        cierre = datos["Close"] if "Close" in datos.columns.get_level_values(0) else datos
        trozos.append(cierre)
        if i + lote < len(tickers):
            time.sleep(pausa)

    precios = pd.concat(trozos, axis=1).sort_index()
    precios = precios.loc[:, ~precios.columns.duplicated()]

    # La descarga en paralelo falla de vez en cuando con "database is locked":
    # yfinance usa una cache sqlite que no tolera bien varios hilos. Los que
    # se caigan se reintentan de uno en uno y sin hilos.
    fallidos = [t for t in tickers if t not in precios.columns
                or precios[t].isna().all()]
    for t in fallidos:
        time.sleep(pausa)
        suelto = yf.download(t, start=inicio, end=fin, auto_adjust=True,
                             progress=False, threads=False)
        if not suelto.empty:
            serie = suelto["Close"]
            precios[t] = serie.iloc[:, 0] if hasattr(serie, "columns") else serie

    precios = precios.reindex(columns=sorted(precios.columns))
    precios.to_parquet(destino)
    return precios


def rendimientos_log(precios: pd.DataFrame) -> pd.DataFrame:
    """Rendimientos logaritmicos a partir de precios ajustados."""
    import numpy as np
    return np.log(precios / precios.shift(1)).iloc[1:]


# ---------------------------------------------------------------------------
# Universo historico
# ---------------------------------------------------------------------------
# La tabla de altas y bajas que Wikipedia publicaba en la misma pagina que los
# constituyentes **ya no esta ahi** (comprobado el 2026-08-23: la pagina solo
# devuelve la tabla de miembros actuales). Se usa en su lugar el conjunto
# publicado en el repositorio fja05680/sp500, que da para cada ticker la fecha
# de alta y la de baja del indice.
URL_UNIVERSO = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                "sp500_ticker_start_end.csv")


def universo_historico(refrescar: bool = False) -> pd.DataFrame:
    """Tickers del S&P 500 con sus fechas de alta y baja en el indice.

    Columnas: ticker, start_date, end_date (nula si sigue dentro).
    Es la pieza que evita el sesgo de supervivencia.
    """
    import requests

    _asegura_directorios()
    destino = DIR_RAW / "sp500_universo.parquet"
    if destino.exists() and not refrescar:
        return pd.read_parquet(destino)

    respuesta = requests.get(URL_UNIVERSO, headers=CABECERA, timeout=60)
    respuesta.raise_for_status()
    df = pd.read_csv(io.StringIO(respuesta.text), parse_dates=["start_date", "end_date"])
    # Yahoo escribe las clases de accion con guion y no con punto: BRK.B es
    # BRK-B. Sin esta conversion esos tickers se descargan vacios.
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df.to_parquet(destino)
    return df


def miembros_en(fecha: str, universo: pd.DataFrame | None = None) -> list[str]:
    """Tickers que pertenecian al indice en una fecha dada."""
    if universo is None:
        universo = universo_historico()
    f = pd.Timestamp(fecha)
    dentro = (universo["start_date"] <= f) & (
        universo["end_date"].isna() | (universo["end_date"] > f))
    return sorted(universo.loc[dentro, "ticker"].tolist())


def miembros_durante(inicio: str, fin: str,
                     universo: pd.DataFrame | None = None) -> list[str]:
    """Tickers que pertenecieron al indice en **todo** el intervalo.

    Se exige pertenencia continua para que todas las series tengan la misma
    longitud, que es la homogeneidad que pide el modelo global. El precio a
    pagar es que se pierde parte de la correccion del sesgo de supervivencia:
    quien fue excluido a mitad del intervalo no entra. Alternativa mas fiel y
    mas costosa: admitir series incompletas y tratar los huecos.
    """
    if universo is None:
        universo = universo_historico()
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    dentro = (universo["start_date"] <= a) & (
        universo["end_date"].isna() | (universo["end_date"] >= b))
    return sorted(universo.loc[dentro, "ticker"].tolist())
