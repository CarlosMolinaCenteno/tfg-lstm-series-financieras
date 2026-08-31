"""Mide el desfase real entre la pagina del PDF y el folio impreso.

No se fia de los desfases anotados: los lee del propio documento. Para
varias paginas del PDF busca el numero de pagina impreso -- primera o
ultima linea con un numero suelto plausible -- y calcula la moda de la
diferencia.
"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess

RAIZ = pathlib.Path(__file__).resolve().parent.parent

PDFS = {
    "Bishop": "libros/Bishop - Pattern Recognition And Machine Learning - Springer 2006 (1).pdf",
    "Goodfellow": "libros/Bengio, Yoshua_ Courville, Aaron_ Goodfellow, Ian J - Deep learning_ adaptive computation and machine learning-The MIT Press (2016).pdf",
    "Sherstinsky": "papers/Sherstinsky.pdf",
    "Cybenko": "papers/Cybenko_Approx_Superpositions_Sigm.pdf",
    "Hochreiter-Schmidhuber": "papers/Hochreiter-Schmidhuber.pdf",
    "Cont": "papers/Cont, R. (2001), Empirical properties of asset returns stylized facts and statistical issues, Quantitative Finance 1.pdf",
    "Fama-1970": "papers/Fama-EfficientCapitalMarkets-1970.pdf",
    "Fama-1991": "papers/The Journal of Finance - December 1991 - FAMA - Efficient Capital Markets  II.pdf",
    "Higham": "papers/Catherine-F-Higham_Desmond-J-Higham.pdf",
    "Hewamalage": "papers/Hewamalage, Bergmeier and Bandara .pdf",
}

SUELTO = re.compile(r"^\s*(\d{1,4})\s*$")


def folios(paginas: list[str]) -> collections.Counter:
    """Diferencia folio_impreso - pagina_pdf, contada sobre todo el PDF."""
    dif = collections.Counter()
    for i, txt in enumerate(paginas, start=1):
        lineas = [l for l in txt.splitlines() if l.strip()]
        if not lineas:
            continue
        for l in (lineas[0], lineas[-1]):
            m = SUELTO.match(l)
            if m:
                dif[int(m.group(1)) - i] += 1
                break
        else:
            # Folio pegado a un encabezado: "412 REVIEW" o "REVIEW 412"
            for l in (lineas[0], lineas[-1]):
                t = l.strip().split()
                for cand in (t[:1], t[-1:]):
                    if cand and cand[0].isdigit() and len(cand[0]) <= 4:
                        dif[int(cand[0]) - i] += 1
                        break
                else:
                    continue
                break
    return dif


def main() -> None:
    print("  fuente                    desfase  (folio impreso - pagina pdf)")
    print("  " + "-" * 66)
    for clave, rel in PDFS.items():
        ruta = RAIZ / "raw" / rel
        if not ruta.exists():
            print("  " + clave.ljust(24) + "  PDF no encontrado")
            continue
        r = subprocess.run(["pdftotext", "-layout", str(ruta), "-"],
                           capture_output=True, timeout=300)
        pags = r.stdout.decode("utf-8", "replace").split("\f")
        d = folios(pags)
        if not d:
            print("  " + clave.ljust(24) + "  sin folios legibles")
            continue
        top = d.most_common(3)
        principal, veces = top[0]
        total = sum(d.values())
        print("  " + clave.ljust(24) + str(principal).rjust(7)
              + "   (" + str(veces) + "/" + str(total) + " paginas"
              + ("; alternativas " + ", ".join(str(v) for v, _ in top[1:])
                 if len(top) > 1 else "") + ")")


if __name__ == "__main__":
    main()
