"""Audita las citas de la memoria contra las fuentes de `raw/`.

    python src/revisa_citas.py [-v]

Tres comprobaciones:

1. **Estructura.** Toda clave citada existe en `ref.bib` y viceversa; toda
   cita lleva localizador.

2. **Anclas de pagina.** Cada pagina citada existe en el PDF de la fuente,
   aplicando el desfase PDF-libro propio de cada una.

3. **Citas textuales.** La comprobacion que de verdad vale. Las citas de la
   memoria estan *traducidas* al castellano, asi que buscar el texto en la
   pagina no funciona: no hay palabras que comparar. Lo que se hace es una
   prueba **relativa**. Se puntua cada pagina de la fuente por su parecido
   con el fragmento -- cifras, simbolos, nombres propios y raices comunes
   entre las dos lenguas, que son lo que sobrevive a la traduccion -- y se
   comprueba que **la pagina mejor puntuada sea la citada**. Da igual que
   la puntuacion absoluta sea baja: lo que importa es que ninguna otra
   pagina se parezca mas. Si la ganadora esta a mas de una pagina de la
   citada, la cita queda marcada para mirarla a mano.

Las fuentes sin PDF paginado se declaran y se saltan.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unicodedata

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CAPS = sorted((RAIZ / "memoria" / "capitulos").glob("*.tex"))
BIB = RAIZ / "memoria" / "ref.bib"

# clave -> (ruta relativa a raw/, desfase pagina_pdf - pagina_citada)
FUENTES = {
    "Bishop": ("libros/Bishop - Pattern Recognition And Machine Learning - Springer 2006 (1).pdf", 20),
    "Goodfellow": ("libros/Bengio, Yoshua_ Courville, Aaron_ Goodfellow, Ian J - Deep learning_ adaptive computation and machine learning-The MIT Press (2016).pdf", 23),
    "Sherstinsky": ("papers/Sherstinsky.pdf", 0),
    "Cybenko": ("papers/Cybenko_Approx_Superpositions_Sigm.pdf", -302),
    "Hochreiter-Schmidhuber": ("papers/Hochreiter-Schmidhuber.pdf", -1734),
    "Cont": ("papers/Cont, R. (2001), Empirical properties of asset returns stylized facts and statistical issues, Quantitative Finance 1.pdf", -222),
    "Fama-1970": ("papers/Fama-EfficientCapitalMarkets-1970.pdf", -381),
    "Fama-1991": ("papers/The Journal of Finance - December 1991 - FAMA - Efficient Capital Markets  II.pdf", -1574),
    "Higham": ("papers/Catherine-F-Higham_Desmond-J-Higham.pdf", -859),
    "Hewamalage": ("papers/Hewamalage, Bergmeier and Bandara .pdf", -387),
}

SIN_PDF = {
    "Hochreiter-1998": "PDF de preimpresion, paginacion distinta; se cita por seccion",
    "Hyndman-Koehler": "PDF de preimpresion, paginacion distinta; se cita por seccion",
    "Guilhoto": "trabajo sin publicar, sin paginacion estable",
    "Werbos-2006": "sin PDF; se cita a la obra entera, sin localizador",
    "Rudin": "libro fuera de raw/; teoremas 1.34, 5.19 y 6.19 verificados a mano",
}

# Citas a la obra entera, sin localizador a proposito.
GLOBALES = {("anexoA-redes.tex", "Bishop"), ("anexoA-redes.tex", "Goodfellow"),
            ("anexoA-redes.tex", "Higham"), ("anexoA-redes.tex", "Guilhoto"),
            ("cap4-gradiente.tex", "Werbos-2006")}

CITA = re.compile(r"\\cite(?:\[([^\]]*)\])?\{([^}]+)\}")
COMILLAS = re.compile(r"«([^»]{25,})»", re.S)
PAGINA = re.compile(r"pp?\.~(\d+)")

VACIAS = {
    "de", "la", "el", "los", "las", "un", "una", "que", "en", "y", "o", "del",
    "al", "se", "es", "son", "con", "por", "para", "no", "lo", "su", "sus",
    "como", "mas", "pero", "esta", "este", "esa", "ese", "ser", "hay", "sobre",
    "entre", "cuando", "si", "ya", "muy", "todo", "todos", "toda", "todas",
    "the", "of", "a", "an", "to", "is", "are", "in", "on", "and", "or", "for",
    "that", "this", "it", "be", "as", "by", "with", "not", "we", "can",
}

_docs: dict[str, list[str]] = {}


def normaliza(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t.lower())


def paginas(clave: str) -> list[str]:
    """Texto normalizado de cada pagina del PDF, indexado desde 1."""
    if clave not in _docs:
        ruta = RAIZ / "raw" / FUENTES[clave][0]
        if not ruta.exists():
            _docs[clave] = []
        else:
            r = subprocess.run(["pdftotext", str(ruta), "-"],
                               capture_output=True, timeout=300)
            bruto = r.stdout.decode("utf-8", "replace").split("\f")
            _docs[clave] = [""] + [normaliza(p) for p in bruto]
    return _docs[clave]


def raices(fragmento: str) -> list[str]:
    """Trozos del fragmento que sobreviven a la traduccion.

    Cifras y simbolos tal cual; y de las palabras largas, su raiz de cinco
    letras, que es lo que suelen compartir el castellano y el ingles en el
    vocabulario tecnico (gradiente/gradient, convexa/convex, recurrente/
    recurrent, eficiencia/efficiency...).
    """
    fuera = []
    for p in normaliza(fragmento).split():
        if p in VACIAS:
            continue
        if p.isdigit():
            fuera.append(p)
        elif len(p) >= 7:
            fuera.append(p[:5])
    return sorted(set(fuera))


def mejor_pagina(clave: str, marcas: list[str]) -> tuple[int, int, int]:
    """(pagina que mas coincide, aciertos, total). Paginas del PDF."""
    docs = paginas(clave)
    mejor, cuantos = 0, -1
    for i, texto in enumerate(docs):
        if i == 0 or not texto:
            continue
        n = sum(1 for m in marcas if m in texto)
        if n > cuantos:
            mejor, cuantos = i, n
    return mejor, cuantos, len(marcas)


def main() -> int:
    verboso = "-v" in sys.argv
    claves_bib = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,",
                                BIB.read_text(encoding="utf-8")))
    citadas: set[str] = set()
    sin_loc: list[str] = []
    fuera_rango: list[str] = []
    ok = revisar = inconcluso = 0
    avisos: list[str] = []

    for f in CAPS:
        texto = re.sub(r"(?m)^%.*", "", f.read_text(encoding="utf-8"))
        nom = f.name

        for m in CITA.finditer(texto):
            loc, claves = m.group(1), m.group(2)
            for c in (k.strip() for k in claves.split(",")):
                citadas.add(c)
                if not loc and (nom, c) not in GLOBALES:
                    sin_loc.append(nom + ":" + str(texto[:m.start()].count(chr(10)) + 1)
                                   + "  " + c)
                if loc and c in FUENTES:
                    docs = paginas(c)
                    for pg in PAGINA.findall(loc):
                        p = int(pg) + FUENTES[c][1]
                        if not (1 <= p < len(docs)):
                            fuera_rango.append(nom + "  " + c + " p. " + pg)

        for m in COMILLAS.finditer(texto):
            mc = CITA.search(texto[m.end():m.end() + 320])
            if not mc or not mc.group(1):
                continue
            clave = mc.group(2).split(",")[0].strip()
            pgs = PAGINA.findall(mc.group(1))
            if clave not in FUENTES or not pgs:
                inconcluso += 1
                continue
            marcas = raices(m.group(1))
            linea = texto[:m.start()].count(chr(10)) + 1
            sitio = nom + ":" + str(linea)
            if len(marcas) < 4:
                inconcluso += 1
                continue
            citada = int(pgs[0]) + FUENTES[clave][1]
            gana, n, tot = mejor_pagina(clave, marcas)
            if n <= 1:
                inconcluso += 1
                if verboso:
                    avisos.append("  [flojo]  " + sitio + "  " + clave
                                  + " p. " + pgs[0] + "  (solo " + str(n)
                                  + "/" + str(tot) + " marcas)")
            elif abs(gana - citada) <= 1:
                ok += 1
            else:
                revisar += 1
                avisos.append(
                    "  [REVISAR] " + sitio + "  " + clave + " cita p. " + pgs[0]
                    + " pero encaja mejor la pag. PDF " + str(gana)
                    + " (libro " + str(gana - FUENTES[clave][1]) + "), "
                    + str(n) + "/" + str(tot) + " marcas\n              «"
                    + " ".join(m.group(1).split())[:88] + "…»")

    print("=" * 68)
    print("1. ESTRUCTURA")
    print("=" * 68)
    huerf = sorted(citadas - claves_bib)
    print("  claves citadas " + str(len(citadas)) + " | entradas en ref.bib "
          + str(len(claves_bib)))
    print("  citadas sin entrada:  " + (", ".join(huerf) if huerf else "ninguna"))
    sobra = sorted(claves_bib - citadas)
    print("  en ref.bib sin citar: " + (", ".join(sobra) if sobra else "ninguna"))
    print("  citas sin localizador: " + (str(len(sin_loc)) if sin_loc else "ninguna"))
    for s in sin_loc:
        print("      " + s)

    print("")
    print("=" * 68)
    print("2. ANCLAS DE PAGINA")
    print("=" * 68)
    print("  paginas fuera del rango del PDF: " + str(len(fuera_rango)))
    for s in fuera_rango:
        print("      " + s)
    for c, motivo in sorted(SIN_PDF.items()):
        if c in citadas:
            print("  [se salta] " + c + ": " + motivo)

    print("")
    print("=" * 68)
    print("3. CITAS TEXTUALES: LA PAGINA CITADA ES LA QUE MEJOR ENCAJA")
    print("=" * 68)
    print("  encajan (mejor pagina = citada, +/- 1):  " + str(ok))
    print("  hay que revisarlas a mano:               " + str(revisar))
    print("  sin marcas suficientes para decidir:     " + str(inconcluso))
    for s in avisos:
        print(s)

    return 1 if (huerf or fuera_rango or revisar) else 0


if __name__ == "__main__":
    sys.exit(main())
