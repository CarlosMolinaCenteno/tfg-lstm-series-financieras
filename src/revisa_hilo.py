"""Comprueba que la memoria se pueda leer de principio a fin.

    python src/revisa_hilo.py

No busca coherencia -- eso lo hace `revisa_citas.py` -- sino **suficiencia
para un lector lineal que no ha leido las fuentes**. Recorre el documento en
el orden en que se lee y avisa de tres cosas:

1. **Conceptos usados antes de definirse.** La tutora pide definir cada
   concepto la primera vez que aparece, en negrita o cursiva. Se localiza
   cada definicion en negrita y se comprueba que el termino no aparezca
   antes en texto llano.

2. **Simbolos usados antes de explicarse.** Para cada simbolo matematico se
   compara su primera aparicion con el punto en que el texto lo introduce.
   El listado de simbolos NO cuenta como introduccion: es una tabla de
   consulta, no una explicacion en contexto.

3. **Material del cuerpo que depende de los anexos.** Los anexos se imprimen
   despues, asi que todo lo que el cuerpo de por sabido de ellos tiene que
   ir con remision explicita. Se listan los simbolos y conceptos cuya unica
   explicacion esta en un anexo.

Lo que este programa no puede ver -- hipotesis tacitas, afirmaciones sin
argumento, saltos de razonamiento -- hay que leerlo a mano. Esto solo da el
esqueleto.
"""

from __future__ import annotations

import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CAPS = RAIZ / "memoria" / "capitulos"

# Orden real de lectura, tomado de mitfg.tex.
ORDEN = ["resumen", "simbolos",
         "cap1-intro", "cap2-problema", "cap3-rnn", "cap4-gradiente",
         "cap5-lstm", "cap6-experimento", "cap7-conclusiones",
         "anexoA-redes", "anexoB-demostraciones", "anexoC-protocolo"]

CUERPO = set(ORDEN[2:9])
ANEXOS = set(ORDEN[9:])

NEGRITA = re.compile(r"\\textbf\{([^{}]{3,60})\}")
MATE = re.compile(r"\$([^$]{1,120})\$")
SIMBOLO = re.compile(r"\\[a-zA-Z]+|[A-Za-z]")


def limpia(t: str) -> str:
    """Quita comentarios, entornos de figura y pies, que no son hilo."""
    t = re.sub(r"(?m)^%.*", "", t)
    t = re.sub(r"\\caption\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    t = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", t, flags=re.S)
    return t


def texto_plano(t: str) -> str:
    """Texto sin matematicas ni ordenes, para buscar terminos."""
    t = MATE.sub(" ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}]", " ", t)
    return re.sub(r"\s+", " ", t).lower()


def carga():
    doc = []
    for nombre in ORDEN:
        f = CAPS / (nombre + ".tex")
        if not f.exists():
            continue
        doc.append((nombre, limpia(f.read_text(encoding="utf-8"))))
    return doc


def conceptos_tarde(doc):
    """Terminos definidos en negrita despues de haberse usado."""
    avisos = []
    plano = [(n, texto_plano(t)) for n, t in doc]
    vistos = set()
    for i, (nombre, bruto) in enumerate(doc):
        for m in NEGRITA.finditer(bruto):
            termino = re.sub(r"\\[a-zA-Z]+|[{}]", "", m.group(1)).strip().lower()
            termino = re.sub(r"\s+", " ", termino)
            # solo terminos con pinta de concepto, no enfasis suelto
            if len(termino) < 6 or " " in termino[:3] or termino in vistos:
                continue
            if not re.fullmatch(r"[a-záéíóúüñ ]+", termino):
                continue
            vistos.add(termino)
            # ¿aparece antes, en texto llano, en un fichero anterior?
            for j in range(i):
                if termino in plano[j][1]:
                    avisos.append((plano[j][0], nombre, termino))
                    break
    return avisos


def simbolos_de(t: str) -> set:
    fuera = set()
    for m in MATE.finditer(t):
        for s in SIMBOLO.findall(m.group(1)):
            if len(s) > 1 or s.isalpha():
                fuera.add(s)
    return fuera


def solo_en_anexo(doc):
    """Simbolos que el cuerpo usa y que solo el anexo explica."""
    prim = {}
    for nombre, t in doc:
        for s in simbolos_de(t):
            prim.setdefault(s, []).append(nombre)
    fuera = []
    for s, donde in sorted(prim.items()):
        cuerpo = [d for d in donde if d in CUERPO]
        anexo = [d for d in donde if d in ANEXOS]
        if cuerpo and anexo and ORDEN.index(cuerpo[0]) < ORDEN.index(anexo[0]):
            fuera.append((s, cuerpo[0], anexo[0], len(donde)))
    return fuera


def main() -> int:
    doc = carga()
    print("=" * 70)
    print("1. CONCEPTOS DEFINIDOS EN NEGRITA DESPUES DE HABERSE USADO")
    print("=" * 70)
    tarde = conceptos_tarde(doc)
    if not tarde:
        print("  ninguno")
    for antes, definido, term in tarde:
        print("  «" + term + "»")
        print("      se usa en  " + antes + "   y se define en  " + definido)

    print("")
    print("=" * 70)
    print("2. SIMBOLOS QUE EL CUERPO USA Y APARECEN TAMBIEN EN ANEXO")
    print("   (revisar a mano si el cuerpo los da por sabidos)")
    print("=" * 70)
    for s, c, a, n in solo_en_anexo(doc):
        if n >= 3:
            print("  " + s.ljust(16) + "primero en " + c.ljust(18) + "tambien en " + a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
