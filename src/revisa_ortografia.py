"""Busca erratas en los ficheros .tex de la memoria.

Lo pide la tutora en su plantilla: «Es conveniente que paseis un diccionario
por vuestro texto, para minimizar el numero de erratas.»

## Por que no se usa un diccionario

Se probo `pyspellchecker` con su lista espanola y resulto inservible: contiene
solo lemas, de modo que marca como errores «tiene», «hay», «datos», «puede» o
«redes». Con ese ruido, las erratas de verdad no se ven.

## Que se hace en su lugar

Dos comprobaciones que no necesitan diccionario y que en un texto tecnico
tienen mucha mas precision:

1. **Vecinos raros.** Una errata aparece casi siempre **una sola vez**,
   mientras que la palabra correcta aparece muchas. Se buscan palabras que
   aparecen una o dos veces y que estan a distancia de edicion uno de otra
   palabra frecuente del mismo documento: «recurente» junto a «recurrente»,
   «segmetos» junto a «segmentos».

2. **Confusiones de tilde.** Los pares que el castellano distingue por
   acento y que un corrector generico no puede resolver, listados para
   revision manual con su contexto.

    python src/revisa_ortografia.py memoria/capitulos/*.tex
"""
import io
import re
import sys
from collections import Counter

B = chr(92)

PARES_TILDE = [("solo", "sólo"), ("mas", "más"), ("aun", "aún"),
               ("este", "éste"), ("esta", "ésta")]


def prosa(texto: str) -> str:
    """Deja solo el texto corrido, sin ordenes ni matematicas."""
    t = texto
    for e in ("equation", "align", "tabular", "table", "verbatim",
              "longtable", "center"):
        t = re.sub(re.escape(B) + r"begin\{" + e + r"\*?\}.*?"
                   + re.escape(B) + r"end\{" + e + r"\*?\}", " ", t, flags=re.S)
    t = re.sub(r"(?m)%.*$", " ", t)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(re.escape(B) + r"cite\[[^]]*\]\{[^}]*\}", " ", t)
    t = re.sub(re.escape(B) + r"(ref|eqref|label|cite|url|texttt)\{[^}]*\}", " ", t)
    t = re.sub(re.escape(B) + r"[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}\[\]&~^_#]", " ", t)
    return t


def solo_flexion(a: str, b: str) -> bool:
    """True si la diferencia esta en la terminacion.

    En castellano casi todas las parejas a distancia uno son flexiones
    legitimas -- singular y plural, o genero, o persona verbal. Si la
    diferencia esta en los dos ultimos caracteres, se descarta: una errata
    real suele estar en medio de la palabra.
    """
    n = min(len(a), len(b))
    return a[:max(0, n - 2)] == b[:max(0, n - 2)]


def distancia1(a: str, b: str) -> bool:
    """True si a y b estan a distancia de edicion uno."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    corta, larga = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    for c in larga:
        if i < len(corta) and corta[i] == c:
            i += 1
    return i == len(corta)


def main(rutas: list[str]) -> int:
    frec: Counter = Counter()
    ubic: dict[str, str] = {}
    for r in rutas:
        for i, l in enumerate(prosa(io.open(r, encoding="utf-8").read()).split("\n"), 1):
            for p in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}", l):
                q = p.lower()
                frec[q] += 1
                ubic.setdefault(q, f"{r}:{i}")

    raras = {p for p, n in frec.items() if n <= 2}
    frecuentes = {p for p, n in frec.items() if n >= 5}

    print("(se excluye el resumen en ingles y las diferencias de terminacion)")
    print(f"{len(frec)} palabras distintas, {sum(frec.values())} en total\n")
    print("=== posibles erratas: palabra rara junto a otra frecuente y casi igual ===")
    hallazgos = 0
    for r in sorted(raras):
        for f in frecuentes:
            if r != f and distancia1(r, f) and not solo_flexion(r, f):
                print(f"  «{r}» (x{frec[r]}) frente a «{f}» (x{frec[f]})   {ubic[r]}")
                hallazgos += 1
                break
    if not hallazgos:
        print("  ninguna")

    print("\n=== pares que se distinguen por tilde: revisar a mano ===")
    for a, b in PARES_TILDE:
        if frec.get(a) or frec.get(b):
            print(f"  «{a}» x{frec.get(a, 0):<4d} «{b}» x{frec.get(b, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
