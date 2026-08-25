"""Comprueba que un fichero .tex no ha sufrido dano por escapes.

Escribir LaTeX con un heredoc de Bash colapsa las barras dobles y convierte
las secuencias de escape en caracteres de control. Ha pasado tres veces en
este proyecto. Ejecutar despues de cada escritura:

    python src/revisa_tex.py memoria/capitulos/*.tex
"""
import io
import sys

CONTROL = {chr(7), chr(8), chr(11), chr(12)}
B = chr(92)


def revisa(ruta: str) -> int:
    lineas = io.open(ruta, encoding="utf-8").read().split("\n")
    fallos = 0

    for i, l in enumerate(lineas, 1):
        d = l.rstrip()
        if d.endswith(B) and not d.endswith(B + B):
            print(f"  {ruta}:{i}  barra suelta al final: ...{d[-40:]!r}")
            fallos += 1
        if any(c in CONTROL for c in l):
            cuales = sorted({hex(ord(c)) for c in l if c in CONTROL})
            print(f"  {ruta}:{i}  caracteres de control {cuales}")
            fallos += 1
        if "**" in l:
            print(f"  {ruta}:{i}  negrita de Markdown en un .tex: ...{d[:60]!r}")
            fallos += 1
    return fallos


if __name__ == "__main__":
    total = sum(revisa(r) for r in sys.argv[1:])
    if total:
        print(f"\n{total} problemas")
        raise SystemExit(1)
    print(f"{len(sys.argv) - 1} ficheros revisados, sin problemas")
