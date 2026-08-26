"""Comprueba que un fichero .tex no ha sufrido dano por secuencias de escape.

Escribir LaTeX desde un heredoc de Bash, o parchearlo con cadenas de Python
que no sean literales, colapsa las barras dobles y **se come la primera letra
de las ordenes que empiezan por una letra de escape**: `\ref` se convierte en
un retorno de carro seguido de `ef`, `\textbf` en un tabulador seguido de
`extbf`, y asi con `\b`, `\a`, `\f` y `\v`.

El dano es silencioso: LaTeX compila y el texto sale mal, o ni siquiera se
nota hasta que alguien lee el PDF. Ha ocurrido cuatro veces en este proyecto.

    python src/revisa_tex.py memoria/capitulos/*.tex memoria/mitfg.tex
"""
import io
import re
import sys

B = chr(92)

# Caracteres de control que nunca deben aparecer en el fuente.
CONTROL = {7: "\a", 8: "\b", 9: "\t", 11: "\v", 12: "\f", 13: "\r"}

# Restos de ordenes a las que se les ha comido la primera letra.
MUTILADAS = [
    ("ef{",      "\ref"),      ("extbf{",   "\textbf"),
    ("exttt{",   "\texttt"),   ("extit{",   "\textit"),
    ("egin{",    "\\begin"),   ("bel{",     "\\label"),
    ("race",     "\brace"),    ("ightarrow", "\rightarrow"),
    ("ho_",      "\rho"),      ("imes",     "\times"),
]


def revisa(ruta: str) -> int:
    texto = io.open(ruta, encoding="utf-8").read()
    fallos = 0

    for i, linea in enumerate(texto.split("\n"), 1):
        etiqueta = f"  {ruta}:{i}"

        # 1. caracteres de control
        for c in linea:
            if ord(c) in CONTROL:
                print(f"{etiqueta}  caracter de control {CONTROL[ord(c)]!r} "
                      f"-> ...{linea.strip()[:60]!r}")
                fallos += 1
                break

        # 2. ordenes mutiladas: el resto aparece sin su barra delante
        for resto, orden in MUTILADAS:
            for m in re.finditer(r'(?<![A-Za-z' + re.escape(B) + r'])' + re.escape(resto), linea):
                print(f"{etiqueta}  parece «{orden}» mutilado: "
                      f"...{linea[max(0, m.start()-25):m.end()+15].strip()!r}")
                fallos += 1

        # 3. barra suelta al final (separador de fila colapsado)
        d = linea.rstrip()
        if d.endswith(B) and not d.endswith(B + B):
            print(f"{etiqueta}  barra suelta al final: ...{d[-45:]!r}")
            fallos += 1

        # 4. porcentaje dentro de modo matematico: babel-spanish lo redefine
        #    y produce "Incompatible glue units", que aborta la compilacion.
        import re as _re
        for m in _re.finditer(r"\$[^$]*" + _re.escape(B) + r"%[^$]*\$", linea):
            print(f"{etiqueta}  % dentro de modo matematico: {m.group(0)!r}")
            fallos += 1

        # 5. negrita o cursiva de Markdown coladas en un .tex
        if "**" in linea:
            print(f"{etiqueta}  negrita de Markdown: ...{d[:60]!r}")
            fallos += 1

    return fallos


if __name__ == "__main__":
    total = sum(revisa(r) for r in sys.argv[1:])
    if total:
        print(f"\n{total} problemas")
        raise SystemExit(1)
    print(f"{len(sys.argv) - 1} ficheros revisados, sin problemas")
