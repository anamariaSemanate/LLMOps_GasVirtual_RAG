"""Revisa que los documentos no contengan nombres reales.

La lista de términos prohibidos está en terminos_prohibidos.txt (uno por línea).
Ese archivo es local y no se sube al repositorio (.gitignore), para no publicar
los nombres que se quieren ocultar.

Uso:  python scripts/verificar_anonimizacion.py
"""
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LISTA = RAIZ / "terminos_prohibidos.txt"
CARPETAS = ["documentos", "datos", "app", "scripts", "README.md"]


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def main():
    if not LISTA.exists():
        print("No existe terminos_prohibidos.txt: no se puede verificar.")
        return 1
    terminos = [normalizar(t.strip()) for t in LISTA.read_text(encoding="utf-8").splitlines() if t.strip()]
    hallazgos = []
    for ruta in CARPETAS:
        p = RAIZ / ruta
        archivos = [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix in {".txt", ".csv", ".py", ".md", ".json"}]
        for f in archivos:
            if f.name in {"terminos_prohibidos.txt", "indice.json", "verificar_anonimizacion.py"}:
                continue
            texto = normalizar(f.read_text(encoding="utf-8", errors="ignore"))
            for t in terminos:
                if re.search(rf"\b{re.escape(t)}\b", texto):
                    hallazgos.append(f"{f.relative_to(RAIZ)}: '{t}'")
    if hallazgos:
        print("Se encontraron términos prohibidos:")
        print("\n".join(hallazgos))
        return 1
    print(f"Sin términos prohibidos ({len(terminos)} términos revisados).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
