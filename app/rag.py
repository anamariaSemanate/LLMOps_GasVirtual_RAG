import os
import json
import time
from pathlib import Path

import numpy as np
import requests
import mlflow

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO_EMBEDDING = "nomic-embed-text"
RUTA_DOCUMENTOS = Path(os.getenv("RUTA_DOCUMENTOS", "/documentos"))
RUTA_INDICE = Path(os.getenv("RUTA_INDICE", "/datos/indice.json"))


# 1. Leer archivos TXT (incluye subcarpetas: consumo, diagnostico, bitacora)
def leer_documentos():
    documentos = []
    for archivo in sorted(RUTA_DOCUMENTOS.rglob("*.txt")):
        documentos.append({
            "fuente": archivo.relative_to(RUTA_DOCUMENTOS).as_posix(),
            "texto": archivo.read_text(encoding="utf-8")
        })
    return documentos


# 2. Dividir textos con solapamiento
def dividir_texto(texto, tamano=500, solape=80):
    if tamano <= solape:
        raise ValueError("El tamaño debe superar el solape")
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        fragmentos.append(texto[inicio:fin])
        if fin == len(texto):
            break
        inicio = fin - solape
    return fragmentos


# 2b. Dividir por párrafos. Cada resumen semanal es un párrafo, así no se corta
# una semana a la mitad. Cada fragmento lleva el título del documento para que
# no pierda a qué estación o tema pertenece.
def dividir_documento(texto, tamano=700, solape=80):
    lineas = texto.strip().splitlines()
    titulo = lineas[0].strip()
    cuerpo = "\n".join(lineas[1:])
    fragmentos = []
    for bloque in [b.strip() for b in cuerpo.split("\n\n") if b.strip()]:
        for parte in dividir_texto(bloque, tamano, solape) if len(bloque) > tamano else [bloque]:
            fragmentos.append(f"{titulo}\n{parte}")
    return fragmentos or [titulo]


# 3. Embeddings mediante Ollama
# nomic-embed-text distingue entre documentos y preguntas con un prefijo
def crear_embeddings(textos, tipo="search_document"):
    respuesta = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODELO_EMBEDDING, "input": [f"{tipo}: {t}" for t in textos]},
        timeout=600
    )
    respuesta.raise_for_status()
    return respuesta.json()["embeddings"]


def crear_embedding(texto, tipo="search_query"):
    return crear_embeddings([texto], tipo)[0]


# 4. Indexar documentos
def indexar_documentos(lote=32):
    inicio = time.perf_counter()
    documentos = leer_documentos()
    pendientes = []
    for documento in documentos:
        for numero, fragmento in enumerate(dividir_documento(documento["texto"])):
            pendientes.append({
                "fuente": documento["fuente"],
                "fragmento": numero + 1,
                "texto": fragmento
            })
    for i in range(0, len(pendientes), lote):
        grupo = pendientes[i:i + lote]
        for item, vector in zip(grupo, crear_embeddings([g["texto"] for g in grupo])):
            item["embedding"] = vector
    RUTA_INDICE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_INDICE.write_text(json.dumps(pendientes, ensure_ascii=False), encoding="utf-8")
    return {
        "documentos": len(documentos),
        "fragmentos": len(pendientes),
        "segundos": round(time.perf_counter() - inicio, 1)
    }


# 5. Comparar vectores
def similitud_coseno(vector_a, vector_b):
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador == 0:
        return 0.0
    return float(np.dot(a, b) / denominador)


# 6. Recuperar los top_k fragmentos
@mlflow.trace(name="recuperar_documentos")
def recuperar_documentos(pregunta, top_k=3):
    inicio = time.perf_counter()
    if not RUTA_INDICE.exists():
        raise FileNotFoundError("Primero debes indexar los documentos")
    indice = json.loads(RUTA_INDICE.read_text(encoding="utf-8"))
    if not indice:
        raise ValueError("El índice está vacío")
    vector_pregunta = crear_embedding(pregunta)
    resultados = []
    for item in indice:
        resultados.append({
            "fuente": item["fuente"],
            "fragmento": item["fragmento"],
            "texto": item["texto"],
            "similitud": similitud_coseno(vector_pregunta, item["embedding"])
        })
    resultados.sort(key=lambda x: x["similitud"], reverse=True)
    return {
        "fragmentos": resultados[:top_k],
        "tiempo_recuperacion": time.perf_counter() - inicio
    }
