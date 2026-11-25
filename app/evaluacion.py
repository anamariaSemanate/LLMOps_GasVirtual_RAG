"""Evaluación del RAG con preguntas de referencia (Pasos 28-30 del tutorial).

Además de guardar las respuestas, como en el tutorial, revisa de forma automática:
  - acierto: la respuesta contiene el dato clave esperado (fecha, cifra o palabra)
  - fuente: el documento esperado está entre los fragmentos recuperados
  - sin evidencia: en las preguntas sin respaldo, el modelo reconoce que no hay información
y registra cada corrida como un run de MLflow para comparar configuraciones.

Se ejecuta dentro del contenedor de la API (no requiere Python en Windows):
    docker compose exec api python evaluacion.py --top-k 3 --temperatura 0.2
"""
import argparse
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import mlflow
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000") + "/preguntar"
DATOS = Path("/datos") if Path("/datos").exists() else Path(__file__).resolve().parents[1] / "datos"
RUTA_PREGUNTAS = DATOS / "preguntas_evaluacion.csv"


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", texto)  # 15.709 -> 15709


def evaluar(fila, datos):
    respuesta = normalizar(datos["respuesta"])
    fuentes = [f["fuente"] for f in datos["fuentes"]]
    if fila["categoria"] == "sin_evidencia":
        return {"acierto": datos["sin_informacion"], "fuente_recuperada": None}
    return {
        "acierto": normalizar(fila["clave"]) in respuesta,
        "fuente_recuperada": fila["fuente_esperada"] in fuentes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--temperatura", type=float, default=0.2)
    p.add_argument("--etiqueta", default="", help="nombre libre del experimento, ej. A_topk")
    args = p.parse_args()

    resultados = []
    with RUTA_PREGUNTAS.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            inicio = time.perf_counter()
            try:
                respuesta = requests.post(
                    API_URL,
                    json={"pregunta": fila["pregunta"], "top_k": args.top_k, "temperatura": args.temperatura},
                    timeout=360
                )
                respuesta.raise_for_status()
                datos = respuesta.json()
                revision = evaluar(fila, datos)
                resultados.append({
                    "id": fila["id"],
                    "categoria": fila["categoria"],
                    "pregunta": fila["pregunta"],
                    "esperada": fila["respuesta_esperada"],
                    "generada": datos["respuesta"],
                    **revision,
                    "fuentes": [f["fuente"] for f in datos["fuentes"]],
                    "similitud_max": max(f["similitud"] for f in datos["fuentes"]),
                    "tokens_entrada": datos["tokens_entrada"],
                    "tokens_salida": datos["tokens_salida"],
                    "duracion": time.perf_counter() - inicio
                })
                marca = "OK " if revision["acierto"] else "MAL"
                print(f"[{marca}] Pregunta {fila['id']} ({fila['categoria']}) en {time.perf_counter() - inicio:.1f} s")
            except requests.RequestException as error:
                print(f"Error en pregunta {fila['id']}: {error}")

    if not resultados:
        print("No hubo resultados")
        return

    n = len(resultados)
    con_fuente = [r for r in resultados if r["fuente_recuperada"] is not None]
    resumen = {
        "exactitud": sum(r["acierto"] for r in resultados) / n,
        "recuperacion_fuente": sum(r["fuente_recuperada"] for r in con_fuente) / max(len(con_fuente), 1),
        "tokens_entrada_promedio": sum(r["tokens_entrada"] for r in resultados) / n,
        "tokens_salida_promedio": sum(r["tokens_salida"] for r in resultados) / n,
        "duracion_promedio": sum(r["duracion"] for r in resultados) / n,
    }
    for cat in sorted({r["categoria"] for r in resultados}):
        grupo = [r for r in resultados if r["categoria"] == cat]
        resumen[f"exactitud_{cat}"] = sum(r["acierto"] for r in grupo) / len(grupo)

    nombre = f"top{args.top_k}_t{args.temperatura}" + (f"_{args.etiqueta}" if args.etiqueta else "")
    ruta = DATOS / f"resultados_{nombre}.json"
    ruta.write_text(json.dumps({"resumen": resumen, "respuestas": resultados}, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    mlflow.set_experiment("LLMOps_GasVirtual_Evaluacion")
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params({"top_k": args.top_k, "temperatura": args.temperatura, "preguntas": n,
                           "etiqueta": args.etiqueta or "-"})
        mlflow.log_metrics(resumen)
        mlflow.log_artifact(str(ruta))

    print("\nResumen:")
    for k, v in resumen.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Resultados en {ruta}")
    print("Evaluacion terminada")


if __name__ == "__main__":
    main()
