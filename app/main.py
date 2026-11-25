import os
import re
import time
from contextlib import asynccontextmanager

import requests
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from rag import indexar_documentos, recuperar_documentos
from metricas import (
    CONSULTAS,
    TOKENS_ENTRADA,
    TOKENS_SALIDA,
    DURACION_TOTAL,
    DURACION_RAG,
    SIMILITUD_RAG,
    SIN_INFORMACION
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODELO = os.getenv("MODELO_LLM", "qwen2.5:1.5b")
EXPERIMENTO = "LLMOps_GasVirtual_RAG"

mlflow.set_tracking_uri(MLFLOW_URL)

PATRON_SIN_INFO = re.compile(
    r"no (existe|hay|tengo|cuento con|se encuentra|contiene|dispongo de) (suficiente )?informaci",
    re.IGNORECASE
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MLflow puede tardar en iniciar: esperamos a que esté disponible
    for intento in range(20):
        try:
            mlflow.set_experiment(EXPERIMENTO)
            print("MLflow conectado")
            break
        except Exception as error:
            if intento == 19:
                raise
            print("Esperando MLflow:", error)
            time.sleep(2)
    yield


app = FastAPI(
    title="LLMOps GasVirtual",
    description="Asistente RAG local para la planeación de gas virtual (datos anonimizados)",
    version="1.0",
    lifespan=lifespan
)


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=1, examples=["¿A qué hora del día consume más gas la estación EST_03?"])
    top_k: int = Field(default=3, ge=1, le=10)
    temperatura: float = Field(default=0.2, ge=0, le=2)


@app.get("/")
def inicio():
    return {"mensaje": "Asistente LLMOps GasVirtual activo"}


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.post("/indexar")
def indexar():
    try:
        return indexar_documentos()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@mlflow.trace(name="generar_respuesta_llm")
def generar_respuesta(pregunta, fragmentos, temperatura):
    contexto = "\n\n".join(
        f"Fuente: {item['fuente']}\nContenido: {item['texto']}"
        for item in fragmentos
    )
    instrucciones = """
Eres un asistente de planeación de estaciones de gas virtual.
Las estaciones se identifican con códigos EST_01 a EST_05 y los volúmenes
están en unidades escaladas (datos anonimizados).
Responde exclusivamente con la información del contexto documental.
Si el contexto no contiene la respuesta, indica que no existe
información suficiente. Responde en español, de forma breve y clara,
y copia las fechas y cifras tal como aparecen en el contexto.
"""
    prompt = f"""
CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:
"""
    respuesta = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODELO,
            "system": instrucciones,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperatura}
        },
        timeout=300
    )
    respuesta.raise_for_status()
    datos = respuesta.json()
    return {
        "respuesta": datos["response"],
        "tokens_entrada": datos.get("prompt_eval_count", 0),
        "tokens_salida": datos.get("eval_count", 0),
        "duracion_ollama": datos.get("total_duration", 0) / 1e9,
        "duracion_generacion": datos.get("eval_duration", 0) / 1e9
    }


@mlflow.trace(name="pipeline_rag")
def ejecutar_rag(datos: Pregunta):
    recuperacion = recuperar_documentos(
        pregunta=datos.pregunta,
        top_k=datos.top_k
    )
    fragmentos = recuperacion["fragmentos"]
    respuesta = generar_respuesta(
        pregunta=datos.pregunta,
        fragmentos=fragmentos,
        temperatura=datos.temperatura
    )
    DURACION_RAG.observe(recuperacion["tiempo_recuperacion"])
    for fragmento in fragmentos:
        SIMILITUD_RAG.observe(fragmento["similitud"])
    TOKENS_ENTRADA.inc(respuesta["tokens_entrada"])
    TOKENS_SALIDA.inc(respuesta["tokens_salida"])
    sin_informacion = bool(PATRON_SIN_INFO.search(respuesta["respuesta"]))
    if sin_informacion:
        SIN_INFORMACION.inc()
    return {
        "pregunta": datos.pregunta,
        "respuesta": respuesta["respuesta"],
        "sin_informacion": sin_informacion,
        "top_k": datos.top_k,
        "temperatura": datos.temperatura,
        "modelo": MODELO,
        "tokens_entrada": respuesta["tokens_entrada"],
        "tokens_salida": respuesta["tokens_salida"],
        "duracion_ollama": respuesta["duracion_ollama"],
        "duracion_generacion": respuesta["duracion_generacion"],
        "tiempo_recuperacion": recuperacion["tiempo_recuperacion"],
        "fuentes": fragmentos
    }


@app.post("/preguntar")
def preguntar(datos: Pregunta):
    inicio = time.perf_counter()
    try:
        resultado = ejecutar_rag(datos)
        CONSULTAS.labels(estado="exito").inc()
        return resultado
    except Exception as error:
        CONSULTAS.labels(estado="error").inc()
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        DURACION_TOTAL.observe(time.perf_counter() - inicio)


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
