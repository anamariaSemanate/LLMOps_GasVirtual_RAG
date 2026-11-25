# LLMOps GasVirtual · Asistente RAG local para la planeación de gas virtual

Aplicación del tutorial *"LLMOps desde cero en Windows con WSL2 y Docker: construye un RAG local con Ollama y monitorea tokens, trazas y rendimiento con MLflow, Prometheus y Grafana"* (J. M. Núñez Velasco) a un caso real: estaciones de gas virtual abastecidas por camión.

Es una **simulación con datos anonimizados**. No se conecta a ningún sistema de la empresa.

## Qué responde el asistente

El RAG consulta tres grupos de documentos:

| Carpeta | Contenido | Ejemplo de pregunta |
|---|---|---|
| `documentos/consumo/` | Perfil y resumen semanal del último año de cada estación, generados desde los datos | ¿A qué hora del día consume más gas la estación EST_03? |
| `documentos/diagnostico/` | Cómo interpretar las alertas del modelo de pronóstico (sesgo, deriva, festivos, calidad de datos) | ¿Por qué es peligroso que el modelo subestime el consumo? |
| `documentos/bitacora/` | Bitácora de los experimentos de modelado del trabajo de grado | ¿Por qué se descartó SARIMA con estacionalidad de 168 horas? |

## Arquitectura

Los mismos cinco servicios del tutorial, con Docker Compose:

| Servicio | Función | Dirección |
|---|---|---|
| Ollama | LLM `qwen2.5:1.5b` y embeddings `nomic-embed-text` | http://localhost:11434 |
| API (FastAPI) | `/indexar`, `/preguntar`, `/metrics` | http://localhost:8000/docs |
| MLflow | Trazas de cada consulta y corridas de evaluación | http://localhost:5000 |
| Prometheus | Métricas de tokens, latencia y similitud | http://localhost:9090 |
| Grafana | Tablero *Monitoreo LLMOps: Ollama y RAG* | http://localhost:3000 |

### Cambios respecto al tutorial

- **Se ejecuta desde PowerShell con Docker Desktop**, sin instalar Ubuntu. Todos los servicios corren en contenedores.
- **Documentos en subcarpetas** y **fragmentación por párrafo**: cada resumen semanal es un párrafo y no se corta a la mitad. Cada fragmento lleva el título del documento, para no perder a qué estación pertenece.
- **Prefijos de `nomic-embed-text`** (`search_document:` / `search_query:`) e indexación por lotes.
- **Métrica nueva** `llm_respuestas_sin_informacion_total`: cuenta las respuestas en las que el modelo dice que no hay información.
- **Evaluación automática**: además de guardar las respuestas, `evaluacion.py` revisa si la respuesta contiene el dato esperado, si se recuperó el documento correcto y si el modelo reconoce las preguntas sin evidencia. Cada corrida queda registrada en MLflow. En las preguntas de consumo, la respuesta correcta se calcula a partir de los datos.
- El tablero de Grafana se carga automáticamente; también se puede crear a mano como indica el Paso 27.

## Anonimización

- **Datos de consumo:** se reutiliza `datos/series_anonimizadas.parquet` del proyecto `mlops-gasvirtual`, con estaciones `EST_01`…`EST_05`, volumen escalado con un factor secreto y fechas desplazadas.
- **Textos:** el diagnóstico y la bitácora se redactaron sin nombres de la empresa, estaciones, municipios, planta ni sistemas internos, y sin fechas reales.
- **Verificación:** `scripts/verificar_anonimizacion.py` busca nombres reales en todos los archivos. La lista de nombres (`terminos_prohibidos.txt`) es local y no se sube, para no publicar justamente lo que se quiere ocultar.

## Cómo ejecutarlo (Windows, PowerShell)

Requisitos: Docker Desktop abierto, unos 8 GB libres y 8 GB de RAM o más.

**1. Levantar los servicios** (desde la carpeta del proyecto)
```powershell
docker compose build
docker compose up -d
docker compose ps
```

**2. Descargar los modelos** (solo la primera vez, unos 1,3 GB)
```powershell
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama list
```

**3. Indexar los documentos**

Desde http://localhost:8000/docs → `POST /indexar` → *Try it out* → *Execute*, o bien:
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/indexar
```

**4. Hacer una pregunta**

Desde http://localhost:8000/docs → `POST /preguntar`, o bien:
```powershell
$cuerpo = @{ pregunta = "¿A qué hora del día consume más gas la estación EST_03?"; top_k = 3; temperatura = 0.2 } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/preguntar -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($cuerpo))
```

**5. Ver trazas, métricas y tablero**
- MLflow → experimento `LLMOps_GasVirtual_RAG` → *Traces*
- Prometheus → http://localhost:9090/targets (debe aparecer `llmops_api` en UP)
- Grafana → `admin` / `admin` → Dashboards → LLMOps → *Monitoreo LLMOps: Ollama y RAG*

**6. Evaluar** (se ejecuta dentro del contenedor, no necesita Python en Windows)
```powershell
docker compose exec api python evaluacion.py --top-k 3 --temperatura 0.2
```
Los resultados quedan en `datos/resultados_*.json` y en MLflow, experimento `LLMOps_GasVirtual_Evaluacion`.

**7. Experimentos del Paso 30**
```powershell
# A. Tamaño del contexto
docker compose exec api python evaluacion.py --top-k 1 --etiqueta A
docker compose exec api python evaluacion.py --top-k 3 --etiqueta A
docker compose exec api python evaluacion.py --top-k 5 --etiqueta A
# B. Temperatura
docker compose exec api python evaluacion.py --temperatura 0   --etiqueta B
docker compose exec api python evaluacion.py --temperatura 0.5 --etiqueta B
docker compose exec api python evaluacion.py --temperatura 1   --etiqueta B
```
- **C. Preguntas sin evidencia:** las preguntas de la categoría `sin_evidencia`, como el precio del gas o el número de conductores.
- **D. Rendimiento:** repetir varias consultas y comparar las latencias con el modelo frío y ya cargado.

En MLflow, el experimento `LLMOps_GasVirtual_Evaluacion` permite comparar las corridas: exactitud, recuperación de la fuente, tokens y duración.

**Detener**
```powershell
docker compose down       # conserva modelos y datos
```
No uses `docker compose down -v`, porque borra los modelos descargados.

## Regenerar los documentos (opcional)

Los servicios corren en Docker y no necesitan instalar nada en Windows. Los scripts de documentos usan las dependencias de `requirements.txt` (las de la API están en `app/requirements.txt`):
```powershell
pip install -r requirements.txt
python scripts/generar_documentos.py
python scripts/verificar_anonimizacion.py
```

## Problemas comunes

| Problema | Revisión |
|---|---|
| Un contenedor se detiene | `docker compose ps -a` |
| Error en la API | `docker compose logs --tail=100 api` |
| MLflow no responde | `docker compose logs --tail=100 mlflow` |
| Puerto 5000 ocupado | Cierra el MLflow del proyecto `mlops-gasvirtual` |
| Ollama no encuentra el modelo | `docker compose exec ollama ollama list` |
| Prometheus aparece DOWN | http://localhost:9090/targets y http://localhost:8000/metrics |
| Grafana sin datos | Haz varias preguntas y amplía la ventana de tiempo |
