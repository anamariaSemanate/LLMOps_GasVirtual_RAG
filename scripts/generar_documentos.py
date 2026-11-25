"""Genera los documentos de consumo del RAG a partir de las series anonimizadas.

Entrada:  datos/series_anonimizadas.parquet (mismo archivo del proyecto mlops-gasvirtual)
Salida:   documentos/consumo/EST_XX_perfil.txt   -> resumen general de cada estación
          documentos/consumo/EST_XX_semanas.txt  -> un párrafo por semana (último año)
          datos/preguntas_evaluacion.csv         -> preguntas con respuesta conocida

Los datos ya vienen anonimizados: estaciones con código, volumen escalado y fechas
desplazadas. Por eso los volúmenes se expresan en "unidades" y no en m³.

Uso:  python scripts/generar_documentos.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos" / "series_anonimizadas.parquet"
SALIDA = RAIZ / "documentos" / "consumo"
PREGUNTAS = RAIZ / "datos" / "preguntas_evaluacion.csv"
SEMANAS = 52

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def pct(x):
    return f"{x:+.1f} %"


def cargar():
    df = pd.read_parquet(DATOS)
    df["fecha"] = df["datetime"].dt.normalize()
    return df


def diario(g):
    d = g.groupby("fecha").agg(consumo=("valor_corregido", "sum"), horas=("valor_corregido", "size"),
                               dia_semana=("dia_semana", "first"), es_festivo=("es_festivo", "max"),
                               mes=("mes", "first"))
    return d[d["horas"] == 24]


def semanas_completas(d):
    """Semanas lunes-domingo con los 7 días completos."""
    lunes = d.index[d["dia_semana"] == 0]
    out = []
    for ini in lunes:
        sem = d.loc[ini:ini + pd.Timedelta(days=6)]
        if len(sem) == 7:
            out.append((ini, sem))
    return out


def parrafo_semana(est, ini, sem, anterior):
    total = sem["consumo"].sum()
    mx, mn = sem["consumo"].idxmax(), sem["consumo"].idxmin()
    festivos = sem[sem["es_festivo"] == 1]
    fest_txt = ("ninguno" if festivos.empty else
                ", ".join(f"{DIAS[int(r.dia_semana)]} {f:%Y-%m-%d}" for f, r in festivos.iterrows()))
    finde = sem[sem["dia_semana"] >= 5]["consumo"].mean()
    entre = sem[sem["dia_semana"] < 5]["consumo"].mean()
    var = f"{pct(100 * (total / anterior - 1))}" if anterior else "sin dato"
    fin = ini + pd.Timedelta(days=6)
    return (
        f"Estación {est}, semana del {ini:%Y-%m-%d} al {fin:%Y-%m-%d}. "
        f"Consumo total de la semana: {total:.0f} unidades. "
        f"Promedio diario: {total / 7:.0f} unidades. "
        f"Día de mayor consumo: {DIAS[int(sem.loc[mx, 'dia_semana'])]} {mx:%Y-%m-%d} con {sem.loc[mx, 'consumo']:.0f} unidades. "
        f"Día de menor consumo: {DIAS[int(sem.loc[mn, 'dia_semana'])]} {mn:%Y-%m-%d} con {sem.loc[mn, 'consumo']:.0f} unidades. "
        f"Festivos en la semana: {fest_txt}. "
        f"Variación frente a la semana anterior: {var}. "
        f"Fin de semana frente a entre semana: {pct(100 * (finde / entre - 1))}."
    ), total


def perfil(est, g, d, sem_ult, sem_prev):
    horas = g.groupby(g["datetime"].dt.hour)["valor_corregido"].mean()
    por_dia = d.groupby("dia_semana")["consumo"].mean()
    laboral = d[(d["es_festivo"] == 0) & (d["dia_semana"] < 5)]["consumo"].mean()
    festivo = d[(d["es_festivo"] == 1) & (d["dia_semana"] < 5)]["consumo"].mean()
    finde = d[d["dia_semana"] >= 5]["consumo"].mean()
    por_mes = d.groupby("mes")["consumo"].mean()
    tot_ult = {ini: s["consumo"].sum() for ini, s in sem_ult}
    s_max, s_min = max(tot_ult, key=tot_ult.get), min(tot_ult, key=tot_ult.get)
    prom_ult, prom_prev = np.mean(list(tot_ult.values())), np.mean([s["consumo"].sum() for _, s in sem_prev])
    efecto_fest = 100 * (festivo / laboral - 1)
    datos = {
        "semana_max": f"{s_max:%Y-%m-%d}", "total_max": f"{tot_ult[s_max]:.0f}",
        "hora_pico": f"{horas.idxmax()}:00", "dia_max": DIAS[int(por_dia.idxmax())],
        "festivo": "aumenta" if efecto_fest > 0 else "disminuye",
        "mes_max": MESES[int(por_mes.idxmax()) - 1],
    }
    texto = f"""PERFIL DE CONSUMO DE LA ESTACIÓN {est}
Datos anonimizados: el volumen está escalado (unidades, no m³) y las fechas están desplazadas.

Periodo con datos de la estación {est}: del {d.index.min():%Y-%m-%d} al {d.index.max():%Y-%m-%d}. Consumo semanal promedio de {est} en el último año: {prom_ult:.0f} unidades. Variación del consumo de {est} en el último año frente al año anterior: {pct(100 * (prom_ult / prom_prev - 1))}.

Semana de mayor consumo de la estación {est} en el último año: la semana del {datos['semana_max']}, con {datos['total_max']} unidades. Semana de menor consumo de {est} en el último año: la semana del {s_min:%Y-%m-%d}, con {tot_ult[s_min]:.0f} unidades.

Patrón diario de la estación {est}: la hora del día con mayor consumo promedio es las {datos['hora_pico']} y la de menor consumo es las {horas.idxmin()}:00. Patrón semanal de {est}: el día de la semana con mayor consumo promedio es el {datos['dia_max']} y el de menor consumo es el {DIAS[int(por_dia.idxmin())]}.

Fin de semana y festivos en la estación {est}: el consumo diario del fin de semana es {pct(100 * (finde / laboral - 1))} frente a un día laboral normal. En los festivos entre semana el consumo diario de {est} {datos['festivo']} ({pct(efecto_fest)}) frente a un día laboral normal.

Estacionalidad anual de la estación {est}: el mes con mayor consumo promedio es {datos['mes_max']} y el de menor consumo es {MESES[int(por_mes.idxmin()) - 1]}.
"""
    return texto, datos


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar()
    rng = np.random.default_rng(7)
    preguntas = []

    for est, g in df.groupby("estacion"):
        d = diario(g)
        todas = semanas_completas(d)
        ult, prev = todas[-SEMANAS:], todas[-2 * SEMANAS:-SEMANAS]

        parrafos, totales, anterior = [], {}, sum(s["consumo"].sum() for s in [todas[-SEMANAS - 1][1]])
        for ini, sem in ult:
            p, total = parrafo_semana(est, ini, sem, anterior)
            parrafos.append(p)
            totales[ini] = total
            anterior = total
        (SALIDA / f"{est}_semanas.txt").write_text(
            f"RESUMEN SEMANAL DE CONSUMO DE LA ESTACIÓN {est} (último año, datos anonimizados)\n\n"
            + "\n\n".join(parrafos) + "\n", encoding="utf-8")

        texto, datos = perfil(est, g, d, ult, prev)
        (SALIDA / f"{est}_perfil.txt").write_text(texto, encoding="utf-8")

        semana = list(totales)[int(rng.integers(5, len(totales) - 5))]
        preguntas += [
            ("consumo", f"¿Cuál fue la semana de mayor consumo de la estación {est} en el último año?",
             f"La semana del {datos['semana_max']} ({datos['total_max']} unidades)", datos["semana_max"], f"consumo/{est}_perfil.txt"),
            ("consumo", f"¿Cuánto gas consumió la estación {est} en la semana del {semana:%Y-%m-%d}?",
             f"{totales[semana]:.0f} unidades", f"{totales[semana]:.0f}", f"consumo/{est}_semanas.txt"),
        ]
        extra = rng.choice(["hora", "dia", "festivo"])
        if extra == "hora":
            preguntas.append(("consumo", f"¿A qué hora del día consume más gas la estación {est}?",
                              f"A las {datos['hora_pico']}", datos["hora_pico"], f"consumo/{est}_perfil.txt"))
        elif extra == "dia":
            preguntas.append(("consumo", f"¿Qué día de la semana tiene el mayor consumo la estación {est}?",
                              f"El {datos['dia_max']}", datos["dia_max"], f"consumo/{est}_perfil.txt"))
        else:
            preguntas.append(("consumo", f"¿Los festivos aumentan o disminuyen el consumo en la estación {est}?",
                              f"Los festivos {datos['festivo']}n el consumo", datos["festivo"][:6], f"consumo/{est}_perfil.txt"))
        print(f"{est}: {len(parrafos)} semanas")

    preguntas += [
        ("diagnostico", "¿Por qué es peligroso que el modelo subestime el consumo?",
         "Porque sobreestima la autonomía y puede llevar a desabastecimiento", "desabastec", "diagnostico/sesgo_negativo.txt"),
        ("diagnostico", "¿Qué indica que el consumo reciente sea mayor que la media de entrenamiento?",
         "Una deriva de nivel; conviene normalizar con medias recientes o reentrenar", "deriva", "diagnostico/deriva_nivel.txt"),
        ("diagnostico", "¿Cuánto aumenta el consumo en Semana Santa?",
         "Entre 19 % y 27 % en algunas estaciones", "19", "diagnostico/festivos_vacaciones.txt"),
        ("diagnostico", "¿Por qué el pronóstico recursivo no alcanza los picos de consumo?",
         "Porque al encadenar predicciones la curva se aplana hacia el promedio", "aplan", "diagnostico/curvas_aplanadas.txt"),
        ("bitacora", "¿Por qué se descartó SARIMA con estacionalidad de 168 horas?",
         "Por el costo de memoria y tiempo: el ajuste no terminó en más de 24 horas", "24", "bitacora/sarima.txt"),
        ("bitacora", "¿Qué modelo se seleccionó finalmente para el pronóstico?",
         "El LSTM global con la variable de la semana anterior (lag)", "lstm", "bitacora/gru_y_seleccion.txt"),
        ("bitacora", "¿Por qué se usan tres semillas en los experimentos?",
         "Porque la misma arquitectura varía mucho entre entrenamientos", "semilla", "bitacora/lstm_mejoras.txt"),
        ("bitacora", "¿Qué aportó la temperatura al modelo LSTM?",
         "Nada: no supera el ruido entre semillas", "no", "bitacora/lstm_mejoras.txt"),
        ("sin_evidencia", "¿Cuál es el precio del metro cúbico de gas?",
         "No existe información suficiente", "", ""),
        ("sin_evidencia", "¿Cuántos conductores tiene la flota de camiones?",
         "No existe información suficiente", "", ""),
    ]
    with PREGUNTAS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "categoria", "pregunta", "respuesta_esperada", "clave", "fuente_esperada"])
        for i, fila in enumerate(preguntas, 1):
            w.writerow([i, *fila])
    print(f"{len(preguntas)} preguntas de evaluación en {PREGUNTAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
