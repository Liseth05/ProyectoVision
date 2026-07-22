# -*- coding: utf-8 -*-
"""
app.py

Backend web (Flask) del Proyecto Final de Vision por Computador.
Todos los modelos y archivos se leen directo de la carpeta QuintoPaso,
la misma carpeta donde vive este app.py.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from flask import Flask, render_template, request

from extraccion_caracteristicas import extraer_vector_combinado, preparar_imagen_cnn

app = Flask(__name__)

# Carpeta donde estan TODOS los archivos: app.py, los modelos, los csv, etc.
CARPETA_BASE = r"D:\PROYECTO\QuintoPaso"
CARPETA_UPLOADS = os.path.join(CARPETA_BASE, "uploads")
os.makedirs(CARPETA_UPLOADS, exist_ok=True)

# -------------------------------------------------------------
# CARGA DE MODELOS (se hace UNA sola vez, al iniciar el servidor)
# -------------------------------------------------------------
print("Cargando modelos, espera un momento...")

svm_model = joblib.load(os.path.join(CARPETA_BASE, "svm_model.pkl"))
rf_model = joblib.load(os.path.join(CARPETA_BASE, "rf_model.pkl"))
scaler = joblib.load(os.path.join(CARPETA_BASE, "scaler.pkl"))

with open(os.path.join(CARPETA_BASE, "clases_clasico.json"), "r") as f:
    clases_clasico = json.load(f)

# OJO: el archivo se llama "modelo_CNN.h5", tal como lo guarda tu script
cnn_model = tf.keras.models.load_model(os.path.join(CARPETA_BASE, "modelo_CNN.h5"))

with open(os.path.join(CARPETA_BASE, "clases_cnn.json"), "r") as f:
    clases_cnn = json.load(f)

print("Modelos cargados correctamente.")
print("Clases (clasico):", clases_clasico)
print("Clases (CNN):", clases_cnn)


def cargar_tabla_comparativa():
    """
    Lee los CSV que tus scripts ya generan y arma una sola tabla
    comparativa. Si algun archivo no existe, esa fila se omite.
    """
    filas = []

    ruta_clasico = os.path.join(CARPETA_BASE, "comparacion_clasificadores.csv")
    if os.path.exists(ruta_clasico):
        df_clasico = pd.read_csv(ruta_clasico)
        filas.append(df_clasico)

    ruta_cnn = os.path.join(CARPETA_BASE, "resultado_CNN.csv")
    if os.path.exists(ruta_cnn):
        df_cnn = pd.read_csv(ruta_cnn)
        filas.append(df_cnn)

    if not filas:
        return pd.DataFrame()

    tabla = pd.concat(filas, ignore_index=True, sort=False)
    return tabla


@app.route("/", methods=["GET"])
def inicio():
    tabla = cargar_tabla_comparativa()
    tabla_html = tabla.to_html(index=False, classes="tabla-resultados", border=0)
    return render_template("index.html", tabla_html=tabla_html, resultado=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    archivo = request.files.get("imagen")

    if archivo is None or archivo.filename == "":
        tabla = cargar_tabla_comparativa()
        tabla_html = tabla.to_html(index=False, classes="tabla-resultados", border=0)
        return render_template("index.html", tabla_html=tabla_html,
                                resultado=None, error="Selecciona una imagen primero.")

    # Guardamos la imagen subida en la carpeta uploads
    ruta_imagen = os.path.join(CARPETA_UPLOADS, archivo.filename)
    archivo.save(ruta_imagen)

    # Leemos la imagen con OpenCV (igual que en tus scripts de entrenamiento)
    img = cv2.imread(ruta_imagen)

    # Verificamos que la imagen se haya podido leer correctamente
    if img is None:
        tabla = cargar_tabla_comparativa()
        tabla_html = tabla.to_html(index=False, classes="tabla-resultados", border=0)
        return render_template(
            "index.html",
            tabla_html=tabla_html,
            resultado=None,
            error="No se pudo leer esa imagen. Prueba con otro archivo JPG o PNG "
                  "(algunas imagenes descargadas de internet vienen en un formato "
                  "distinto al que indica su extension)."
        )

    # ---------- Prediccion con SVM y Random Forest ----------
    vector = extraer_vector_combinado(img)
    vector_escalado = scaler.transform([vector])

    # predict_proba devuelve la probabilidad de CADA clase para esta imagen
    # (por eso el resultado es distinto en cada prediccion, no esta fijo)
    proba_svm = svm_model.predict_proba(vector_escalado)[0]
    proba_rf = rf_model.predict_proba(vector_escalado)[0]

    indice_svm = np.argmax(proba_svm)
    indice_rf = np.argmax(proba_rf)

    pred_svm = svm_model.classes_[indice_svm]
    pred_rf = rf_model.classes_[indice_rf]

    confianza_svm = round(float(proba_svm[indice_svm]) * 100, 2)
    confianza_rf = round(float(proba_rf[indice_rf]) * 100, 2)

    # ---------- Prediccion con la CNN ----------
    entrada_cnn = preparar_imagen_cnn(img)
    probabilidades = cnn_model.predict(entrada_cnn, verbose=0)[0]
    indice_cnn = np.argmax(probabilidades)
    pred_cnn = clases_cnn[indice_cnn]
    confianza_cnn = round(float(probabilidades[indice_cnn]) * 100, 2)

    resultado = {
        "imagen": archivo.filename,
        "svm": pred_svm,
        "confianza_svm": confianza_svm,
        "rf": pred_rf,
        "confianza_rf": confianza_rf,
        "cnn": pred_cnn,
        "confianza_cnn": confianza_cnn,
    }

    tabla = cargar_tabla_comparativa()
    tabla_html = tabla.to_html(index=False, classes="tabla-resultados", border=0)

    return render_template("index.html", tabla_html=tabla_html, resultado=resultado)


if __name__ == "__main__":
    app.run(debug=True, port=5000)