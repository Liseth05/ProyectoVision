import os
import cv2
import time
import numpy as np
import pandas as pd
import psutil
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

DATASET = r"D:\PROYECTO\DOCUMENTACION\Imagenes_Nuevas"
IMG_SIZE = (128,128)
carpeta_salida = r"D:\PROYECTO\CuartoPaso"
os.makedirs(carpeta_salida, exist_ok=True)

##ABRIMOS LOS DATOS OBTENIDOS ANTERIORES DE HU
def hu_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments)
    hu = -np.sign(hu)*np.log10(
        np.abs(hu)+1e-10
    )
    return hu.flatten()
# #ABRIMOS LOS DATOS OBTENIDOS ANTERIORES DE SIFT
def sift_features(img):
    gray=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )
    sift=cv2.SIFT_create()
    keypoints, descriptors=sift.detectAndCompute(
        gray,
        None
    )
    if descriptors is None:
        return np.zeros(128)
    return np.mean(
        descriptors,
        axis=0
    )
#ABRIMOS LOS DATOS OBTENIDOS ANTERIORES DE ORB
def orb_features(img):
    gray=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )
    orb=cv2.ORB_create()
    keypoints,descriptors=orb.detectAndCompute(
        gray,
        None
    )
    if descriptors is None:
        return np.zeros(32)
    return np.mean(
        descriptors,
        axis=0
    )
def load_dataset():
    X=[]
    y=[]
    start=time.time()
    for label,folder in enumerate(os.listdir(DATASET)):
        path=os.path.join(
            DATASET,
            folder
        )
        if os.path.isdir(path):
            for img_name in os.listdir(path):
                img_path=os.path.join(
                    path,
                    img_name
                )
                img=cv2.imread(img_path)
                if img is None:
                    continue
                img=cv2.resize(
                    img,
                    IMG_SIZE
                )
                hu=hu_features(img)
                sift=sift_features(img)
                orb=orb_features(img)
                features=np.concatenate(
                    [
                        hu,
                        sift,
                        orb
                    ]
                )
                X.append(features)
                y.append(folder)
    extraction_time=time.time()-start
    return np.array(X),np.array(y),extraction_time
#EVALUACION DEL MODELO
def evaluate_model(model,name,X_test,y_test):
    start=time.time()
    prediction=model.predict(X_test)
    prediction_time=time.time()-start
    acc=accuracy_score(
        y_test,
        prediction
    )
    precision=precision_score(
        y_test,
        prediction,
        average="weighted"
    )
    recall=recall_score(
        y_test,
        prediction,
        average="weighted"
    )
    f1=f1_score(
        y_test,
        prediction,
        average="weighted"
    )
    print("\n==========================")
    print(name)
    print("==========================")
    print(
        classification_report(
            y_test,
            prediction
        )
    )
    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",f1)
    #VISUALIZAMOS LA MATRIZ DE CONFUSION
    cm=confusion_matrix(
        y_test,
        prediction
    )
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(
        "Matriz de confusión - "+name
    )
    plt.xlabel(
        "Predicción"
    )
    plt.ylabel(
        "Real"
    )
    plt.show()
    return [
        name,
        acc,
        precision,
        recall,
        f1,
        prediction_time
    ]
print("CARACTERISTICAS OBTENIDAS")
X,y,feature_time=load_dataset()
print("\nCaracterísticas:")
print(X.shape)
#Separamos para entrenamiento y test
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Normalizamos los valores
scaler=StandardScaler()
X_train=scaler.fit_transform(
    X_train
)
X_test=scaler.transform(
    X_test
)
resultados=[]
#ALGORTIMO SVM
inicio=time.time()
svm=SVC(
    kernel="rbf",
    probability=True
)
svm.fit(
    X_train,
    y_train
)
svm_time=time.time()-inicio
res=evaluate_model(
    svm,
    "SVM",
    X_test,
    y_test
)
res.append(svm_time)
resultados.append(res)
#ALGORTIMO RANDOM FOREST
inicio=time.time()
rf=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(
    X_train,
    y_train
)
rf_time=time.time()-inicio
res=evaluate_model(
    rf,
    "Random Forest",
    X_test,
    y_test
)
res.append(rf_time)
resultados.append(res)
#TABLA DE VALORES
tabla=pd.DataFrame(
    resultados,
    columns=[
        "Modelo",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Tiempo_prediccion",
        "Tiempo_entrenamiento"
    ]
)
print("\nRESULTADOS FINALES")
print(tabla)
tabla.to_csv(
    os.path.join(carpeta_salida, "comparacion_clasificadores.csv"),
    index=False
)

##
# Guardamos SVM, Random Forest y el scaler directo en QuintoPaso
carpeta_web = r"D:\PROYECTO\QuintoPaso"
os.makedirs(carpeta_web, exist_ok=True)

joblib.dump(svm, os.path.join(carpeta_web, "svm_model.pkl"))
joblib.dump(rf, os.path.join(carpeta_web, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(carpeta_web, "scaler.pkl"))

# Guardamos las clases en el mismo orden que usa el modelo (mismo formato que usa la CNN)
clases_clasico = sorted(list(set(y)))
with open(os.path.join(carpeta_web, "clases_clasico.json"), "w") as f:
    json.dump(clases_clasico, f)

print("SVM, Random Forest y scaler guardados en:", carpeta_web)
##

# Memoria utilizada
ram=psutil.virtual_memory().used/1024/1024
print("\nCosto computacional")
print(
    "Tiempo extracción características:",
    feature_time,
    "segundos"
)
print(
    "RAM utilizada aproximada:",
    round(ram,2),
    "MB"
)
