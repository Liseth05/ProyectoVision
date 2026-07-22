import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



DATASET = r"D:\PROYECTO\DOCUMENTACION\Imagenes_Nuevas"
IMG_SIZE = 128
carpeta_salida = r"D:\PROYECTO\QuintoPaso"
os.makedirs(carpeta_salida, exist_ok=True)

#CARGAMOS LAS IMAGENES
def imagenes_cargadas():
    images=[]
    labels=[]
    classes=sorted(os.listdir(DATASET))
    print("Clases encontradas:")
    print(classes)
    for label,folder in enumerate(classes):
        folder_path=os.path.join(
            DATASET,
            folder
        )
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path=os.path.join(
                    folder_path,
                    img_name
                )
                img=cv2.imread(img_path)
                if img is None:
                    continue
                img=cv2.resize(
                    img,
                    (IMG_SIZE,IMG_SIZE)
                )
                # Normalizamos a 0-1
                img=img/255.0
                images.append(img)
                labels.append(label)
    return np.array(images),np.array(labels),classes
#DATASET DE LAS IMAGENES
print("\nIMAGENES:")
X,y,classes=imagenes_cargadas()
print("\nCantidad imágenes:")
print(X.shape)
print("\nCantidad clases:")
print(len(classes))
# Convertir etiquetas
y=to_categorical(
    y,
    num_classes=len(classes)
)
# División entrenamiento prueba
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#RED NEURONAL Y SUS CAPAS
model=Sequential()
# Primera capa convolucional
model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(IMG_SIZE,IMG_SIZE,3)
    )
)
model.add(
    MaxPooling2D(
        (2,2)
    )
)
# Segunda capa
model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)
model.add(
    MaxPooling2D(
        (2,2)
    )
)
# Tercera capa
model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)
model.add(
    MaxPooling2D(
        (2,2)
    )
)
# Clasificador
model.add(
    Flatten()
)
model.add(
    Dense(
        128,
        activation="relu"
    )
)
model.add(
    Dropout(
        0.5
    )
)
model.add(
    Dense(
        len(classes),
        activation="softmax"
    )
)
#DESAROLLO MODELO
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
#TRAIN
print("\nEntrenamiento de CNN")
inicio=time.time()
history=model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(
        X_test,
        y_test
    )
)
training_time=time.time()-inicio
#DATOS DE PREDICCION
inicio=time.time()
pred=model.predict(
    X_test
)
prediction_time=time.time()-inicio
y_pred=np.argmax(
    pred,
    axis=1
)
y_real=np.argmax(
    y_test,
    axis=1
)
#METRICAS DE EVALUACION
accuracy=accuracy_score(
    y_real,
    y_pred
)
precision=precision_score(
    y_real,
    y_pred,
    average="weighted"
)
recall=recall_score(
    y_real,
    y_pred,
    average="weighted"
)
f1=f1_score(
    y_real,
    y_pred,
    average="weighted"
)
print("\n========================")
print("RESULTADOS CNN")
print("========================")
print(
    classification_report(
        y_real,
        y_pred,
        target_names=classes
    )
)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1-score:",f1)
#MATRIZ DE CONFUSION
cm=confusion_matrix(
    y_real,
    y_pred
)
plt.figure(
    figsize=(7,6)
)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.title(
    "Matriz de confusión CNN"
)
plt.xlabel(
    "Predicción"
)
plt.ylabel(
    "Real"
)
plt.show()
#DATOS DEL MODELO
resultado=pd.DataFrame(
    {
        "Modelo":["CNN"],
        "Accuracy":[accuracy],
        "Precision":[precision],
        "Recall":[recall],
        "F1":[f1],
        "Tiempo_entrenamiento":[training_time],
        "Tiempo_prediccion":[prediction_time]
    }
)
resultado.to_csv(
    os.path.join(carpeta_salida, "resultado_CNN.csv"),
    index=False
)
#Guardamos nuestro modelo
model.save(
    os.path.join(carpeta_salida, "modelo_CNN.h5")
)
#Guardamos el orden de las clases (importante para la app web)
with open(os.path.join(carpeta_salida, "clases_cnn.json"), "w") as f:
    json.dump(classes, f)

print("\nTiempo de entrenamiento:",
      training_time,
      "segundos")
print("\nTiempo de predicción:",
      prediction_time,
      "segundos")
print("\nModelos guardados en:", carpeta_salida)
print("\nFINALIZACION")

