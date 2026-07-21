##Aplicar tres algoritmos de extracción
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time


ruta_dataset = r"D:\PROYECTO\DOCUMENTACION\Imagenes_Nuevas"
carpeta_salida = r"D:\PROYECTO\TercerPaso"
os.makedirs(carpeta_salida, exist_ok=True)

clases = [
    "Afros",
    "Europeos",
    "Indigenas",
    "Mestizos"
]
#Creamos SIFT y ORB
sift = cv2.SIFT_create()
orb = cv2.ORB_create(
    nfeatures=1000
)
#Guardamos datos
datos_hu = []
datos_sift = []
datos_orb = []
#Almacenamos tiempos de ejeucion
tiempos = []
mostrar = True
print("*"*70)
print("EXTRACCIÓN DE CARACTERÍSTICAS")
print("*"*70)
#APLICAMOS EN TODO EL DATASET
for clase in clases:
    carpeta = os.path.join(
        ruta_dataset,
        clase
    )
    contador = 0
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(
            (".jpg",".jpeg",".png")
        ):
            ruta = os.path.join(
                carpeta,
                archivo
            )
            imagen = cv2.imread(
                ruta,
                cv2.IMREAD_GRAYSCALE
            )
            if imagen is None:
                continue
            #MOMENTOS HU
            inicio_hu = time.time()
            momentos = cv2.moments(
                imagen
            )
            hu = cv2.HuMoments(
                momentos
            ).flatten()
            fin_hu = time.time()
            tiempo_hu = fin_hu - inicio_hu
            datos_hu.append({
                "Clase": clase,
                "Imagen": archivo,
                "HU1": hu[0],
                "HU2": hu[1],
                "HU3": hu[2],
                "HU4": hu[3],
                "HU5": hu[4],
                "HU6": hu[5],
                "HU7": hu[6],
                "Tiempo_HU": tiempo_hu
            })
            #SIFT
            inicio_sift = time.time()
            kp_sift, des_sift = sift.detectAndCompute(
                imagen,
                None
            )
            fin_sift = time.time()
            tiempo_sift = fin_sift - inicio_sift
            cantidad_sift = (
                len(kp_sift)
                if kp_sift is not None
                else 0
            )
            datos_sift.append({
                "Clase": clase,
                "Imagen": archivo,
                "Keypoints_SIFT": cantidad_sift,
                "Tiempo_SIFT": tiempo_sift
            })
            #ORB
            inicio_orb = time.time()
            kp_orb, des_orb = orb.detectAndCompute(
                imagen,
                None
            )
            fin_orb = time.time()
            tiempo_orb = fin_orb - inicio_orb
            cantidad_orb = (
                len(kp_orb)
                if kp_orb is not None
                else 0
            )
            datos_orb.append({
                "Clase": clase,
                "Imagen": archivo,
                "Keypoints_ORB": cantidad_orb,
                "Tiempo_ORB": tiempo_orb
            })
            #TIEMPOS DE EJECUCION
            tiempos.append({
                "Clase": clase,
                "Imagen": archivo,
                "Tiempo_HU": tiempo_hu,
                "Tiempo_SIFT": tiempo_sift,
                "Tiempo_ORB": tiempo_orb
            })
            #EJEMPLO REALIZANDO A 1 IMAGEN
            if mostrar:
                imagen_color = cv2.cvtColor(
                    imagen,
                    cv2.COLOR_GRAY2RGB
                )
                img_sift = cv2.drawKeypoints(
                    imagen_color,
                    kp_sift,
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                img_orb = cv2.drawKeypoints(
                    imagen_color,
                    kp_orb,
                    None,
                    color=(0,255,0)
                )
                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1)
                plt.imshow(imagen_color)
                plt.title("Imagen")
                plt.axis("off")
                plt.subplot(1,3,2)
                plt.imshow(img_sift)
                plt.title(
                    f"SIFT: {cantidad_sift} puntos"
                )
                plt.axis("off")
                plt.subplot(1,3,3)
                plt.imshow(img_orb)
                plt.title(
                    f"ORB: {cantidad_orb} puntos"
                )
                plt.axis("off")
                plt.show()
                print("\nMomentos de Hu:")
                for i,valor in enumerate(hu):
                    print(
                        f"HU{i+1}: {valor:.10f}"
                    )
                mostrar=False
            contador += 1
    print(
        f"{clase}: {contador} imágenes realizadas"
    )
#ALMACENAMOS LA INFORMACION EN DATAFRAMES
df_hu = pd.DataFrame(datos_hu)
df_sift = pd.DataFrame(datos_sift)
df_orb = pd.DataFrame(datos_orb)
df_tiempos = pd.DataFrame(tiempos)
#ALMACENAR EN UN CSV
df_hu.to_csv(
    os.path.join(carpeta_salida, "dataset_HU.csv"),
    index=False
)
df_sift.to_csv(
    os.path.join(carpeta_salida, "dataset_SIFT.csv"),
    index=False
)
df_orb.to_csv(
    os.path.join(carpeta_salida, "dataset_ORB.csv"),
    index=False
)
df_tiempos.to_csv(
    os.path.join(carpeta_salida, "tiempos_extraccion.csv"),
    index=False
)
#RESULTADOS
print("\n")
print("*"*70)
print("RESULTADOS DEL PROCESAMIENTO")
print("*"*70)
comparacion = pd.DataFrame({
    "Metodo":[
        "Hu",
        "SIFT",
        "ORB"
    ],
    "Numero_caracteristicas":[
        7,
        round(
            df_sift["Keypoints_SIFT"].mean(),
            2
        ),
        round(
            df_orb["Keypoints_ORB"].mean(),
            2
        )
    ],
    "Formato":[
        "Vector 7 valores",
        "Descriptores SIFT 128 dimensiones",
        "Descriptores ORB binarios"
    ],
    "Tiempo_promedio_segundos":[
        round(
            df_tiempos["Tiempo_HU"].mean(),
            6
        ),
        round(
            df_tiempos["Tiempo_SIFT"].mean(),
            6
        ),
        round(
            df_tiempos["Tiempo_ORB"].mean(),
            6
        )
    ]
})
print(comparacion)
comparacion.to_csv(
    os.path.join(carpeta_salida, "comparacion_caracteristicas.csv"),
    index=False
)
print("\n")
print("ARCHIVOS GENERADOS:")
print("-------------------")
print("dataset_HU.csv")
print("dataset_SIFT.csv")
print("dataset_ORB.csv")
print("tiempos_extraccion.csv")
print("comparacion_caracteristicas.csv")
print("\nTotal imágenes realizadas:")
print(
    len(df_hu)
)