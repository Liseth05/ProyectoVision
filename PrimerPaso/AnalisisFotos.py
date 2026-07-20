##Descargar y analizar el conjunto de imágenes (N. de instancias, N.
##clases, características de las imágenes)
import os
import cv2
from collections import Counter

#Ruta interna de mis archivos
ruta_dataset = r"D:\PROYECTO\DOCUMENTACION\DATASET"
excluir = ("Metadata",)

#Extensiones de imagenes que pueden exitir en el dataset
extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
#Almacenamos los datos e inicializamos el conteo de imagenes de cada instancia
clases = []
total_imagenes = 0
formatos = []
resoluciones = []
canales = []
#Imprimir los resultados de el dataset
print("=" * 50)
print("RESULTADOS DEL DATASET")
print("=" * 50)

for carpeta in os.listdir(ruta_dataset):

    ruta_carpeta = os.path.join(ruta_dataset, carpeta)
    #Quitar las carpetas que no son clases
    if (
        os.path.isdir(ruta_carpeta)
        and carpeta not in excluir
        and not carpeta.startswith(".")
    ):
        clases.append(carpeta)
        contador = 0
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith(extensiones):
                #Contador de las imagenes
                contador += 1
                total_imagenes += 1
                #Guardamos formatos de las imagenes
                formatos.append(os.path.splitext(archivo)[1].lower())
                #Leemos las imagenes
                ruta_imagen = os.path.join(ruta_carpeta, archivo)
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    alto, ancho = imagen.shape[:2]
                    resoluciones.append((ancho, alto))
                    if len(imagen.shape) == 3:
                        canales.append(imagen.shape[2])
                    else:
                        canales.append(1)
        print(f"\nClase: {carpeta}")
        print(f"Número de imágenes: {contador}")
#Sacamos un resumen total de todo el data set analizado
print("\n" + "=" * 50)
print("RESUMEN TOTAL DEL DATASET")
print("=" * 50)

print(f"Número total de instancias: {total_imagenes}")
print(f"Número total de clases: {len(clases)}")

print("\nClases encontradas:")
for clase in clases:
    print(f"- {clase}")

print("\nFormatos de imagen encontrados:")
for formato, cantidad in Counter(formatos).items():
    print(f"{formato}: {cantidad}")
#Resolucion de cada
print("\nResoluciones encontradas:")
for resolucion, cantidad in Counter(resoluciones).items():
    print(f"{resolucion[0]} x {resolucion[1]}: {cantidad}")
#Analizamos que canales de colores dispone las imagenes
print("\nNúmero de canales:")
for canal, cantidad in Counter(canales).items():
    if canal == 3:
        print(f"RGB (3 canales): {cantidad}")
    elif canal == 1:
        print(f"Escala de grises (1 canal): {cantidad}")