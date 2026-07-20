##Aplicar técnicas para mejorar el contraste, eliminación de ruido y
##umbralización de las imágenes.
import os
import cv2
import matplotlib.pyplot as plt

ruta_dataset = r"D:\PROYECTO\DOCUMENTACION\DATASET"

#Carpeta con imagenes procesadas
data_nuevo = r"D:\PROYECTO\DOCUMENTACION\Imagenes_Nuevas"
#Creamos la carpeta para guardar imagenes si no existiera
os.makedirs(data_nuevo, exist_ok=True)
#Ingresamos las clases de el data set
clases = ["Afros", "Europeos", "Indigenas", "Mestizos"]
#Esta variable nos ayudara a mostrar un ejemplo
visualizacion = True
print("*"*60)
print("EJEMPLO DE TECNICAS DE CONTRASTE")
print("*"*60)

total = 0
#Aplicamos a todo el data set las tecnicas de mejoramiento y umbralizacion
for clase in clases:
    path_origen = os.path.join(ruta_dataset, clase)
    path_destino = os.path.join(data_nuevo, clase)
    os.makedirs(path_destino, exist_ok=True)
    contador = 0
    for archivo in os.listdir(path_origen):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            path_imagen = os.path.join(path_origen, archivo)
            imagen = cv2.imread(path_imagen)
            if imagen is None:
                continue
            #Aplicamos las tecnicas
            # Escala de grises
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            # Eliminación de ruido
            blur = cv2.GaussianBlur(gris, (5,5), 0)
            # Mejora del contraste
            contraste = cv2.equalizeHist(blur)
            # Umbralización automática
            _, umbral = cv2.threshold(
                contraste,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            #Almacenamos las imagenes
            path_guardar = os.path.join(path_destino, archivo)
            cv2.imwrite(path_guardar, umbral)
            contador += 1
            total += 1
            #Mostramos un ejemplo
            if visualizacion:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(imagen_rgb)
                plt.title("Imagen Original")
                plt.axis("off")
                plt.subplot(1,2,2)
                plt.imshow(umbral, cmap="gray")
                plt.title("Imagen Aplicada Umbralizacion")
                plt.axis("off")
                plt.show()
                visualizacion = False
    print(f"{clase}: {contador} imágenes umbralizadas")