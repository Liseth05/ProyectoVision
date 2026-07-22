#Contiene las MISMAS funciones de extraccion de caracteristicas que se
#usaron en ClasificacionSVM_RF.py para entrenar SVM y Random Forest.


import cv2
import numpy as np

IMG_SIZE = (128, 128)


def hu_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(th)
    hu = cv2.HuMoments(moments)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.flatten()


def sift_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)


def orb_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(32)
    return np.mean(descriptors, axis=0)


def extraer_vector_combinado(img):
    img = cv2.resize(img, IMG_SIZE)
    hu = hu_features(img)
    sift = sift_features(img)
    orb = orb_features(img)
    return np.concatenate([hu, sift, orb])


def preparar_imagen_cnn(img, tam=128):
    img = cv2.resize(img, (tam, tam))
    img = img / 255.0
    return np.expand_dims(img, axis=0)