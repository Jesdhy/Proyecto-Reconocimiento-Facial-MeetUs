"""
Routes and views for the flask application.
"""
#Bibliotecas
from DETECCION_ROSTROS import app
from flask import Flask, render_template, Response, url_for
import cv2
import os
import pygame
import threading
import numpy as np

if __name__ == '__main__':
    app.run(debug=True)

        
#Direccion principal por defecto
@app.route('/')
#Se crea una dirección INICIO
@app.route('/INICIO')
#funcion
def index1():
    #render_template llama a mi archivo html
    return render_template('index1.html')

@app.route('/INICIO/PAG')
#funcion
def pagina():
    #render_template llama a mi archivo html de la pagina del gobierno
    return render_template('menu.html')

@app.route('/INICIO/ABOUT')
#funcion
def contacto():
    #render_template llama a mi archivo html de la informacion de nosotr@s
    return render_template('contact.html')



# Ruta de la carpeta que contiene las imágenes de referencia de los rostros
carpeta_imagenes = "C:/Users/Jessy/Desktop/DeteccionRostros/Michael"

# Cargar el clasificador pre-entrenado de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar las imágenes de referencia de los rostros
rostros_referencia = []
nombres_imagenes = []
#El for Revisa cada elemento de la carpeta que tenemos
for nombre_archivo in os.listdir(carpeta_imagenes):
    #combina la ruta de la carpeta con el nombre del archivo
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    #veridica si es un archivo valido y entra al bucle
    if os.path.isfile(ruta_imagen):
        #Carga la imagen de la ruta 
        imagen_referencia = cv2.imread(ruta_imagen)
        #Convierte la imagen a  esacalas de grises
        imagen_referencia_gris = cv2.cvtColor(imagen_referencia, cv2.COLOR_BGR2GRAY)
        #guarda las imagenes que se encuentran en escala de grises
        rostros_referencia.append(imagen_referencia_gris)
        #quita la extension jpg de la imagen y solo toma el nombre
        nombres_imagenes.append(os.path.splitext(nombre_archivo)[0])
     
def reproducir_sonido(ruta):
    pygame.init()
    sound = pygame.mixer.Sound(ruta)
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))
    pygame.quit()

def generar_fotogramas():
    #prende la cámara, 0 porque es predeterminado por el sistema
    video = cv2.VideoCapture(0)
   
    while True:
        #lee el fortograma del video
        ret, frame = video.read()
        #convierte la imagen en escalas de grises 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen de la cámara con el clasificador face_cascade, detecta los rostros grises
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        #Extrae solo regines que le interecen del rostro detectado
        #X es la posicion, Y esquina sup izquierda, W ancho, H altura
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Comparar cada rostro detectado con las imágenes de referencia
            for i, rostro_referencia in enumerate(rostros_referencia):
                #redimensiona la región
                rostro_gris_redimensionado = cv2.resize(roi_gray, (rostro_referencia.shape[1], rostro_referencia.shape[0]))
                #calcula la diferencia entre pixceles
                diferencia = cv2.absdiff(rostro_referencia, rostro_gris_redimensionado)
                #np calcula la suma de esta matriz
                suma_diferencia = np.sum(diferencia)
                #Cantidad de referencia para que detecte mi rosto
                umbral = 1000000

                if suma_diferencia < umbral:
                    nombre_imagen = nombres_imagenes[i]
                    cv2.putText(frame, "Rostro Similar: " + nombre_imagen, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                    ruta_del_sonido = 'D:\PROGRAMAS FILE\PYTHON\proyectos\DETECCION_ROSTROS\DETECCION_ROSTROS\Sonido\Advertencia.wav'
                    threading.Thread(target=reproducir_sonido, args=(ruta_del_sonido,)).start()
                    
                    break
                else:
                    cv2.putText(frame, "Rostro Desconocido: ", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break
            #Marco de prueba para reconocer el rostro     
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()


@app.route('/video_rostro')
def video_rostro():
    # Retorna el contenido de la transmisión en tiempo real
    return Response(generar_fotogramas(), mimetype='multipart/x-mixed-replace; boundary=frame')



