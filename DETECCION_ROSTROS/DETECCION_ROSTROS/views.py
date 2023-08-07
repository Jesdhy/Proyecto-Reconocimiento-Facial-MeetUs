"""
Routes and views for the flask application.
"""

from DETECCION_ROSTROS import app
from flask import Flask, render_template, Response
import cv2
import os
import pygame
import threading
import numpy as np



if __name__ == '__main__':
    app.run(debug=True)

#Definir rutas
@app.route('/')

@app.route('/INICIO')
def index1():
    return render_template('index1.html')

carpeta_imagenes = "C:/Users/Jessy/Desktop/DeteccionRostros"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

rostros_referencia = []
nombres_imagenes = []

#Extrae el nombre de las imagenes
for nombre_archivo in os.listdir(carpeta_imagenes):
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    if os.path.isfile(ruta_imagen):
        imagen_referencia = cv2.imread(ruta_imagen)
        imagen_referencia_gris = cv2.cvtColor(imagen_referencia, cv2.COLOR_BGR2GRAY)
        rostros_referencia.append(imagen_referencia_gris)
        nombres_imagenes.append(os.path.splitext(nombre_archivo)[0])

#Reproduce el sonido 
def reproducir_sonido(ruta):
    pygame.init()
    sound = pygame.mixer.Sound(ruta)
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))
    pygame.quit()

#Camara
def generar_fotogramas():
    video = cv2.VideoCapture(0)
   
 #Reconcimiento Facial
    while True:
       
        exito, frame = video.read()
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        Rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in Rostros:
            reg_gris = gris[y:y+h, x:x+w]
            reg_color = frame[y:y+h, x:x+w]

            #comparar el rostro detectado con las imagenes de referencia
            for i, rostro_referencia in enumerate(rostros_referencia):
                rostro_gris_redimensionado = cv2.resize( reg_gris, (rostro_referencia.shape[1], rostro_referencia.shape[0]))            
                diferencia = cv2.absdiff(rostro_referencia, rostro_gris_redimensionado)
                suma_diferencia = np.sum(diferencia)
              
                umbral = 1000000

                #Mostrara el mensaje si reconoce o no el rostro 
                if suma_diferencia < umbral:
                    nombre_imagen = nombres_imagenes[i]
                    cv2.putText(frame, "Rostro Similar: " + nombre_imagen, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                    ruta_del_sonido = 'D:\PROGRAMAS FILE\PYTHON\proyectos\DETECCION_ROSTROS\DETECCION_ROSTROS\Sonido\Advertencia.wav'
                    threading.Thread(target=reproducir_sonido, args=(ruta_del_sonido,)).start()
                    break
                else:
                    cv2.putText(frame, "Rostro Desconocido: ", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break
           
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        exito, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()


@app.route('/video_rostro')
def video_rostro():
    # Retorna el contenido de la transmisión en tiempo real
    return Response(generar_fotogramas(), mimetype='multipart/x-mixed-replace; boundary=frame')

