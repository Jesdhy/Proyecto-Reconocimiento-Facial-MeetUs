from datetime import datetime
from flask import Flask, render_template, Response
import cv2
import os
import numpy as np

# Obtiene la ruta completa del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construye la ruta completa al archivo XML del clasificador Haar Cascade
cascade_path = os.path.join(current_dir, 'cascades', 'haarcascade_eye.xml')

# Carga el clasificador Haar Cascade
eye_cascade = cv2.CascadeClassifier(cascade_path)

@app.route('/')
@app.route('/DEFINICION PROYECTO')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='DEFINICION',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/ACERCA DEL PROYECTO')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='ACERCA DEL PROYECTO',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/PYTHON')
def PROYECTO():
    """proyecto"""
    return render_template(
        'about.html',
        title='ACERCA DEL PROYECTO',
        year=datetime.now().year,
        message='Your application description page.'

    )

# Ruta de la carpeta que contiene las imágenes de referencia de los rostros
carpeta_imagenes = "D:\PROGRAMAS FILE\PYTHON\proyectos\DETECCION_ROSTROS\DETECCION_ROSTROS\Base_Datos"

# Cargar el clasificador pre-entrenado de detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar las imágenes de referencia de los rostros
rostros_referencia = []
nombres_imagenes = []

for nombre_archivo in os.listdir(carpeta_imagenes):
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
    if os.path.isfile(ruta_imagen):
        imagen_referencia = cv2.imread(ruta_imagen)
        imagen_referencia_gris = cv2.cvtColor(imagen_referencia, cv2.COLOR_BGR2GRAY)
        rostros_referencia.append(imagen_referencia_gris)
        nombres_imagenes.append(os.path.splitext(nombre_archivo)[0])

def generar_fotogramas():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen de la cámara
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Comparar cada rostro detectado con las imágenes de referencia
            for i, rostro_referencia in enumerate(rostros_referencia):
                rostro_gris_redimensionado = cv2.resize(roi_gray, (rostro_referencia.shape[1], rostro_referencia.shape[0]))
                diferencia = cv2.absdiff(rostro_referencia, rostro_gris_redimensionado)
                suma_diferencia = np.sum(diferencia)
                umbral = 1000000

                if suma_diferencia < umbral:
                    nombre_imagen = nombres_imagenes[i]
                    cv2.putText(frame, "Rostro Similar: " + nombre_imagen, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        # Retorna el contenido de la transmisión en tiempo real
        return Response(generar_fotogramas(), mimetype='multipart/x-mixed-replace; boundary=frame')


    if __name__ == '__main__':
        app.run(debug=True)