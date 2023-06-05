'''' En el cmd: waitress-serve --listen=192.168.3.143:5000 server:app '''

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load('data/modelo.pkl')
#modelo = joblib.load('temp_gpt/data/modelo.pkl')


# Crear la aplicación Flask
app = Flask(__name__)


# Definir la ruta del servidor para responder a las solicitudes POST
@app.route('/predict', methods=['POST'])
def predict():

    # Obtener la fecha del cuerpo de la solicitud
    fecha_str = request.json['fecha']
    print(fecha_str)
    
    # Preparar los datos para la predicción
    X = int(fecha_str)
    X = np.array(X).reshape(1, -1)
    
    # Realizar la predicción
    prediccion = modelo.predict(X)
    print(prediccion)

    # Devolver la respuesta en formato JSON
    return jsonify({'temperatura': prediccion[0]})


# Iniciar el servidor
if __name__ == '__main__':
    app.run(host='192.168.3.143', port=5000)
