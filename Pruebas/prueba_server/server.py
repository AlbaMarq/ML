"""Filename: server.py
"""
import pandas as pd
# from sklearn.externals import joblib
import joblib 
import tensorflow as tf
from flask import Flask, has_request_context, jsonify, request

from utiles import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	"""API request
	"""
	try:
		req_json = request.get_json()
		input = pd.read_json(req_json, orient='records')
	except Exception as e:
		raise e

	if input.empty:
		return(has_request_context())
	else:
		#Load the saved model
		print("Cargar el modelo...")
		loaded_model = cargarModeloSiEsNecesario()

		print("Hacer Pronosticos")
		continuas = input[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
		predictions = loaded_model.predict([input['weekday'], input['month'], continuas])

		print("Transformando datos")
		loaded_scaler = load_object('data/scaler_time_series.pkl')
		inverted = loaded_scaler.inverse_transform(predictions)
		inverted = inverted.astype('int32')

		final_predictions = pd.DataFrame(inverted)
		final_predictions.columns = ['ventas']
		
		print("Enviar respuesta")
		responses = jsonify(predictions=final_predictions.to_json(orient="records"))
		responses.status_code = 200
		print("Fin de Peticion")
		
		return (responses)

global_model = None

def cargarModeloSiEsNecesario():
	global global_model
	if global_model is not None:
		print('Modelo YA cargado')
		return global_model
	else:
		# global_model = crear_modeloEmbeddings()
		# global_model.load_weights('prueba/data/pesos.h5')
		global_model = tf.keras.models.load_model('data/red_time_series.h5')
		global_model.load_weights('data/pesos.h5')
		print('Modelo Cargado')
		return global_model
		