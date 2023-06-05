import requests
from datetime import date

# Definir la URL del servidor
url = 'http://192.168.3.143:5000/predict'

# Definir la fecha de la predicción
fecha = date(2022, 1, 10)
fecha_esp = fecha.strftime('%d-%m-%Y')

# Realizar la solicitud POST al servidor
respuesta = requests.post(url, json={'fecha': fecha.strftime('%Y%m%d')})

# Imprimir la respuesta completa
print(respuesta.text)

# Obtener la temperatura de la respuesta en formato JSON
temperatura = respuesta.json()['temperatura']

# Imprimir la temperatura
print(f'Temperatura predicha para el {fecha_esp}: {temperatura:.2f} grados')
