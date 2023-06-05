"""Creando un servidor Flask
"""

from flask import Flask

app = Flask(__name__)

@app.route('/users/<string:nombre>')
def hello_world(nombre=None):

	return ("Hola {}!".format(nombre))

# Para ejecutar la app en el servidor: 
# 1. Abrimos el cmd
# 2. Nos metemos en el directorio donde está el archivo de la aplicación
# 3. ejecutamos el siguiente comando: waitress-serve --listen=*:8000 mi_server:app