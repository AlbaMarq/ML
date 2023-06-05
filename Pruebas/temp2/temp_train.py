import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.model_selection import train_test_split
import datetime
from datetime import datetime
from datetime import date
import numpy as np

# Cargar los datos desde un archivo CSV
data = pd.read_csv('temp_gpt/data/temp3.csv')
data['fecha'] = pd.to_datetime(data['fecha'], format='%d/%m/%Y')
print(data.head)

# Preparar los datos para el entrenamiento
X = data['fecha'].apply(lambda x: datetime.strftime(x,"%Y%m%d"))
X = X.astype('int64').values.reshape(-1, 1)
print(X)
y = data['temperatura'].astype('float32').values
print(y)

# Separar los datos de entrenamiento y test
test_size = 0.3
seed = 7 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)


# Evaluar el modelo
acc = modelo.score(X_test, y_test)
print('Exactitud: {:.2f}%'.format(acc * 100))


# Guardar el modelo en un archivo
joblib.dump(modelo, 'temp_gpt/data/modelo.pkl')


# Predecir un nuevo dato
nuevo = date(2023,5,11)
nuevo1 = nuevo.strftime('%Y%m%d')
nuevo2 = int(nuevo1)
nuevo3 = np.array(nuevo2).reshape(1, -1)
nuevo4 = nuevo.strftime('%d-%m-%Y')

prediccion = modelo.predict(nuevo3)
print(type(prediccion))
print(f'La temperatura estimada para el día {nuevo4} es {float(prediccion):.2f} grados.')