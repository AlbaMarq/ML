#  Modelo de ML basándonos en el ejercicio de Pronóstico de Series Temporales que hace un pronóstico de ventas con redes neuronales con Embeddings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import data

# Importa las funciones declaradas en otro archivo
from utiles import *

df = pd.read_csv('prueba/time_series.csv', header=None, index_col=0, names=['fecha','unidades'], sep=',')
#df['weekday']=[x.weekday() for x in df.index]
#df['month']=[x.month for x in df.index]
df = df.reset_index()
#df['weekday'] = pd.to_datetime(df['fecha']).apply(lambda x: x.weekday())
df['weekday'] = pd.DatetimeIndex(df['fecha']).weekday
df['month'] = pd.DatetimeIndex(df['fecha']).month
print(df.head())

EPOCHS=40
PASOS=7

scaler = MinMaxScaler(feature_range=(-1, 1))

reframed = transformar(df, scaler)

reordenado = reframed[['weekday','month','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)','var1(t)']]
reordenado.dropna(inplace=True)

training_data = reordenado.drop('var1(t)', axis=1)
target_data = reordenado['var1(t)']
cant = len(df.index)
valid_data = training_data[cant-30:cant]
valid_target = target_data[cant-30:cant]

training_data = training_data[0:cant]
target_data = target_data[0:cant]
print(training_data.shape, target_data.shape, valid_data.shape, valid_target.shape)
print(training_data.head())

model = crear_modeloEmbeddings()

continuas = training_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
valid_continuas = valid_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]

history = model.fit([training_data['weekday'],training_data['month'],continuas], target_data, epochs=EPOCHS,
                 validation_data=([valid_data['weekday'],valid_data['month'],valid_continuas],valid_target))

results = model.predict([valid_data['weekday'],valid_data['month'],valid_continuas])

print( 'Resultados escalados',results )
inverted = scaler.inverse_transform(results)
print( 'Resultados',inverted )

# guardamos los objetos que necesitaremos mas tarde
save_object('prueba/data/scaler_time_series.pkl', scaler)
model.save('prueba/data/red_time_series.h5')
model.save_weights('prueba/data/pesos.h5')

# # cargamos cuando haga falta
# # loaded_model = load_object('prueba/data/red_time_series.h5')
# loaded_scaler = load_object('prueba/data/scaler_time_series.pkl')
# loaded_model = crear_modeloEmbeddings()
# loaded_model.load_weights('prueba/data/pesos.h5')

# # results = loaded_model.predict([valid_data['weekday'],valid_data['month'],valid_continuas])
# # print( 'Resultados escalados',results )
# # loaded_scaler = load_object('prueba/scaler_time_series.pkl')
# # inverted = loaded_scaler.inverse_transform(results)
# # print( 'Resultados',inverted )