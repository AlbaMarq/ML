## LIBRERÍAS

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utiles import *

## CARGAR LOS DATOS

df = pd.read_csv('prueba/time_series.csv', sep=',', header=None, index_col=0, names=['fecha','unidades'])
print(df.head())

# Separar la fecha
df = df.reset_index()
#df['weekday'] = pd.to_datetime(df['fecha']).apply(lambda x: x.weekday())
df['weekday'] = pd.DatetimeIndex(df['fecha']).weekday
df['month'] = pd.DatetimeIndex(df['fecha']).month
print(df.head())

# EPOCHS=40
# PASOS=7

# scaler = MinMaxScaler(feature_range=(-1, 1))

# reframed = transformar(df, scaler)
# print(reframed)

# reordenado = reframed[ ['weekday','month','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)','var1(t)'] ]
# reordenado.dropna(inplace=True)
# print(reordenado)

# training_data = reordenado.drop('var1(t)', axis=1)
# target_data = reordenado['var1(t)']
# cant = len(df.index)
# valid_data = training_data[cant-30:cant]
# valid_target = target_data[cant-30:cant]

# training_data = training_data[0:cant]
# target_data = target_data[0:cant]
# print(training_data.shape, target_data.shape, valid_data.shape, valid_target.shape)
# print(training_data.head())

# model = crear_modeloEmbeddings()