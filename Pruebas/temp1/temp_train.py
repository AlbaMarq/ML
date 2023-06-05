'''' LIBRERÍAS '''

from funciones import * # Importa las funciones declaradas en otro archivo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Para escalar los datos


'''' CARGAR DATOS '''

filename = 'temperatura/temp.csv'
names = ['fecha','temperatura']
df = pd.read_csv(filename, names=names)
print(df.shape)
print(df.head())



'''' PREPROCESAR LOS DATOS '''

# Separar la fecha
df['dia'] = pd.DatetimeIndex(df['fecha']).day
df['mes'] = pd.DatetimeIndex(df['fecha']).month
print(df.head())

# Transfomar los datos (método escalamiento)
scaler = MinMaxScaler(feature_range=(0, 1))    # Rango de escalamiento
reframed = transformar(df, scaler)

# Reordenar columnas
reordenado = reframed[['dia','mes','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)','var1(t)']]
reordenado.dropna(inplace=True)

# Datos de entrenamiento
training_data = reordenado.drop('var1(t)', axis=1)
target_data = reordenado['var1(t)']
cant = len(df.index)
# Datos de validación
valid_data = training_data[cant-30:cant]
valid_target = target_data[cant-30:cant]

training_data = training_data[0:cant]
target_data = target_data[0:cant]



''''' CREAR MODELO '''

model = crear_modelo()



'''' ENTRENAR MODELO '''

continuas = training_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
valid_continuas = valid_data[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]

history = model.fit([training_data['dia'],training_data['mes'],continuas], target_data, epochs=EPOCHS,
                 validation_data=([valid_data['dia'],valid_data['mes'],valid_continuas],valid_target))

results = model.predict([valid_data['dia'],valid_data['mes'],valid_continuas])

print( 'Resultados escalados',results )
inverted = scaler.inverse_transform(results)
print( 'Resultados',inverted )



'''' GUARDAR '''

save_object('temperatura/data/scaler_time_series.pkl', scaler)
model.save('temperatura/data/red_time_series.h5')
model.save_weights('temperatura/data/pesos.h5')