import pickle
from api_train_model import *
from sklearn import preprocessing

# # definimos funciones de guardar y cargar
# def save_object(filename, object):
#  with open(''+filename, 'wb') as file:
#     pickle.dump(object, file)
 
# def load_object(filename):
#  with open(''+filename ,'rb') as f:
#     loaded = pickle.load(f)
#  return loaded
 
# # guardamos los objetos que necesitaremos mas tarde
# save_object('prueba/data/scaler_time_series.pkl', scaler)
# model.save('prueba/data/red_time_series.h5')
# model.save_weights("prueba/data/pesos.h5")
 
# cargamos cuando haga falta
# loaded_model = load_object('prueba/data/red_time_series.h5')
loaded_scaler = load_object('prueba/data/scaler_time_series.pkl')
loaded_model = crear_modeloEmbeddings()
loaded_model.load_weights('prueba/data/pesos.h5')

results = loaded_model.predict([valid_data['weekday'],valid_data['month'],valid_continuas])
print( 'Resultados escalados',results )
loaded_scaler = load_object('prueba/scaler_time_series.pkl')
inverted = loaded_scaler.inverse_transform(results)
print( 'Resultados',inverted )