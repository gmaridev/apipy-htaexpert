from django.http import HttpResponse
from keras.models import Model,model_from_json
from keras.models import Sequential
from django.shortcuts import render
#from keras.layers.core import Dense
from tensorflow.keras.layers import Activation, Dense, Dropout
from keras import regularizers
import pandas as pd
from keras.models import load_model

import numpy as np
def entrenar():
  training = pd.read_csv("apipy/datos.csv",sep=";")

  columnsI = ["x0","x1","x2","x3","x4"]
  x_input = training[list(columnsI)].values
  columnsO = ["T0","T1","T2","T3"]
  y_target = training[list(columnsO)].values

  model = Sequential()
  model.add(Dense(20,input_dim=5,activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
  model.add(Dense(20,activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
  model.add(Dense(20, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(4,activation='sigmoid'))

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

  model.fit(x_input,y_target,epochs=900)

  model_json = model.to_json()
  with open("apipy/model.json","w") as json_file:
    json_file.write(model_json)
  model.save_weights("apipy/model.h5")
  
def respuesta(modelo, X):
    # Convertir la entrada X a un array de NumPy con la forma (1, 5)
    X = np.array([X])
    
    # Realizar la predicción
    respuesta0 = modelo.predict(X)
    
    # Redondear los valores de la predicción
    respuesta1 = [round(i) for i in respuesta0[0]]
    
    D = ""
    P = 0.0
    
    # Determinar la respuesta basada en la salida del modelo
    if respuesta1[0] == 1:
        D = "Ninguna"
        P = respuesta0[0][0]
    elif respuesta1[1] == 1:
        D = "Baja"
        P = respuesta0[0][1]
    elif respuesta1[2] == 1:
        D = "Media"
        P = respuesta0[0][2]
    elif respuesta1[3] == 1:
        D = "Alta"
        P = respuesta0[0][3]
    
    return D, str(round(P, 3))
def agregar(x0,x1,x2,x3,x4,D):
  T = T0 = T1 = T2 = T3 = 0;
  if D=="Ninguna":
    T , T0 = 0,1    
  if D=="Baja":
    T , T1 = 1,1
  if D=="Media":
    T , T2 = 2,1
  if D=="Alta":
    T , T3 = 3,1
  data = {
      'x0':[x0],
      'x1':[x1],
      'x2':[x2],
      'x3':[x3],
      'x4':[x4],
      'T' :[T],
      'T0':[T0],
      'T1':[T1],
      'T2':[T2],
      'T3':[T3]      
      }
  df = pd.DataFrame(data)
  df.to_csv("apipy/datos.csv",mode='a',sep=";",index=False, header=False)
  
def recibe(request, x0, x1, x2, x3, x4, A):
    A = int(A)
    if A == 0:
        X = [float(x0), float(x1), float(x2), float(x3), float(x4)]

        # Cargar el modelo desde el archivo model.h5
        loaded_model = load_model("apipy/model.h5")
        
        # Compilar el modelo si es necesario
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        D, P = respuesta(loaded_model, X)
        agregar(x0, x1, x2, x3, x4, D)
        resultado = D + "-" + P
        print(resultado)
    else:
        entrenar()
        resultado = "OK"
    return HttpResponse(resultado)