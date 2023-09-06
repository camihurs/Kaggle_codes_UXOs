#Este archivo usa una regresión logística para clasificar entre un cilindro de metal y una roca con forma cilíndrica
#Utiliza los datos que se usan en el paper con el mismo nombre, de Gorman y Sejnowski, de 1988


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""Las líneas de código que mencionas se utilizan para suprimir las advertencias en Python. El módulo `warnings` permite a
los programadores emitir advertencias y controlar cómo se manejan las advertencias¹. La función `warnings.filterwarnings('ignore')`
se utiliza para ignorar todas las advertencias, lo que significa que no se mostrarán en la consola cuando se ejecute el código³. Sin embargo,
es importante tener en cuenta que ignorar todas las advertencias puede no ser una buena práctica, ya que algunas advertencias pueden ser
importantes y deben ser atendidas.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Import the dataset
#La r antes de la ruta es para que no tome los \ como caracter de escape
df = pd.read_csv(r"C:\Users\camih\Documents\DOCTORADO\Doctorado Belgica\PAPERS\Analysis of Hidden Units in a Layered Network\archive\sonar.all-data.csv")
print(df.head())


#Encontramos la distribución de la etiqueta ("Label")
print(df['Label'].value_counts())
#M: Metal
#R: Rock

#Convertimos la columna "Label" en forma numérica
df['Label']=df['Label'].replace({'R':0,'M':1})
print(df.head())

#Encontramos el número de características y observaciones
print(df.shape)

#Revisamos si hay observaciones faltantes en alguna de las columnas
print(df.isnull().sum())

#Obtenemos los estadísticos descriptivos de los datos
print(df.describe().round(2))

#Dividimos los datos en features y label (características y etiqueta)
x = df.drop(columns='Label', axis=1)
y=df['Label']

#Dividimos los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2, test_size=0.2)
print(x.shape,x_train.shape,x_test.shape)

#Llamamos al modelo y entrenamos la regresión logística con los datos de entrenamiento
model = LogisticRegression()
model.fit(x_train,y_train)

#Predecimos las etiquetas para el dataset de entrenamiento
x_train_prediction = model.predict(x_train)

#Evaluamos el modelo
#Accuracy en los datos de entrenamiento
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print(training_data_accuracy)

#Ahora predecimos el valor de las etiquetas usando el modelo para el dataset de test
x_test_prediction = model.predict(x_test)

#Ahora vamos a evaluar el accuracy en los datos de prueba
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print(test_data_accuracy)


#Hacemos ahora un sistema predictivo----------------------------------------
input_data = df.iloc[0,:-1]#Tomamos la primera fila, y todas las columnas excepto la última
print(input_data)
#Transformamos ese objeto que es una serie en un arregle de numpy
input_data_as_numpy_array = np.asarray(input_data)
print(input_data_as_numpy_array)
print(input_data_as_numpy_array.shape) #Arroja (60,)
#Cambiamos la forma del arreglo de numpy ya que estamos prediciendo para una instancia
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)
print(input_data_reshaped.shape)#Arroja (1,60)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("El objeto es una roca")
else:
    print("El objeto es un cilindro de metal")