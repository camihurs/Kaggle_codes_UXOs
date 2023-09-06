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