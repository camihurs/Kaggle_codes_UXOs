"""The task is to train a network to discriminate between sonar signals bounced off a metal cylinder and 
those bounced off a roughly cylindrical rock
De esta página se pueden obtener tal vez más datasets: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
"""

import pandas as pd

#Cargamos el dataset
df = pd.read_csv(r"C:\Users\camih\Documents\DOCTORADO\Doctorado Belgica\PAPERS\Analysis of Hidden Units in a Layered Network\archive\sonar.all-data.csv")

#Obtenemos algunas características útiles del dataset