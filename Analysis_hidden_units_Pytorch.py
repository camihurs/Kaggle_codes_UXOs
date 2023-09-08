"""The task is to train a network to discriminate between sonar signals bounced off a metal cylinder and
those bounced off a roughly cylindrical rock
De esta página se pueden obtener tal vez más datasets: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
"""
#https://www.kaggle.com/code/cleverds/pytorch-sonar-data-classification-practice

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

#Cargamos el dataset
df = pd.read_csv(r"C:\Users\camih\Documents\DOCTORADO\Doctorado Belgica\PAPERS\Analysis of Hidden Units in a Layered Network\archive\sonar.all-data.csv")

#Obtenemos algunas características útiles del dataset
print(f"The dataset contains {len(df)} samples and {len(df.columns)-1} features")
print(df.head())

"""Building a accurate machine learning model heavily relies on quality of the data fed into it for training and testing. Analyzing how the data is
distributed and how relevant they are for predicting the outcome is an essential part of designing an accurate model. We do not expect that all
features in a real-world dataset will be important for predicting the outcome, and we can use some dimensionality reduction and feature selection
techniques to filter out those features that contain little information for the model. We seek for those features that has high variance but excluding
the features that are highly correlated, thus redundant for training the model. A lighter dataset will result in a simpler and more powerfull model,
with greater generalization capability when seeing new data.

To get some initial descriptive statistics of the dataset, the DataFrame method DataFrame.describe() shows the data central tendency, dispersion and 
quartile information.
"""

#Estadística descriptiva
print(df.describe())

"""
We can notice that the standard deviations (std) are quite high, showing that the data is greatly dispersed around the mean. It is commonly believed that 
features with low variance contributes little to the model performance and with high variance contributes more. However this is far from truth because this 
notion disconsiders that features might come in different units (e.g. meters and millimeters) that might differ greatly in terms of mean and std values. 
However, these features can be important even in a narrow range when predicting the outcome of a model.
"""

######Calculamos lla varianza de cada columna, aunque antes ya habíamos obtenido la desviación
#estándar
feat_var = df.var(numeric_only=True, skipna=True)
print(feat_var)

#Calculamos nuevamente la media de las columnas que representan la "intensidad" de las frecuencias
feat_mean = df.mean(numeric_only = True, skipna=True)
print(feat_mean)

disp = sns.lineplot(data=feat_mean, color="g", legend="brief", label="Mean")
disp = sns.lineplot(data=feat_var, color="r", legend="brief", label="Variance")
disp.set_xlabel("Features")
sns.despine() #remove top and right spines


"""
Checking if it is a balanced dataset. Imbalanced datasets are bad for model training because the model becomes biased towards the majority label.
"""

sns.countplot(x="Label", data=df).set_title("Label distribution")
sns.despine() #remove top and right spines

"""
The labels' distribution is almost balanced. We can also take a look at some density estimates of three variables. Instead of using an histogram to
approximate the probability density function, we will use kernel density estimation (KDE) that "smooths" the histogram and present a continuous density estimate.
"""

#creating subplots
fig, axs = plt.subplots(10,6,figsize=(15,15))#10 filas y 6 columnas
plt.subplots_adjust(hspace=0.7, wspace=0.3)
plt.suptitle("Feature kernel density distribution", fontsize=18)
sns.despine()

#iterating through dataset to plot in axes
for feat, ax in zip(df.columns, axs.ravel()):
    sns.kdeplot(data=df[feat], fill=True, ax=ax, color="royalblue")


"""
Checking the relationship between some randomly chosen features.
"""
#Pearson's correlation coefficient Entre la Frecuencia 1 y la Frecuencia 2
r,p = stats.pearsonr(x=df["Freq_1"], y=df["Freq_2"])
print("Pearson's correlation coefficiente: R={:.3f}, P={:.3f}".format(r,p))

disp = sns.jointplot(data=df, x="Freq_1", y="Freq_2", kind="reg")

"""
La conclusión estadística que mencionas indica que se ha calculado el coeficiente de
correlación de Pearson entre dos variables y se ha obtenido un valor de R=0.736. Esto
sugiere que existe una correlación positiva moderadamente fuerte entre las dos variables,
lo que significa que cuando una variable aumenta, la otra también tiende a aumentar.

El valor de P=0.000 indica que la correlación es estadísticamente significativa, lo que
significa que es muy poco probable que la correlación observada se deba al azar. En otras
palabras, podemos estar bastante seguros de que existe una relación real entre las dos
variables.
"""

#Pearson's correlation coefficient entre la Frecuencia 20 y la 30
r,p = stats.pearsonr(x=df["Freq_20"], y=df["Freq_30"])
print("Pearson's correlation coefficient: R={:.3f}, P={:.3f}".format(r,p))

disp = sns.jointplot(data=df, x="Freq_20", y="Freq_30", kind="reg")
plt.show()

"""
It seems that we have some highly correlated features. Some of these features could be
excluded from training to get a simpler and lighter classification model. Eliminating highly
correlated features can make our dataset less "noisy" and our model simpler. A correlation
heatmap plot can show correlated variables and the correlation degree, direction (positive
or negative correlation).

To plot the heatmap we might use a divergent color map to clearly show the positive and
negative correlations.
"""

df_corr = df.corr(numeric_only=True) #numeric_only excludes the target label column
heatplot = sns.heatmap(data=df_corr, vmin=-1, vmax=1, cmap="Spectral")

"""
There're both positive and negative correlations between variables, as we can see in the
heatmap. I guess it looks like an green alien :-|. We can get a mask of absolute correlations
to pick variables. When training the model, it can be useful to set different correlation
thresholds and see how model's accuracy behaves. For now lets set the threshold to 1 to
get the full dataset for training and testing.
"""

#get the mask (upper half of correlation matrix, absolute)
upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
print(upper.head())

#filter correlation by threshold
threshold = 1
cols_to_drop = [col for col in upper.columns if any(upper[col]>threshold)]
print(
    f"There are {len(cols_to_drop)} features with correlation higher than {threshold}."
)

#drop correlated columns
df.drop(columns=cols_to_drop, axis=1, inplace=True)
df.head()

#store the numer of remaining features (will be used to define the input size during training)
num_features = len(df.columns)-1 #features minus label columns, lo usamos más abajo en la red neuronal
print(f"There are {num_features} remaining in the dataset.")

"How the data look like after dropping highly correlated columns:"

sns.heatmap(df.corr(numeric_only=True),vmin=-1, vmax=1, cmap="Spectral")



#Ahora sí empieza la parte de Pytorch###############################

"""
First, lets split the training data into a feature matrix and a target
label vector. Here we convert the labels into numbers using LabelEncoder.
"""
#split into features vector and labels vector
x = df.iloc[:,:-1]#Todas las filas y todas las columnas menos la última
y = df.iloc[:, -1]#Todas las filas y la última columna

#Encoding labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print(f"Labels: {encoder.classes_} = [Metal, Rock] encoded into:")
print(y)

#converting features and labels to tensors

x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32).reshape(-1,1)

print(f"features dim: {x.shape}")
print(f"labels dim: {y.shape}")

"""
Designing Feedforward Neural Network (FNN) models¶
The model is built using the PyTorch nn module, which is the base class for neural networks built in PyTorch
and allows nested tree structure where submodules can be assigned as regular attributes. Just a refresher:

A FNN in PyTorch has fully-connected layers as attributes and a forward method defining the network architecture
and how the data will do the forward pass through the network.

attributes: variables stored in an instance or class
methods: functions encoded in an instance or class

In a neural network the number of layers and the number of neurons in each layer are hyperparameters, i.e.,
parameters hard-coded before training. A deep learning model has more than one hidden layer, while models with
more parameters in the layers is a wider model. Here we test both to determine which one will give more accurate predictions.
"""

#defining the wide neural network structure (a single hidden layer)########################
import torch.nn as nn

#The neural network child class is derived from nn.Module base class
#In the constructor, declare all the layers of the network

class Wide(nn.Module):
    #instantiate all modules (their attributes and methods)
    def __init__(self):
        #initialize attributes and methods of the parent class
        super().__init__()
        #input layer for 60 variables (60 units or neurons) and 180 output units; el 60 corresponde a las 60 columnas de las frecuencias
        # 180 corresponde a la cantidad de neuronas de la capa oculta
        self.hidden = nn.Linear(in_features=num_features, out_features=180)

        #activation of the layer (breaking linearity)
        self.relu = nn.ReLU()

        #the output es a real number for binary classification...
        self.output = nn.Linear(in_features=180, out_features=1)

        #...and the sigmoid takes the input (1) tensor and squeeze (reescale) it to [0,1] range
        #representing the probability of the target label of a given sample.
        #class 1=P, class 2=1-P (class 1)
        #Note: sigmoid is used for binary classification, softmax is an extension of sigmoid for multiclass problems
        self.sigmoid = nn.Sigmoid()

        #The forward function defines the neural network structure, with numero of units (neurons), activations,
        #regularizations, outputs...
        #Then, here we define how the network will be run from input to output:

    def forward(self,x):
        #taking the input, computing weights and applying non-linearity
        x = self.relu(self.hidden(x))

        #taking the output of the previous layer and squeezing it to the range [0,1]
        x = self.sigmoid(self.output(x))
        return x #x is the probability of class 1, while class 2 is (1-x)

wide_model = Wide()
print(wide_model)

#defining the deep neural network structure (more than one hidden layer)

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=num_features, out_features=60)#primera capa oculta de 60 neuronas
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60,60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60,60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60,1)
        self.sigmoid = nn.Sigmoid()

        