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

#defining the deep neural network structure (more than one hidden layer)##########################################
#Tres capas ocultas, todas con función de activación ReLU
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

    def forward(self,x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

#Instantiate the models and move them to the GPU
#¿Eso de mover a la GPU es automático?

deep_model = Deep()
print(deep_model)


#Testing the data flow through the networks with random data, the result will be around 0.5 probability bacause
#the network is not trained yet and data is just blah blah blah for the model

random_data = torch.rand(num_features)
print(random_data)

result_wide = wide_model(random_data)
result_deep = deep_model(random_data)

print(f"\nResult Wide: {result_wide}")
print(f"Result Deep: {result_deep}")


"""
Defining the FNN training loop
After defining the network attributes and the forward method we can create the training
loop, which will feed batches of data to the model, get the output to calculate the loss
and using this loss in the backward pass to update the learnable parameters (i.e. weights
and biases) of the network in each iteration (batch).

The loss function calculates the difference between the predicted output and the ground
truth labels. The optimizer adjust the parameters of the network towards minimization of
the difference between the predictions and the actual labels.

We define both the loss function and optimizers in the training loop to instantiate them
in each iteration, allowing unbiased comparison between the performance of the models.
"""

from tqdm.notebook import tqdm_notebook
import copy

#tqdm makes it easy to implement a progress bar to monitor the training status and time
tqdm_notebook.pandas()
"""
La línea de código tqdm_notebook.pandas() registra la función progress_apply de tqdm con
el método apply de pandas. Esto permite mostrar una barra de progreso cuando se aplica una
función a un DataFrame o una Serie de pandas utilizando el método apply. Después de
ejecutar esta línea de código, puedes utilizar el método progress_apply en lugar del
método apply para ver una barra de progreso mientras se aplica la función

La función tqdm_notebook crea una barra de progreso para Jupyter Notebook y se utiliza
junto con un bucle for para iterar sobre los datos de entrenamiento en lotes. El parámetro
batch_start especifica el rango de índices para iterar, mientras que unit y mininterval
controlan la apariencia y el comportamiento de la barra de progreso. El parámetro
disable se utiliza para desactivar la visualización de la barra de progreso
"""

def model_train(model, x_train, y_train, x_test, y_text, n_epochs=250, batch_size=10):
    #two labels, binary cross-entropy loss
    criterion = nn.BCELoss() #definimos la función de pérdida

    #Adam is a stochastic gradient descent optimizer that requires little memory and
    #parameter tuning
    optim = torch.optim.Adam(model.parameters(),lr=0.0001)

    batch_start = torch.arange(start=0,end=len(x_train), step=batch_size)

    #keeps the best model
    best_acc = -np.inf #strating at negative infinity

    best_weights = None #hold the best learnt parameters

    #training loop (epoch counter)
    for epoch in range(n_epochs):
        model.train() #set the model to training mode (e.g. activates dropout layers)
        with tqdm_notebook(
            batch_start, unit="batch", mininterval=0, disable=True
        ) as bar:
            bar.set_description(f"Epoch: {epoch}")
            for start in bar:
                #get the batch
                x_batch = x_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]

                #FNN forward pass: obtain predictions and loss of each training batch
                y_pred = model(x_batch) #predicts labels for training batch
                loss = criterion(y_pred,y_batch) #calculate the loss of the batch

                #FNN backward pass:
                optim.zero_grad()#zero the parameter gradients of previous runs
                loss.backward() # accumulates dloss/dx for every parameter that requires_grad=True
                #update weigths using the accumulated loss stored in parameter_x.grad

                optim.step()

                #print training progress (accuracy and loss)
                #using accuracy but for highly imbalanced datasets used balanced accuracy
                #compute metrics after the optimization step to obtain metrics for the batch
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))

                #calculates the accuracy after each epoch, i.e. when all the training data have
                #passed through the network.
                #model.eval(): turns off parts of the model (e.g. dropout layers) used for training
                #that aren't used in inference mode.
                #torch.no_grad(): disable autograd and may speed up computation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
            acc = float((y_pred.round()==y_test).float().mean())
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())

    #returns the best model (based on the best accuracy)
    model.load_state_dict(best_weights)
    return best_acc

"""k-fold Cross-Validation
With the data preprocessed and the network architectures on hands, we can carry out a
competition between these two approaches to determine the best estimator.

Note: We cannot test the model with the same data we've used for training the model
because the model will simply repeat the labels of samples already seen. We have to
separate a fraction of the full dataset to present it to the model and see if the
model can generalize well for samples never seen before.

Here we will seek how accurate are the predictions of a test set after a number
of iterations between these two models. The metrics will be recorded for later evaluation,
but the best model can be used right the way because we will encode its instantiation and
training.

The competition between the two networks is mediated by k-fold cross-validation from
scikit-learn. The k-fold cross-validation allows dynamic k-fold splits of the data into
training and test sets. It is similar to the Leave One Out strategy. Doing this we avoid
choosing a model due to a good result just by chance arising from splitting the
training data. Here, we assume that the data is Independent and Identically
Distributed (i.i.d.), meaning that they where obtained using the same process and each
 sample doesn't have a "memory" of past samples.

The k-fold cross-validation is good to choose parameters for the model. In this case,
 the network architecture is the parameter we're testing.

For this task we can use StratifiedKFold from Sklearn, which returns samples's indices
to split the data while preserving the percentage of samples of each label among the
splits. We can set the number of splits and a integer seed to control randomness and
ensure reproducibility during model evaluation. First, lets see the StratifiedKFold
behavior when generating indices:
"""

from sklearn.model_selection import StratifiedKFold, train_test_split

###Esto es sólo una prueba para el funcionamiento del stratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Tamaño de x: ", x.shape)
print("Tamaño de y: ", y.shape)
for i, (train_idx, test_idx) in enumerate(kfold.split(x,y)):
    print(f"Split {i}: ")
    print(f"    Train: index={train_idx[:10]}")#Solo mostramos 10 elementos, 10 filas de x
    print(f"    Test: index={test_idx[:10]}")#Igual para y


######Ahora sí viene lo de verdad

#split the data into train and test sets
x_train, x_test, y_train, t_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=123)

#setting kfold parameters
kfold = StratifiedKFold(n_splits5, random_state=42, shuffle=True)