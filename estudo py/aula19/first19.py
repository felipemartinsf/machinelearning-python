from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula5\census.pkl'
with open (path,'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

rede_neural_census = MLPClassifier(max_iter=1000, tol=0.000010,hidden_layer_sizes=(55,55))
rede_neural_census.fit(x_census_treinamento,y_census_treinamento)
previsoes = rede_neural_census.predict(x_census_teste)
print(accuracy_score(y_census_teste,previsoes)) 