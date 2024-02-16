import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import numpy as np
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula5\census.pkl'
with open (path,'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)
naive_census = GaussianNB()
naive_census.fit(x_census_treinamento,y_census_treinamento)
previsoes = naive_census.predict(x_census_teste)
print(accuracy_score(y_census_teste,previsoes)) #esse algoritmo nao foi daora nao taxa bem baixa
#quando nao executava o escalonamento ele ficava com 70%, entao as vezes vale a pena ir testando pra ver
