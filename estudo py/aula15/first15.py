from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula5\census.pkl'
with open (path,'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)
regression_census = LogisticRegression(random_state=1)
regression_census.fit(x_census_treinamento,y_census_treinamento)
previsoes = regression_census.predict(x_census_teste)
accuracy = accuracy_score(y_census_teste,previsoes)
print(accuracy)