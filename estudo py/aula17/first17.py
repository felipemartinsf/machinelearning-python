from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula5\census.pkl'
with open (path,'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)
svm_census = SVC(kernel='linear',random_state=1, C=1.0)
svm_census.fit(x_census_treinamento,y_census_treinamento)
predict = svm_census.predict(x_census_teste)
accuracy = accuracy_score(y_census_teste,predict)
print(accuracy)
