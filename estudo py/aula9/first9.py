from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
random_forest_cr = RandomForestClassifier(n_estimators=62, criterion='entropy', random_state=0) #n_estimators é o numero de arvores...
random_forest_cr.fit(x_credit_treinamento,y_credit_treinamento)
previsoes = random_forest_cr.predict(x_credit_teste)
accuracy = accuracy_score(y_credit_teste,previsoes)
print(accuracy)