from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pickle
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
arvore_credot = DecisionTreeClassifier(criterion='entropy',random_state=0)
arvore_credot.fit(x_credit_treinamento,y_credit_treinamento)
previsoes = arvore_credot.predict(x_credit_teste)
accuracy = accuracy_score(y_credit_teste,previsoes)
print(accuracy)