from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
import numpy as np
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
svm_credit =  SVC(kernel='rbf',random_state=1, C=2.0)
svm_credit.fit(x_credit_treinamento,y_credit_treinamento)
previsoes = svm_credit.predict(x_credit_teste)
accuracy = accuracy_score(y_credit_teste,previsoes)
print(accuracy)