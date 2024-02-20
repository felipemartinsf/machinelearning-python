import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
#nao precisa realizar um treinamento porque nao gera modelo
knn_credit = KNeighborsClassifier(n_neighbors=5)
knn_credit.fit(x_credit_treinamento,y_credit_treinamento) #aqui eh criado o banco de dados.
previsoes = knn_credit.predict(x_credit_teste)
accuracy = accuracy_score(y_credit_teste,previsoes)
print(accuracy)