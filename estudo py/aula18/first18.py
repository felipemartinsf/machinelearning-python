from sklearn.neural_network import MLPClassifier #mulyi layer perceptron
import pickle
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)


rede_neural_credit = MLPClassifier(max_iter=1000, tol=0.0000100, hidden_layer_sizes=(100,100)) #max_iter é o que vai encontrar o menor erro possível. verbose mostra todas as iteracoes e o erros dela, tol é 
#quanto ele testa basicamente antes de fala o melhor resultado. diminuir faz ele rodar mais 
rede_neural_credit.fit(x_credit_treinamento,y_credit_treinamento)
previsoes = rede_neural_credit.predict(x_credit_teste)
accuracy = accuracy_score(y_credit_teste,previsoes)
print(accuracy)