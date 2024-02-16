import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import numpy as np
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\estudo\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)

naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento) #essas duas linhas ele treina o algoritmo ja kkk como pode
previsoes = naive_credit_data.predict(x_credit_teste)

resposta = accuracy_score(y_credit_teste,previsoes)
print(f'{resposta*100}%')
respostsa2 = confusion_matrix(y_credit_teste,previsoes)

cm = ConfusionMatrix(naive_credit_data)
cm.fit(x_credit_treinamento,y_credit_treinamento)
cm.score(x_credit_teste,y_credit_teste)

print(classification_report(y_credit_teste,previsoes)) #ele eh meio ruim em verificar quem nao paga o empréstimo (0.64)