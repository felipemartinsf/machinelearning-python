from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.naive_bayes import GaussianNB

path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\estudo\estudo py\aula3\risco_credito.csv'
base_risco_credito = pd.read_csv(path)

x_risco = base_risco_credito.iloc[:,0:4].values #busca do 0 ate 0 3
y_risco = base_risco_credito.iloc[:,4].values #busca o 4
label_enconder_historia = LabelEncoder()
label_enconder_divida = LabelEncoder()
label_enconder_garantia = LabelEncoder()
label_enconder_renda = LabelEncoder()

x_risco[:,0] = label_enconder_historia.fit_transform(x_risco[:,0])
x_risco[:,1] = label_enconder_divida.fit_transform(x_risco[:,1])
x_risco[:,2] = label_enconder_garantia.fit_transform(x_risco[:,2])
x_risco[:,3] = label_enconder_renda.fit_transform(x_risco[:,3])
#nao precisa fazer a aplicacao do onehot
#ruim = 2 / desconhecida = 1 / boa = 1
#alta = 0/ 
#nenhuma = 1 /adequada = 0 
#maior 35 = 2 /menor que 15 = 0

naive_risco_credito = GaussianNB() #distribuição estatística
naive_risco_credito.fit(x_risco,y_risco) #geracao da tabela de probabilidaes.

previsao = naive_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsao)