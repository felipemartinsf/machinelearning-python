''''
Glossário:
ML = MACHINE LEARNING
Machine Learning está dentro de IA
Machine learning é mais matématica, treinando algoritmos
Data mining é utilizar o algoritmo de machine learning para um fundamento
Rede neural é um tipo de machine learning
Deep learning é uma rede neural com muito mais dados
Big data é um imenso volume de dados
Métodos preditivos(ou supervisionada):
-Classificação: dividir  em classes usando ML (rotulos)
-Regressão:preve valores numéricos (numeros)
Métodos descritivos(ou não supervisionada):
-Associação: achar padrões e tentar prever se uma coisa gera outra.
-Agrupamento: juntar pessoas em grupos para descrever as necessidades delas.
-Outliers: encontrar desvios do padrão para análise.
-Padrões sequencias: parecido com associação mas é mais sequencial.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle



caminho_arquivo = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\estudo\estudo py\aula1\credit_data.csv'

#visualização básica de gráficos ------------------------------------------
base_credit = pd.read_csv(caminho_arquivo)
#print(np.unique(base_credit['default'],return_counts=True)) #contar valores únicos de um array. se return_counts tiver True ele conta o numero de vezes
contador = sns.countplot(x = base_credit['default'])#grafico de contagem
histograma = plt.hist(x=base_credit['age'])#cria um histograma, uma representacao grafica da distribuicao de frequencia
grafico = px.scatter_matrix(base_credit,dimensions=['age','income'],color='default') #grafico de dispersao, irá misturar os parametros e mostrar um gráfico. o color pinta confome as informacoes de default
#grafico.show() #.show mostra o grafico, é necessario fazer isso para mostrar ele.


#tratamento de dados --------------------------------------- iloc localiza por coluna e linha e loc sómente por coluna
base_credit_age_lost = base_credit.drop(base_credit[base_credit['age'] < 0].index) # isso apaga  a linha toda em qe a idade for negativa
media = base_credit_age_lost['age'].mean() #media das idades
print(media)
base_credit.loc[base_credit['age']<0] = media #onde ta bugado[
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True) #aqui altera na propria tabela os erros, por causa do inplace = true. ele procura onde ta vazio e automaticamente preenche com a media
#base_credit.loc[pd.isnul(base_credit['age'])] #aqui ve onde as idades estao nulas...
#print(base_credit.loc[base_credit['clientid']==29])


#previsores e classes ------------ previsores = x e classes = y. queremos prever as classes
x_credit = base_credit.iloc[:,1:4].values #: significa pra pegar todos. o value é pra formatar do jeito certo
y_credit = base_credit.iloc[:,4].values #isso pega todos os valores da ultima coluna
#padronização ou normalização para fazer os numreos estarem na mesma escala (padronizacao quando tem muito outlier e normalizacao quanto ta mais tranquilo)

#escalonamento
scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit) #aqui padroniza tudo e deixa numa escala bem parecida, pra que seja mais facil pro programa entender
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit,y_credit,test_size=0.25,random_state=0)
#passei 25% do banco para testes aqui,

with open ('credit.pkl',mode='wb') as f:
    pickle.dump([x_credit_treinamento,y_credit_treinamento,x_credit_teste,y_credit_teste],f)