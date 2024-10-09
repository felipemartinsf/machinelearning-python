 #regressão logistíca
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split

#base_plano_saude = pd.read_csv(r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\unifesp\git\machinelearning-python\estudo py\aula22\plano_saude.csv')
#x_plano_saude = base_plano_saude.iloc[:,0].values
#y_plano_saude = base_plano_saude.iloc[:,1].values 

#corr = np.corrcoef(x_plano_saude,y_plano_saude) # isso aqui calcula a correlação entre as variaveis. mt util 
#print(corr)

#x_plano_saude = x_plano_saude.reshape(-1,1) #agora está no formato de matriz, necessaria para o sklearn regressor funcionar
#regressor = LinearRegression()
#regressor.fit(x_plano_saude,y_plano_saude)
#regressor.intercept_ indica o b0
#regressor.coef_ indica o b1
#previsoes = regressor.predict(x_plano_saude) # sem dividir em teste para testar somente como fica o gráfico
#x_plano_saude = x_plano_saude.ravel()
#grafico = px.scatter(x= x_plano_saude, y = y_plano_saude)
#grafico.add_scatter(x = x_plano_saude, y= previsoes, name = 'Regression')
#grafico.show()


#visualizador = ResidualsPlot(regressor)
#visualizador.fit(x_plano_saude,y_plano_saude)
#visualizador.poof() # esse gráfico são os residuais. mostra a distancia acumulada da distancia dos valores reais para os calculados.

base_casa_preco = pd.read_csv(r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\unifesp\git\machinelearning-python\estudo py\aula22\house_prices.csv')
#print(base_casa_preco.describe()) #mostra algumas informacoes importantes sobre a base. bom pra conhecer melhor
#print(base_casa_preco.isnull().sum()) # mostra por coluna a soma de vazios ( ta tudo normal, n precisa fazer tratamento )
#print(base_casa_preco.select_dtypes(include='number').corr()) # printa  a correlacao coluna x coluna
#figura = plt.figure(figsize=(20,20))
#sns.heatmap(base_casa_preco.select_dtypes(include='number').corr(), annot=True)
#plt.show() # mostra o heatmap com a correlação
x_casas = base_casa_preco.iloc[:,5:6].values
y_casas =  base_casa_preco.iloc[:,2].values

xCasasTreinamento, xCasasTeste, yCasasTreinamento, yCasasTeste = train_test_split(x_casas,y_casas,test_size=0.2,random_state=0)
regressorSimple = LinearRegression()
regressorSimple.fit(xCasasTreinamento,yCasasTreinamento)
previsoesCasa = regressorSimple.predict(xCasasTeste)
grafico = px.scatter(x= xCasasTreinamento, y = yCasasTreinamento)
grafico.add_scatter(x = xCasasTeste, y= previsoesCasa, name = 'Regression')
grafico.show()