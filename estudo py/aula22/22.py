 #regressão logistíca
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
base_plano_saude = pd.read_csv(r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\unifesp\git\machinelearning-python\estudo py\aula22\plano_saude.csv')
x_plano_saude = base_plano_saude.iloc[:,0].values
y_plano_saude = base_plano_saude.iloc[:,1].values 

corr = np.corrcoef(x_plano_saude,y_plano_saude) # isso aqui calcula a correlação entre as variaveis. mt util 
print(corr)

x_plano_saude = x_plano_saude.reshape(-1,1) #agora está no formato de matriz, necessaria para o sklearn regressor funcionar
regressor = LinearRegression()
regressor.fit(x_plano_saude,y_plano_saude)
#regressor.intercept_ indica o b0
#regressor.coef_ indica o b1
previsoes = regressor.predict(x_plano_saude) # sem dividir em teste para testar somente como fica o gráfico
x_plano_saude = x_plano_saude.ravel()
grafico = px.scatter(x= x_plano_saude, y = y_plano_saude)
grafico.add_scatter(x = x_plano_saude, y= previsoes, name = 'Regression')
grafico.show()


visualizador = ResidualsPlot(regressor)
visualizador.fit(x_plano_saude,y_plano_saude)
visualizador.poof() # esse gráfico são os residuais. mostra a distancia acumulada da distancia dos valores reais para os calculados.
