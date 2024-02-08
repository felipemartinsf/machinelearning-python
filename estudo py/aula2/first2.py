import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle


caminho_arquivo = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\estudo\estudo py\aula2\census.csv'
base_census = pd.read_csv(caminho_arquivo)

x_census = base_census.iloc[:,0:14].values
y_census = base_census.iloc[:,14].values

#aqui vai acontecer o seguinte: tem mt string nos valores e a gente precisa usar elas pra fazer calculo. por isso, a gente vai transformar as strings em numeros específicos
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder() 
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:,1] = label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:,3] = label_encoder_education.fit_transform(x_census[:,3])
x_census[:,5] = label_encoder_marital.fit_transform(x_census[:,5])
x_census[:,6] = label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:,7] = label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:,8] = label_encoder_race.fit_transform(x_census[:,8])
x_census[:,9] = label_encoder_sex.fit_transform(x_census[:,9])
x_census[:,13] = label_encoder_country.fit_transform(x_census[:,13])
#desvantages do label enconder: as coisas nao ficam na mesma escala e isso acaba dando pesos diferentes

onehot_census = ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),[1,3,5,6,7,8,9,13])],remainder='passthrough')
x_census = onehot_census.fit_transform(x_census).toarray()
#aqui, inves de dar um valor numerico pra as coisas ele vai dar um valor de 01 e criar mais colunas. assim representa qual é por 00001 etc

scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census,y_census,test_size=0.15,random_state=0)


with open ('census.pkl',mode='wb') as f:
    pickle.dump([x_census_treinamento,y_census_treinamento,x_census_teste,y_census_teste],f)
