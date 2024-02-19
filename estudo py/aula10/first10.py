import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula5\census.pkl'
with open (path,'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)
arvore_random = RandomForestClassifier(n_estimators=250,random_state=0,criterion='entropy')
arvore_random.fit(x_census_treinamento,y_census_treinamento)
previsoes = arvore_random.predict(x_census_teste)
accuracy = accuracy_score(y_census_teste,previsoes)
print(accuracy)