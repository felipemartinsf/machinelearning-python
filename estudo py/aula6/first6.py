from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula6\risco_credito.pkl'
with open(path,'rb') as f:
    x_risco,y_risco = pickle.load(f)

arvore_risco = DecisionTreeClassifier(criterion='entropy')
#criterion é como ele classifica.
arvore_risco.fit(x_risco,y_risco)

previsoes = arvore_risco.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes)