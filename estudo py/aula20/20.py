from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\unifesp\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
#
x_credit = np.concatenate((x_credit_treinamento,x_credit_teste),axis=0)
y_credit = np.concatenate((y_credit_treinamento,y_credit_teste),axis=0)

parametros = {'criterion':['gini','entropy'],
              'splitter':['best','random'],
              'min_samples_split':[2,4,10],
              'min_samples_leaf':[1,5,10]}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=parametros)
grid_search.fit(x_credit,y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)
#serve para achar os melhores parâmetros das opcoes que voce colocar