from sklearn.model_selection import cross_val_score, KFold
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
path= r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\unifesp\git\machinelearning-python\estudo py\aula4\credit.pkl'
with open(path, 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)
y_credit_treinamento = np.array(y_credit_treinamento).astype(int)
x_credit = np.concatenate((x_credit_treinamento,x_credit_teste),axis=0)
y_credit = np.concatenate((y_credit_treinamento,y_credit_teste),axis=0)
#cross_val_score verifica qual algoritmo é melhor que o outro
resultados_arvore = []
resultados_random_forest = []
resultados_knn =[]
resultados_logistica = []
resultados_svm = []
resultados_neural = [] #faço isso que eu fiz para cada um dos resultados, ai vejo no final qual foi melhor

for i in range(30): #30 testes é um valor padrão.
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    arvore = DecisionTreeClassifier(criterion ='entropy',min_samples_leaf=1,min_samples_split=5,splitter='best')
    score = cross_val_score(arvore, x_credit, y_credit, cv=kfold)
    resultados_arvore.append(score.mean())

    random = RandomForestClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=5,n_estimators=10)
    score = cross_val_score(random, x_credit, y_credit, cv=kfold)
    resultados_random_forest.append(score.mean())

    knn = KNeighborsClassifier()
    score = cross_val_score(knn, x_credit, y_credit, cv=kfold)
    resultados_knn.append(score.mean())

    logistic = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
    score = cross_val_score(logistic, x_credit, y_credit, cv=kfold)
    resultados_logistica.append(score.mean())

    svm = SVC(C=2.0, kernel='rbf')
    score = cross_val_score(svm, x_credit, y_credit, cv=kfold)
    resultados_svm.append(score.mean())

    neural = MLPClassifier(activation='relu',batch_size=56,solver='adam'     )
    score = cross_val_score(neural, x_credit, y_credit, cv=kfold)
    resultados_neural.append(score.mean())
    


resultados = pd.DataFrame({'Arvore':resultados_arvore, 'Random':resultados_random_forest,
                           'KNN':resultados_knn, 'logistica':resultados_logistica,
                           'SVM':resultados_svm, 'neural':resultados_neural})

print(resultados)
resultados.describe()