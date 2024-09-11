from sklearn.model_selection import cross_val_score, KFold
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
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

    #neural = MLPClassifier(activation='relu',batch_size=56,solver='adam', max_iter=400  , learning_rate_init=0.001)
    #score = cross_val_score(neural, x_credit, y_credit, cv=kfold)
    #resultados_neural.append(score.mean())
    

'''resultados = pd.DataFrame({'Arvore':resultados_arvore, 'Random':resultados_random_forest,
                           'KNN':resultados_knn, 'logistica':resultados_logistica,
                           'SVM':resultados_svm, 'neural':resultados_neural})'''

#print(resultados)
#resultados.describe()

#teste de normalidade nos resultados de cada um dos algoritmos
alpha = 0.05 #valor padrao do teste de shapiro. se o segundo valor retornado pelo shapiro for maior que o alpha, quer dizer que os dados nao estao em distribuicao normal. dessa formma, nao da pra aplicar ANOVA e TUKEY


print(shapiro(resultados_arvore), shapiro(resultados_random_forest), shapiro(resultados_knn), shapiro(resultados_logistica), shapiro(resultados_svm))

#sns.displot(resultados_random_forest, kind='kde')
#plt.show()

a, p = f_oneway(resultados_arvore, resultados_random_forest, resultados_logistica, resultados_knn, resultados_svm)
#isso é o teste anova, que verifica se o resultado dos algoritmos é o mesmo. se não for, da pra fazer o tukey para ver o melhor
resultados_algoritmos = {
    'accuracy': np.concatenate([resultados_arvore, resultados_random_forest, resultados_logistica, resultados_knn, resultados_svm]),
    'algoritmo': np.concatenate([
        ['arvore'] * 30,
        ['random'] * 30,
        ['logistica'] * 30,
        ['knn'] * 30,
        ['svm'] * 30
    ])
}
resultados_df = pd.DataFrame(resultados_algoritmos)
compara_algoritmos = MultiComparison(resultados_algoritmos['accuracy'],resultados_algoritmos['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd() #tukey para verificar o melhor
#teste_estatistico.plot_simultaneous() #plota o gráfico

#pickle.dump(svm, open('svm.sav','wb')) #aqui salva o algoritmo num .sav

novo_registro = x_credit[0]
novo_registro = novo_registro.reshape(1,-1)

resposta_arvore = arvore.predict(novo_registro)
resposta_knn = knn.predict(novo_registro) 
resposta_svm = svm.predict(novo_registro)

count_paga = 0 
count_deve = 0

if resposta_arvore[0] == 1:
    count_deve +=1
else:
    count_paga +=1

if resposta_knn[0] == 1:
    count_deve +=1
else:
    count_paga +=1

if resposta_svm[0] == 1:
    count_deve +=1
else:
    count_paga +=1

if count_paga> count_deve: #aqui ele vai considerar a maioria, olhando as respostas dos tres classicos
    print('Ele vai pagar')
elif count_paga == count_deve:
    print('Indecisão')
else:
    print('Não paga')