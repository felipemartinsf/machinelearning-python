from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pickle
import numpy as np
path = r'C:\Users\FFranci8\OneDrive - JNJ\Área de Trabalho\git\machinelearning-python\estudo py\aula6\risco_credito.pkl'
with open(path,'rb') as f:
    x_risco,y_risco = pickle.load(f)

x_risco = np.delete(x_risco,[2,7,11],axis=0)
y_risco = np.delete(x_risco,[2,7,11],axis=0)
#apagamos os moderados
logistic_risco = LogisticRegression(random_state=1)
logistic_risco.fit(x_risco,y_risco)
print(logistic_risco.intercept_) #b0
print(logistic_risco.coef_) #b1,b2,b3,b4
previsoes1 = logistic_risco.predict([0,0,1,2],[2,0,0,0])