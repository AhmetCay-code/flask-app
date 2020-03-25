import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


db = pd.read_csv("C:/Users/Ahmet/Desktop/API/data.csv",sep = ';' )


logreg = LogisticRegression(solver='liblinear',multi_class='ovr')



X = db.drop(["MESANE-ILTIHABI","BOBREK-ILTIHABI"],axis=1)
y = db['MESANE-ILTIHABI']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 )




A = db.drop(["MESANE-ILTIHABI","BOBREK-ILTIHABI"],axis=1)
b = db['BOBREK-ILTIHABI']
A_train, A_test, b_train, b_test = train_test_split(A, b,test_size=0.2 )



new_input = [[389,0,1,1,0,0]]





m1 = logreg
m1.fit(X_train, y_train)

pickle.dump(m1, open('model1.pkl','wb'))

m2 = logreg
m2.fit(A_train, b_train)

pickle.dump(m2, open('model2.pkl','wb'))

