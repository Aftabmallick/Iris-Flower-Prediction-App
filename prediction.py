import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv("Iris.csv")
x=data.drop("Species",axis=1)
x=x.drop("Id",axis=1)
y=data["Species"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knowledge_base=KNeighborsClassifier(n_neighbors=1)
knowledge_base.fit(x_train,y_train)
file='model.json'
pickle.dump(knowledge_base, open(file, 'wb'))
pred=knowledge_base.predict(x_test)
