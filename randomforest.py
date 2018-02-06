import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

#Here is the data in excel. 
df=pd.read_excel("svm.xlsx",'data')


print(type(df))

X = df[['bir','iki']]
y = df['renk']

X=X.as_matrix()
y=y.as_matrix()

print(X)
print(y)

X=X.astype(int)
#C=0.7, kernel='linear' hyperparameter
clf = RandomForestClassifier(n_estimators=4) #decision_function_shape='ovo')
# Fit Support Vector Machine Classifier

clf.fit(X, y) 
test= np.array([19,2])
test=test.reshape(1,2)

print(clf.predict(test))

# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X, 
                      y=y,
                      clf=clf, 
                      legend=2)

plt.show()
