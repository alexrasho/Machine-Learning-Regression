
# coding: utf-8

# Mashup of Machine Learning Projects using Linear Regression and Decision Trees

# In[33]:

from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()
clf = tree.DecisionTreeClassifier() #Store Classifier Object
clfIris = clf.fit(iris.data, iris.target) #Train the Tree with sample data from iris
plabel1 = clf.predict([[5, 3.1, 1.2, 0.4]]) #prediction example with classificiation tree from training data
plabel2 = clf.predict([[6.2, 6, 8, 10]]) #second prediction example
print(plabel1) #label generated as integer
print(plabel2) #label generated as integer

data = tree.export_graphviz(clfIris, out_file=None) 
graph = graphviz.Source(data) #create graphical representation of classification tree
graph.render("TheIrisTree") #Render to File System


# In[115]:

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as pl

rng = np.random.RandomState(100)
X = np.sort(10 * rng.rand(50, 1), axis=0) #randomly generate 50 X values between 0,100
y = np.cos(X).ravel() #create target variables


for i in range (1, y.size, 5): #add some error noise to target values
    PolarRand = 1 * rng.rand(1) #randomly negate or add a value of 2
    if PolarRand < 1:
        y[i] += 2 * rng.rand(1)
    else:
        y[i] -= 2 * rng.rand(1)

regr_1 = DecisionTreeRegressor(max_depth=3)
regr_2 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)

X_test = np.arange(0, 10, 1)[:, np.newaxis] #create test 
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

pl.figure()
pl.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="target")
pl.plot(X_test, y_1, color="blue",
         label="max_depth=3", linewidth=2)
pl.plot(X_test, y_2, color="green", label="max depth=8", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression of Cosine")
pl.legend()
pl.show()
print("Max depth = 8 overfits heavily while max_depth = 3 underfits")


# In[166]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0)
clf.fit(X, y)

clf2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=2)
clf2.fit(X, y)

print("Feature Informativeness (Forest 1):", clf.feature_importances_) #Show how informative each feature is in the Random Forest Tree
print("Predicted Label (Forest 1):", clf.predict([[.5, .2, 1, 0]])) #Predict a label for a test data

print("Feature Informativeness (Forest 2):", clf2.feature_importances_) #Show how informative each feature is in the Random Forest Tree
print("Predicted Label (Forest 2):", clf2.predict([[.5, .2, 1, 0]])) #Predict a label for a test data

print("\nDecision Path (Forest 1): \n\n", clf.decision_path(X)) #Show which nodes are passing through data


# In[117]:

#1.
data = np.loadtxt('/Users/alexrasho/Downloads/female200.csv', delimiter=',')
x = np.loadtxt('/Users/alexrasho/Downloads/female200.csv', delimiter=',', usecols=(0,))
y = np.loadtxt('/Users/alexrasho/Downloads/female200.csv', delimiter=',', usecols=(1,))

(m,b0) = np.polyfit(x, y, 1) #1st order polynomial linear regression fit of Olympic time
yhat = np.polyval([m, b0], x) #predict
pl.plot(x, yhat)
pl.xlabel("Year")
pl.ylabel("Time")
pl.title("Olympic Time Linear Regression")
pl.scatter(x,y)
pl.show()

mse = np.square(np.subtract(y, yhat)).mean() #mean square error of line
print('mse = ', mse) 


# In[118]:

(m,b0,b1,b2) = np.polyfit(x, y, 3) #Third Order Polynomial Fit
yhat = np.polyval([m, b0, b1, b2], x)
pl.plot(x, yhat)
pl.scatter(x,y)
pl.show()

mse = np.square(np.subtract(y, yhat)).mean()
print("mse = ", mse)


# In[119]:

(m,b0,b1,b2,b3,b4) = np.polyfit(x, y, 5) #5th Order fit
yhat = np.polyval([m, b0, b1, b2, b3, b4], x)
pl.plot(x, yhat)
pl.scatter(x,y)
pl.show()

mse = np.square(np.subtract(y, yhat)).mean()
print(mse)


# In[120]:

x = np.loadtxt('/Users/alexrasho/Downloads/female200.csv', delimiter=',', usecols=(0,))
y = np.loadtxt('/Users/alexrasho/Downloads/female200.csv', delimiter=',', usecols=(1,))
x = np.reshape(x, (16,1))
y = np.reshape(y, (16,1))
import pylab as plt
x = (x-1948)/4 #Normalizing the values of X

maxorder = 5 
x_test = np.linspace(0,16,30)[:,None] #Generate x_test data
X = np.ones_like(x)
X_test = np.ones_like(x_test)
for i in range(1,maxorder+1): #Preparing and applying X data transformations
    X = np.hstack((X,x**i))
    X_test = np.hstack((X_test,x_test**i))

for lamb in [0,.01,0.1,1,10,100]: #Using L2 Regularization with Varying lambda values
    w = np.linalg.solve(np.dot(X.T,X) + x.size*lamb*np.identity(maxorder+1),np.dot(X.T,y))
    f_test = np.dot(X_test,w)
    plt.figure()
    plt.plot(x_test,f_test,'b-',linewidth=2)
    plt.plot(x,y,'ro')
    title = '$\lambda=$%g'%lamb
    plt.title(title)

plt.show()


# In[ ]:



