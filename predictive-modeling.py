# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:02:12 2021

@author: jonat
"""

### Predictive Modeling ###

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

dataset = pd.read_csv('marketing_campaign.csv', sep='\t')
dataset.dropna()

# Identify parents
dataset["Children"]=dataset["Kidhome"]+dataset["Teenhome"]
dataset["Is_Parent"] = np.where(dataset.Children> 0, 1, 0)

# Calculate monthly spend for products
import datetime as dt
dataset["Dt_customer"] = pd.to_datetime(dataset['Dt_Customer'])
dataset["Month_customer"] = 12 * (2015 - dataset.Dt_customer.dt.year) + (1 - dataset.Dt_customer.dt.month)

dataset['MntWinesMonth'] = dataset['MntWines'] / dataset['Month_customer'] 
dataset['MntFruitsMonth'] = dataset['MntFruits'] / dataset['Month_customer']  
dataset['MntMeatProductsMonth'] = dataset['MntMeatProducts'] / dataset['Month_customer'] 
dataset['MntFishProductsMonth'] = dataset['MntFishProducts'] / dataset['Month_customer'] 
dataset['MntSweetProductsMonth'] = dataset['MntSweetProducts'] / dataset['Month_customer'] 

# Convert total purchases to proportions 
dataset["TotalPurchases"] = dataset["NumCatalogPurchases"] + dataset["NumStorePurchases"] + dataset["NumWebPurchases"]
dataset["CatalogProp"] = dataset["NumCatalogPurchases"] / dataset["TotalPurchases"]
dataset["DiscountProp"] = dataset["NumDealsPurchases"] / dataset["TotalPurchases"]

Plot1 = ["Income", "Age", "Total_Spent", "Is_Parent"]
sns.pairplot(dataset[Plot1], hue = 'Is_Parent')

# droping outliers
dataset = dataset[(dataset['Age']<100)]
dataset = dataset[(dataset['Income']<600000)]

target1 = ['Is_Parent']
features1 = ['MntWinesMonth', 'MntFruitsMonth', 'MntMeatProductsMonth', 'MntFishProductsMonth', 'MntSweetProductsMonth',
             'CatalogProp', 'DiscountProp']
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset[features1],dataset[target1], test_size= 0.2, random_state = 121)

# Support Vector Machines

from sklearn.svm import SVC
clf_svc = SVC(kernel = 'rbf', random_state=0)
clf_svc.fit(X_train1, y_train1)

y_pred1 = clf_svc.predict(X_test1)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
accuracy_score(y_test1, y_pred1)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train1, y_train1)

y_pred1 = clf_dt.predict(X_test1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
accuracy_score(y_test1, y_pred1)

from matplotlib import pyplot
importance = clf_dt.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# Neural Network

from sklearn.neural_network import MLPClassifier

clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf_nn.fit(X_train1, y_train1)

y_pred1 = clf_nn.predict(X_test1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
accuracy_score(y_test1, y_pred1)

# Fourth Model: kNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train1, y_train1)
y_pred1 = knn.predict(X_test1)
cm = confusion_matrix(y_test1, y_pred1)
print(cm)
accuracy_score(y_test1, y_pred1)

