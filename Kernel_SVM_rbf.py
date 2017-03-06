#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:38:12 2017

@author: DK
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
dataset=pd.read_csv("/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Classification/SVM/Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]].values          #classification to be based on Age and EstimatedSalary
Y=dataset.iloc[:,4].values

#test and training sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.75,random_state=0)

#feature scale
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)

#SVM classification
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,Y_train)

#prediction
y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

#visualisation
from matplotlib.colors import ListedColormap
X_set,Y_set=X_test,Y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,1].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

#plotting the contour line
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

#plotting all points
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0],
                X_set[Y_set ==j,1],
                c=ListedColormap(("red","green"))(i),label=j)
plt.title('Kernel_SVM_rbf Classification')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    

