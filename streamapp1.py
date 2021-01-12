# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:38:52 2021

@author: Anurag

The following is an streamlit app built for deploying machine learning models to 
the web

"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


st.title("Multi Dataset ")

st.write(""" Exporing
         """)
         
         
data_set = st.sidebar.selectbox("Select dataset", ("Iris", "Breast cancer","Wine dataset") )     

         
         
classifiers_set = st.sidebar.selectbox("Select classifiers", ("Knn","Svm","Random forest") )     


def get_dataset(dataName):
    if dataName == "Iris":
        data = datasets.load_iris()
    elif dataName == "Breast cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data
    y = data.target
    return X,y
     

X,y = get_dataset(data_set)
st.write("shape of dataset", X.shape)    
st.write( "Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    param =dict()
    if clf_name == "Knn":
      K = st.sidebar.slider("K" , 1,15)
      param["k"] = K
    elif clf_name == "Svm":
        C = st.sidebar.slider("C", 0.01 , 10.0 )
        param["C"] = C
    else:
        
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_estimators = st.sidebar.slider("n_estimators", 1,100)
        param["max_depth"] = max_depth
        param["n_estimators"] = n_estimators
        
    return param  

param =  add_parameter_ui(classifiers_set)      

def get_classifier(clf_name , param):
    
    if clf_name == "Knn":
     clf = KNeighborsClassifier(n_neighbors=param["k"])
     
    elif clf_name == "Svm":
       clf = SVC(C = param["C"])
    else:    
      clf = RandomForestClassifier(n_estimators= param["n_estimators"],
                                   max_depth= param["max_depth"],
                                   random_state=42)
    
    return clf

clf = get_classifier(classifiers_set,param)

#Dooing clasification

from sklearn.model_selection import train_test_split

X_train,X_test ,y_train,y_test = train_test_split(X ,y ,
                                                  test_size= 0.2,
                                                  random_state=42 )

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifiers_set}" )
st.write(f"accuracy = {acc}")

#plot

from sklearn.decomposition import PCA


pca =PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y ,alpha=0.8,cmap="viridis" )
plt.xlabel("Principal component 1")
plt.ylabel("principal component 2")
plt.colorbar()

st.pyplot(fig)

#save model

import pickle

with open('stream1', 'wb') as f:
    pickle.dump(clf,f)

with open('stream1', 'rb') as f:
    mp = pickle.load(f)

savmp= mp.predict(X_test)
accne = accuracy_score(y_test,savmp)
st.write(f"accuracy new = {accne}")
