# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:10:45 2019

@author: nari9
"""


# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
#from pyeeg import first_order_diff, hjorth
from joblib import dump
from sklearn.model_selection import GridSearchCV

# In[2]:


def returnClassLabels(data):

    labels = data['label'].unique()

    labeldict = dict()

    for i, val in enumerate(labels):
        labeldict[val] = i

    print (labeldict)

    y = data['label'].values

    outs = [labeldict[char] for char in y]
    
    return outs


# In[3]:


def scale_and_build(inputs,data8WL,datanewVAR,datanewRMS,datanewZC,datanewHJORTH):
    new_outs = returnClassLabels(inputs)
    #new_inputs = datanewFINAL
    #print (datanewFINAL.head(2))
    
    new_inputs = inputs[['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']]
    new_inputs['ch1WL'] = data8WL['ch1']
    new_inputs['ch2WL'] = data8WL['ch2']
    new_inputs['ch3WL'] = data8WL['ch3']
    new_inputs['ch4WL'] = data8WL['ch4']
    new_inputs['ch5WL'] = data8WL['ch5']
    new_inputs['ch6WL'] = data8WL['ch6']
    new_inputs['ch7WL'] = data8WL['ch7']
    new_inputs['ch8WL'] = data8WL['ch8']
    
    
    new_inputs['ch1SSC'] = datanewZC['ch1']
    new_inputs['ch2SSC'] = datanewZC['ch2']
    new_inputs['ch3SSC'] = datanewZC['ch3']
    new_inputs['ch4SSC'] = datanewZC['ch4']
    new_inputs['ch5SSC'] = datanewZC['ch5']
    new_inputs['ch6SSC'] = datanewZC['ch6']
    new_inputs['ch7SSC'] = datanewZC['ch7']
    new_inputs['ch8SSC'] = datanewZC['ch8']
    
    
    new_inputs['ch1VAR'] = datanewVAR['ch1']
    new_inputs['ch2VAR'] = datanewVAR['ch2']
    new_inputs['ch3VAR'] = datanewVAR['ch3']
    new_inputs['ch4VAR'] = datanewVAR['ch4']
    new_inputs['ch5VAR'] = datanewVAR['ch5']
    new_inputs['ch6VAR'] = datanewVAR['ch6']
    new_inputs['ch7VAR'] = datanewVAR['ch7']
    new_inputs['ch8VAR'] = datanewVAR['ch8']
    
    
    new_inputs['ch1RMS'] = datanewRMS['ch1']
    new_inputs['ch2RMS'] = datanewRMS['ch2']
    new_inputs['ch3RMS'] = datanewRMS['ch3']
    new_inputs['ch4RMS'] = datanewRMS['ch4']
    new_inputs['ch5RMS'] = datanewRMS['ch5']
    new_inputs['ch6RMS'] = datanewRMS['ch6']
    new_inputs['ch7RMS'] = datanewRMS['ch7']
    new_inputs['ch8RMS'] = datanewRMS['ch8']
    new_inputs['ch8RMS'] = datanewRMS['ch8']
    
    
    new_inputs['ch11H'] = datanewHJORTH['ch11']
    new_inputs['ch12H'] = datanewHJORTH['ch12']
    new_inputs['ch21H'] = datanewHJORTH['ch21']
    new_inputs['ch22H'] = datanewHJORTH['ch22']
    new_inputs['ch31H'] = datanewHJORTH['ch31']
    new_inputs['ch32H'] = datanewHJORTH['ch32']
    new_inputs['ch41H'] = datanewHJORTH['ch41']
    new_inputs['ch42H'] = datanewHJORTH['ch42']
    new_inputs['ch51H'] = datanewHJORTH['ch51']
    new_inputs['ch52H'] = datanewHJORTH['ch52']
    new_inputs['ch61H'] = datanewHJORTH['ch61']
    new_inputs['ch62H'] = datanewHJORTH['ch62']
    new_inputs['ch71H'] = datanewHJORTH['ch71']
    new_inputs['ch72H'] = datanewHJORTH['ch72']
    new_inputs['ch81H'] = datanewHJORTH['ch81']
    new_inputs['ch82H'] = datanewHJORTH['ch82|']
    
    
    #new_inputs.to_csv('datanewFINAL.csv', index=False)
    
    
    print ('Length : ', len(new_inputs))
    print('Scaling')
    scaler = StandardScaler()
    scaler.fit(new_inputs)
    new_inputs = scaler.transform(new_inputs)                                  #Change made here adding extra n
    
    print('Scaling Done')
    print ('Finding Principal Components')
    
    #pca = PCA(n_components=14)
    #pca.fit(new_inputs)                             #Change made here adding extra n
    #new_inputs = pca.transform(new_inputs)
    print (new_inputs[0])
    print ('Principal Components Done')
    train_data,test_data,train_labels,test_labels = train_test_split(new_inputs, np.array(new_outs), test_size = 0.3)
        
        
    neigh = KNeighborsClassifier(n_neighbors = 10, metric = 'manhattan' ,weights = 'distance',n_jobs=-1)
            #neigh1 = KNeighborsClassifier(n_neighbors = i, weights = 'distance')
    
    
    #clf = svm.SVC(C=8,kernel='rbf', gamma='auto', verbose=True)
                #print ('Neighbors : ', i)
    print('Fitting Data')
    neigh.fit(train_data, train_labels)
                    #neigh1.fit(train_data, train_labels)
    print('Fitting Done')
    val = neigh.predict(test_data)
                    #val1 = neigh1.predict(test_data)
    print('Accuracy Score : ' ,accuracy_score(val,test_labels))
    print(classification_report(test_labels,val))
            #print('Accuracy Score 2: ' ,accuracy_score(val1,test_labels))
            #print(classification_report(test_labels,val1))
    
    #dump(pca,'pcaModel.joblib')
    dump(scaler,'scalerModel.joblib')
    dump(neigh,'svmModel.joblib')

# In[ ]:


inputs = pd.read_csv('dataMAV.csv')
inputs1 = pd.read_csv('dataWL.csv')
inputs2 = pd.read_csv('dataMZC.csv')
inputs3 = pd.read_csv('dataRMS.csv')
inputs4 = pd.read_csv('dataVAR.csv')
inputs5 = pd.read_csv('dataHJORTH.csv')
scale_and_build(inputs,inputs1,inputs2,inputs3,inputs4,inputs5)

