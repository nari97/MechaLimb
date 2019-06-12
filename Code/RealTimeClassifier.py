# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:20:17 2019

@author: nari9
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:10:00 2019

@author: nari9
"""
#import pickle
import sys
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
#import queue
from collections import Counter
import os
from sklearn.metrics import accuracy_score
import RPi.GPIO as GPIO
model = load('/home/pi/Desktop/FinalProject/FinalTest/svmModel.joblib')
GPIO.setmode(GPIO.BCM)
scalerModel = load('/home/pi/Desktop/FinalProject/FinalTest/scalerModel.joblib')
#inc_data = pd.read_csv('C:\\Users\\nari9\\Documents\\Mechalimb\\Data8Channels\\S1-Delsys-15Class\\R_R1.csv',header=None)
#vote_list = [i for i in range(0,15)]
print ("Loaded Libraries")
def mav(col):
    val = 0.0
    
    for ele in col:
        val +=np.absolute(ele)
        
    val = val/len(col)
    return val

def wl(col):
    val = 0.0
    col = col.values
    for i in range(1,len(col)):
        val +=np.absolute(col[i] - col[i-1])
    return val

def var(col):
    return np.var(col)

def ssc(col):
    
    val = 0
    
    for i in range(1,len(col)):
        if col[i-1]<0 and col[i]>0:
            val = val+1
        elif col[i-1]>0 and col[i]<0:
            val = val+1
    return val

def rms(col):
    
    col = col.values
    return np.sqrt(np.mean(col**2))

def first_order_diff(X):
    """ Compute the first order difference of a time series.

        For a time series X = [x(1), x(2), ... , x(N)], its first order 
        difference is:
        Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]
        
    """
    D=[]
    
    for i in range(1,len(X)):
        D.append(X[i]-X[i-1])

    return D

def hjorth(X, D = None):
    """ Compute Hjorth mobility and complexity of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.

    Parameters
    ----------

    X
        list
        
        a time series
    
    D
        list
    
        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """
    
    if D is None:
        D = first_order_diff(X)

    D.insert(0, X[0]) # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = np.sum(np.array(X) ** 2)
    M4 = 0;
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n
    
    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)

def extract_features(df):

    inputs = [];
    df = df.reset_index(drop=True)
    val1 = mav(df.iloc[:,0].T)
    #print (val1)
    val2 = mav(df.iloc[:,1].T)
    val3 = mav(df.iloc[:,2].T)
    val4 = mav(df.iloc[:,3].T)
    val5 = mav(df.iloc[:,4].T)
    val6 = mav(df.iloc[:,5].T)
    val7 = mav(df.iloc[:,6].T)
    val8 = mav(df.iloc[:,7].T)
    
    '''
    
    inputs['ch1MAV'] = val1
    inputs['ch2MAV'] = val2
    inputs['ch3MAV'] = val3
    inputs['ch4MAV'] = val4
    inputs['ch5MAV'] = val5
    inputs['ch6MAV'] = val6
    inputs['ch7MAV'] = val7
    inputs['ch8MAV'] = val8
    '''
    inputs.append(val1)
    inputs.append(val2)
    inputs.append(val3)
    inputs.append(val4)
    inputs.append(val5)
    inputs.append(val6)
    inputs.append(val7)
    inputs.append(val8)
    
    val1 = wl(df.iloc[:,0].T)
    val2 = wl(df.iloc[:,1].T)
    val3 = wl(df.iloc[:,2].T)
    val4 = wl(df.iloc[:,3].T)
    val5 = wl(df.iloc[:,4].T)
    val6 = wl(df.iloc[:,5].T)
    val7 = wl(df.iloc[:,6].T)
    val8 = wl(df.iloc[:,7].T)
    '''
    inputs['ch1WL'] = val1
    inputs['ch2WL'] = val2
    inputs['ch3WL'] = val3
    inputs['ch4WL'] = val4
    inputs['ch5WL'] = val5
    inputs['ch6WL'] = val6
    inputs['ch7WL'] = val7
    inputs['ch8WL'] = val8
    '''
    inputs.append(val1)
    inputs.append(val2)
    inputs.append(val3)
    inputs.append(val4)
    inputs.append(val5)
    inputs.append(val6)
    inputs.append(val7)
    inputs.append(val8)
    
    
    val1 = ssc(df.iloc[:,0].T)
    val2 = ssc(df.iloc[:,1].T)
    val3 = ssc(df.iloc[:,2].T)
    val4 = ssc(df.iloc[:,3].T)
    val5 = ssc(df.iloc[:,4].T)
    val6 = ssc(df.iloc[:,5].T)
    val7 = ssc(df.iloc[:,6].T)
    val8 = ssc(df.iloc[:,7].T)
    '''
    inputs['ch1SSC'] = val1
    inputs['ch2SSC'] = val2
    inputs['ch3SSC'] = val3
    inputs['ch4SSC'] = val4
    inputs['ch5SSC'] = val5
    inputs['ch6SSC'] = val6
    inputs['ch7SSC'] = val7
    inputs['ch8SSC'] = val8
    '''
    inputs.append(val1)
    inputs.append(val2)
    inputs.append(val3)
    inputs.append(val4)
    inputs.append(val5)
    inputs.append(val6)
    inputs.append(val7)
    inputs.append(val8)
    
    val1 = var(df.iloc[:,0].T)
    val2 = var(df.iloc[:,1].T)
    val3 = var(df.iloc[:,2].T)
    val4 = var(df.iloc[:,3].T)
    val5 = var(df.iloc[:,4].T)
    val6 = var(df.iloc[:,5].T)
    val7 = var(df.iloc[:,6].T)
    val8 = var(df.iloc[:,7].T)
    
    inputs.append(val1)
    inputs.append(val2)
    inputs.append(val3)
    inputs.append(val4)
    inputs.append(val5)
    inputs.append(val6)
    inputs.append(val7)
    inputs.append(val8)
    
    val1 = rms(df.iloc[:,0].T)
    val2 = rms(df.iloc[:,1].T)
    val3 = rms(df.iloc[:,2].T)
    val4 = rms(df.iloc[:,3].T)
    val5 = rms(df.iloc[:,4].T)
    val6 = rms(df.iloc[:,5].T)
    val7 = rms(df.iloc[:,6].T)
    val8 = rms(df.iloc[:,7].T)
    '''
    inputs['ch1RMS'] = val1
    inputs['ch2RMS'] = val2
    inputs['ch3RMS'] = val3
    inputs['ch4RMS'] = val4
    inputs['ch5RMS'] = val5
    inputs['ch6RMS'] = val6
    inputs['ch7RMS'] = val7
    inputs['ch8RMS'] = val8
    '''
    inputs.append(val1)
    inputs.append(val2)
    inputs.append(val3)
    inputs.append(val4)
    inputs.append(val5)
    inputs.append(val6)
    inputs.append(val7)
    inputs.append(val8)
    
    val11,val12 = hjorth(df.iloc[:,0].T)
    val21,val22 = hjorth(df.iloc[:,1].T)
    val31,val32 = hjorth(df.iloc[:,2].T)
    val41,val42 = hjorth(df.iloc[:,3].T)
    val51,val52 = hjorth(df.iloc[:,4].T)
    val61,val62 = hjorth(df.iloc[:,5].T)
    val71,val72 = hjorth(df.iloc[:,6].T)
    val81,val82 = hjorth(df.iloc[:,7].T)
    '''
    inputs['ch11H'] = val11
    inputs['ch12H'] = val12
    inputs['ch21H'] = val21
    inputs['ch22H'] = val22
    inputs['ch31H'] = val31
    inputs['ch32H'] = val32
    inputs['ch41H'] = val41
    inputs['ch42H'] = val42
    inputs['ch51H'] = val51
    inputs['ch52H'] = val52
    inputs['ch61H'] = val61
    inputs['ch62H'] = val62
    inputs['ch71H'] = val71
    inputs['ch72H'] = val72
    inputs['ch81H'] = val81
    inputs['ch82H'] = val82
    '''
    inputs.append(val11)
    inputs.append(val12)
    inputs.append(val21)
    inputs.append(val22)
    inputs.append(val31)
    inputs.append(val32)
    inputs.append(val41)
    inputs.append(val42)
    inputs.append(val51)
    inputs.append(val52)
    inputs.append(val61)
    inputs.append(val62)
    inputs.append(val71)
    inputs.append(val72)
    inputs.append(val81)
    inputs.append(val82)
    return inputs

    

#inputs = model.predict(pcaModel.transform(scalerModel.transform(extract_features(inc_data))))
#print (inputs)
#print(mav(inc_data.iloc[0:1600,0].T))

#new_list = []

def move_finger(finger):
    print ('In')
    if finger == 1: #Little
        GPIO.setup(6, GPIO.OUT)
        
        pwm_servo = GPIO.PWM(6, 50)
        pwm_servo.start(13)
        i=2
        while True:
            pwm_servo.ChangeDutyCycle(i)
            i=i+1
            time.sleep(0.05)
            if(i>13):
                break
    if finger==2: #Ring
        GPIO.setup(6, GPIO.OUT)
        
        pwm_servo = GPIO.PWM(6, 50)
        pwm_servo.start(13)
        i=2
        while True:
            pwm_servo.ChangeDutyCycle(i)
            i=i+1
            time.sleep(0.05)
            if(i>13):
                break
    if finger==3: #Middle
        GPIO.setup(5, GPIO.OUT)
        
        pwm_servo = GPIO.PWM(5, 50)
        pwm_servo.start(13)
        i=2
        while True:
            pwm_servo.ChangeDutyCycle(i)
            i=i+1
            time.sleep(0.05)
            if(i>13):
                break
    if finger==4: #Index
        GPIO.setup(17, GPIO.OUT)
        
        pwm_servo = GPIO.PWM(17, 50)
        pwm_servo.start(13)
        i=13
        while True:
            pwm_servo.ChangeDutyCycle(i)
            i=i-1
            time.sleep(0.05)
            if(i<2):
                break
    if finger==5: #Thumb
        GPIO.setup(27, GPIO.OUT)
        
        pwm_servo = GPIO.PWM(27, 50)
        pwm_servo.start(13)
        i=13
        while True:
            pwm_servo.ChangeDutyCycle(i)
            i=i-1
            time.sleep(0.05)
            if(i<2):
                break
    
def return_class(inc_data):
    print ("Entered")
    vote_list = []
    j=0
    prev = ""
    while j<len(inc_data[:9600]):
        inputs = extract_features(inc_data[j:j+1600][:])
        inputs = scalerModel.transform(np.reshape(inputs,(1,-1)))
        #inputs = pcaModel.transform(inputs)
        #print (inputs)
        val = model.predict(inputs)
        #print ('Model Voted : ', val)
        if (len(vote_list)<10):
            vote_list.append(val[0])
        else:
            new_list = vote_list
            new_list.append(val[0])
            #print (new_list)
            #print (Counter(new_list[-11:]).most_common())
            vote_list.append(Counter(new_list[-11:]).most_common(1)[0][0])
            if j%1600==0:
                if prev != vote_list[-1]:
                    move_finger(vote_list[-1])
                    prev = vote_list[-1]
                
        print ('For ',j,'to',j+1600,':',vote_list[-1])
        j = j+800
        
    

def real_test():
    #out = []
    
    val = pd.read_csv(sys.argv[1])
    return_class(val)
    #return out

real_test()
#move_finger(5)
