
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
import scipy.io


# In[6]:


import pandas as pd
import os


# In[7]:


def mav(col):
    """
        Calculate Mean Absolute Variance
    """
    val = 0.0
    
    for ele in col:
        val +=np.absolute(ele)
        
    val = val/len(col)
    return val

def wl(col):
    """
        Calculate Wavelength
    """
    val = 0.0
    col = col.values
    for i in range(1,len(col)):
        val +=np.absolute(col[i] - col[i-1])
    return val

def var(col):
    """
        Calculate Variance
    """
    return np.var(col)

def zc(col):
    """
        Calculate Zero Crossing
    """
    val = 0
    col = col.reset_index(drop=True)
    for i in range(1,len(col)):
        if col[i-1]<0 and col[i]>0:
            val = val+1
        elif col[i-1]>0 and col[i]<0:
            val = val+1
    return val

def rms(col):
    """
        Calculate Root Mean Square
    """
    col = col.values
    return np.sqrt(np.mean(col**2))

def first_order_diff(X):
	""" Compute the first order difference of a time series.

		For a time series X = [x(1), x(2), ... , x(N)], its	first order 
		difference is:
		Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]
		
	"""

	D=[]
	#print(X)
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
	X = X.reset_index(drop=True)
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


# The 2nd Dataset contains 8 Channels per Sample, making more number of features available for classification

# In[21]:


def data8mav(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        #dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                val1 = mav(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val2 = mav(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val3 = mav(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val4 = mav(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val5 = mav(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val6 = mav(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val7 = mav(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val8 = mav(df.iloc[j:j+1600,7])
                
                #print ('Ch8 done')
                #dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata


# In[22]:
def data8wl(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        #dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                
                val1 = wl(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val2 = wl(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val3 = wl(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val4 = wl(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val5 = wl(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val6 = wl(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val7 = wl(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val8 = wl(df.iloc[j:j+1600,7])
                #print ('Ch8 done')
                #dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata

def data8zc(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        #dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                val1 = zc(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val2 = zc(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val3 = zc(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val4 = zc(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val5 = zc(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val6 = zc(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val7 = zc(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val8 = zc(df.iloc[j:j+1600,7])
                #print ('Ch8 done')
                #dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata

def data8var(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        #dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                val1 = var(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val2 = var(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val3 = var(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val4 = var(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val5 = var(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val6 = var(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val7 = var(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val8 = var(df.iloc[j:j+1600,7])
                
                #print ('Ch8 done')
                #dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata

def data8rms(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        #dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                val1 = rms(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val2 = rms(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val3 = rms(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val4 = rms(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val5 = rms(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val6 = rms(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val7 = rms(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val8 = rms(df.iloc[j:j+1600,7])
                #print ('Ch8 done')
                #dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata

def data8hjorth(finaldata,path):
    
    """
        Label Headers
    """
    label_headers = ['hc_','l_l','r_r','m_m','i_i','t_t','t_l','t_r','t_m','t_i','i_m','imr','m_r','mrl','r_l']
    dataset = dict()
    
    """
        Get all files in folder
    """
    
    for val in label_headers:
        dataset[val] = []
        
    """
        For every file in the folder, append Headers to the dataset
    """
    for f in os.listdir(path):
        #print (f)
        header = f[0] +f[1]+ f[2]
        header = header.lower()
        d = pd.read_csv(path+f,header = None)
    #ch1 = d.iloc[:,0]
    #ch2 = d.iloc[:,1]
        try:
            dataset[header].append(d)
        except:
            i=0
    #print (dataset[0])
    newdata = dict()
        
    for keys,values in dataset.items():
        newdata[keys] = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
    
    i=0
    for keys in dataset.keys():
        dd = pd.DataFrame(columns = ['ch11','ch12','ch21','ch22','ch31','ch32','ch41','ch42','ch51','ch52','ch61','ch62','ch71','ch72','ch81','ch82|','label']);
        #dd = pd.DataFrame(columns =['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label'])
        
        for x,df in enumerate(dataset[keys]):
            j=0
            while j+1600<len(df):
                
                val11,val12 = hjorth(df.iloc[j:j+1600,0])
                    #print ('Ch1 done')
                val21,val22 = hjorth(df.iloc[j:j+1600,1])
                    #print ('Ch2 done')
                val31,val32 = hjorth(df.iloc[j:j+1600,2])
                    #print ('Ch3 done')
                val41,val42 = hjorth(df.iloc[j:j+1600,3])
                    #print ('Ch4 done')
                val51,val52 = hjorth(df.iloc[j:j+1600,4])
                    #print ('Ch5 done')
                val61,val62 = hjorth(df.iloc[j:j+1600,5])
                    #print ('Ch6 done')
                val71,val72 = hjorth(df.iloc[j:j+1600,6])
                    #print ('Ch7 done')
                val81,val82 = hjorth(df.iloc[j:j+1600,7])
                #print ('Ch8 done')
                dd.loc[i] = [val11,val12,val21,val22,val31,val32,val41,val42,val51,val52,val61,val62,val71,val72,val81,val82,keys]
                #dd.loc[i] = [val1,val2,val3,val4,val5,val6,val7,val8,keys]
                i= i+1
                j = j+800
            #print (i)
        newdata[keys] = dd
    
    #print (newdata)
    for keys in newdata.keys():
        d = pd.DataFrame(newdata[keys])
        #print (d)
        finaldata = pd.concat([finaldata,d],axis=0)
    
    return finaldata

def return_data(method):
    
    finaldata = pd.DataFrame(columns = ['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','label']) 
    if method == 'mav':
        print('Currently doing mav')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8mav(finaldata , path)
            finaldata.to_csv('dataMAV.csv')
    if method == 'wl':
        print('Currently doing wl')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8wl(finaldata , path)
            finaldata.to_csv('dataWL.csv')
            
    if method == 'zc':
        print('Currently doing zc')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8zc(finaldata , path)
            finaldata.to_csv('dataMZC.csv')
            
    if method == 'var':
        print('Currently doing var')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8var(finaldata , path)
            finaldata.to_csv('dataVAR.csv')
            
    if method == 'rms':
        print('Currently doing rms')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8rms(finaldata , path)
            finaldata.to_csv('dataRMS.csv')

    if method == 'hjorth':
        print('Currently doing hjorth')
        for val in os.listdir('C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\'):
        
            path = 'C:\\Users\\nari9\\Documents\\MechaLimb\\Data8Channels\\' + val + '\\'
            finaldata = data8hjorth(finaldata , path)
            finaldata.to_csv('dataHJORTH.csv')

# In[23]:


return_data('mav')
return_data('wl')
return_data('zc')
return_data('var')
return_data('rms')
return_data('hjorth')