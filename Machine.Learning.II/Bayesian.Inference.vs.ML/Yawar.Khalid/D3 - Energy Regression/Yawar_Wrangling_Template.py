#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as mano
import datetime
import string
from sklearn.impute import KNNImputer
import tkinter as tk
from tkinter import filedialog
import pylab 
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, auc, log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier




class yawar_wrangling_KNN:
    
    def read_file(filename):
        try:
            df= pd.read_csv(filename, skipinitialspace=True)
        except:
            df= pd.read_csv(filename, skipinitialspace=True)

        return df
    
    
    def summary_df(df,n):
    
        print('shape:',df.shape)
    
        print('\ncol_types:\n',df.dtypes)
    
        print('\nhead:\n',df.head(n))
    
        print('\ntail:\n',df.tail(n))
    
    
    def drop_col(df,n):
        for i in df:
            y=df[i].isnull().sum()
            w=y/len(df[i])*100
            print(w)
            if w>n:
                df.drop(i,axis=1,inplace=True)
    
    def filter_rows(df,col,string):
        df.drop(df.loc[df[col]==string].index, inplace=True)
        return df
    
    def check_mv(df):
        print(df.isnull().sum().sort_values(ascending=False))
        
    def mano_analysis(df):
        mano.bar(df)

        mano.matrix(df)

        mano.heatmap(df, figsize=(12,6))
        
    def cater_mv(df,col,col2,ld='forward'):
        df[col2] = df[col].interpolate(method ='linear', limit_direction =ld)
        return df
    
    
    def num_analysis(df,col):
    
        df[col] = df[col].astype(float)

        #describe
        print(df[col].describe())

        #Histogram
        df[col].hist(grid=True, bins=20, rwidth=0.9,
                       color='#9400D3')
        plt.title('Histogram Analysis')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

        #Box Plot
        plt.figure(figsize=(10,10))
        plt.boxplot(x=df[col],widths=0.8)

        #QQ Plot
        #stats.probplot(df[col], dist="norm", plot=pylab)
        #pylab.show()

        qqplot(df[col], line='s')
        plt.show()


        output = anderson(df[col])
        print('Output: %.4f' % output.statistic)

        p = 0
        for i in range(len(output.critical_values)):
            sl, cv = output.significance_level[i], output.critical_values[i]
            if output.statistic < output.critical_values[i]:
                print('%.4f: %.4f, data looks normal (cant reject H0)' % (sl, cv))
            else:
                print('%.4f: %.4f, data not look normal (reject H0)' % (sl, cv))
                
                
    def val_count(df):
    
        c = df.columns

        for col in c:
            print(df[col].value_counts())

            
    def barchart(df,col):
        df[col].value_counts().plot(kind='bar')
        print(df[col].value_counts())
        plt.title('Distribution of %s'%col, fontsize=15)
        
        
        
    def tcast(df,col,type_):
        df[col] = df[col].astype(type_)
        
    
   

    def rem_char(df,col,char):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.rstrip()
        df[col].replace({char : np.nan}, inplace=True)
        
        
    def date_analyze(df,col):
        df['Quarter'] = df[col].apply(lambda df: df.quarter)
        df['Year'] = df[col].apply(lambda df: df.year)
        df['Month'] = df[col].apply(lambda df: df.month)
        df['Week'] = df[col].apply(lambda df: df.week)
        df['Day'] = df[col].apply(lambda df: df.day)
        
        
    def deep_copy(df):
        return df.copy(deep=True)
    
    def label_encode(df, col):
        oe = LabelEncoder()
        oe.fit(df[col])
        data_enc = oe.transform(df[col])
        return data_enc
    
    
    def heat_map(df):
    
        mano.heatmap(df, figsize=(12,6))

        corrmatrix = df.corr()
        f, axis = plt.subplots(figsize =(25, 15)) 
        sns.heatmap(corrmatrix, ax = axis, linewidths = 0.2)
    
    
    def scatter_(df,col1,col2):
        plt.scatter(df[col1], df[col2], c='indigo')
        plt.show()
        
    
   
    def KNN(X,y,n=0.2):
        if input(print('Enter c for Classifer or r for Regression'))=='c':
            if input(print('Which Classifier? Hint - KNN'))=='KNN':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n)

                scaler = StandardScaler()
                scaler.fit(X_train)

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                classifier = KNeighborsClassifier(n_neighbors=int(input(print('Neighbours:'))))
                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)

                print(confusion_matrix(y_test, y_pred))
                print(classification_report(y_test, y_pred))
            
            
            error = []
            # Calculating error for K values between 1 and 35
            for i in range(1, 50):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(X_train, y_train)
                pred_i = knn.predict(X_test)
                error.append(np.mean(pred_i != y_test))
                
                
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
                     markerfacecolor='blue', markersize=10)
            plt.title('Error Rate K Value')
            plt.xlabel('K Value')
            plt.ylabel('Mean Error')


# In[ ]: