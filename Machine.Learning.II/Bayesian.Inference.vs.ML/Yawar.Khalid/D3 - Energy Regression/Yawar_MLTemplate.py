#!/usr/bin/env python
# coding: utf-8

# Dear students,
# 
# Based on the template I uploaded yesterday, please update your current template as follows:
# 
# 1) include all possible wrangling functions, including t-tests, chi-squared, anova, correlation heatmaps, normality tests (one of them), missing value solutions, transformations etc.
# 
# 2) include all classification and regression algorithms
# 
# 3) FS: feature selection algorithms (I included RFFS, RFE, MI and PCA but you can add only RFFS as for me, it works good).
# 
# 4) CV: cross validation approaches (include stratified K-fold definitely)
# 
# 5) REG: regularization (lasso, ridge regression) methods (include Lasso definitely).
# 
# 6) Include methods for addressing class imbalance
# 
# 7) Include any other helper/logistic function you want to add
# 
# 8) Include function for executing ML (classification) without FS, REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# 9) Include function for executing ML (classification) with FS and without REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# 10) Include function for executing ML (classification) with REG and without FS and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# 11) Include function for executing ML (classification) with CV and without FS and REG. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# 12) Include function for executing ML (classification) with CV and with FS and with REG. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# NB: 8-12 above can be merged as a single function as well
# 
# 
# 
# 
# Output Required:
# 
# 1) Completed template
# 
# 2) Completed thorough wrangling for all three datasets and include interpretations.
# 
# 3) Filled up Results.Template excel attached.
# 
# 4) A detailed commentary on the results regarding performance comparison of algorithms and the effect of REG, CV, FS etc. 
# 
# You can also use different methods of CV, REG and FS to add breadth to your assignment and for bonus marks. Record all results in the template.

# In[98]:


#pip install xgboost
#conda install -c conda-forge lightgbm
#conda update -n base -c defaults conda
#sudo pip install imblearn


# In[99]:


try:
    from xgboost import XGBClassifier
except:
    print("Failed to import xgboost, make sure you have xgboost installed")
    print("Use following command to install it: pip install xgboost")
    XGBClassifier = None

try:
    import lightgbm as lgb
except:
    print("Failed to import lightgbm, make sure that you have lightgbm installed")
    print("Use following command to install it: conda install -c conda-forge lightgbm")
    lgb = None


# In[175]:


#!/usr/bin/env python
# coding: utf-8

#import basic modules
import pandas as pd 
import numpy as np
import seaborn as sb
import math
import warnings
import matplotlib.pyplot as plt        
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing

#import feature selection modules
from sklearn.feature_selection import mutual_info_classif,RFE,RFECV
from sklearn.feature_selection import mutual_info_regression

#import classification modules
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# import regression modules
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor

#import split methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

#import performance scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error, r2_score

#import regularization libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean

# import scaling
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sb.set(color_codes=True, font_scale=1.2)

#importing class balancing libraries
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# In[141]:


#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as mano
import datetime
import string
from sklearn.impute import KNNImputer
get_ipython().run_line_magic('matplotlib', 'inline')
import tkinter as tk
from tkinter import filedialog
import pylab 
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
import seaborn as sb

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")
sb.set(color_codes=True, font_scale=1.2)

from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier, LassoCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# ## 1) include all possible wrangling functions, including t-tests, chi-squared, anova, correlation heatmaps, normality tests (one of them), missing value solutions, transformations etc.

# In[101]:


##F1: loading data in a dataframe (either CSV or Excel - can be generalized for databases)
def read_csv(filename):
    try:
        df= pd.read_csv(filename, skipinitialspace=True)
    except:
        df= pd.read_csv(filename, skipinitialspace=True)
        
    return df


# In[102]:


# F2: checking shape, column types, and see the first/last 'n' rows using head/tail 
#(where n is one of the arguments of F2)

def summary_df(df,n):
    
    print('shape:',df.shape)
    
    print('\ncol_types:\n',df.dtypes)
    
    print('\nhead:\n',df.head(n))
    
    print('\ntail:\n',df.tail(n))


# In[103]:


# F3: remove unnecessary/useless columns 
#(based on results of F2 and your background knowledge and the problem to be solved)
# e.g., identifiers, multiple primary keys, extra KPI like GMROI in sales which is the same for the whole year etc


def cleaningup(df, to_date=[], to_numeric=[], cols_to_delete=[], fill_na_map={}, cols_to_drop_na_rows=[], cols_to_interpolate=[]):
    """
    We will perform all the generic cleanup stuff in this function,
    Data specific stuff should be handled by driver program.
    
    Mandatory Parameter:
    df : Dataframe to be cleaned
    
    Optional Parameters:
    to_date:  List of columns to convert to date
    to_numeric:  List of columns to convert to numeric
    cols_to_delete: All the useless columns that we need to delete from our dataset
    fill_na_map:  A dictionary containing map for column and a value to be filled in missing places
                    e.g. {'age': df['age'].median(), 'city': 'Karachi'}
    cols_to_drop_na_rows: List of columns where missing value in not tolerable and we couldn't risk predicting                     value for it, so we drop such rows.
    cols_to_interpolate: List of columns where missing values have to be replaced by forward interpolation
    """
    
    # columns to convert to date format
    def change_type_to_date(df, to_date):
        # Deal with incorrect data in date column
        for i in to_date:
            df[i] = pd.to_datetime(df[i], errors='coerce')
        return df
    
    # columns to convert to numerical format
    def change_type_to_numeric(df, to_numeric):
        # Deal with incorrect data in numeric columns
        for i in to_numeric:
            df[i] = pd.to_numeric(df[i], errors='coerce')
        return df
    
    # columns to delete
    def drop_useless_colums(df, cols_to_delete):
        # Drop useless columns before dealing with missing values
        for i in cols_to_delete:
            df = df.drop(i, axis=1)
        return df
    
    # delete columns that have more than n% MV
    def drop_by_mvpercent(df,n):
        for i in df:
            y=df[i].isnull().sum()
            w=y/len(df[i])*100
            print(w)
            
            if w>n:
                df.drop(i,axis=1,inplace=True)
        
        return df
    
    #drop all rows which contain more than 40% missing values
    def drop_useless_rows(df):
        min_threshold = math.ceil(len(df.columns)*0.4)
        df = df.dropna(thresh=min_threshold)
        return df
    
    # drop rows in which columns specified by the driver program has missing values
    def drop_na_rows(df, cols_to_drop_na_rows):
        for i in cols_to_drop_na_rows:
            df = df.drop(df[df[i].isnull()].index)
        return df
    
    # Deal with missing values according to map, e.g., {'age': df['age'].median(), 'city': 'Karachi'}
    def fill_na_vals(df, fill_na_map):
        for col,val in fill_na_map.items():
            df[col].fillna(val, inplace=True)
        return df
    
    # Deal with missing values according to the interpolation
    def fill_na_interpolate(df, cols_to_interpolate):
        for i in cols_to_interpolate:
            df[i] = df[i].interpolate(method ='linear', limit_direction ='forward')
        return df
    
    try:
        df = change_type_to_date(df, to_date)
        df = change_type_to_numeric(df, to_numeric)
        df = drop_useless_colums(df, cols_to_delete)
        df = drop_useless_rows(df)
        df = drop_na_rows(df, cols_to_drop_na_rows)
        df = fill_na_vals(df, fill_na_map)
        df = fill_na_interpolate(df, cols_to_interpolate)
        print("df is all cleaned up..")
        return df
    except Exception as e:
        print("Failed to perform cleanup, exception=%s" % str(e))
    finally:
        return df


# In[104]:


#F4: remove rows containing a particular value of a given column, e.g., in smoking_status column, 
# i don't want to consider non-smokers in my ML problem so I remove all these rows.

def filter_rows(df,col,string):
    df.drop(df.loc[df[col]==string].index, inplace=True)
    return df


# In[105]:


#F5: determine the missing values in the whole dataset

def check_mv(df):
    print(df.isnull().sum().sort_values(ascending=False))


# In[106]:


#F6: analyze missing values of one or more columns using mano module

def mano_analysis(df):
    mano.bar(df)
    
    mano.matrix(df)
    
    mano.heatmap(df, figsize=(12,6))


# In[107]:


#F7: cater for missing values 
#(input the column with missing value, and the method through which you want to cater for the missing values)

#the following function will replace specified charachter ex: N/A, 0, - with NaN.

def cater_mv(df,col,col2,ld='forward'):
        df[col2] = df[col].interpolate(method ='linear', limit_direction =ld)
        return df


# In[108]:


#string column analysis analysis
def stringcolanalysis(df):
    stringcols = df.select_dtypes(exclude=[np.number, "datetime64"])
    fig = plt.figure(figsize = (8,10))
    for i,col in enumerate(stringcols):
        fig.add_subplot(4,2,i+1)
        fig.savefig('Categorical.png')
        df[col].value_counts().plot(kind = 'bar', color='black' ,fontsize=10)
        plt.tight_layout()
        plt.title(col)

#numerical analysis
def numcolanalysis(df):
    numcols = df.select_dtypes(include=np.number)
    
    # Box plot for numerical columns
    for col in numcols:
        fig = plt.figure(figsize = (5,5))
        sb.boxplot(df[col], color='grey', linewidth=1)
        plt.tight_layout()
        plt.title(col)
        plt.savefig("Numerical.png")
    
    # Lets also plot histograms for these numerical columns
    df.hist(column=list(numcols.columns),bins=25, grid=False, figsize=(15,12),
                 color='#86bf91', zorder=2, rwidth=0.9)


# In[109]:


#F8: Function for numerical data analysis - 
#includes histogram, boxplot, qqplot, describe, and statistical tests for normality

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


# In[110]:


#F9: Function for categorical data analysis - includes value counts, and bar charts

def val_count(df, col):
    
    c = df.columns

    for col in c:
        print(df[col].value_counts())

        
    df[col].value_counts().plot(kind='bar')
    print(df[col].value_counts())
    plt.title('Distribution of %s'%col, fontsize=15)


# In[111]:


#F10: Function to change the type of any column (input col name and the type you want)

def tcast(df,col,type_):
    df[col] = df[col].astype(type_)


# In[112]:


#F11: Function to change the discretizations of a particular catergorical column, 
#     e.g., rename the values, remove space between value names etc.

def rem_char(df,col,char):
    df[col] = df[col].astype(str)
    df[col] = df[col].str.rstrip()
    df[col].replace({char : np.nan}, inplace=True)


# In[113]:


#F12: Function for data analysis - extract year, month etc., subtract dates etc. 
    #(this function cannot be specified exactly so just add what you believe are the basic things
    
def date_analyze(df,col):
    df['Quarter'] = df[col].apply(lambda df: df.quarter)
    df['Year'] = df[col].apply(lambda df: df.year)
    df['Month'] = df[col].apply(lambda df: df.month)
    df['Week'] = df[col].apply(lambda df: df.week)
    df['Day'] = df[col].apply(lambda df: df.day)


# In[114]:


#F13: function to make a deep copy of a dataframe

def deep_copy(df):
    return df.copy(deep=True)


# In[115]:


#F14: function to encode categorical into numerical (label, ordinal, or onehot)

# Apply label encoding on specified columns
def apply_label_encoding(df, cols=[]):
    le = preprocessing.LabelEncoder()
    for i in cols:
        le.fit(df[i])
        df[i] = le.transform(df[i])
    return df


# One-Hot/dummy encoding on specified columns
def onehotencoding(df):
    df = pd.get_dummies(df)
    return df

# One Hot encoding with Pandas categorical dtype
def onehotencoding_v2(df, cols=[]):
    for col in cols:
        df[col] = pd.Categorical(df[col])
        dfDummies = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, dfDummies], axis=1)
        df = df.drop(col, axis=1)
    return df


# In[172]:


#Train Test Split: splitting manually
def traintestsplit(df,split,random=None, label_col=''):
    #make a copy of the label column and store in y
    y = df[label_col].copy()
    
    #now delete the original
    X = df.drop(label_col,axis=1)
    
    #manual split
    trainX, testX, trainY, testY= train_test_split(X, y, test_size=split, random_state=random)
    return X, trainX, testX, trainY, testY

#helper function which only splits into X and y
def XYsplit(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    return X,y


# In[150]:


#F15: function to split dataframe into X (predictors) and y (label), apply standard scaling on X, apply the desired ML algorithm and output the results:

#input dataframe
#input the algo name (e.g., decisiontree)
#input whether this is a classification task or a regression task (then you should select either decisiontreeclassifier or decisiontreeregressor within the function)
#for classification, output confusion matrix, AUC, logloss and classification report
#for regression, output MAE, MSE, R-squared and adjusted R-squared

def KNN_x(X,y,n=0.2):
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


# In[117]:


#F17: Function to generate correlation heatmaps

def heat_map(df):
    
    mano.heatmap(df, figsize=(12,6))
    
    corrmatrix = df.corr()
    f, axis = plt.subplots(figsize =(25, 15)) 
    sb.heatmap(corrmatrix, ax = axis, linewidths = 0.2)


# In[118]:


#F18: Function to generate scatter plot

def scatter_(df,col1,col2):
    plt.scatter(df[col1], df[col2], c='indigo')
    plt.show()


# In[119]:


#F19: Correlation Analysis

def correlation_anlysis(df):
    # NOTE: If label column is non-numeric, 'encode' it before calling this function 
    numcols = df.select_dtypes(include=np.number)
    corr = numcols.corr()
    ax = sb.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sb.diverging_palette(20, 220, n=200),
    square=True
    )
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')


# In[120]:


#F20 Tests:::

def anova_test(df,a,b):
    fvalue, pvalue = stats.f_oneway(df[a], df[b])
    print('f val:',fvalue)
    print('p val:', pvalue)
    
    
def normalcy_test(df, col):
    
    output = anderson(df[col])
    print('Output: %.4f' % output.statistic)
    
    p = 0
    for i in range(len(output.critical_values)):
        sl, cv = output.significance_level[i], output.critical_values[i]
        if output.statistic < output.critical_values[i]:
            print('%.4f: %.4f, data looks normal (cant reject H0)' % (sl, cv))
        else:
            print('%.4f: %.4f, data not look normal (reject H0)' % (sl, cv))
            
def homogenity_test(df,c1,c2,c3):
    print("%.4f - %.4f" % levene(df[c1],df[c2],df[c3]))


# ## 2) include all classification and regression algorithms

# In[151]:


# Helper function to provide list of supported algorithms for Classification
def get_supported_algorithms():
    covered_algorithms = [LogReg, KNN, GradientBoosting, AdaBoost,
                          SVM, DecisionTree, RandomForest, NaiveBayes,
                          MultiLayerPerceptron]
    if XGBClassifier:
        covered_algorithms.append(XgBoost)
    if lgb:
        covered_algorithms.append(LightGbm)
    return covered_algorithms


# In[152]:


# Classification Algorithms

def LogReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LogisticRegression()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def KNN(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = KNeighborsClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def GradientBoosting(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def AdaBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def SVM(trainX, testX, trainY, testY, svmtype="SVC", verbose=True, clf=None):
    # for one vs all
    if not clf:
        if svmtype == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def DecisionTree(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def RandomForest(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = RandomForestClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def NaiveBayes(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GaussianNB()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def MultiLayerPerceptron(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def XgBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = XGBClassifier(random_state=1,learning_rate=0.01)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def LightGbm(trainX, testX, trainY, testY, verbose=True, clf=None):
    d_train = lgb.Dataset(trainX, label=trainY)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)
    return validationmetrics(clf,testX,testY,verbose=verbose)


# In[148]:


# Validation metrics for classification
def validationmetrics(model, testX, testY, verbose=True):   
    predictions = model.predict(testX)
    
    if model.__class__.__module__.startswith('lightgbm'):
        for i in range(0, predictions.shape[0]):
            predictions[i]= 1 if predictions[i] >= 0.5 else 0
    
    #Accuracy
    accuracy = accuracy_score(testY, predictions)*100
    
    #Precision
    precision = precision_score(testY, predictions, pos_label=1, labels=[0,1])*100
    
    #Recall
    recall = recall_score(testY, predictions,pos_label=1,labels=[0,1])*100
    
    #get FPR (specificity) and TPR (sensitivity)
    fpr , tpr, _ = roc_curve(testY, predictions)
    
    #AUC
    auc_val = auc(fpr, tpr)
    
    #F-Score
    f_score = f1_score(testY, predictions)
    
    if verbose:
        print("Prediction Vector: \n", predictions)
        print("\n Accuracy: \n", accuracy)
        print("\n Precision of event Happening: \n", precision)
        print("\n Recall of event Happening: \n", recall)
        print("\n AUC: \n",auc_val)
        print("\n F-Score:\n", f_score)
        #confusion Matrix
        print("\n Confusion Matrix: \n", confusion_matrix(testY, predictions,labels=[0,1]))
        # plot the roc curve for the model
        plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
    
    res_map = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc_val": auc_val,
                "f_score": f_score,
                "model_obj": model
              }
    return res_map


# In[124]:


# Helper function to provide list of supported algorithms for Regression
def get_supported_algorithms_reg():
    covered_algorithms = [LinearReg, RandomForestReg, PolynomialReg, SupportVectorRegression,
                          DecisionTreeReg, GradientBoostingReg, AdaBooostReg, VotingReg]
    return covered_algorithms


# In[173]:


# Helper function to run all algorithms provided in algo_list over given dataframe, without cross validation
# By default it will run all supported algorithms 
def run_algorithms(df, label_col, algo_list=get_supported_algorithms(), feature_list=[]):
    """
    Run Algorithms with manual split
    
    """
    # Lets make a copy of dataframe and work on that to be on safe side 
    _df = df.copy()
    
    if feature_list:
        impftrs = feature_list
        impftrs.append(label_col)
        _df = _df[impftrs]
    
    _df, trainX, testX, trainY, testY = traintestsplit(_df, 0.2, 91, label_col=label_col)
    algo_model_map = {}
    for algo in algo_list:
        print("============ " + algo.__name__ + " ===========")
        res = algo(trainX, testX, trainY, testY)
        algo_model_map[algo.__name__] = res.get("model_obj", None)
        print ("============================== \n")
    
    return algo_model_map


# In[125]:


# Regression Algorithms
def LinearReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LinearRegression()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def RandomForestReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = RandomForestRegressor(n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def PolynomialReg(trainX, testX, trainY, testY, degree=3, verbose=True, clf=None):
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(trainX)
    poly.fit(X_poly, trainY)
    if not clf:
        clf = LinearRegression() 
    clf.fit(X_poly, trainY)
    return validationmetrics_reg(clf, poly.fit_transform(testX), testY, verbose=verbose)

def SupportVectorRegression(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = SVR(kernel="rbf")
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def DecisionTreeReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def GradientBoostingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def AdaBooostReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostRegressor(random_state=0, n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def VotingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100)
    sv = SVR(kernel="rbf")
    dt = DecisionTreeRegressor()
    gb = GradientBoostingRegressor()
    ab = AdaBoostRegressor(random_state=0, n_estimators=100)
    if not clf:
        clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)


# In[126]:


#Validation metrics for Regression algorithms
def validationmetrics_reg(model,testX,testY, verbose=True):
    predictions = model.predict(testX)
    
    # R-squared
    r2 = r2_score(testY,predictions)
    
    # Adjusted R-squared
    r2_adjusted = 1-(1-r2)*(testX.shape[0]-1)/(testX.shape[0]-testX.shape[1]-1)
    
    # MSE
    mse = mean_squared_error(testY,predictions)
    
    #RMSE
    rmse = math.sqrt(mse)
    
    if verbose:
        print("R-Squared Value: ", r2)
        print("Adjusted R-Squared: ", r2_adjusted)
        print("RMSE: ", rmse)
    
    res_map = {
                "r2": r2,
                "r2_adjusted": r2_adjusted,
                "rmse": rmse,
                "model_obj": model
              }
    return res_map


# ## 3) FS: feature selection algorithms (I included RFFS, RFE, MI and PCA but you can add only RFFS as for me, it works good).

# In[127]:


# Helper function to get fetaure importance metrics via Random Forest Feature Selection (RFFS)
def RFfeatureimportance(df, trainX, testX, trainY, testY, trees=35, random=None, regression=False):
    if regression:
        clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
    else:
        clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(trainX,trainY)
    #validationmetrics(clf,testX,testY)
    res = pd.Series(clf.feature_importances_, index=df.columns.values).sort_values(ascending=False)*100
    print(res)
    return res

def MachineLearningwithRFFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY,
                              trees=10, regression=regression)

    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}


# Helper function to select important features via RFFS, run supported ML algorithms over dataset with manual split and measure accuracy without Cross Validation - select features with importance >=threshold
def MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY, trees=10, regression=regression)
    
    impftrs = list(res[res > threshold].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}





# In[161]:


# ## Mutual Information Feature Selection (MIFS) 
# MachineLearningwithMIFS() => Helper function to select important features and run supported ML algorithms over dataset

# mutualinformation()  => Helper function to get fetaure importance metrics via Mutual Information Classifier/Regressor.
def mutualinformation(df, label_col, regression=False):
    df_cpy = df.copy()
    y = df_cpy[label_col].copy()
    X = df_cpy.drop(label_col,axis=1)
    if regression:
        mutual_info = mutual_info_regression(X,y,random_state=35)
    else:
        mutual_info = mutual_info_classif(X,y,random_state=35)
    results = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)*100
    print(results)
    return results


# Helper function to select important features via MIFS, run supported ML algorithms over dataset with manual split and measure accuracy, with CV ... select features with importance >=threshold
def MachineLearningwithMIFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    res = mutualinformation(df_cpy, label_col=label_col, regression=regression)
    
    #include all selected features in impftrs
    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}

# Helper function to select important features via MIFS, run supported ML algorithms over dataset with manual split and measure accuracy, without CV ... select features with importance >=threshold
def MachineLearningwithMIFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    res = mutualinformation(df_cpy, label_col=label_col, regression=regression)
    
    #include all selected features in impftrs
    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}



# In[162]:


# Helper function to provide list of classification algorithms to be used for recursive elimination feature selection
def get_supported_algorithms_refs():
    algo_list = [LogisticRegression, GradientBoostingClassifier, AdaBoostClassifier,
                          DecisionTreeClassifier, RandomForestClassifier]
    return algo_list

# Helper function to provide list of regression algorithms to be used for recursive elimination feature selection
def get_supported_reg_algorithms_refs():
    algo_list = [LinearRegression, RandomForestRegressor,
                 DecisionTreeRegressor, GradientBoostingRegressor, AdaBoostRegressor]
    return algo_list

# Helper function to select important features via REFS, run supported ML algorithms over dataset with manual split and measure accuracy, with CV ... select features with importance >=threshold
# flexible enough to use any algorithm for recursive feature elimination and any alogorithm to run on selected features
def GenericREFS_CV(df, label_col,
                algo_list=get_supported_algorithms(),
                regression=False,
                re_algo=RandomForestClassifier,
                **kwargs):
    
    X,y = XYsplit(df, label_col)
    clf = re_algo(**kwargs)
    selector = RFECV(estimator=clf, step=1, cv=10)
    selector = selector.fit(X,y)
    feature_list = X.columns[selector.support_].tolist()
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=feature_list, cross_valid_method=cross_valid_method)
    return {"selected_features": feature_list, "results": results}

# Helper function to select important features via REFS, run supported ML algorithms over dataset with manual split and measure accuracy, without CV ... select features with importance >=threshold
# flexible enough to use any algorithm for recursive feature elimination and any alogorithm to run on selected features
def GenericREFS(df, label_col,
                algo_list=get_supported_algorithms(),
                re_algo=RandomForestClassifier,
                **kwargs):
    
    X,y = XYsplit(df, label_col)
    clf = re_algo(**kwargs)
    selector = RFE(estimator=clf, step=1)
    selector = selector.fit(X,y)
    feature_list = X.columns[selector.support_].tolist()
    
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=feature_list)
    return {"selected_features": feature_list, "results": results}


# In[ ]:


# Helper function to perform feature selection using PCA. It runs supported algorithms with over specified components and mesure performance stats, with Cross Validation

def PCA_FS_CV(df, label_col, n_components, algo_list=get_supported_algorithms(), regression=False):
    df_cpy = df.copy()
    X,y = XYsplit(df_cpy, label_col)
    
    cross_valid_method = cross_valid_kfold if regression else cross_valid_stratified_kf 
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for X_train,y_train,X_test,y_test in cross_valid_method(X, y, split=10):
            # First we need to normalize the data
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            # Now perform PCA
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
            # apply algo on this fold and save result for later usage
            res_algo = algo(X_train, X_test, y_train, y_test, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
            
        algo_model_map[algo.__name__] = clf
        
    
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] =  algo_model_map[algo]
    return {"n_components": n_components, "results": algo_model_map}


# ## 4) CV: cross validation approaches (include stratified K-fold definitely)

# In[131]:


# #### For Cross Validation, lets create generator functions for different cross validation techniques, this will help us run an iterator over all folds
def cross_valid_kfold(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for KFold cross validation
    
    """
    kf = KFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
    
def cross_valid_repeated_kf(X, y, split=10, random=None, repeat=10):
    """
    Generator function for Repeated KFold cross validation
    
    """
    kf = RepeatedKFold(n_splits=split, random_state=random, n_repeats=repeat)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
        

def cross_valid_stratified_kf(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for Stratified KFold cross validation
    
    """
    skf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in skf.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY


def cross_valid_strat_shuffle_kf(X, y, split=10, random=None):
    """
    Generator function for StratifiedShuffle cross validation
    
    """
    sss = StratifiedShuffleSplit(n_splits=split, random_state=random)
    for train_index, test_index in sss.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY


# In[132]:


# With stratified kfold validation support
def run_algorithms_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with cross validation
    
    """
    _df = df.copy()
    X,y = XYsplit(_df, label_col)
    
    # Select features if specified by driver program
    if feature_list:
        X = X[feature_list]
    
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for trainX,trainY,testX,testY  in cross_valid_method(X, y, split=10):
            res_algo = algo(trainX, testX, trainY, testY, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
                
        algo_model_map[algo.__name__] = clf
            
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] = algo_model_map[algo]
    
    return score_map


# ## 5) REG: regularization (lasso, ridge regression) methods (include Lasso definitely).
# 
# REGULARIZTION NOT REQUIRED

# ## 6) Include methods for addressing class imbalance

# In[181]:


def check_balance(df,col):
    target_count = df[col].value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
    
    target_count.plot(kind='bar', title='Count of Class');

def rand_undersample(df,label_col):
        
    y = df[label_col].copy()
    
    #now delete the original
    X = df.drop(label_col,axis=1)
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X, y)
    print(Counter(y_over))
    X_over['result']=y_over
    return X_over
    
def rand_oversample(df,label_col):
   #make a copy of the label column and store in y
    y = df[label_col].copy()
    
    #now delete the original
    X = df.drop(label_col,axis=1)
    ros = RandomOverSampler(sampling_strategy='minority')
    A_resampled, b_resampled = ros.fit_resample(X, y)
    print(sorted(Counter(b_resampled).items()))
    A_resampled['result']=b_resampled
    return A_resampled


# ## 7) Include any other helper/logistic function you want to add

# In[134]:


#Added in respective code above


# ## 8) Include function for executing ML (classification) without FS, REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve

# In[167]:


#first split dataframe into x (predictors) and y (label)

def classification_ML(x, y, n= 0.2):
    
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=n)
    
    c= str(input('Which Classifier?\n\n[LogReg, KNN, GradientBoosting, AdaBoost, SVM, DecisionTree,\nRandomForest,NaiveBayes, MultiLayerPerceptron, XgBoost, LightGbm]\n'))
    
    
    if c =='LogReg':
        print('applying Logistic Regression Classification algorithm without FS, REG or CV....\n\n')
        LogReg(trainX, testX, trainY, testY)
        
    elif c =='KNN':
        print('applying KNN algorithm without FS, REG or CV....\n\n')
        KNN(trainX, testX, trainY, testY)
        
    elif c =='GradientBoosting':
        print('applying GradientBoosting algorithm without FS, REG or CV....\n\n')
        GradientBoosting(trainX, testX, trainY, testY)
        
    elif c =='AdaBoost':
        print('applying AdaBoost algorithm without FS, REG or CV....\n\n')
        AdaBoost(trainX, testX, trainY, testY)
        
    elif c =='SVM':
        print('applying SVM algorithm without FS, REG or CV....\n\n')
        SVM(trainX, testX, trainY, testY)
        
    elif c =='DecisionTree':
        print('applying DecisionTree algorithm without FS, REG or CV....\n\n')
        DecisionTree(trainX, testX, trainY, testY)
        
    elif c =='RandomForest':
        print('applying RandomForest algorithm without FS, REG or CV....\n\n')
        RandomForest(trainX, testX, trainY, testY)
        
    elif c =='NaiveBayes':
        print('applying NaiveBayes algorithm without FS, REG or CV....\n\n')
        NaiveBayes(trainX, testX, trainY, testY)
        
    elif c =='MultiLayerPerceptron':
        print('applying MultiLayerPerceptron algorithm without FS, REG or CV....\n\n')
        MultiLayerPerceptron(trainX, testX, trainY, testY)
        
    elif c =='XgBoost':
        print('applying XgBoost algorithm without FS, REG or CV....\n\n')
        XgBoost(trainX, testX, trainY, testY)
        
    elif c =='LightGbm':
        print('applying LightGbm algorithm without FS, REG or CV....\n\n')
        LightGbm(trainX, testX, trainY, testY)
        
    else:
        print('\ninvalid classification algo entered ... check exact match from list')


# ## 9) Include function for executing ML (classification) with FS and without REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve

# In[168]:


def classification_ML_FS(df,col,threshold=5,algo_list=get_supported_algorithms(), regression=False, re_algo=RandomForestClassifier,**kwargs):
      
    FS= str(input('Which FS Algo?\n\n[RFFS, MIFS, REFS]\n'))
    
    
    if FS =='RFFS':
        print('applying FS with RFFS and then applying all ML algos....\n\n')
        MachineLearningwithRFFS(df, col , threshold=5,
                                    algo_list=get_supported_algorithms())
   
    elif FS =='MIFS':
        print('applying FS with MIFS and then applying all ML algos....\n\n')
        MachineLearningwithMIFS(df, col , threshold=5,
                                    algo_list=get_supported_algorithms())

    elif FS =='REFS':
            print('applying FS with REFS and then applying all ML algos....\n\n')
            GenericREFS(df, col, algo_list=get_supported_algorithms(), re_algo=RandomForestClassifier,**kwargs)
             
    else:
        print('\ninvalid FS algo entered ... check exact match from list')


# ## 10) Include function for executing ML (classification) with REG and without FS and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 
# #REG NOT REQUIRED

# ## 11) Include function for executing ML (classification) with CV and without FS and REG. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve

# In[170]:


def classification_ML_CV(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with cross validation
    
    """
    _df = df.copy()
    X,y = XYsplit(_df, label_col)
    
    # Select features if specified by driver program
    if feature_list:
        X = X[feature_list]
    
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for trainX,trainY,testX,testY  in cross_valid_method(X, y, split=10):
            res_algo = algo(trainX, testX, trainY, testY, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
                
        algo_model_map[algo.__name__] = clf
            
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] = algo_model_map[algo]
    
    return score_map


# ## 12) Include function for executing ML (classification) with CV and with FS and with REG. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve
# 

# In[169]:


#Executing all ML functions with FS and CV. Reg not applied since not required as per instructions relayed on whatsapp

def classification_ML_FS_CV(df,col,threshold=5,algo_list=get_supported_algorithms(), regression=False, re_algo=RandomForestClassifier,**kwargs):
      
    FS_CV= str(input('Which FS Algo?\n\n[RFFS_CV, MIFS_CV, REFS_CV]\n'))
    
    
    if FS_CV =='RFFS_CV':
        print('applying FS with RFFS and then running ML algos with CV....\n\n')
        MachineLearningwithRFFS_CV(df, col , threshold=5,
                                    algo_list=get_supported_algorithms())
   
    elif FS_CV =='MIFS_CV':
        print('applying FS with MIFS and then running all ML algos with CV....\n\n')
        MachineLearningwithMIFS_CV(df, col , threshold=5,
                                    algo_list=get_supported_algorithms())

    elif FS_CV =='REFS_CV':
            print('applying FS with REFS and then running all ML algos with CV....\n\n')
            GenericREFS_CV(df, col, algo_list=get_supported_algorithms(), re_algo=RandomForestClassifier,**kwargs)
             
    else:
        print('\ninvalid FS algo entered ... check exact match from list')

