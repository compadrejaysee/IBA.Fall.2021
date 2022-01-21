#!/usr/bin/env python
# coding: utf-8

# # Wrangling Functions

# In[1]:


def load_data(filetype):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyodbc
    valid = {'sql','csv','excel'}
    if filetype not in valid:
        raise ValueError("filetype must be one of %r." % valid)
    if filetype=='csv':
        path=input('Enter path of file')
        df=pd.read_csv(path)
    elif filetype=='excel':
        path=input('Enter path of file')
        df=pd.read_excel(path)
    else:
        query=input('Enter sql query')
        #connect_params=input('Enter connection parameters')
        conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-AIS9287\SQLEXPRESS;'
                      'Database=ML_PROJECT;'# Parameters can be altered accordingly
                      'Trusted_Connection=yes;')
        cursor = conn.cursor()
        df = pd.read_sql_query(query,conn)
    return df


# In[2]:


def overview(df,head,n):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print('The shape is :',df.shape)
    print('The column data types are: \n')
    print(df.dtypes)
    print('\n')
    valid = {'head','tail'}
    if head not in valid:
        raise ValueError("Either head or tail")
    if head=='head':
        print('The first '+str(n)+' rows are :')
        display(df.head(n))
    else:
        print('The last '+str(n)+' rows are :')
        display(df.tail(n))
    


# In[3]:


def missing_details(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    b = pd.DataFrame()
    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)
    b['Missing value count']=df.isnull().sum()
    b['N unique value'] = df.nunique()
    return b 


# ### T Test

# In[4]:


def ttest(group1,group2,related=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyodbc
    from scipy import stats
    if related==False:
        stat,p=stats.ttest_ind(group1,group2)
        if p<0.05:
            print('P value is:',p)
            print('Sampling error does not exist')
        else:
            print('P value is:',p)
            print('Sampling error exists')
    else:
        stat,p=stats.ttest_rel(group1,group2)
        if p<0.05:
            print('P value is:',p)
            print('Sampling error does not exist')
        else:
            print('P value is:',p)
            print('Sampling error exists')


# ### Chi square test

# In[5]:


def chi_square_test(df,*cols):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyodbc
    from scipy.stats import chi2_contingency
    x=[]
    for i in cols:
        x.append(i)
    target=input('Enter target variable')
    for i in x:
        print(i)
        data=pd.crosstab(df[i],df[target])
        stat, p, dof, expected = chi2_contingency(data)

        # interpret p-value
        alpha = 0.05
        print("p value is " + str(p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (H0 holds true)')
        print('\n')


# ### ANOVA

# In[6]:


def anova(df,continous,categorical):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    lm=ols(continous+'~'+categorical,data=df).fit()
    table=sm.stats.anova_lm(lm)
    display(table)


# ### Correlation Heatmaps

# In[7]:


def corr_heatmap(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    corr=df.corr()
    plt.subplots(figsize=(12,12))
    sns.heatmap(corr,annot=True)


# ### Normality Test

# In[8]:


def normality_test(df,*cols):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import pylab as py
    for i in cols:
        x.append(i)
    sm.qqplot(df[column], line ='s') 
    py.show()
    from scipy.stats import skew
    from scipy.stats import kurtosis
    method=input('Enter normality test')
    if method=='Shapiro Wilk Test':
        for column in x:
            print(column)
            sm.qqplot(df[column], line ='s') 
            py.show()
            display(df[column].describe())
            from scipy.stats import skew
            from scipy.stats import kurtosis
            Skew=skew(df[column])
            Kurtosis=kurtosis(df[column])
            print('Skew is',Skew)
            print('Kurtosis is',Kurtosis)
            from scipy.stats import shapiro
            stat,p=shapiro(df[column])
            print('stat=%.3f,p=%.3f\n'%(stat,p))
            if p>0.05:
                print('Normally distributed according to shapiro test')
            else:
                print('Not Normally distributed according to shapiro test')
            print('\n')
    elif method=='Anderson':
        for column in x:
            print(column)
            sm.qqplot(df[column], line ='s') 
            py.show()
            display(df[column].describe())
            from scipy.stats import skew
            from scipy.stats import kurtosis
            Skew=skew(df[column])
            Kurtosis=kurtosis(df[column])
            print('Skew is',Skew)
            print('Kurtosis is',Kurtosis)
            from scipy.stats import anderson
            result=anderson(df[column])
            print('stat=%.3f'%(result.statistic))
            for i in range(len(result.critical_values)):
                sig_level,cric_value=result.significance_level[i],result.critical_values[i]
                if result.statistic<cric_value:
                    print(f"Probably Normally distributed {cric_value} critical value at {sig_level} level of significance")
                else:
                    print(f"Probably Not Normally distributed {cric_value} critical value at {sig_level} level of significance")
            print('\n')
    elif method=='Chi square':
        for column in x:
            print(column)
            sm.qqplot(df[column], line ='s') 
            py.show()
            display(df[column].describe())
            from scipy.stats import skew
            from scipy.stats import kurtosis
            Skew=skew(df[column])
            Kurtosis=kurtosis(df[column])
            print('Skew is',Skew)
            print('Kurtosis is',Kurtosis)
            from scipy.stats import chisquare
            stat,p=chisquare(df[column])
            print('stat=%.3f,p=%.3f\n'%(stat,p))
            if p>0.05:
                print('Normally distributed according to chisquare test')
            else:
                print('Not Normally distributed according to chisquare test')
            print('\n')
    elif method=='Lilliefors':
        for column in x:

            print(column)
            sm.qqplot(df[column], line ='s') 
            py.show()
            display(df[column].describe())
            from scipy.stats import skew
            from scipy.stats import kurtosis
            Skew=skew(df[column])
            Kurtosis=kurtosis(df[column])
            print('Skew is',Skew)
            print('Kurtosis is',Kurtosis)
            from statsmodels.stats.diagnostic import lilliefors
            stat,p=lilliefors(df[column])
            print('stat=%.3f,p=%.3f\n'%(stat,p))
            if p>0.05:
                print('Normally distributed according to lilliefors test')
            else:
                print('Not Normally distributed according to lilliefors test')
            print('\n')
    elif method=='Kolmogorov':
        for column in x:
            print(column)
            sm.qqplot(df[column], line ='s') 
            py.show()
            display(df[column].describe())
            from scipy.stats import skew
            from scipy.stats import kurtosis
            Skew=skew(df[column])
            Kurtosis=kurtosis(df[column])
            print('Skew is',Skew)
            print('Kurtosis is',Kurtosis)
            from scipy.stats import kstest
            stat,p=kstest(df[column],'norm')
            print('stat=%.3f,p=%.3f\n'%(stat,p))
            if p>0.05:
                print('Normally distributed according to Kolmogorov-Smirnov test')
            else:
                print('Not normally distributed according to Kolmogorov-Smirnov test')
            print('\n')
    else:
        raise ValueError('Method must be one of Shapiro Wilk Test,Anderson,Chi square,Lilliefors,Kolmogorov')


# ### Missing Value solution

# In[9]:


def handle_mv(df,column,method):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    if method=='mean':
        df[column].fillna(df[column].mean(),inplace=True)
    elif method=='median':
        df[column].fillna(df[column].median(),inplace=True)
    elif method=='mode':
        df[column].fillna(df[column].mode()[0],inplace=True)
    elif method=='value':
        x=input('Enter value to replace null values')
        df[column].fillna(x,inplace=True)
    elif method=='interpolation':
        x=input('forward or backward interpolation?')
        if x=='forward':
            df[column] = df['column'].interpolate(method ='linear', limit_direction ='forward')
        elif x=='backward':
            df[column] = df['column'].interpolate(method ='linear', limit_direction ='backward')
        else:
            raise ValueError('Select on from forward or backward')
    elif method=='KNN':
        from sklearn.impute import KNNImputer
        neighbors=int(input('Enter number of number of neighbors'))
        df2=df.copy()
        imputer = KNNImputer (n_neighbors=neighbors)
        y=[]
        n=int(input('Enter the number of columns to fit KNN imputer'))
        for i in range(n):
            x=input('Enter name of column')
            y.append(x)
        y.append(column)
        df[y] = imputer.fit_transform(df[y])
        x_axis=input('x axis label')
        y_axis=input('y axis label')
        nulls=df2[x_axis].isna()+df2[y_axis].isna()
        df.plot(x=x_axis,y=y_axis,kind='scatter',alpha=0.5,c=nulls,cmap='rainbow')
        plt.show()
    elif method=='group_mode':        
        x=input('Enter name of second column')
        df2=df[[column,x]]
        df2[column]=df2.groupby(x).transform(lambda group:group.fillna(group.mode()[0]) )
        df[column]=df2[column]
    elif method=='drop': 
        df.dropna(axis=0, inplace=True)
    else:
        raise ValueError('method must be one of mean,median,mode,value,interpolation,KNN,group_mode or drop')


# ### Transformations

# In[10]:


def remove_cols(df,*columns):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols=[]
    for i in columns:
        cols.append(i)
    df.drop(columns=cols,inplace=True)
    


# In[11]:


def drop_rows(df,column,value):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    df.drop(df.loc[df[column]==value].index,inplace=True)


# In[12]:


def missing_details(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    b = pd.DataFrame()
    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)
    b['Missing value count']=df.isnull().sum()
    b['N unique value'] = df.nunique()
    return b 


# In[13]:


def missing_analysis(df,*cols):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import missingno as mn
    x=[]
    for i in cols:
        x.append(i)
    df2=df[x]
    chart_type=input('Enter the type of chart required')
    if chart_type=='bar':
        mn.bar(df2)
    elif chart_type=='matrix':
        mn.matrix(df2)
    elif chart_type=='heatmap':
        mn.heatmap(df2)
    elif chart_type=='dendrogram':
        mn.dendrogram(df2)
    else:
        raise ValueError('Chart type must be on of bar,matrix,heatmap or dendrogram')


# In[14]:


def categorical_analysis(df,column):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(df[column].value_counts())
    plt.subplots(figsize=(12,6))
    sns.countplot(df[column])


# In[15]:


def numerical_analysis(df,column):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import pylab as py
    sns.distplot(df[column])
    plt.show()
    sns.boxplot(df[column])
    plt.show()
    sm.qqplot(df[column], line ='s') 
    py.show()
    display(df[column].describe())
    from scipy.stats import skew
    from scipy.stats import kurtosis
    Skew=skew(df[column])
    Kurtosis=kurtosis(df[column])
    print('Skew is',Skew)
    print('Kurtosis is',Kurtosis)
    if Skew>-2 and Skew<2 and Kurtosis>-7 and Kurtosis<7:
        print('Can be considered normally distributed on the basis of skewness and kurtosis')
    else:
        print('Cannot be considered normally distributed on the basis of skewness and kurtosis')
    print('\n')
    print('Shapiro Wilk Test for Normality')
    from scipy.stats import shapiro
    stat,p=shapiro(df[column])
    print('stat=%.3f,p=%.3f\n'%(stat,p))
    if p>0.05:
        print('Normally distributed according to shapiro test')
    else:
        print('Not Normally distributed according to shapiro test')
    print('\n')
    print('Anderson Test for Normality')
    from scipy.stats import anderson
    result=anderson(df[column])
    print('stat=%.3f'%(result.statistic))
    for i in range(len(result.critical_values)):
        sig_level,cric_value=result.significance_level[i],result.critical_values[i]
        if result.statistic<cric_value:
            print(f"Probably Normally distributed {cric_value} critical value at {sig_level} level of significance")
        else:
            print(f"Probably Not Normally distributed {cric_value} critical value at {sig_level} level of significance")
    print('\n')
    print('Chi square Normality test')
    from scipy.stats import chisquare
    stat,p=chisquare(df[column])
    print('stat=%.3f,p=%.3f\n'%(stat,p))
    if p>0.05:
        print('Normally distributed according to chisquare test')
    else:
        print('Not Normally distributed according to chisquare test')
    print('\n')
    print('Lilliefors Test for Normality')
    from statsmodels.stats.diagnostic import lilliefors
    stat,p=lilliefors(df[column])
    print('stat=%.3f,p=%.3f\n'%(stat,p))
    if p>0.05:
        print('Normally distributed according to lilliefors test')
    else:
        print('Not Normally distributed according to lilliefors test')
    print('\n')
    print('Kolmogorov-Smirnov test')
    from scipy.stats import kstest
    stat,p=kstest(df[column],'norm')
    print('stat=%.3f,p=%.3f\n'%(stat,p))
    if p>0.05:
        print('Normally distributed according to Kolmogorov-Smirnov test')
    else:
        print('Not normally distributed according to Kolmogorov-Smirnov test')


# In[16]:


def change_type(df,column,t):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    if t=='integer':
        df[column]=pd.to_numeric(df[column],downcast='integer')
    elif t=='float':
        df[column]=pd.to_numeric(df[column],downcast='float')
    elif t=='datetime':
        df[column]=pd.to_datetime(df[column])
    elif t=='object':
        df[column]=df[column].astype('object')
    else:
        raise ValueError('Must be one of integer,float,datetime,object')
        


# In[17]:


def change_descr(df,column,value=None,new_value=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    x=input('Remove white space? Y or N')
    if x=='Y':
        df[column]=df[column].str.replace(' ','')
    elif x=='N':
        df[column]=df[column].replace(value,new_value)
    else:
        raise ValueError('Y or N only')
    y=input('change column name? Y or N')
    if y=='Y':
        w=input('Remove with space? Y or N')
        if w=='N':
            z=input('Enter new column name')
            df.rename(columns={column:z},inplace=True)
        elif w=='Y':
            for i in df.columns:
                
                c=i.replace(' ','')
                df.rename(columns={i:c},inplace=True)
                #df.rename(columns=lambda x: x.strip())
    else:
        pass


# In[18]:


def deep_copy(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    df2=df.copy(deep=True)
    return df2


# In[19]:


def encode(df,*columns):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols=[]
    for i in columns:
        cols.append(i)
    encode_type=input('Enter encode type')
    if encode_type=='label':
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        for i in cols:
            df[i]=le.fit_transform(df[[i]])
        return df
    elif encode_type=='ordinal':
        for i in cols:

            ord_dict={}
            print(i)
            n=int(input("enter a number of unique values present in the column: "))
            for j in range(n):
                key=input('Unique value for the column')
                value=int(input('Enter ordinal number for the value'))
                ord_dict[key]=value
            x=df[i].map(ord_dict)
            df[i]=x
        return df
    elif encode_type=='one-hot':
        df2=pd.get_dummies(df,columns=cols)
        
        return df2
    else:
        raise Error('Invalid encode type, select one from label,ordinal,one-hot')


# # Classification Algorithms

# In[20]:


def LogReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = LogisticRegression()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def KNN(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = KNeighborsClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def GadientBoosting(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = GradientBoostingClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def AdaBoost(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = AdaBoostClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn import svm
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        if svm_type == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
    clf.fit(xtrain,ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    accuracy=accuracy_score(ytest, y_pred)*100
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    AUC=auc(fpr, tpr)*100
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    precision=precision_score(ytest, y_pred)*100
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    recall=recall_score(ytest,y_pred)*100
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    f1=f1_score(ytest,y_pred)*100
    print('logistic loss :\n',log_loss(ytest,y_pred))
    log=log_loss(ytest,y_pred)
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    class_report=classification_report(ytest,y_pred)
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()
    return accuracy,AUC,precision,recall,f1,log,class_report


def DecisionTree(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = DecisionTreeClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def RandomForest(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = RandomForestClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def NaiveBayes(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = GaussianNB()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def MultiLayerPerceptron(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def XgBoost(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        clf  = XGBClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()

def LightGbm(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    
    if not clf:
        clf  = LGBMClassifier()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    fpr , tpr, _ = roc_curve(ytest, y_pred)
    print("AUC  (%): \n",auc(fpr, tpr)*100)
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    print('logistic loss :\n',log_loss(ytest,y_pred))
    print(algorithm + '\n'+classification_report(ytest,y_pred))
    #print(confusion_matrix(ytest, y_pred))
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()


# # Regression Algorithms

# In[21]:


def LinearReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import math
    if not clf:
        clf  = LinearRegression()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def RandomForestReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    import math
    if not clf:
        clf  = RandomForestRegressor(n_estimators=100)
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def PolynomialReg(xtrain, xtest, ytrain, ytest,degree=3, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import math
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(xtrain)
    poly.fit(X_poly, ytrain)
    if not clf:
        clf = LinearRegression() 
    clf.fit(X_poly, ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def SupportVectorRegression(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.svm import SVR
    from sklearn import metrics
    import math
    if not clf:
        clf  = SVR(kernel="rbf")
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def DecisionTreeReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import metrics
    import math
    if not clf:
        clf  = DecisionTreeRegressor()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def GradientBoostingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import metrics
    import math
    if not clf:
        clf  = GradientBoostingRegressor()
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def AdaBooostReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn import metrics
    import math
    if not clf:
        clf  = AdaBoostRegressor(random_state=0, n_estimators=100)
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)

    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()

def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn import metrics
    import math
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100)
    sv = SVR(kernel="rbf")
    dt = DecisionTreeRegressor()
    gb = GradientBoostingRegressor()
    ab = AdaBoostRegressor(random_state=0, n_estimators=100)
    if not clf:
        clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
    clf.fit(xtrain , ytrain)
    y_pred=clf.predict(xtest)
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
    print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
    print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
    rmse = math.sqrt(mse)
    print('The RMSE is ',rmse)
    mse=metrics.mean_squared_error(ytest, y_pred)
    mae=metrics.mean_absolute_error(ytest, y_pred)
    r2=metrics.r2_score(ytest,y_pred)
    n=len(xtrain)
    k=len(x.columns)
    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
    print('The adjusted R Squared Error is',r2_adj)
    plt.scatter(ytest, y_pred, c = 'green') 
    plt.xlabel("True Value") 
    plt.ylabel("Predicted value") 
    plt.title("True value vs predicted value") 
    plt.show()
    return mse,mae,rmse,r2,r2_adj


# # Cross validation

# In[22]:


def cross_valid_kfold(x, y, split=10, random=None, shuffle=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold 
    kf = KFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(x):
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        yield xtrain,ytrain,xtest,ytest


# In[23]:


def cross_valid_repeated_kf(x, y, split=10, random=None, repeat=10):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RepeatedKFold
    kf = RepeatedKFold(n_splits=split, random_state=random, n_repeats=repeat)
    for train_index, test_index in kf.split(x):
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        yield xtrain,ytrain,xtest,ytest


# In[24]:


def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(x, y):
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        yield xtrain,ytrain,xtest,ytest

def cross_valid_strat_shuffle_kf(x, y, split=10, random=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=split, random_state=random)
    for train_index, test_index in sss.split(x, y):
        xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        yield xtrain,ytrain,xtest,ytest


# # Regularization

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Feature Selection

# In[25]:


def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    if regression:
        clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
    else:
        clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(xtrain,ytrain)
    #validationmetrics(clf,testX,testY)
    res = pd.Series(clf.feature_importances_, index=df.columns.values).sort_values(ascending=False)*100
    print(res)
    return res


# # Dimensionality Reduction

# In[26]:


def PCA_plot(df,label,start,end,gap,ib=True,kernel='rbf'):
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import model_selection
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostClassifier(),
    #"SVM": svm.SVC(kernel=kernel),
    'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'light Gradient Boosting':LGBMClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier()}
    from sklearn.preprocessing import scale 
    for key,classifier in dict_classifiers.items():
        X = df.drop([label], axis=1)
        y = df[[label]]

        #scale predictor variables
        pca = PCA()
        X_reduced = pca.fit_transform(scale(X))
        X_reduced=pd.DataFrame(X_reduced)
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in X_reduced.columns:
            X_reduced[i] = ss.fit_transform(X_reduced[[i]])
        #display(X_reduced)
        X_reduced=X_reduced.to_numpy()

        #define cross validation method
        print(key)

        clf = classifier
        acc= []
        pre=[]
        re=[]
        f1=[]
        results=pd.DataFrame(columns=['Number of components','Accuracy','Precision','Recall','F1 score']) 
        for idx,i in tqdm(enumerate(np.arange(start,end,gap))):
            results.loc[idx,'Number of components']=i   
            accuracy = cross_val_score(clf,X_reduced[:,:i], y, cv=10, scoring='accuracy').mean()
            acc.append(accuracy)
            results.loc[idx,'Accuracy']=accuracy
            precision = cross_val_score(clf,X_reduced[:,:i], y, cv=10, scoring='precision').mean()
            pre.append(precision)
            results.loc[idx,'Precision']=precision

            recall = cross_val_score(clf,X_reduced[:,:i], y, cv=10, scoring='recall').mean()
            re.append(recall)
            results.loc[idx,'Recall']=recall
            f1score = cross_val_score(clf,X_reduced[:,:i], y, cv=10, scoring='f1').mean()
            f1.append(f1score)
            results.loc[idx,'F1 score']=f1score
        display(results)

        # Plot cross-validation results 
        if ib==True:
            plt.subplots(figsize=(15, 5))
            plt.subplot(1,3,1)
            plt.plot(np.arange(start,end,gap),f1)
            plt.xlim(start-1,end)
            plt.xlabel('Number of Principal Components')
            plt.ylabel('f1 score')
            plt.title('target')
        else:
            
            plt.subplots(figsize=(15, 5))
            plt.subplot(1,3,1)
            plt.plot(np.arange(start,end,gap),acc)
            plt.xlim(start-1,end)
            plt.xlabel('Number of Principal Components')
            plt.ylabel('accuracy')
            plt.title('target')
        #plt.show()
        plt.subplot(1,3,2)
        plt.plot(np.arange(start,end,gap),pre)
        plt.xlim(start-1,end)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Precision')
        plt.title('target')
        #plt.show()
        plt.subplot(1,3,3)
        plt.plot(np.arange(start,end,gap),re)
        plt.xlim(start-1,end)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Recall')
        plt.title('target')
        #plt.tight_layout(4)
        plt.show()
        print('\n')
        


# In[27]:


def pca_transform(df,label,n):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    x = df.drop([label], axis=1)
    y = df[[label]]
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    x = ss.fit_transform(x)
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n)
    pca_df = pca.fit_transform(x)
    pca_df=pd.DataFrame(pca_df)
    pca_df[label]=y
    return pca_df


# In[28]:


def lda(df,label,n):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    x = df.drop([label], axis=1)
    y = df[[label]]
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    for i in x.columns:
        x[i]=ss.fit_transform(x[[i]])
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda=LinearDiscriminantAnalysis(n_components=n)
    lda.fit(x,y)
    X=lda.transform(x)
    X=pd.DataFrame(X)
    X.rename(columns={0:'Feature 1'},inplace=True)
    X[label]=y
    return X


# # Class Imbalance Handling

# In[29]:


def over_sample(df,label,ratio=0.5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy=ratio)
    from sklearn.model_selection import train_test_split
    x=df.drop([label], axis=1)
    y=df[label]
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
    return x_res,y_res


# In[30]:


def under_sample(df,label,ratio=0.5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from imblearn.under_sampling import RandomUnderSampler
    oversample = RandomUnderSampler(sampling_strategy=ratio)
    from sklearn.model_selection import train_test_split
    x=df.drop([label], axis=1)
    y=df[label]
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
    return x_res,y_res


# # Miscellaneous

# In[31]:


def knn_plot(df,label,classification=True,over_sample=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor
    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from sklearn.preprocessing import StandardScaler
    x=df.drop([label], axis=1)
    y=df[label]
    ss=StandardScaler()
    for i in x.columns:
        x[i] = ss.fit_transform(x[[i]])
    if over_sample==False:
        from sklearn.model_selection import train_test_split
        xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
        
            
    else:
        from imblearn.over_sampling import RandomOverSampler
        oversample = RandomOverSampler(sampling_strategy=0.5)
        from sklearn.model_selection import train_test_split
        xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
        x_res, y_res = oversample.fit_resample(xtrain, ytrain)
        xtrain=x_res
        ytrain=y_res
    if classification==True:

        error=[]
        for j in range(1,35):
            knn=KNeighborsClassifier(n_neighbors=j)
            knn.fit(xtrain,ytrain)
            pred_i=knn.predict(xtest)
            error.append(np.mean(pred_i!=ytest))
        plt.figure(figsize=(12,6))
        plt.plot(range(1,35),error,color='red',marker='o',markersize=10)
        plt.title('Error rate K value')
        plt.xlabel('K value')
        plt.ylabel('Mean Error')
    else:
        error=[]
        for j in range(1,35):
            knn=KNeighborsRegressor(n_neighbors=j)
            knn.fit(xtrain,ytrain)
            pred_i=knn.predict(xtest)
            error.append(np.mean(pred_i!=ytest))
        plt.figure(figsize=(12,6))
        plt.plot(range(1,35),error,color='red',marker='o',markersize=10)
        plt.title('Error rate K value')
        plt.xlabel('K value')
        plt.ylabel('Mean Error')


# # ML algorithms without FS,CV

# In[32]:


def fit_model(df,label,classification=True,min_max=False,svm_type = "Linear",kernel='poly'):
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(probability=True,kernel=kernel)
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
        AUC=metrics.auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                

                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    fpr , tpr, _ = roc_curve(ytest, y_pred)
                    print("AUC  (%): \n",auc(fpr, tpr)*100)
                    auc_score=auc(fpr, tpr)*100
                    df_results.loc[count,'AUC']= auc_score
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_without_CV_FS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(xtrain,ytrain)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(xtrain)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithm with FS

# In[33]:


def fit_model_RFS(df,label,threshold=5,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        print('false positive rate',fpr)
        print('true positive rate',tpr)
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    xtrain , xtest , ytrain, ytest =train_test_split(x2,y,test_size=0.20,random_state=42 )
    
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_FS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(xtrain,ytrain)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(xtrain)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithm with CV

# In[34]:


def fit_model_CV(df,label,split=10,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(kernel='poly')
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xr , ytrain, yr =train_test_split(x,y,test_size=0.50,random_state=42 )
    
    xcv , xtest , ycv, ytest =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(xtrain,ytrain)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(xtrain)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithm with CV and FS

# In[35]:


def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
def fit_model_CV_FS(df,label,split=10,classification=True,cross_valid_method=cross_valid_stratified_kf,threshold=5,min_max=False,svm_type='Linear',kernel='poly'):
    def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
       # print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
       # print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
       # print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
       # print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
       # print('The adjusted R Squared Error is',r2_adj)
        '''plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()'''
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        #print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        #print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        #print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        #print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        #print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        #print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        #print(algorithm + '\n'+classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        '''visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()'''
        return accuracy,AUC,precision,recall,f1,log,class_report,fpr,tpr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    xtrain , xr , ytrain, yr =train_test_split(x2,y,test_size=0.50,random_state=42 )
    
    xcv , xtest , ycv, ytest =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV_FS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain , ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                    s.append(mse)
                    a.append(mae)
                    rm.append(rmse)
                    r.append(r2)
                    ra.append(r2_adj)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                print('The adjusted R Squared Error is',np.mean(ra))
                
            else:
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain , ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    
                    regressor.fit(xtrain,ytrain)
                    y_pred=regressor.predict(xtest)
                    mse=metrics.mean_squared_error(ytest, y_pred)
                    s.append(mse)
                    mae=metrics.mean_absolute_error(ytest, y_pred)
                    a,append(mae)
                    rmse = math.sqrt(mse)
                    rm.append(rmse)
                    r2=metrics.r2_score(ytest,y_pred)
                    r.append(r2)
                    rmse = math.sqrt(mse)
                    
                    n=len(xtrain)
                    k=len(x.columns)
                    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                    ra.append(r2_adj)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                
                print('The adjusted R Squared Error is',np.mean(ra))
                
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
            count+=1
        return df_results


# # ML algorithms with oversampling without FS,CV

# In[36]:


def fit_model_os(df,label,ratio=0.5,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy=ratio)
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(x_res,y_res)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(x_res,y_res)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(x_res,y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_OS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(x_res,y_res)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(xtrain)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithms with undersampling without FS,CV

# In[37]:


def fit_model_us(df,label,ratio=0.5,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy=ratio)
    x_res, y_res = undersample.fit_resample(xtrain, ytrain)
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(x_res,y_res)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(x_res,y_res)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_US.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(x_res,y_res)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(xtrain)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithm with oversampling and FS

# In[38]:


def fit_model_RFS_os(df,label,ratio=0.5,threshold=5,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    xtrain , xtest , ytrain, ytest =train_test_split(x2,y,test_size=0.20,random_state=42 )
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy=ratio)
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(x_res,y_res)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(x_res,y_res)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_RFS_OS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(x_res,y_res)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(x_res)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # ML algorithm with undersampling and FS

# In[39]:


def fit_model_RFS_us(df,label,ratio=0.5,threshold=5,classification=True,min_max=False,svm_type='Linear',kernel='poly'):
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
        print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
        print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
        print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
        print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
        print('The adjusted R Squared Error is',r2_adj)
        plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        print(classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()
        return accuracy,AUC,precision,recall,f1,log,class_report
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    xtrain , xtest , ytrain, ytest =train_test_split(x2,y,test_size=0.20,random_state=42 )
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy=ratio)
    x_res, y_res = undersample.fit_resample(xtrain, ytrain)
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(x_res,y_res)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(x_res,y_res)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(x_res, y_res)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_RFS_US.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            else:
                
                regressor.fit(x_res,y_res)
                y_pred=regressor.predict(xtest)
                print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
                print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
                print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
                rmse = math.sqrt(mse)
                r2=metrics.r2_score(ytest,y_pred)
                rmse = math.sqrt(mse)
                mse=metrics.mean_squared_error(ytest, y_pred)
                mae=metrics.mean_absolute_error(ytest, y_pred)
                n=len(x_res)
                k=len(x.columns)
                r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                print('The adjusted R Squared Error is',r2_adj)
                plt.scatter(ytest, y_pred, c = 'green') 
                plt.xlabel("True Value") 
                plt.ylabel("Predicted value") 
                plt.title("True value vs predicted value") 
                plt.show()
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = mae
                df_results.loc[count,'Mean Squared Error'] = mse
                df_results.loc[count,'RMSE'] = rmse
                df_results.loc[count,'R squared Error'] = r2
                df_results.loc[count,'Adjusted R Squared Error'] = r2_adj
            count+=1
        return df_results


# # Ml algorithm with oversampling and CV

# In[40]:


def cross_valid_stratified_kf(x, y,ratio=0.5, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
def fit_model_CV_OS(df,label,ratio=0.5,split=10,classification=True,cross_valid_method=cross_valid_stratified_kf,min_max=False,svm_type='Linear',kernel='poly'):
    def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
       # print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
       # print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
       # print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
       # print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
       # print('The adjusted R Squared Error is',r2_adj)
        '''plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()'''
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        #print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        #print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        #print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        #print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        #print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        #print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        #print(algorithm + '\n'+classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        '''visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()'''
        return accuracy,AUC,precision,recall,f1,log,class_report,fpr,tpr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    #xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
   # for xtrain , xtest , ytrain, ytest in cross_valid_method(x,y,split=split):
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy=ratio)
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)    
    
    xtrain , xr , ytrain, yr =train_test_split(x_res,y_res,test_size=0.50,random_state=42 )
    
    xcv , x_test , ycv, y_test =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV_OS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain  , ytrain,xtest, ytest in cross_valid_method(x,y,split=split):
                    from imblearn.over_sampling import RandomOverSampler
                    oversample = RandomOverSampler(sampling_strategy=ratio)
                    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
                    mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                    s.append(mse)
                    a.append(mae)
                    rm.append(rmse)
                    r.append(r2)
                    ra.append(r2_adj)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                print('The adjusted R Squared Error is',np.mean(ra))
                
            else:
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain , ytrain,xtest, ytest in cross_valid_method(x,y,split=split):
                    from imblearn.over_sampling import RandomOverSampler
                    oversample = RandomOverSampler(sampling_strategy=ratio)
                    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
                    regressor.fit(x_res,y_res)
                    y_pred=regressor.predict(xtest)
                    mse=metrics.mean_squared_error(ytest, y_pred)
                    s.append(mse)
                    mae=metrics.mean_absolute_error(ytest, y_pred)
                    a,append(mae)
                    rmse = math.sqrt(mse)
                    rm.append(rmse)
                    r2=metrics.r2_score(ytest,y_pred)
                    r.append(r2)
                    rmse = math.sqrt(mse)
                    
                    n=len(xtrain)
                    k=len(x.columns)
                    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                    ra.append(r2_adj)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                
                print('The adjusted R Squared Error is',np.mean(ra))
                
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
            count+=1
        return df_results


# # ML algorithm with undersampling and CV

# In[41]:


def cross_valid_stratified_kf(x, y,ratio=0.5, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
def fit_model_CV_US(df,label,ratio=0.5,split=10,classification=True,cross_valid_method=cross_valid_stratified_kf,min_max=False,svm_type='Linear',kernel='poly'):
    def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
       # print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
       # print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
       # print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
       # print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
       # print('The adjusted R Squared Error is',r2_adj)
        '''plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()'''
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        #print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        #print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        #print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        #print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        #print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        #print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        #print(algorithm + '\n'+classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        '''visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()'''
        return accuracy,AUC,precision,recall,f1,log,class_report,fpr,tpr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    #from sklearn.model_selection import train_test_split
    #xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
   # for xtrain , xtest , ytrain, ytest in cross_valid_method(x,y,split=split):
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy=ratio)
    x_res, y_res = undersample.fit_resample(xtrain, ytrain)   
    
    xtrain , xr , ytrain, yr =train_test_split(x_res,y_res,test_size=0.50,random_state=42 )
    
    xcv , x_test , ycv, y_test =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV_US.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain  , ytrain,xtest, ytest in cross_valid_method(x,y,split=split):
                    from imblearn.over_sampling import RandomUnderSampler
                    oversample = RandomUnderSampler(sampling_strategy=ratio)
                    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
                    mse,mae,rmse,r2,r2_adj=VotingReg(x_res, xtest, y_res, ytest, verbose=True, clf=None)
                    s.append(mse)
                    a.append(mae)
                    rm.append(rmse)
                    r.append(r2)
                    ra.append(r2_adj)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                print('The adjusted R Squared Error is',np.mean(ra))
                
            else:
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain , ytrain,xtest, ytest in cross_valid_method(x,y,split=split):
                    from imblearn.over_sampling import RandomUnderSampler
                    oversample = RandomUnderSampler(sampling_strategy=ratio)
                    x_res, y_res = oversample.fit_resample(xtrain, ytrain)
                    regressor.fit(x_res,y_res)
                    y_pred=regressor.predict(xtest)
                    mse=metrics.mean_squared_error(ytest, y_pred)
                    s.append(mse)
                    mae=metrics.mean_absolute_error(ytest, y_pred)
                    a,append(mae)
                    rmse = math.sqrt(mse)
                    rm.append(rmse)
                    r2=metrics.r2_score(ytest,y_pred)
                    r.append(r2)
                    rmse = math.sqrt(mse)
                    
                    n=len(xtrain)
                    k=len(x.columns)
                    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                    ra.append(r2_adj)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                
                print('The adjusted R Squared Error is',np.mean(ra))
                
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
            count+=1
        return df_results


# # ML Algorithm with oversampling,FS and CV

# In[42]:


def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
def fit_model_CV_FS_OS(df,label,ratio=0.5,split=10,classification=True,cross_valid_method=cross_valid_stratified_kf,threshold=5,min_max=False,svm_type='Linear',kernel='poly'):
    def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
       # print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
       # print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
       # print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
       # print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
       # print('The adjusted R Squared Error is',r2_adj)
        '''plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()'''
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        #print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        #print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        #print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        #print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        #print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        #print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        #print(algorithm + '\n'+classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        '''visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()'''
        return accuracy,AUC,precision,recall,f1,log,class_report,fpr,tpr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
   # for xtrain , xtest , ytrain, ytest in cross_valid_method(x,y,split=split):
        
    xtrain , xtest , ytrain, ytest =train_test_split(x2,y,test_size=0.20,random_state=42 )
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(sampling_strategy=ratio)
    x_res, y_res = oversample.fit_resample(xtrain, ytrain)    
    
    xtrain , xr , ytrain, yr =train_test_split(x_res,y_res,test_size=0.50,random_state=42 )
    
    xcv , x_test , ycv, y_test =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV_OS_FS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.item():
            if key=='Voting Regression':
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain  , ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                    s.append(mse)
                    a.append(mae)
                    rm.append(rmse)
                    r.append(r2)
                    ra.append(r2_adj)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                print('The adjusted R Squared Error is',np.mean(ra))
                
            else:
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain, ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    
                    regressor.fit(xtrain,ytrain)
                    y_pred=regressor.predict(xtest)
                    mse=metrics.mean_squared_error(ytest, y_pred)
                    s.append(mse)
                    mae=metrics.mean_absolute_error(ytest, y_pred)
                    a,append(mae)
                    rmse = math.sqrt(mse)
                    rm.append(rmse)
                    r2=metrics.r2_score(ytest,y_pred)
                    r.append(r2)
                    rmse = math.sqrt(mse)
                    
                    n=len(xtrain)
                    k=len(x.columns)
                    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                    ra.append(r2_adj)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                
                print('The adjusted R Squared Error is',np.mean(ra))
                
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
            count+=1
        return df_results


# # ML Algorithm with undersampling,FS and CV
# 

# In[43]:


def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
def fit_model_CV_FS_US(df,label,ratio=0.5,split=10,classification=True,cross_valid_method=cross_valid_stratified_kf,threshold=5,min_max=False,svm_type='Linear',kernel='poly'):
    def cross_valid_stratified_kf(x, y, split=10, random=None, shuffle=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
        for train_index, test_index in kf.split(x, y):
            xtrain, xtest = x.iloc[train_index], x.iloc[test_index] 
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            yield xtrain,ytrain,xtest,ytest
    def RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        if regression:
            clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
        else:
            clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
        clf.fit(xtrain,ytrain)
        #validationmetrics(clf,testX,testY)
        res = pd.Series(clf.feature_importances_, index=xtrain.columns.values).sort_values(ascending=False)*100
        print(res)
        return res
    def VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import VotingRegressor
        from sklearn import metrics
        import math
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        sv = SVR(kernel="rbf")
        dt = DecisionTreeRegressor()
        gb = GradientBoostingRegressor()
        ab = AdaBoostRegressor(random_state=0, n_estimators=100)
        if not clf:
            clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
        clf.fit(xtrain , ytrain)
        y_pred=clf.predict(xtest)
        mse=metrics.mean_squared_error(ytest, y_pred)
       # print('Mean Absolute Error is ',metrics.mean_absolute_error(ytest, y_pred))
       # print('Mean Squared Error is ',metrics.mean_squared_error(ytest, y_pred))
       # print('The R squared Error is ',metrics.r2_score(ytest,y_pred))
        rmse = math.sqrt(mse)
       # print('The RMSE is ',rmse)
        mse=metrics.mean_squared_error(ytest, y_pred)
        mae=metrics.mean_absolute_error(ytest, y_pred)
        r2=metrics.r2_score(ytest,y_pred)
        n=len(xtrain)
        k=len(x.columns)
        r2_adj=1-((1-r2)*(n-1)/(n-k-1))
       # print('The adjusted R Squared Error is',r2_adj)
        '''plt.scatter(ytest, y_pred, c = 'green') 
        plt.xlabel("True Value") 
        plt.ylabel("Predicted value") 
        plt.title("True value vs predicted value") 
        plt.show()'''
        return mse,mae,rmse,r2,r2_adj
    def SVM(xtrain, xtest, ytrain, ytest, svmtype="Linear", verbose=True, clf=None):
        import pandas as pd
        import numpy as np
        from sklearn import metrics
        from sklearn.metrics import classification_report
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn import svm
        from sklearn.metrics import auc 
        from sklearn.metrics import log_loss
        from matplotlib import pyplot
        from yellowbrick.classifier import ConfusionMatrix
        if not clf:
            if svm_type == "Linear":
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC()
        clf.fit(xtrain,ytrain)
        y_pred=clf.predict(xtest)
        #print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
        accuracy=accuracy_score(ytest, y_pred)*100
        fpr , tpr, _ = roc_curve(ytest, y_pred)
        #print("AUC  (%): \n",auc(fpr, tpr)*100)
        AUC=auc(fpr, tpr)*100
        #print("Precision: \n",precision_score(ytest, y_pred)*100)
        precision=precision_score(ytest, y_pred)*100
        #print("Recall (%): \n",recall_score(ytest, y_pred)*100)
        recall=recall_score(ytest,y_pred)*100
        #print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
        f1=f1_score(ytest,y_pred)*100
        #print('logistic loss :\n',log_loss(ytest,y_pred))
        log=log_loss(ytest,y_pred)
        #print(algorithm + '\n'+classification_report(ytest,y_pred))
        class_report=classification_report(ytest,y_pred)
        #print(confusion_matrix(ytest, y_pred))
        '''visualizer = ConfusionMatrix(clf)
        visualizer.fit(xtrain, ytrain)  
        visualizer.score(xtest,ytest)
        g = visualizer.poof()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.show()'''
        return accuracy,AUC,precision,recall,f1,log,class_report,fpr,tpr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
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
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    # import regression modules
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import VotingRegressor
    from sklearn.model_selection import cross_val_score

    from yellowbrick.classifier import ConfusionMatrix
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    import math
    import warnings
    warnings.filterwarnings('ignore')
    import math
    dict_classifiers = {
    
    "Logistic Regression":LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestClassifier(),
    "xgboost":XGBClassifier(),
    #"Ada Boost":AdaBoostClassifier(),
    "SVM": svm,
    #'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    #'light Gradient Boosting':LGBMClassifier(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier()
    }
    dict_regression={
    
    "Lnear Regression":LinearRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeRegressor(),
   # "Naive Bayes": GaussianNB(), 
    "Random Forest": RandomForestRegressor(n_estimators=100),
    #"xgboost":XGBClassifier(),
    "Ada Boost":AdaBoostRegressor(random_state=0,n_estimators=1000),
    "SVR": SVR(kernel='rbf'),
   # 'MLP Classifier':MLPClassifier(hidden_layer_sizes=5),
    'Voting Regression':VotingRegressor,
    "Gradient Boosting": GradientBoostingRegressor()}
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.20,random_state=42 )
    if classification==True:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=False)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
    else:
        res=RFfeatureimportance(df, xtrain, xtest, ytrain, ytest, trees=35, random=None, regression=True)
        impftrs = list(res[res > threshold].keys())
        print ("Selected Features =" + str(impftrs))
        x2=x[impftrs]
   # for xtrain , xtest , ytrain, ytest in cross_valid_method(x,y,split=split):
        
    
    xtrain , xtest , ytrain, ytest =train_test_split(x2,y,test_size=0.20,random_state=42 )
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy=ratio)
    x_res, y_res = undersample.fit_resample(xtrain, ytrain)   
    
    xtrain , xr , ytrain, yr =train_test_split(x_res,y_res,test_size=0.50,random_state=42 )
    
    xcv , x_test , ycv, y_test =train_test_split(xr,yr,test_size=0.40,random_state=42 )
    print('_____Cross Validation_____')
    if classification==True:
        #df_results = pd.DataFrame(data=np.zeros(shape=(11,6)), columns = ['classifier','Accuracy','Precision','Recall','f1score','log_loss'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                accuracy=cross_val_score(clf, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(clf, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(clf, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(clf, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(clf, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
            else:
                accuracy=cross_val_score(classifier, xcv, ycv,scoring='accuracy', cv=split)
                #y_pred=classifier.predict(xtest)
                print("Mean accuracy  (%): \n", (accuracy.mean())*100)
                auc=cross_val_score(classifier, xcv, ycv,scoring='roc_auc', cv=split)
                print("Mean AUC  (%): \n",(auc.mean())*100)
                precision=cross_val_score(classifier, xcv, ycv,scoring='precision', cv=split)
                print("Mean Precision: \n",(precision.mean())*100)
                recall=cross_val_score(classifier, xcv, ycv,scoring='recall', cv=split)
                print("Mean Recall (%): \n",(recall.mean())*100)
                f1score=cross_val_score(classifier, xcv, ycv,scoring='f1', cv=split)
                print("Mean f1 score (%): \n",(f1score.mean())*100)
                #print('logistic loss :\n',log_loss(ytest,y_pred))
                #print(classification_report(ytest,y_pred))
                #print(confusion_matrix(ytest, y_pred))
                
            count+=1
        #return df_results
    print('______Test_______')
    if classification==True:
        df_results = pd.DataFrame(data=np.zeros(shape=(11,5)), columns = ['classifier','Precision','Recall','AUC','Accuracy'])
        count = 0
        for key,classifier in dict_classifiers.items():
            print('---------'+key+'------------')
            if key=='SVM':
                if svm_type == "Linear":
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(probability=True,kernel=kernel)
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
               # fpr , tpr, _ = roc_curve(ytest, y_pred)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
                
                df_results.loc[count,'Accuracy'] = accuracy
                if svm_type == "Linear":
                    print('AUC score cannot be calculated for linear SVM')
                else:
                    
                    ypred_prob=clf.predict_proba(xtest)[:,1]
                    fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                    random_probs = [0 for i in range(len(ytest))]
                    p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                    auc_score = roc_auc_score(ytest, ypred_prob)
                    df_results.loc[count,'AUC']= auc_score
                    print("AUC  (%): \n",auc_score*100)
                    gmeans = np.sqrt(tpr * (1-fpr))
                    ix = np.argmax(gmeans)
                    print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(class_report)
                visualizer = ConfusionMatrix(clf)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                if svm_type == "Linear":
                    pass
                else:
                    
                    plt.plot(fpr, tpr,label=key)

                    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                    # title
                    plt.title('ROC curve')
                    # x label
                    plt.xlabel('False Positive Rate')
                    # y label
                    plt.ylabel('True Positive rate')

                    plt.legend(loc='best')

                    plt.show()
            else:
                classifier.fit(xtrain,ytrain)
                y_pred=classifier.predict(xtest)
                print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
                #fpr , tpr, _ = roc_curve(ytest, y_pred)
                #print("AUC  (%): \n",metrics.auc(fpr, tpr)*100)
                print("Precision: \n",precision_score(ytest, y_pred)*100)
                print("Recall (%): \n",recall_score(ytest, y_pred)*100)
                print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
                print('logistic loss :\n',log_loss(ytest,y_pred))
                print(classification_report(ytest,y_pred))
                accuracy=accuracy_score(ytest, y_pred)*100
                #AUC=metrics.auc(fpr, tpr)*100
                precision=precision_score(ytest, y_pred)*100
                recall=recall_score(ytest,y_pred)*100
                f1=f1_score(ytest,y_pred)*100
                log=log_loss(ytest,y_pred)
                df_results.loc[count,'classifier'] = key       
                df_results.loc[count,'Precision'] = precision
               # df_results.loc[count,'Precision'] = precision
                df_results.loc[count,'Recall'] = recall
               # df_results.loc[count,'AUC']= AUC
                df_results.loc[count,'Accuracy'] = accuracy
                ypred_prob=classifier.predict_proba(xtest)[:,1]
                fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
                random_probs = [0 for i in range(len(ytest))]
                p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
                auc_score = roc_auc_score(ytest, ypred_prob)
                df_results.loc[count,'AUC']= auc_score
                print("AUC  (%): \n",auc_score*100)
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
                print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

                #df_results.loc[count,'f1score'] = f1
                #df_results.loc[count,'log_loss'] = log
                #print(confusion_matrix(ytest, y_pred))
                visualizer = ConfusionMatrix(classifier)
                visualizer.fit(xtrain, ytrain)  
                visualizer.score(xtest,ytest)
                g = visualizer.poof()
                plt.plot(fpr, tpr,label=key)

                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')

                plt.legend(loc='best')

                plt.show()
            count+=1
        display(df_results)
        #df_results.to_csv('E:/Drive/ML1/Untitled Folder/Algorithm_results/ML_algorithm_with_CV_US_FS.csv',index=False)
        return df_results
    else:
        df_results = pd.DataFrame(data=np.zeros(shape=(7,6)), columns = ['Regressor','Mean Absolute Error','Mean Squared Error','RMSE','R squared Error','Adjusted R Squared Error'])
        count = 0
        for key,regressor in dict_regressors.items():
            if key=='Voting Regression':
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain  , ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    mse,mae,rmse,r2,r2_adj=VotingReg(xtrain, xtest, ytrain, ytest, verbose=True, clf=None)
                    s.append(mse)
                    a.append(mae)
                    rm.append(rmse)
                    r.append(r2)
                    ra.append(r2_adj)
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                print('The adjusted R Squared Error is',np.mean(ra))
                
            else:
                s=[]
                a=[]
                rm=[]
                r=[]
                ra=[]
                for xtrain, ytrain,xtest, ytest in cross_valid_method(x2,y,split=split):
                    
                    regressor.fit(xtrain,ytrain)
                    y_pred=regressor.predict(xtest)
                    mse=metrics.mean_squared_error(ytest, y_pred)
                    s.append(mse)
                    mae=metrics.mean_absolute_error(ytest, y_pred)
                    a,append(mae)
                    rmse = math.sqrt(mse)
                    rm.append(rmse)
                    r2=metrics.r2_score(ytest,y_pred)
                    r.append(r2)
                    rmse = math.sqrt(mse)
                    
                    n=len(xtrain)
                    k=len(x.columns)
                    r2_adj=1-((1-r2)*(n-1)/(n-k-1))
                    ra.append(r2_adj)
                print('Mean Absolute Error is ',np.mean(a))
                print('Mean Squared Error is ',np.mean(s))
                print('The R squared Error is ',np.mean(r))
                
                print('The adjusted R Squared Error is',np.mean(ra))
                
                df_results.loc[count,'Regressor'] = key       
                df_results.loc[count,'Mean Absolute Error'] = np.mean(a)
                df_results.loc[count,'Mean Squared Error'] = np.mean(s)
                df_results.loc[count,'RMSE'] = np.mean(rm)
                df_results.loc[count,'R squared Error'] = np.mean(r)
                df_results.loc[count,'Adjusted R Squared Error'] = np.mean(ra)
            count+=1
        return df_results


# In[44]:


def SVM_classifier(df,label,kernel='poly' ,svmtype="SVC",over_sample=False,under_sample=False,name=None,clf=None,min_max=False):
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn import svm
    from sklearn.metrics import auc 
    from sklearn.metrics import log_loss
    from matplotlib import pyplot
    from yellowbrick.classifier import ConfusionMatrix
    if not clf:
        if svmtype == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC(probability=True,kernel=kernel)
    from sklearn.preprocessing import StandardScaler
    x=df.drop([label], axis=1)
    y=df[label]
    if min_max==True:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        for i in x.columns:
            x[i] = mm.fit_transform(x[[i]])
    else:
        from sklearn.preprocessing import StandardScaler
        ss=StandardScaler()
        for i in x.columns:
            x[i] = ss.fit_transform(x[[i]])
    if over_sample==True:
        from imblearn.over_sampling import RandomOverSampler
        oversample = RandomOverSampler(sampling_strategy=0.5)
        from sklearn.model_selection import train_test_split
        xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.50,random_state=42 )
        x_res, y_res = oversample.fit_resample(xtrain, ytrain)
        x_train=x_res
        y_train=y_res
    elif under_sample==True:
        from imblearn.under_sampling import RandomUnderSampler
        undersample = RandomUnderSampler(sampling_strategy=0.5)
        from sklearn.model_selection import train_test_split
        xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.50,random_state=42 )
        x_res, y_res = undersample.fit_resample(xtrain, ytrain)
        x_train=x_res
        y_train=y_res
    else:
        from sklearn.model_selection import train_test_split
        xtrain , xtest , ytrain, ytest =train_test_split(x,y,test_size=0.50,random_state=42 )

    clf.fit(xtrain,ytrain)
    y_pred=clf.predict(xtest)
    print("Accuracy  (%): \n", accuracy_score(ytest, y_pred)*100)
    accuracy=accuracy_score(ytest, y_pred)*100
    #fpr , tpr, _ = roc_curve(ytest, y_pred)
    #print("AUC  (%): \n",auc(fpr, tpr)*100)
    #AUC=auc(fpr, tpr)*100
    print("Precision: \n",precision_score(ytest, y_pred)*100)
    precision=precision_score(ytest, y_pred)*100
    print("Recall (%): \n",recall_score(ytest, y_pred)*100)
    recall=recall_score(ytest,y_pred)*100
    print("f1 score (%): \n",f1_score(ytest, y_pred)*100)
    f1=f1_score(ytest,y_pred)*100
    print('logistic loss :\n',log_loss(ytest,y_pred))
    log=log_loss(ytest,y_pred)
    print(classification_report(ytest,y_pred))
    class_report=classification_report(ytest,y_pred)
    #print(confusion_matrix(ytest, y_pred))
    '''visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()'''
    import pickle
    filename = name
    pickle.dump(model, open(filename, 'wb'))
    if svm_type == "Linear":
        print('AUC score cannot be calculated for linear SVM')
    else:

        ypred_prob=clf.predict_proba(xtest)[:,1]
        fpr, tpr, thresh = roc_curve(ytest, ypred_prob, pos_label=1)
        random_probs = [0 for i in range(len(ytest))]
        p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)
        auc_score = roc_auc_score(ytest, ypred_prob)
       
        print("AUC  (%): \n",auc_score*100)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresh[ix], gmeans[ix]))

    #df_results.loc[count,'f1score'] = f1
    #df_results.loc[count,'log_loss'] = log
    #print(class_report)
    visualizer = ConfusionMatrix(clf)
    visualizer.fit(xtrain, ytrain)  
    visualizer.score(xtest,ytest)
    g = visualizer.poof()
    if svm_type == "Linear":
        pass
    else:

        plt.plot(fpr, tpr,label=key)

        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')

        plt.show()
    return accuracy,auc_score,precision,recall,f1,log,class_report


# In[45]:


def scatter(df,column1,column2):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(12,12))
    sns.scatterplot(df[column1],df[column2])


# # Clustering

# In[46]:


def elbow_plot_kmeans(df3,yellow=True):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from sklearn.cluster import KMeans


    from yellowbrick.cluster import KElbowVisualizer

    if yellow==True:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(4,12))

        visualizer.fit(df3)       
        visualizer.show()    
    else:
        Sum_of_squared_distances = []
        K = range(1,21)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df3)
            Sum_of_squared_distances.append(km.inertia_)
        plt.figure(figsize=(20,5))
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()


# In[47]:


def kmeans(n,data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n)
    model.fit(data)
    kmeans_labels = model.labels_
    return model,kmeans_labels


# In[48]:


def SI_score(data,labels):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import silhouette_score
    print('Silhouette score: {}'.format(silhouette_score(data, labels, 
                                           metric='euclidean')))


# In[49]:


def cluster_viz(df,label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    t=TSNE(n_components=2)
    X_2d = t.fit_transform(df)
    sns.scatterplot(X_2d[:,0], X_2d[:,1],hue=label)
    plt.show()
    t2=TSNE(n_components=3)
    


# In[50]:


def cluster_viz3d(df,label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    t2=TSNE(n_components=3)
    X_3d = t2.fit_transform(df)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(projection="3d")

    ax.scatter3D(X_3d[:,0], X_3d[:,1], X_3d[:,2],c=label, cmap='rainbow')


# In[51]:


def dendrogram(df2):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.cluster.hierarchy as sch

    plt.figure(figsize=(20,10))
    dendrogram = sch.dendrogram(sch.linkage(df2, method='complete'))


# In[52]:


def hierarchical_clustering(n):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n) 
    model.fit(df2)
    hac_labels = model.labels_
    return model,hac_labels


# In[53]:


def dbscan(df2,eps=0.5,min_samples=5):
    from sklearn.cluster import DBSCAN
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(df2)
    dblabels = db.labels_
    return db,dblabels


# In[54]:


def meanshift(df2,quantile=0.3):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn import cluster

    bandwidth=estimate_bandwidth(df2, quantile=quantile)

    ms = MeanShift(bandwidth = bandwidth) 
    ms.fit(df2)
    mslabels = ms.labels_
    return ms,mslabels


# In[55]:


def optics(df2):
    from sklearn.cluster import OPTICS
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    op = OPTICS()
    op.fit(df2)
    oplabels= op.labels_
    return op,oplabels


# In[56]:


def birch(df2):
    from sklearn.cluster import Birch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    bi = Birch()
    bi.fit(df2)
    bilabels= bi.labels_
    return bi,bilabels


# In[57]:


def ap(df2):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import AffinityPropagation
    from sklearn.metrics.pairwise import euclidean_distances
    preference=euclidean_distances(df2, df2).max()
    af = AffinityPropagation()
    clustering = af.fit(df2)
    af.get_params()
    aflabels = af.predict(df2)
    return af,aflabels


# In[58]:


def gmm(df2,n):
    from sklearn.mixture import GaussianMixture
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42 )
    gmm.fit(df2)
    gmmlabels = gmm.predict(df2)
    return gmm,gmmlabels


# # MCMC

# In[1]:


def plot_ppc(model,trace):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pymc3 as pm
    import arviz as az
    prior_predictive = pm.sample_prior_predictive(model=model)

    posterior_predictive = pm.sample_posterior_predictive(model=model,trace=trace)
    dataset = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive,prior=prior_predictive)
    az.plot_ppc(dataset)


# In[3]:


def pred(model,trace,xtest,ytest):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pymc3 as pm
    import arviz as az
    with model:
        pm.set_data({'x': xtest})
        y_pred = pm.sample_posterior_predictive(trace)
    predictions=np.mean(y_pred['yl'],axis=0)
    predictions2=[]
    for i in predictions:
        if i>0.5:
            predictions2.append(1)
        else:
            predictions2.append(0)
    from sklearn.metrics import classification_report
    print(classification_report(ytest, predictions2))


# In[ ]:




