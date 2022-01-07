from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chisquare
from sklearn import tree
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
import imblearn
import missingno as mano
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import functools
import operator

#F1: loading data in a dataframe (either CSV or Excel - can be generalized for databases)
def import_data(path_to_file):
    file_ext = path_to_file.split(".")[-1].lower()
    if file_ext == "csv":
        return pd.read_csv(path_to_file)
    elif file_ext == "xlsx" or file_ext=="xls":
        return pd.read_excel(path_to_file)
    elif file_ext == "json":
        return pd.readjson(path_to_file)

#F2: checking shape, column types, and see the first/last 'n' rows using head/tail (where n is one of the arguments of F2)
# df =import_data("E:\Documents\Education\IBA\Courses\Spring 2021\ML 1\Quiz1\House.Price.csv")

def data_summary(df,n):
    print("###### Shape ######")
    print(df.shape)
    print("###### Dimensions ######")
    print(df.ndim)
    print("###### Dtypes ######")
    print(df.dtypes)
    print("###### head ######")
    print(df.head(n))
    print("###### tail ######")
    print(df.tail(n))

# data_summary(df,5)

#F3: remove unnecessary/useless columns (based on results of F2 and your background knowledge and the problem to be solved), e.g., identifiers, multiple primary keys, extra KPI like GMROI in sales which is the same for the whole year etc.

def drop_columns(df, list_of_col):
    return df.drop(list_of_col,axis=1)

#F4: remove rows containing a particular value of a given column, e.g., in smoking_status column, I don't want to consider non-smokers in my ML problem so I remove all these rows.

def drop_rows(df, colname, row_value):
    return df[~df[colname].isin([row_value])]

#F5: determine the missing values in the whole dataset

def missing_values(df):
    print(df.isnull().sum())


# F6: analyze missing values of one or more columns using mano module

def analyze_missing(df):
    mano.matrix(df)
    mano.bar(df)

def analyze_missing_select(df,chart="matrix"):
    if chart == "matrix":
        mano.matrix(df)
    else:
        mano.bar(df)

#F7: cater for missing values (input the column with missing value, and the method through which you want to cater for the missing values)
def median_impute(df, colname):
  median = df[colname].median()
  df[colname].fillna(median, inplace=True)

def show_histogram(df, list_of_num_cols, hist_bins):
    print("####### histogram ########")
    for i in list_of_num_cols:
        df[i].hist(grid=True, bins=hist_bins)
        plt.xlabel(i)
        plt.ylabel('Value')
        plt.title(i)
        plt.show()

def show_boxplot(df, list_of_num_cols ):
    print("####### boxplot ########")
    for i in list_of_num_cols:
        df.boxplot(column=i, sym='o', return_type='axes')
        plt.show()

def show_qqplot(df, list_of_num_cols):
    print("####### QQPLOT ########")
    for i in list_of_num_cols:
        sm.qqplot(df[i], line ='45')
        plt.title(i)
        plt.show()

def show_test_normality(df,list_of_num_cols):
    print("########Test for Normality#########")
    for i in list_of_num_cols:
        stat, p_value = chisquare(df[i])
        print( i + " --> p value = " + str(p_value))



def chi_squared_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = chi2_contingency(contingency)
    prob = 0.90
    # interpret p-value
    alpha = 1.0 - prob
    if p <= alpha:
	    print('Dependent (reject H0)')
    else:
	    print('Independent (fail to reject H0)')
    


#F8: Function for numerical data analysis - includes histogram, boxplot, qqplot, describe, and statistical tests for normality
def numerical_analysis(df, list_of_num_cols, hist_bins='auto', plots="all"):
    if plots =="all":
        print(df[list_of_num_cols].describe())
        show_histogram(df,list_of_num_cols,hist_bins)
        show_boxplot(df,list_of_num_cols)
        show_qqplot(df, list_of_num_cols)
        show_test_normality(df,list_of_num_cols)

    elif plots=="histogram":
        show_histogram(df,list_of_num_cols,hist_bins)
    elif plots=="boxplot":
        show_boxplot(df,list_of_num_cols)
    elif plots=="qqplot":
        show_qqplot(df, list_of_num_cols)
    else:
        show_test_normality(df,list_of_num_cols)


#F9: Function for categorical data analysis - includes value counts, and bar charts
def categ_analysis(df, list_of_columns):
    for i in list_of_columns:
        print(df[i].value_counts())
        # print(df[i].value_counts().plot.bar())
        print("\n")
        # sns.set(style="darkgrid")
        sns.barplot(df[i].value_counts().index, df[i].value_counts().values)
        plt.title(i)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel(i, fontsize=12)
        plt.show()
# categ_analysis(df, ["Street", "MSZoning"])

#F10: Function to change the type of any column (input col name and the type you want)
def change_type(df, column, type_col):
    return df[column].astype(type_col)


#F11: Function to change the discretizations of a particular catergorical column, e.g., rename the values, remove space between value names etc.
def change_data(x):
    y = x.strip().split(" ")
    z = "".join(y).tolower()

def change_discretize(df, column_name ):
    return df[column_name].apply(change_data)


#F12: Function for data analysis - extract year, month etc., subtract dates etc. (this function cannot be specified exactly so just add what you believe are the basic things
def basic_data(df, list_of_columns, new_column_name= "", method="add", values="num"):
    if method == "add":
        df[new_column_name] = df[list_of_columns].sum(axis=1)

# basic_data(df, ["MoSold", "SalePrice"] )
# df

#F13: function to make a deep copy of a dataframe
def deep_copy(df):
    return df.copy(deep=True)

#F14: function to encode categorical into numerical (label, ordinal, or onehot)
def encode_categ(df, column, type_enc):
    if type_enc == "label":
        df[column] = df[column].astype('category')
        return df[column].cat.codes
    elif type_enc == "onehot":
        return pd.get_dummies(df,columns=[column])


#F15: function to split dataframe into X (predictors) and y (label), apply standard scaling on X, apply the desired ML algorithm and output the results:
# Â - input dataframe
# Â - input the algo name (e.g., decisiontree)
# Â - input whether this is a classification task or a regression task (then you should select either decisiontreeclassifier or decisiontreeregressor within the function)
# Â - for classification, output confusion matrix, AUC, logloss and classification report
# Â - for regression, output MAE, MSE, R-squared and adjusted R-squaredÂ 
# Â - NB: you can add more metrics if available

def splitxy(df, label):
    x_predictor = df.drop(label,axis=1)
    y_label = df[label]
    return x_predictor, y_label

def standard_scale(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_test(df, predictors, labels, standard_scaling=False):
    #predictors, labels = splitxy(df, label)
    X_train, X_test, y_train, y_test = train_test_split(predictors, labels, test_size=0.20)
    if standard_scaling:
        X_train, X_test = standard_scale(X_train, X_test)
    return X_train, X_test, y_train, y_test


#feature importance RFFS
def RFfeatureimportance(df, x, y, trees=35, random=None, regression=False):
    if regression:
        clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
    else:
        clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(x,y)
    importance = clf.feature_importances_
    res = pd.Series(importance, index=x.columns.values).sort_values(ascending=False)
    return res



def cross_valid_kfold(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for KFold cross validation
    
    """
    kf = KFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY

def cross_valid_stratified_kf(X, y, split=5, random=None, shuffle=False):
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

def address_class_imbalance(predictors, labels, imbalance_method="RandomOversample"):
    if imbalance_method == "SMOTE":
        oversample = imblearn.over_sampling.SMOTE()
    else:
        oversample = imblearn.RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(predictors, labels)

    return X_over, y_over




def validatemetrics_class(y_pred, y_test, model_name):
    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    #Precision
    try:
        precision = precision_score(y_test, y_pred, pos_label=1, labels=[0,1])

        #Recall
        recall = recall_score(y_test, y_pred,pos_label=1,labels=[0,1])

        #get FPR (specificity) and TPR (sensitivity)
        fpr , tpr, _ = roc_curve(y_test, y_pred)

        #AUC
        auc_val = auc(fpr, tpr)

        #F-Score
        f_score = f1_score(y_test, y_pred)
        result = {
                "model_type": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc_val": auc_val,
                "f_score": f_score
                
                }
    except ValueError:
        # if len(e.args) > 0 and e.args[0] in "Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].":
        precision = precision_score(y_test, y_pred, pos_label=1, labels=[0,1], average="weighted")
        #Recall
        recall = recall_score(y_test, y_pred,pos_label=1,labels=[0,1], average="weighted")
        #get FPR (specificity) and TPR (sensitivity)
        # fpr , tpr, _ = roc_curve(y_test, y_pred)
        #AUC
        # auc_val = auc(fpr, tpr)
        #F-Score
        f_score = f1_score(y_test, y_pred, average="weighted")
        result = {
            "model_type": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            # "auc_val": auc_val,
            "f_score": f_score
            
            }
    return result

########## classification algos #####################
def DecisionTree_class(df,X_train, y_train, X_test, y_test,regulariztaion=True):
    model_name = "Decision Tree Classifier"
    print(model_name)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return validatemetrics_class(y_pred, y_test, model_name)

def KNN_class(df, X_train, y_train, X_test, y_test,regulariztaion=True):
    model_name = "KNN"
    print(model_name)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return validatemetrics_class(y_pred, y_test, model_name)


def NaiveBayes_class(df,X_train, y_train, X_test, y_test,regulariztaion=True):
    model_name = "Naive Bayes Classifier"
    print(model_name)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    return validatemetrics_class(y_pred, y_test, model_name)


def Log_reg(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "Logistic Regression"
    print(model_name)
    if regulariztaion:
        clf = LogisticRegression()
    else:
        clf = LogisticRegression(C=1e10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)

def MLP_class(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "Multi-Layer Perceptron"
    print(model_name)
    if regulariztaion:
        clf = MLPClassifier(hidden_layer_sizes=5)
    else:
        clf = MLPClassifier(hidden_layer_sizes=5, alpha=1e10)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)

def SVM(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "Support Vector Machine"
    print(model_name)
    if regulariztaion:
        clf = svm.SVC(C=0.1)
    else:
        clf = svm.SVC(C=1)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)

# Random forest classifier
def RandomForest_class(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "Random Forest Classifier"
    print(model_name)
    clf  = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)

def XGBoost(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "XGBoost"
    print(model_name)
    clf = XGBClassifier(random_state=1,learning_rate=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)

def AdaBoost(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "AdaBoost"
    print(model_name)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return validatemetrics_class(y_pred, y_test, model_name)


def LightGbm(df,X_train, y_train, X_test, y_test, regulariztaion=True):
    model_name = "LightGbm"
    print(model_name)
    d_train = lgb.Dataset(X_train, label=y_train)
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
    y_pred = clf.predict(X_test)
    for i in range(0, y_pred.shape[0]):
        y_pred[i]= 1 if y_pred[i] >= 0.5 else 0
    return validatemetrics_class(y_pred, y_test, model_name)



def run_all_class_algorithm(df, X_train, y_train, X_test, y_test ):
    pass
def run_all_class_algorithm_cv(df, X_train, y_train, X_test, y_test):
    pass


algo_list = [ 
                KNN_class,
                NaiveBayes_class,
                Log_reg  ,
                MLP_class,
                SVM,
                DecisionTree_class,
                RandomForest_class,
                XGBoost,
                AdaBoost,
                LightGbm

]

def MachineLearning_class_algo(df, predictors, labels, feature_selection=False, feature_threshold=100, cross_val=False, regularization=False):
    filtered_col = predictors.copy(deep=True)
    if feature_selection:
        res = RFfeatureimportance(df, predictors, labels, trees=35, random=None, regression=False)
        less_imp_features = list(res[res <= feature_threshold].index)
        # filtered_col = predictors.copy(deep=True)
        print("##############  total columns : " + str(len(filtered_col.columns)))
        filtered_col = drop_columns(filtered_col, less_imp_features)
        # X_train, X_test, y_train, y_test = train_test(df, filtered_col, labels, standard_scaling=True)
    
    if cross_val:
        results = []
        multi_result = []
        for algo in algo_list:
            print("######## Running Model")
            print("##############  total columns : " + str(len(filtered_col.columns)))
            for  X_train, y_train, X_test, y_test in cross_valid_stratified_kf(filtered_col, labels):
                # print(y_train)
                sub_model_result = algo(df,X_train, y_train, X_test, y_test)
                multi_result.append(sub_model_result)
            # print(multi_result)
            results.append( {
                "model_type": multi_result[0]["model_type"],
                "accuracy" : functools.reduce(lambda metric1,metric2 : metric1 + metric2, [i["accuracy"] for i in multi_result])/len(multi_result),
                "precision": functools.reduce(lambda metric1,metric2 : metric1 + metric2, [i["precision"] for i in multi_result])/len(multi_result),
                "recall": functools.reduce(lambda metric1,metric2 : metric1+ metric2, [i["recall"] for i in multi_result])/len(multi_result),
                "auc_val": functools.reduce(lambda metric1,metric2 : metric1 + metric2, [i["auc_val"] for i in multi_result])/len(multi_result),
                "f_score": functools.reduce(lambda metric1,metric2 : metric1 + metric2, [i["f_score"] for i in multi_result])/len(multi_result),
            })
            # print(results)
            multi_result = []
        
    # print(len(filtered_col.columns))
    else: 
        print("##############  total columns : " + str(len(filtered_col.columns)))
        results = []
        X_train, X_test, y_train, y_test = train_test(df, filtered_col, labels, standard_scaling=True)
        for algo in algo_list:
            one_model_result = algo(df,X_train, y_train, X_test, y_test, regularization)
            results.append(one_model_result)
    return results




################## old function (to be duscarded) ##########################
def predict_model(df, algo_name, class_or_reg, predictors, labels, n=0, get_score=0):
    X_train, X_test, y_train, y_test = train_test(df, predictors, labels, standard_scaling=True)
    if algo_name == "decisiontree":
        if class_or_reg == "classification":
            
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("####### confusion matrix ########")
            print(metrics.confusion_matrix(y_test, y_pred))
            print("######## confusion report #########")
            print(metrics.classification_report(y_test, y_pred))
            print( "###### log loss ###########")
            print('log loss : ', metrics.log_loss(y_test, y_pred))
            print("######### AUC ########")
            print(roc_auc_score(y_test, y_pred))
        else:
            regressor = tree.DecisionTreeRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            

    

    elif algo_name == "LinearRegression":
        regr = LinearRegression()
        y_pred = regr.fit(X_train, y_train)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    else:
        models = "KNN"
        classifier = KNeighborsClassifier(n_neighbors=n)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print("####### confusion matrix ########")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("######## confusion report #########")
        print(metrics.classification_report(y_test, y_pred))
        print( "###### log loss ###########")
        print('log loss : ', metrics.log_loss(y_test, y_pred))
        print("######### AUC ########")
        print(metrics.roc_auc_score(y_test, y_pred))
        if get_score == 1:
            return metrics.accuracy_score(y_test, y_pred)




#F16: Function to apply ANOVA and output results
def apply_anova(df, list_of_columns):
    models = ols(list_of_columns[0] + "~+C(" + list_of_columns[1] + ")", data=df).fit()
    tables=anova_lm(models, typ=2)
    return tables

#F17: Function to generate correlation heatmaps
def corr_heatmap(df):
    corrmatrix = df.corr()
    f, axis = plt.subplots(figsize =(15, 10)) 
    sns.heatmap(corrmatrix, ax = axis, linewidths = 0.05)

#F18: Function to generate scatter plot

def gen_scatterploy(df, list_of_columns):
    plt.scatter(df[list_of_columns[0]], df[list_of_columns[1]], c='green')
    plt.show()


def write_to_file(results, file_name="results", sheet_name='Sheet1'):
    df = pd.DataFrame(results)
    df = df[["model_type","precision", "recall", "auc_val", "accuracy", "f_score"]]
    df.to_excel(file_name,sheet_name)
