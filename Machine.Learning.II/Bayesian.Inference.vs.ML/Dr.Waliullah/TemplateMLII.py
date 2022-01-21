import numpy as np
import pandas as pd
pd.set_option('max_columns', 105)
#import matplotlib.pyplot as plt


from scipy import stats
from scipy.stats import skew
from math import sqrt
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
# plotly
#import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingRegressor
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import chart_studio.plotly
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style(
    style="darkgrid", 
    rc={"axes.facecolor": ".9", "grid.color": ".8"}
)
sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")

import arviz as az
import patsy
import pymc3 as pm
from pymc3 import glm
from sklearn.model_selection import train_test_split

plt.rcParams["figure.figsize"] = [7, 6]
plt.rcParams["figure.dpi"] = 100

from sklearn.linear_model import LinearRegression
#import performance scores
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import seaborn as sns
import math
from math import *
#Stats Libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
import theano

# import scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()

SEED=42

################################################################################################################
    
# Load Data
def load_data(file_name):
    def readcsv(file_name):
        return pd.read_csv(file_name)
    def readexcel(file_name):
        return pd.read_excel(file_name)
    def readjson(file_name):
        return pd.read_json(file_name)
    func_map = {
        "csv": readcsv,
        "xls": readexcel,
        "xlsx": readexcel,
        "txt": readcsv,
        "json": readjson
    }
    
    # default reader = readcsv
    reader = func_map.get("csv")
    
    for k,v in func_map.items():
        if file_name.endswith(k):
            reader = v
            break
    return reader(file_name)

###############################################################################################################################################
###################################################################################################
#Missing Values Analysis and Plot Function
def missing_values_display(df):
    df_null = pd.DataFrame()
    df_null['missing'] = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
    trace = go.Bar(x = df_null.index, 
                    y = df_null['missing'],
                    name="df", 
                    text = df_null.index)



    data = [trace]

    layout = dict(title = "NaN in Dataset", 
                  xaxis=dict(ticklen=10, zeroline= False),
                  yaxis=dict(title = "number of rows", side='left', ticklen=10,),                                  
                  legend=dict(orientation="v", x=1.05, y=1.0),
                  autosize=False, width=750, height=500,
                  barmode='stack'
                  )

    fig = dict(data = data, layout = layout)
    iplot(fig)
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    #display(missing_data.head(20))
    display(missing_data)
###################################################################################################
### Numerical and Categorical features
def numirical_catagorical_calumns(df):
    numerical_feats = df.dtypes[df.dtypes != "object"].index
    print("Number of Numerical features: ", len(numerical_feats))

    categorical_feats = df.dtypes[df.dtypes == "object"].index
    print("Number of Categorical features: ", len(categorical_feats))
    print("*"*100)
    print('Numrical Columns')
    print(df[numerical_feats].columns)
    print("*"*100)
    print('Catagorical Columns')
    print(df[categorical_feats].columns)
    return numerical_feats, categorical_feats
#############################################################################################
#data cleaning function
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
   

    #delete column with are highly correlated
    def drop_highcorrelated_colums(df, corr_threshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in df.columns:
                        del df[colname] # deleting the column from the dataset
        return df

    #drop all rows which contain more than 40% missing values
    def drop_useless_rows(df):
        min_threshold = math.ceil(len(df.columns)*0.4)
        df = df.dropna(thresh=min_threshold)
        return df
    #######################################
    
    #########################################
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
    def fill_na_mean(df):
        df.fillna(df.mean(), inplace=True)
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
##################################################################################################
#######################################################################################################
# Get Best Parameters and Score in Grid Search
def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score

###################################################################################################################################################
###################################################################################################################################################
def Num_Cat_Col(df,label_col):
    data1=df.drop(label_col,axis=1)
    numerical_features   =data1.select_dtypes(exclude=['object']).columns.tolist()
    print('numerical_features')
    print(numerical_features)
    categorical_features = data1.select_dtypes(include=['object']).columns.tolist()
    print('categorical_features')
    print(categorical_features)
    return numerical_features,categorical_features

###################################################################################################################
# Function for computing correlation of Features with Label
#correlation with labl

def corr_x_y(df,LABEL_COL):
    df_corr = df.corrwith(df[LABEL_COL]).abs().sort_values(ascending=False)[2:]

    data = go.Bar(x=df_corr.index, 
                  y=df_corr.values )
      
    layout = go.Layout(title = "Correlation of Features to Label",  
                       xaxis = dict(title = ''), 
                       yaxis = dict(title = 'correlation'),
                       autosize=False, width=750, height=500,)

    fig = dict(data = [data], layout = layout)
    iplot(fig)

###################################################################################################################
# Distribution of the target variable

from scipy.stats import skew, kurtosis
import statistics
def dist_label(df,LABEL_COL):
    mean=df[LABEL_COL].mean()
    std=statistics.stdev(df[LABEL_COL])
    sk=df[LABEL_COL].skew()
    kr=df[LABEL_COL].kurtosis()
    print('mean:',mean)
    print('standard deviation:',std)
    print('skewness:',sk)
    print('kurtosis:',kr)
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, 
                          subplot_titles=[LABEL_COL])
    trace_1 = go.Histogram(x=df[LABEL_COL], name=LABEL_COL)

    fig.append_trace(trace_1, 1, 1)

    iplot(fig)

###################################################################
# Functions for Scatter plots
#plots Display Functions

def plotly_scatter_x_y(df_plot, val_x, val_y):
    
    value_x = df_plot[val_x] 
    value_y = df_plot[val_y]
    
    trace_1 = go.Scatter( x = value_x, y = value_y, name = val_x, 
                         mode="markers", opacity=0.8 )

    data = [trace_1]
    
    plot_title = val_y + " vs. " + val_x
    
    layout = dict(title = plot_title, 
                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),
                  yaxis=dict(title = val_y, side='left'),                                  
                  legend=dict(orientation="h", x=0.4, y=1.0),
                  autosize=False, width=750, height=500,
                 )

    fig = dict(data = data, layout = layout)
    iplot(fig)

def plotly_scatter_x_y_color(df_plot, val_x, val_y, val_z):
    
    value_x = df_plot[val_x] 
    value_y = df_plot[val_y]
    value_z = df_plot[val_z]
    
    trace_1 = go.Scatter( 
                         x = value_x, y = value_y, name = val_x, 
                         mode="markers", opacity=0.8, text=value_z,
                         marker=dict(size=6, color = value_z, 
                                     colorscale='Jet', showscale=True),                        
                        )
                            
    data = [trace_1]
    
    plot_title = val_y + " vs. " + val_x
    
    layout = dict(title = plot_title, 
                  xaxis=dict(title = val_x, ticklen=5, zeroline= False),
                  yaxis=dict(title = val_y, side='left'),                                  
                  legend=dict(orientation="h", x=0.4, y=1.0),
                  autosize=False, width=750, height=500,
                 )

    fig = dict(data = data, layout = layout)
    iplot(fig)
    
def plotly_scatter_x_y_catg_color(df, val_x, val_y, val_z):
    
    catg_for_colors = sorted(df[val_z].unique().tolist())

    fig = { 'data': [{ 'x': df[df[val_z]==catg][val_x],
                       'y': df[df[val_z]==catg][val_y],    
                       'name': catg, 
                       'text': df[val_z][df[val_z]==catg], 
                       'mode': 'markers',
                       'marker': {'size': 6},
                      
                     } for catg in catg_for_colors       ],
                       
            'layout': { 'xaxis': {'title': val_x},
                        'yaxis': {'title': val_y},                    
                        'colorway' : ['#a9a9a9', '#e6beff', '#911eb4', '#4363d8', '#42d4f4',
                                      '#3cb44b', '#bfef45', '#ffe119', '#f58231', '#e6194B'],
                        'autosize' : False, 
                        'width' : 750, 
                        'height' : 600,
                      }
           }
  
    iplot(fig)
    
def plotly_scatter3d(data, feat1, feat2, target) :

    df = data
    x = df[feat1]
    y = df[feat2]
    z = df[target]

    trace1 = go.Scatter3d( x = x, y = y, z = z,
                           mode='markers',
                           marker=dict( size=5, color=y,               
                                        colorscale='Viridis',  
                                        opacity=0.8 )
                          )
    data = [trace1]
    camera = dict( up=dict(x=0, y=0, z=1),
                   center=dict(x=0, y=0, z=0.0),
                   eye=dict(x=2.5, y=0.1, z=0.8) )

    layout = go.Layout( title= target + " as function of " +  
                               feat1 + " and " + feat2 ,
                        autosize=False, width=700, height=600,               
                        margin=dict( l=15, r=25, b=15, t=30 ) ,
                        scene=dict(camera=camera,
                                   xaxis = dict(title=feat1),
                                   yaxis = dict(title=feat2),
                                   zaxis = dict(title=target),                                   
                                  ),
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
#Scatter plot against multiple variables
def scatter_multi_col(df_plot, x_col_vals, y_col_vals,title):
    value_x = df_plot[x_col_vals] 
    value_y = df_plot[y_col_vals]
    nr_rows=int(len(x_col_vals)/3)
    nr_cols=3
    fig = tools.make_subplots(rows=nr_rows, cols=nr_cols, print_grid=False,
                          subplot_titles=x_col_vals)
                                                                
    for row in range(1,nr_rows+1):
        for col in range(1,nr_cols+1):
            i = (row-1) * nr_cols + col-1
            trace = go.Scatter(x = df_plot[x_col_vals[i]],
                               y = df_plot[y_col_vals],
                               name=x_col_vals[i],
                               mode="markers",
                               opacity=0.8)
            fig.append_trace(trace, row, col,)
 
                                                                                                  
    fig['layout'].update(height=700, width=900, showlegend=False,
                     title='SalePrice' + ' vs. Area features')
    iplot(fig)                     
    return fig
######################################################################################
# Box plot with string variables

def plotly_boxplots_sorted_by_yvals(df, catg_feature, sort_by_target):
    
    df_by_catg   = df.groupby([catg_feature])
    sortedlist_catg_str = df_by_catg[sort_by_target].median().sort_values().keys().tolist()
    
    
    data = []
    for i in sortedlist_catg_str :
        data.append(go.Box(y = df[df[catg_feature]==i][sort_by_target], name = i))

    layout = go.Layout(title = sort_by_target + " vs " + catg_feature, 
                       xaxis = dict(title = catg_feature), 
                       yaxis = dict(title = sort_by_target))

    fig = dict(data = data, layout = layout)
    return fig

# Box plot with numirical variables
def plotly_boxplots_num_yvals(df, num_feature, LABEL_COL):
    trace = []
    for name, group in df[[LABEL_COL, num_feature]].groupby(num_feature):
        trace.append( go.Box( y=group[LABEL_COL].values, name=name ) )
    
    layout = go.Layout(title=num_feature, 
                       xaxis=dict(title=num_feature,ticklen=5, zeroline= False),
                       yaxis=dict(title=LABEL_COL, side='left'),
                       autosize=False, width=750, height=500)

    fig = go.Figure(data=trace, layout=layout)
    iplot(fig)
    return fig

#######################################################################################################
#Finad Numerical and Catgorical Columns
def numirical_catagorical_calumns(df):
    numerical_feats = df.dtypes[df.dtypes != "object"].index
    print("Number of Numerical features: ", len(numerical_feats))

    categorical_feats = df.dtypes[df.dtypes == "object"].index
    print("Number of Categorical features: ", len(categorical_feats))
    print("*"*100)
    print('Numrical Columns')
    print(df[numerical_feats].columns)
    print("*"*100)
    print('Catagorical Columns')
    print(df[categorical_feats].columns)
    return numerical_feats, categorical_feats
    
#################################################################################

#basic analysis
def basicanalysis(df):
    print("Shape is:\n", df.shape)
    print("\n Columns are:\n", df.columns)
    print("\n Types are:\n", df.dtypes)
    print("\n Statistical Analysis of Numerical Columns:\n", df.describe())

#string column analysis analysis
def stringcolanalysis(df):
    stringcols = df.select_dtypes(exclude=[np.number, "datetime64"])
    fig = plt.figure(figsize = (8,15))
    for i,col in enumerate(stringcols):
        fig.add_subplot(15,2,i+1)
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
        sns.boxplot(df[col], color='grey', linewidth=1)
        plt.tight_layout()
        plt.title(col)
        plt.savefig("Numerical.png")
    
    # Lets also plot histograms for these numerical columns
    df.hist(column=list(numcols.columns),bins=25, grid=False, figsize=(15,12),
                 color='#86bf91', zorder=2, rwidth=0.9)

# Perform correlation analysis over numerical columns
def correlation_anlysis(df):
    # NOTE: If label column is non-numeric, 'encode' it before calling this function 
    numcols = df.select_dtypes(include=np.number)
    corr = numcols.corr()
    ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    )
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
    
    
    
def correlation(df):
    numcols = df.select_dtypes(include=np.number)
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
    
###############################

###############################################################################################################################
# T-Test
# Conducted t-test for between all numeric columns in dataset

def t_test(df):
    print ("t-test for equality of mean between all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    print(num_columns)

    for i in range(len(num_columns)-1):
        for j in range(i+1,len(num_columns)):
            col1 = num_columns[i]
            col2 = num_columns[j]
            t_val, p_val = stats.ttest_ind(df[col1], df[col2])
            print("(%s,%s) => t-value=%s, p-value=%s" % (num_columns[i], num_columns[j], str(t_val), str(p_val)))   
####################3
# Normality Test for all Numeric Columns
def Normality_test(df):
    print ("Normality Test for all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    print(num_columns)

    for i in range(len(num_columns)):
        col1 = num_columns[i]
        stat, p_val = shapiro(df[col1])
        print('Normaility Test for Column:', col1)
        print('Statistics=%.3f, p_value=%.3f' % (stat, p_val))
        #print("(%s,%s) => Statistics=%s, p-value=%s" % (str(stat), str(p_val)))
        alpha = 0.05
        if p_val > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        
#######################3

#######################################
#Chi Square Test for indepence
def chisquare_test(df):
    print ("Chisquare-test for Independence between all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    #print(num_columns)

    for i in range(len(num_columns)-1):
        for j in range(i+1,len(num_columns)):
            col1 = num_columns[i]
            col2 = num_columns[j]
            crosstab = pd.crosstab(df[col1], df[col2])

            stat, p, dof, expected = chi2_contingency(crosstab)
            #t_val, p_val = stats.ttest_ind(df[col1], df[col2])
            print("(%s,%s) => chisqr-value=%s, p-value=%s" % (num_columns[i], num_columns[j], str(stat), str(p)))
            alpha = 0.05
            if p <= alpha:
                print('Dependent (reject H0)')
            else:
                print('Independent (H0 holds true)')
     
    
############################################
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
################################################################################################
################################################################################################


####################################
#Helper Function for Tranformation

#MinMax Transformation
def MinMax_Transformation(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    scaler = MinMaxScaler()
    scaled_features = MinMaxScaler().fit_transform(X.values)
    df1= pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    z1=pd.DataFrame(df1)
    z2=pd.DataFrame(y)
    df= pd.concat([z1,z2], axis=1)
    return df 
#Standard Transformation
def Standard_Transformation(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    scaled_features = StandardScaler().fit_transform(X.values)
    df1= pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    z1=pd.DataFrame(df1)
    z2=pd.DataFrame(y)
    df= pd.concat([z1,z2], axis=1)
    return df 


#####################################
#################################################################################################################
#FS: feature selection algorithms (I included RFFS, RFE, MI and PCA but you can add only RFFS as for me, it works good).Ã‚Â
# Helper function to get fetaure importance metrics via Random Forest Feature Selection (RFFS)

#for regression
def RFfeatureimportance_reg(df, LABEL_COL, split,random, tree,threshold):
    
    df1=df.drop(LABEL_COL,axis=1)
    
    X_train, X_test, y_train, y_test=traintestsplit(df,split,random, LABEL_COL)

    #for Random Forest Regression
    
    
    clf  = RandomForestRegressor(n_estimators=tree, random_state=random)
    clf.fit(X_train, y_train)
    
    res_rf = pd.Series(clf.feature_importances_, index=df1.columns.values).sort_values(ascending=False)*100
    #print(res_rf)
    impftrs_rf = list(res_rf[res_rf > threshold].keys())
    resf1=pd.Series(impftrs_rf)
    impftrs_rf.append(LABEL_COL)
    RF_df = df[impftrs_rf]
    print('Random Forest FS Analysis')
    print('Selected Features')
    print(resf1)
    return RF_df

#################################################################################################################
# Train Test Split for ML Algorithms


def traintestsplit(df,split,random=SEED, LABEL_COL=''):
    #make a copy of the label column and store in y
    y = df[LABEL_COL].copy()
    
    #now delete the original
    X = df.drop(LABEL_COL,axis=1)
    
    #manual split
    trainX, testX, trainY, testY= train_test_split(X, y, test_size=split, random_state=random)
    return trainX, testX, trainY, testY

###################################################################################