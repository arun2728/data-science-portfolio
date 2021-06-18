#!/usr/bin/env python
# coding: utf-8

# # Mobile Price Classification
# 
# *The dataset for this project originates from the [Classify Mobile Price Range](https://www.kaggle.com/iabhishekofficial/mobile-price-classification)*
# <br>
# 
# ### Context: 
# 
# Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc. He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
# 
# Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
# 
# ### Problem Statement: 
# 
# In this problem you do not have to predict actual price but a price range indicating how high the price is
# 
# 
# ### Attribute Information
# 
# The data is contains values of different features of a mobile collected from different sources. The objective is caluclate price range of a mobile.
# 
# #### Input variables:
# 
# 1) **battery_power**: Total energy a battery can store in one time measured in mAh
# 
# 2) **blue**: Has bluetooth or not
# 
# 3) **clock_speed**: speed at which microprocessor executes instructions
# 
# 4) **dual_sim**: Has dual sim support or not
# 
# 5) **fc**: Front Camera mega pixels
# 
# 6) **four_g**: Has 4G or not
# 
# 7) **int_memory**: Internal Memory in Gigabytes
# 
# 8) **m_dep**: Mobile Depth in cm
# 
# 9) **mobile_wt**: Weight of mobile phone
# 
# 10) **n_cores**: Number of cores of processor
# 
# 11) **pc**: Primary Camera mega pixels
# 
# 12) **px_height**: Pixel Resolution Height
# 
# 13) **px_width**: Pixel Resolution Width
# 
# 14) **ram**: Random Access Memory in Mega Bytes
# 
# 15) **sc_h**: Screen Height of mobile in cm
# 
# 16) **sc_w**: Screen Width of mobile in cm
# 
# 17) **talk_time**: longest time that a single battery charge will last when you are
# 
# 18) **three_g**: Has 3G or not
# 
# 19) **touch_screen**: Has touch screen or not
# 
# 20) **wifi**: Has wifi or not
# 
# #### Output variable (desired target):
# 
# 21) **price_range**: This is the target variable with value of 0 (cheap), 1 (mid priced), 2 (costly) and 3(expensive).
# 

# <a id ='toc'></a>
# # Table of Contents
# 
# 1. **[Environment Setup](#environment_setup)**
#     - 1.1 - **[Install Package](#install_packages)**
#     - 1.2 - **[Load Dependencies](#import_packages)**
# 2. **[Load dataset](#load_data)**
# 3. **[Data Types and Dimensions](#Data_Types)**
# 4. **[Data Preprocessing](#data_preprocessing)**
#     - 4.1 - [Data Cleaning](#data_cleaning)
#     - 4.2 - [Exploratory Analysis](#exploratory_analysis)
#         - 4.2.1 - [Numeric features](#numerical_features)
#         - 4.2.2 - [Categorical features](#categorical_features)
#         - 4.2.3 - [Analysis report](#report)
#     - 4.3 - [Feature Selection](#Feature_Selection)
#     - 4.4 - [Data Transformation](#data_transformation) 
#         - 4.4.1 - [Normalization](#normalization)
#         - 4.4.2 - [Split the dataset](#split_the_dataset)
# 5. **[Model Development](#model_development)**
#     - 5.1 - [KNN](#KNN)
#     - 5.2 - [Random Forest](#random_forest)
#     - 5.3 - [Naive Bayes](#Naive_Bayes)
#     - 5.4 - [Gradient Boosting](#GBM)
# 6. **[Model Comparision](#model_cmp)**  
# 7. **[Conclusion](#conclusion)**

# <a id ='environment_setup'></a>
# ## 1. Environment Setup
# 
# [goto toc](#toc)

# <a id ='install_packages'></a>
# ### 1.1. Install Packages
# 
# Install required packages
# 
# [goto toc](#toc)

# In[130]:


# Install pandas
get_ipython().system(' pip install pandas')

# Install matplotlib
get_ipython().system(' pip install matplotlib')

# Install seaborn
get_ipython().system(' pip install seaborn ')

# Install sklearn
get_ipython().system(' pip install sklearn')

# Install tqdm to visualize iterations
get_ipython().system(' pip install tqdm')


# <a id ='import_packages'></a>
# ### 1.2. Load Dependencies
# 
# Import required packages
# 
# [goto toc](#toc)

# In[123]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# Set default setting of seaborn
sns.set()


# In[124]:


# Import required function for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Import the required function for normalization
from sklearn.preprocessing import StandardScaler

# Import train and test split function
from sklearn.model_selection import train_test_split


# In[125]:


# Import Classifiers to be used

# Import Grid Search Cross Validation for tunning
from sklearn.model_selection import GridSearchCV

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Import Naive bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Import Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# In[126]:


# Import packages to calculate performance of the models
from sklearn import metrics

# Function to compute confusion metric
from sklearn.metrics import confusion_matrix

# Function to generate classification report
from sklearn.metrics import classification_report


# In[127]:


# To save the model import pickle 
import pickle


# <a id ='load_data'></a>
# ## 2. Load dataset
# 
# Read data from mobile_price.csv file using pandas method read_csv().
# 
# [goto toc](#toc)

# In[6]:


# read the data
raw_data = pd.read_csv('data/mobile_price.csv')

# print the first five rows of the data
raw_data.head()


# <a id ='Data_Types'></a>
# ## 3. Data Types and Dimensions
# 
# [goto toc](#toc)

# In[7]:


# check the data types of the features
raw_data.info()


# **Note:**
# 
# Features like **blue**, **dual_sim**, **four_g**, **n_cores**, **three_g**, **touch_screen**, **wifi** and **price_range** are actually categorical in nature but are represemted as numeric so we need to convert them for better analysis.

# In[8]:


# create copy of the dataframe
data = raw_data.copy()

# Create list of features to be converted into category
features = ['blue', 'dual_sim', 'four_g', 'n_cores', 'three_g', 'touch_screen', 'wifi', 'price_range']

# Convert numeric to categorical
for col in features:
    data[col] = pd.Categorical(data[col])
    
# Check for datatypes
data.info()


# In[9]:


# Get categorical features
categorical_features = data.select_dtypes('category').columns.values.tolist()

# Get nuemric features
numerical_features = [col for col in data.columns.values if col not in categorical_features]


# In[11]:


print("Mobile Price Classification Data Set has \033[4m\033[1m{}\033[0m\033[0m data points with \033[4m\033[1m{}\033[0m\033[0m variables each.".format(*raw_data.shape))
print(f"Numeric features: \033[4m\033[1m{len(numerical_features)}\033[0m\033[0m \nCategorical features: \033[4m\033[1m{len(categorical_features)}\033[0m\033[0m")


# <a id='data_preprocessing'></a>
# ## 4. Data Preprocessing
# 
# 
# *Data preprocessing is a data mining technique which is used to transform the raw data in a useful and efficient format.*
# 
# [...goto toc](#toc)

# <a id='data_cleaning'></a>
# ## 4.1. Data Cleaning
# 
# *Data cleaning* refers to preparing data for analysis by removing or modifying data that is incomplete, irrelevant, duplicated, or improperly formatted.
# 
# [...goto toc](#toc)

# ### Missing Data Treatment
# 
# If the missing values are not handled properly we may end up drawing an inaccurate inference about the data. Due to improper handling, the result obtained will differ from the ones where the missing values are present.

# In[12]:


# get the count of missing values
missing_values = data.isnull().sum()

# print the count of missing values
print(missing_values)


# **Note: There are no missing values in the dataset so we can proceed further**

# <hr style="border:1px solid gray"> </hr>
# <h3><center>Summary</center></h3>
# <hr style="border:1px solid gray"> </hr>
# 
# | Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Missing Values |
# | :-: | :-: | :-: | :-: | :-: |
# | 2000  | 21 | 13 | 8  | Null |

# <a id='exploratory_analysis'></a>
# ## 4.2. Exploratory Analysis
# 
# The preliminary analysis of data to discover relationships between measures in the data and to gain an insight on the trends, patterns, and relationships among various entities present in the data set with the help of statistics and visualization tools is called Exploratory Data Analysis (EDA). 
# 
# Exploratory data analysis is cross-classified in two different ways where each method is either graphical or non-graphical. And then, each method is either univariate, bivariate or multivariate.
# 
# [...goto toc](#toc)

# <a id='numerical_features'></a>
# ### 4.2.1. Numerical Features
# 
# *Analysis of only numeric features*
# 
# [...goto toc](#toc)

# In[15]:


# Get only numeric features for analysis
numeric_data = data[numerical_features]
numeric_data.head()


# In[87]:


# PLot KDE for all features

# Set size of the figure
plt.figure(figsize=(20,35))

# Iterate on list of features
for i, col in enumerate(numerical_features):
    if numeric_data[col].dtype != 'object':
        ax = plt.subplot(9, 2, i+1)
        kde = sns.kdeplot(numeric_data[col], ax=ax)
        plt.xlabel(col)
        
# Save the plot
plt.savefig("Numeric_Features_1.png")

# Show plots        
plt.show()


# In[94]:


# Function to plot a numeric feature
def plot_numeric_feature(data, numerical_features):
    # Iterate throw each feature
    for feature in numerical_features:
        print("-"*150)
        print(f"Feature : \033[4m\033[1m{feature}\033[0m\033[0m")
        print("-"*150)
        
        # Create subplots figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram
        sns.histplot(data=data, x=feature, ax = axes[0])
        
        # Boxplot
        sns.boxplot(y = feature , x = 'price_range', data = data, ax = axes[1] )
        
        # Displot of given feature with respect to output variable
        sns.displot(data=data, x=feature, hue="price_range", multiple="stack", kind="kde")
        
        # Save the plot
        fig.savefig(f"{feature}_Feature.png")


        # Show all plots
        plt.show()


# In[95]:


plot_numeric_feature(data, numerical_features)


# **Note:**
# 
# - **battery_power**, **int_memory**, **m_depth**, **mobile_wt**, **pc**, **px_width**, **ram**, **sc_h** and **talktime** are features with data concentrated toward the center and their extremes are less in quantity.
# 
# - **sc_w**, **px_h**, **fc** and **clock_speed** features are right skewed i.e mean > median > mode
# 

# #### Correlation

# In[19]:


# check correlation
corr = data.corr(method = 'spearman')
corr


# In[93]:


# correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns_plot = sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f', ax=ax)

# Save the plot
f.savefig("correlation.png")


# **Note:** Features **fc**, **pc** are highly correlated

# <a id='categorical_features'></a>
# ### 4.4.2. Categorical Features
# 
# *Analysis of categorical features*
# 
# [...goto toc](#toc)

# In[21]:


# Get only categorical features for analysis
categorical_data = data[categorical_features]
categorical_data.head()


# In[96]:


def plot_categorical_features(data, categorical_features):
    
    names, count  = {'blue': 'bluetooth', 'dual_sim': "Dual Sim", "four_g":"4G", "n_cores":"No. of Cores", 
         "three_g":"3G", "touch_screen":"Touch Screen", "wifi":"WiFi"}, 1
    
    for feature in categorical_features:
        
        if feature == "price_range":
            continue
            
        print("-"*150)
        print(f"Feature : \033[4m\033[1m{names[feature]}\033[0m\033[0m")
        print("-"*150)
        
        labels = ["no", "yes"] if feature != "n_cores" else data[feature].unique().tolist()
        
        # Create subplots figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        if count % 2 == 0:
            # Plot countplot of feature with respect to target
            sns.countplot(x = 'price_range', data = data, hue=feature, ax = axes[0], palette='rainbow')

            # Plot pie chart to show distribution of feature
            axes[1].pie(data[feature].value_counts().values, labels = labels, autopct='%1.1f%%',startangle=90)
            axes[1].set_xlabel(names[feature], size=22)

        else:
            # Plot pie chart to show distribution of feature
            axes[0].pie(data[feature].value_counts().values, labels = labels, autopct='%1.1f%%',startangle=90)
            axes[0].set_xlabel(names[feature], size=22)
            
            # Plot countplot of feature with respect to target
            sns.countplot(x = 'price_range', data = data, hue=feature, ax = axes[1], palette='rainbow')   
        
        # Increase the counter
        count += 1
        
        # Save features
        fig.savefig(f"{feature}_feature.png")
        
        # Show all plots
        plt.show()
        
        


# In[97]:


plot_categorical_features(categorical_data, categorical_features)


# ### Analyzing target feature

# In[120]:


# Create subplots figure
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot countplot of feature with respect to target
sns.countplot(x = 'price_range', data = data, ax = axes[0], palette='rainbow')

# Plot pie chart to show distribution of feature

labels = ['cheap', 'mid priced', 'costly', 'expensive']

axes[1].pie(categorical_data.price_range.value_counts().values, labels = labels, autopct='%1.1f%%',startangle=90)
axes[1].set_ylabel('Price Range', size=22)

# Save plot
fig.savefig('target_feature.png')

# Show all plots
plt.show()


# <a id='report'></a>
# ### 4.2.3. Analysis Report
# 
# [...goto toc](#toc)
# 

# <hr style="border:1.2px solid gray"> </hr>
# <h3><center><u>Analysis Report</u></center></h3>
# <hr style="border:1.2px solid gray"> </hr>
# 
# | Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature | Missing Values |
# | :-: | :-: | :-: | :-: | :-: | :-: |
# | 2000  | 21 | 13 | 8  | price_range | Null |
# 
# <hr>
# 
# <h4><center><u>Data Types</u></center></h4>
# 
# | Sr.No. | Column | Data type |
# | :-: | :- | :- |
# | 1  | battery_power                                    | int64 |
# | 2  | blue                          | category | 
# | 3  | clock_speed                        | float64 | 
# | 4  | dual_sim                 | category | 
# | 5  | fc           | int64 |  
# | 6  | four_g                     | category | 
# | 7  | int_memory                     | int64 | 
# | 8  | m_dep                         | float64 | 
# | 9  | mobile_wt                 | int64 | 
# | 10  | n_cores                   | category | 
# | 11 | pc                 | int64 | 
# | 12 | px_height                | int64 | 
# | 13 | px_width              | int64 | 
# | 14 | ram       | int64 | 
# | 15 | sc_h | int64 | 
# | 16 | sc_w           | int64 | 
# | 17 | talk_time                 | int64 |
# | 18 | three_g                 | category |
# | 19 | touch_screen                 | category |
# | 20 | wifi                 | category |
# | 21 | price_range                 | category |
# 
# <hr>
# 
# <h4><center><u>Exploratory Data Analysis</u></center></h4>
# <hr>
# 
# #### Numeric Features 
# 
# - Generally cheaper phones have low front camera mega pixels as compared to others.
# - Costly and expensive phones generally have higher battery and internal memory. 
# - Additionaly, they are lighter as compared to others.
# - Pixel resolution increases as price range increases but clock speed doesnot show much deviation with respect to price.
# - There is no significant variation in phones price with respect to screen width and height.
# - front camera and primary camera mega pixels are higly correlated to each other.
# - **RAM** is the most important feature to predict price among all the features which is also true in practical scenario.
# 
# <hr>
# 
# #### Categorical Features
# 
# - Features like **blue**, **dual_sim**, **four_g**, **three_g**, **touch_screen**, **wifi** are binary in nature and are equally distributed 
#     - 1 : Yes
#     - 0 : No
# - Number of cores are ranging from 1 to 8
# - Majority of expensive phone have features like wifi, bluetooth, dual sim and 4G.
# - 3G features is available in all types of phones. But in the given dataset it is biased that 70% of phones don't have 3G service.
# - Target feature price_range has four different categories
#     - 0 : cheap
#     - 1 : mid-priced
#     - 2 : costly
#     - 3 : expensive
# - Additionally target feature is balanced

# **Note:**
# 
# There is no need to encode categorical features as they are encoded by defualt. We just need to change there datatype using pandas function called *pd.to_numeric()*. But feature **n_cores** need to **one-hot encoded** as it has eight different categories.

# In[25]:


# One-hot encode n_cores feature
no_cores = pd.get_dummies(data['n_cores'], prefix='cores', drop_first=True)

# Convert features to numeric
data_preprocessed = data.drop('n_cores', axis = 1).apply(pd.to_numeric, axis = 1)

# Concatenate with original feature
data_preprocessed = data_preprocessed.join(no_cores)
data_preprocessed.head()


# <a id='Feature_Selection'></a>
# ## 4.3. Feature Selection
# 
# Since there are all together **27** independent features we will perform feature selection to eliminate curse of dimensionality.
# 
# We will be using **f_classif as feature selector** because our **features** are **quantitative** i.e numeric and **target** feature is **categorical**. f_classif is a feature selector that computes the **ANOVA F-value** between each feature and the target vector.
# 
# [...goto toc](#toc)
# 

# In[27]:


# Seperate independent features and target feature
X, y = data_preprocessed.drop(['price_range'], axis = 1), data_preprocessed['price_range']


# In[28]:


# Calculate f-score and p-value 
f_statistic, p_values = f_classif(X, y)

# Create a dataframe to record score
d = pd.DataFrame()

# Save features
d['feature'] = X.columns

# record F-score
d['fscore'] = f_statistic

# record p-values
d['pvalue'] = p_values

# Sort based of f-score
d.sort_values(by = 'fscore', ascending=False)


# **Note:**
# 
# We can see that **RAM** has the **highest F-score** and **minimum p-value** which is expected. From *22* features we will be selecting top *10* features based on their *f-score* using *SelectKBest* method.

# In[29]:


# Perform feature selection we will select best 10 features
fvalue_Best = SelectKBest(f_classif, k = 10)

# Fit and transform feature selector on given dataset
X_best = fvalue_Best.fit_transform(X, y)


# In[30]:


print(f'Original dataset have \033[4m\033[1m{X.shape[1]}\033[0m\033[0m features.\nAfter feature selection dataset have \033[4m\033[1m{X_best.shape[1]}\033[0m\033[0m features.')


# In[31]:


# The list of your K best features
mask = fvalue_Best.get_support() 

# Get list of selected features
selected_features = [feature for bool_val, feature in zip(mask, X.columns.values.tolist()) if bool_val]

# print best features
print("Selected features are : ", selected_features)


# <a id='data_transformation'></a>
# ## 4.4. Data Transformation
# 
# [...goto toc](#toc)
# 
# 

# <a id='normalization'></a>
# ### 4.4.1 Normalization
# 
# *Normalization is used to scale the data of an attribute so that it falls in a smaller range, such as -1.0 to 1.0 or 0.0 to 1.0. It is generally useful for classification algorithms.*
# 
# We will use *Standard Scaler* to perform normalization.
# 
# [...goto toc](#toc)
# 

# In[33]:


# Initilize scaler
scaler = StandardScaler()

# fit the scaler
scaler.fit(X_best)


# In[34]:


# Transform the dataset
X_normal = scaler.fit_transform(X_best)


# <a id='split_the_dataset'></a>
# ### 4.4.2. Split dataset
# 
# We will be splitting the dataset into train and test set with **70-30** split
# 
# [...goto toc](#toc)
# 

# In[36]:


# let us now split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size = 0.3, random_state=42)

# print the shape of 'x_train'
print("X_train : ",X_train.shape)

# print the shape of 'x_test'
print("X_test : ",X_test.shape)

# print the shape of 'y_train'
print("y_train : ",y_train.shape)

# print the shape of 'y_test'
print("y_test : ",y_test.shape)


# <a id='model_development'></a>
# ## 5. Model Development
# 
# We will be training different classification model and choose the one with best performance
# 
# [...goto toc](#toc)
# 

# <a id="KNN"> </a>
# ### 5.1. KNN
# 
# *To find optimal value of **K** we will be performing hyperparameter tuning using **Grid Search Cross Validation**.*
# 
# [...goto toc](#toc)

# In[39]:


# Hyperparameter tuning

# Initialize a knn object
knn = KNeighborsClassifier()

# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 6)}


# In[40]:


# Perform gridsearch
knn_gscv = GridSearchCV(knn, param_grid, cv=5)

# fit the data
knn_gscv.fit(X_train, y_train)


# In[41]:


# predict the values
y_pred_knn  = knn_gscv.predict(X_test)


# In[42]:


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)

# label the confusion matrix  
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3'],index=['Actual:0','Actual:1','Actual:2','Actual:3'])

# set sizeof the plot
plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False)
plt.show()


# In[44]:


# Generate classiffication report

# accuracy measures by classification_report()
result = classification_report(y_test, y_pred_knn)

# print the result
print(result)


# In[51]:


# Tabulate the result

# create a list of column names
cols = ['Model', 'Precision Score', 'Recall Score','Accuracy Score','f1-score']

# creating an empty dataframe of the colums
result_tabulation = pd.DataFrame(columns = cols)

# compiling the required information
knn_estimator = pd.Series({'Model': "KNN",
                 'Precision Score': metrics.precision_score(y_test, y_pred_knn, average="macro"),
                 'Recall Score': metrics.recall_score(y_test, y_pred_knn, average="macro"),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred_knn),
                  'f1-score':metrics.f1_score(y_test, y_pred_knn, average="macro")})



# appending our result table
result_tabulation = result_tabulation.append(knn_estimator , ignore_index = True)

# view the result table
result_tabulation


# <a id="random_forest"> </a>
# ## 5.2 Random Forest
# 
# [...goto toc](#toc)

# In[53]:


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


# In[54]:


# Predicting the Test set results
y_pred_random = classifier.predict(X_test)


# In[55]:


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_random)

# label the confusion matrix  
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3'],index=['Actual:0','Actual:1','Actual:2','Actual:3'])

# set sizeof the plot
plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False)
plt.show()


# In[57]:


# Generate classification report

# accuracy measures by classification_report()
result = classification_report(y_test, y_pred_random)

# print the result
print(result)


# In[60]:


# create the result table for all scores
random_forest_metrics = pd.Series({'Model': "Random Forest",
                 'Precision Score': metrics.precision_score(y_test, y_pred_random, average="macro"),
                 'Recall Score': metrics.recall_score(y_test, y_pred_random, average="macro"),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred_random),
                  'f1-score':metrics.f1_score(y_test, y_pred_random, average="macro")})



# appending our result table
result_tabulation = result_tabulation.append(random_forest_metrics , ignore_index = True)

# view the result table
result_tabulation


# <a id="Naive_Bayes"> </a>
# ## 5.3 Naive Bayes
# 
# [...goto toc](#toc)

# In[61]:


# build the model
GNB = GaussianNB()

# fit the model
GNB.fit(X_train, y_train)


# In[62]:


# predict the values
y_pred_GNB  = GNB.predict(X_test)


# In[63]:


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_GNB)

# label the confusion matrix  
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3'],index=['Actual:0','Actual:1','Actual:2','Actual:3'])

# set sizeof the plot
plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False)
plt.show()


# In[64]:


# Generate classiffication report

# accuracy measures by classification_report()
result = classification_report(y_test, y_pred_GNB)

# print the result
print(result)


# In[65]:


# create the result table for all scores
GNB_metrices = pd.Series({'Model': "Naive Bayes",
                 'Precision Score': metrics.precision_score(y_test, y_pred_GNB, average="macro"),
                 'Recall Score': metrics.recall_score(y_test, y_pred_GNB, average="macro"),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred_GNB),
                  'f1-score':metrics.f1_score(y_test, y_pred_GNB, average="macro")})



# appending our result table
result_tabulation = result_tabulation.append(GNB_metrices , ignore_index = True)

# view the result table
result_tabulation


# <a id="GBM"> </a>
# ## 5.4 Gradient Boosting
# 
# We will be performing hyperparameter tuning
# 
# [...goto toc](#toc)

# In[68]:


# Choose the best Hyperparameters
# We have chosen, learning_rate, max_depth and the n_estimators.

# Define hyperparameters
parameters = {
    "n_estimators":[5,50,250,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]
}

# Call the Boosting classifier constructor
gbc = GradientBoostingClassifier()


# In[69]:


# Use the GridSearhCV() for the cross -validation
cv = GridSearchCV(gbc,parameters,cv=5)

# Fit the data
cv.fit(X_train, y_train)


# In[70]:


# Function to display best parameters
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[71]:


# Display best parameters
display(cv)


# **Note:**
# 
# Best parameters are: 
# - learning_rate = 1
# - max_depth = 1
# - n_estimators = 500

# In[101]:


# Train the classifier
GBM = GradientBoostingClassifier(learning_rate=1, max_depth=1, n_estimators=500)

# fit on data
GBM.fit(X_train, y_train)


# In[112]:


# Create a dataframe to store importance of features
feature_importance = pd.DataFrame()
feature_importance['feature'] = selected_features
feature_importance['Importance'] = GBM.feature_importances_

# Sort in decreasing order
feature_importance = feature_importance.sort_values(ascending = False, by = 'Importance').reset_index(drop=True)
feature_importance


# In[119]:


# Plot feature importance
fig, ax = plt.subplots(figsize = (18,18))

# Barplot
sns.barplot(x = 'feature' , y = 'Importance', data = feature_importance, ax = ax)

# Add title
ax.set_title("Feature Importance in Gradient Boosting Model")

# Save the plot
fig.savefig('Feature_importance.png')

# Show the plot
plt.show()


# In[103]:


# predict the values
y_pred_gbm  = GBM.predict(X_test)


# In[81]:


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_gbm)

# label the confusion matrix  
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3'],index=['Actual:0','Actual:1','Actual:2','Actual:3'])

# set sizeof the plot
plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False)
plt.show()


# In[82]:


# Generate classification report

# accuracy measures by classification_report()
result = classification_report(y_test, y_pred_gbm)

# print the result
print(result)


# In[77]:


# create the result table for all scores
GBM_metrices = pd.Series({'Model': "Gradient Boosting",
                 'Precision Score': metrics.precision_score(y_test, y_pred_gbm, average="macro"),
                 'Recall Score': metrics.recall_score(y_test, y_pred_gbm, average="macro"),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred_gbm),
                  'f1-score':metrics.f1_score(y_test, y_pred_gbm, average="macro")})



# appending our result table
result_tabulation = result_tabulation.append(GBM_metrices , ignore_index = True)

# view the result table
result_tabulation


# <a id="model_cmp"> </a>
# ## 6. Model Comparision
# 
# [...goto toc](#toc)

# In[78]:


result_tabulation


# **Note: We can see that Gradient Boosting Method has outperformed.**

# In[84]:


best_model = GBM


# ### Save the model

# In[121]:


pickle.dump(best_model, open("mobile_price_predictor.sav", "wb"))


# <a id="conclusion"> </a>
# 
# [...goto toc](#toc)
# 
# 

# <hr style="border:1.2px solid gray"> </hr>
# <h3><center><u>Conclusion</u></center></h3>
# <hr style="border:1.2px solid gray"> </hr>
# 
# During the analysis the given problem of predicting the price range we found that the problem is a multiclass classiffication problem and the dataset given is balanced with respect different categories of target feature (price_range).
# 
# We built different classifier for the given problem like KNN, Naive Bayes, Random Forest but **Gradient Boosting Classifier** outperformed other classifiers with test accuracy of **91.8%** and a f1-score of **0.917**. 
# 
# Additionaly, while extracting feature importance from trained gradient boosting classifier we found that the feature **RAM** is the most important feature to predict price among all the features which is also true in practical scenario.
# 
# <hr>
# <h4><center>Best Model</center></h4>
# <hr>
# 
# | Model | Precision Score | Recall Score | Accuracy Score | f1-score |
# | :-: | :-: | :-: | :-: | :-: |
# | Gradient Boosting	  | 0.917786 | 0.917883 | 0.918333  | 0.917677 |
# 
# <hr>

# In[ ]:




