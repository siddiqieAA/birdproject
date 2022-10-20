import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

import folium
from folium.plugins import HeatMap
import plotly.express as px
import seaborn as sns

## Import LabelEncoder from sklearn
from sklearn.preprocessing import LabelEncoder
#Calculate by using PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data=pd.read_csv('/content/bird.csv')
dataset = pd.read_csv('/content/bird.csv')
df = pd.read_csv('/content/bird.csv')

data

data.shape

data.columns

data.dtypes

data_catagorical=data.select_dtypes(include=['object']).columns.tolist()
data_catagorical

data.describe()

data.info()

#Check for Null or missing values in the datset
data.isnull().sum()

dataset.duplicated().any()

#remove column 'id' as it is no relevant for data analysis
datanew = data.drop(['id'], axis = 1)
datanew.info()

sns.countplot(data=datanew, x = 'bird_type');

datanew.groupby('bird_type').size().plot(kind = 'pie', autopct = '%.0f%%', label = '');

sns.boxplot(data = datanew, x = 'bird_type', y = 'humerus_length');

sns.boxplot(data = datanew, x = 'bird_type', y = 'humerus_width');

sns.boxplot(data = datanew, x = 'bird_type', y = 'ulna_length');

sns.boxplot(data = datanew, x = 'bird_type', y = 'ulna_width');

sns.boxplot(data = datanew, x = 'bird_type', y = 'femur_length');

sns.boxplot(data = datanew, x = 'bird_type', y = 'femur_width');

sns.boxplot(data = datanew, x = 'bird_type', y = 'tibiotarsus_length');

sns.boxplot(data = datanew, x = 'bird_type', y = 'tibiotarsus_width');

sns.boxplot(data = datanew, x = 'bird_type', y = 'tarsus_length');

sns.boxplot(data = datanew, x = 'bird_type', y = 'tarsus_width');

#Taking numerical columns
matrix = dataset.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=1, square=True, annot=True)

#Setting the value for X and Y
x = dataset[['id']]
y = dataset['bird_type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(x_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(x_test)

print('Accurancy of Logistic Regression (without PCA) on training set: {:.2f}'.format(logregwithoutpca.score(x_train, y_train)))
print('Accurancy of Logistic Regression (without PCA) on testing set: {:.2f}'.format(logregwithoutpca.score(x_test, y_test)))
