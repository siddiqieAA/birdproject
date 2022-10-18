import numpy as np
import pandas as pd
import streamlit as st
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

st.write(data=pd.read_csv('bird.csv'))

st.pyplot(data)

st.write(dcopy=data.copy())

st.pyplot(dcopy)

st.write(dcopy.shape)

st.write(dcopy.columns)

st.write(dcopy.dtypes)






