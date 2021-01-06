# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ____
# ### K Nearest Neighbors and scaling in Python 
# #### Author: Alex Fields
# 
# Using classified data that has no inherit meaning to the user, we are to decipher using KNN, if we can predict the target class.
# This is to showcase the KNN and StandardScaler methods for my portfolio in Data Science

# %%
#importing necessary libraries for data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
#reading in the data
df = pd.read_csv('Classified Data', index_col=0)
df.head()


# %%
#highest value in series
df.idxmax()


# %%
#lowest value in series
df.idxmin()


# %%
from sklearn.preprocessing import StandardScaler


# %%
#initializing a scaler object
scaler = StandardScaler()


# %%
#fitting the model with all features
scaler.fit(df.drop('TARGET CLASS',axis=1))


# %%
#scales all feature variables by removed y variable
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# %%
scaled_features


# %%
df_feature = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feature


# %%
#model selection
from sklearn.model_selection import train_test_split

#Feature vs predictor selection
X = df_feature
y = df['TARGET CLASS']

#train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# %%
from sklearn.neighbors import KNeighborsClassifier


# %%
#initiating KNN variable to analysis
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train) #fits model


# %%
#predicts neighbor from test set 
pred = knn.predict(X_test)


# %%
from sklearn.metrics import classification_report


# %%
#shows classification matrix of success
print(classification_report(y_test, pred))


# %%
#showing what are error rate is compared to 40 possible K's
error_rate = []

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# %%
#elbow method for finding optimal K number
plt.figure(figsize=(15,10))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor = 'red', markersize = 10)
plt.title=('Error Rate vs K Value')
plt.xlabel=('K')
plt.ylabel=('Error Rate')


# %%
#refitting model for new K to see if raises scores
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))


# %%
#another method for finding lowest K number
x = list(range(1,40))
z = error_rate
predicted_K = dict(zip(x,z))

#initializing all as pandas DF's
df1=pd.DataFrame(x,columns=["Values"])
df2=pd.DataFrame(z,columns=["Values"])

predicted_K=pd.concat((df1,df2))

print(predicted_K)


# %%
#will show the absolute lowest min value for K's
predicted_K['Values'].min() 


# %%
#input values filter of less than or equal to min value
predicted_K.query('Values <= 0.043333333333333335')

# index 33 has the lowest K


