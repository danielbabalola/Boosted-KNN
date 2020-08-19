#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import preprocessing


# In[ ]:


#to read in the training set
df = pd.read_csv("Datasets/letter_train.csv") 
df.head() 


# In[ ]:


features = df.drop(columns=['Letter Value'])


# In[ ]:


features.head()


# In[ ]:


#to split the training set into training and validation
X_train, X_test, y_train, y_test = train_test_split(features, df['Letter Value'], test_size = 0.25)


# In[ ]:


#plotting training and test error as K grows. Adjust the range to determine k
train_error = [] 
test_error = []
  
# Will take some time 
for i in range(1, 11): 
      
    knn = KNeighborsClassifier(n_neighbors = i) 
    knn.fit(X_train, y_train) 
    train_error.append(1 - (knn.score(X_train, y_train)))
    test_error.append(1 - (knn.score(X_test, y_test))) 
  
plt.figure(figsize =(20,12)) 
plt.plot(range(1, 11), train_error, label = 'Training Error', 
         color ='blue', linestyle ='dashed', marker ='o', markerfacecolor ='red', markersize = 10) 
plt.plot(range(1, 11), test_error, label = 'Validation Error', 
         color ='green', linestyle ='dashed', marker ='x', markerfacecolor ='red', markersize = 10) 

plt.legend()
plt.title('Training Error vs. Validation Error') 
plt.xlabel('k-value') 
plt.ylabel('Error') 

#to export actual value
df3=pd.DataFrame(data={"Training Error":train_error,"Test Error": test_error})
df3.to_csv("new error as k grows.csv")


# In[ ]:


#plotting training and test error at k=3 as n grows
train_error = [] 
test_error = []

df1 = df.sample(n=10, replace = False, random_state=1)
f1 = df1.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f1, df1['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df3 = df.sample(n=100, replace = False, random_state=3)
f3 = df3.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f3, df3['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df4 = df.sample(n=200, replace = False, random_state=4)
f4 = df4.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f4, df4['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df5 = df.sample(n=500, replace = False, random_state=5)
f5 = df5.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f5, df5['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df6 = df.sample(n=1000, replace = False, random_state=6)
f6 = df6.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f6, df6['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df7 = df.sample(n=5000, replace = False, random_state=7)
f7 = df7.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f7, df7['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 

df8 = df.sample(n=10000, replace = False, random_state=8)
f8 = df8.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f8, df8['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


df9 = df.sample(n=12000, replace = False, random_state=9)
f9 = df9.drop(columns=['Letter Value'])

X_train, X_test, y_train, y_test = train_test_split(f9, df9['Letter Value'], test_size = 0.25) #random_state=
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
train_error.append(1 - (knn.score(X_train, y_train)))
test_error.append(1 - (knn.score(X_test, y_test))) 


plt.figure(figsize =(20, 12)) 
plt.plot([10,100,200,500,1000,5000,10000,12000],test_error, label = 'Validation Error', 
         color ='blue', linestyle ='dashed', marker ='o', markerfacecolor ='red', markersize = 10) 
plt.plot([10,100,200,500,1000,5000,10000,12000],train_error, label = 'Training Error', 
         color ='green', linestyle ='dashed', marker ='x', markerfacecolor ='red', markersize = 10) 

plt.legend()
plt.title('Training Error vs. Validation Error') 
plt.xlim(0, 12000)
plt.xlabel('n') 
plt.ylabel('Error')  

#to export actual value
df3=pd.DataFrame(data={"Training Error": train_error,"Test Error": test_error})
df3.to_csv("new error as n grows.csv")


# In[ ]:


#one time training on test data
df_t = pd.read_csv("Datasets/letter_test.csv") 
features_t = df_t.drop(columns=['Letter Value'])
labels_t =  df_t['Letter Value']

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train) 
pred = knn.predict(features_t)
true_test_error = 1 - (knn.score(features_t, labels_t))

true_test_error


# In[ ]:


#exporting actual values of prediction and label
export =pd.DataFrame(data={"Actual Labels":labels_t,"Predicted Labels": pred})
export.to_csv("actual vs predicted labes.csv")

