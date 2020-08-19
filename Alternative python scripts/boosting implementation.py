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


#reading in the training data
df = pd.read_csv("Datasets/boosting_train.csv") 
df.head() 


# In[ ]:


features = df.drop(columns=['Letter Value'])
features.head()


# In[ ]:


#attempt to implement boosting following adaboost algorithm
x_train, x_test, y_train, y_test = train_test_split(features, df['Letter Value'], test_size = 0.25)

training_sample = pd.DataFrame(data = x_train, columns = ['x-box','y-box','width','height','onpix','x-bar','y-bar','x2bar','y2bar','xybar',
                                  'x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx'])
training_label = pd.DataFrame(data = y_train, columns = ['Letter Value'])
tr_sample_alone = training_sample

training_sample.insert(0, "Letter Value", training_label) 


# In[ ]:


prototype = []
p_features = []
p_label = []
current_error = 100
iteration = 1

#adjust prototype size by changing the value of the while check
while len(prototype) <= 10:
    print('Length at the beginning:', len(prototype))
    print('Current error:', current_error)
    print('Current Iteration:', iteration)

    #1.select n points at random
    random = training_sample.sample(n=10, replace = False)
    training_error = []
    
    #2a. find the errors of each of these points with the prototype set
    for ind in random.index:
        row = random.loc[ind]
        
        #extract each sample individualy from n points and test with Prototype set
        feature = [row['x-box'],row['y-box'],row['width'],row['height'],row['onpix'],row['x-bar'],row['y-bar'],row['x2bar'],
                      row['y2bar'],row['xybar'],row['x2ybr'],row['xy2br'],row['x-ege'],row['xegvy'],row['y-ege'],row['yegvx']]
        label = row['Letter Value']
        
        p_features.append(feature)
        p_label.append(label)
    
        p_features_pd = pd.DataFrame(data = p_features, columns = ['x-box','y-box','width','height','onpix','x-bar','y-bar',
                                     'x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx'])
        p_label_pd = pd.DataFrame(data = p_label, columns = ['Letter Value'])
        
        knn = KNeighborsClassifier(n_neighbors = 1) 
        knn.fit(p_features_pd, p_label_pd.values.ravel())
        
        training_error.append(1 - (knn.score(x_train, y_train)))
        
        p_label.pop()
        p_features.pop()
        
        
    #2b. Find the point with the lowest error    
    iteration += 1
    a = min(training_error) 
    print ('Lowest training error out of ten random points is: ',a)

    #check for an actual decrease in the prototype error
    if current_error < a and iteration <= 500 :
        continue

    index_a = training_error.index(a)
    
    #to find its position in 'random'
    random.insert(0, "index1", random.index)
    random_array = random.to_numpy()
    point = random_array[index_a]
    print ('Attribute value of selected candidate: ', point)
    
    #3. add this point to set P    
    #3a first check for duplicates
    for i in range(0,(len(prototype))):
        if (point[0] == ((prototype[i])[0])):
            continue
    
    #3b then append to set p
    prototype.append(point)
    p_label.append(point[1])
    p_features.append(point[2:])
    print('Round completed:', len(prototype))
    print()

    current_error = a
    iteration = 1
    
    #4. repeat the process at another 10 points, this time adding points that have the lowest decrease in error
    #5. stop when P is size n
    #this is one iteration


# In[ ]:


#to find a way of getting these points as training data vs x_test
p_boost = pd.DataFrame(data = prototype, columns = ['index','Letter Value','x-box','y-box','width','height','onpix','x-bar','y-bar','x2bar','y2bar','xybar',
                                  'x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx'])
p_boost = p_boost.set_index('index')
p_boost


# In[ ]:


features_boost = p_boost.drop(columns=['Letter Value'])
features_boost.head()


# In[ ]:


label_boost = p_boost['Letter Value']
label_boost


# In[ ]:


#you're going to train this with validation(test) data
boosted_train_error = []

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(features_boost, label_boost)
        
#boosted_train_error.append(1 - (knn.score(x_train, y_train)))
boosted_train_error = 1 - (knn.score(x_train, y_train))
print('Train error:',boosted_train_error)
boosted_val_error = 1 - (knn.score(x_test, y_test))
print('Validation error:',boosted_val_error)


# In[ ]:


#export this prototype set
p_boost.to_csv("boosted data.csv")


# In[ ]:


#train boosted set on final test
df_t = pd.read_csv("Datasets/boosting_test.csv") 
features_t = df_t.drop(columns=['Letter Value'])
labels_t =  df_t['Letter Value']

pred = knn.predict(features_t)
true_test_error = 1 - (knn.score(features_t, labels_t))

true_test_error


# In[ ]:


#exporting actual values of prediction and label
export =pd.DataFrame(data={"Actual Labels":labels_t,"Predicted Labels": pred})
export.to_csv("actual vs predicted labes.csv")


# In[ ]:




