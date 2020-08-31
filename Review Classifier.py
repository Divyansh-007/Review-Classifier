# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:48:19 2020

@author: Dell
"""

# Importing all the libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Importing the data
data = pd.read_csv('yelp.csv')

# Exploring the data
data.head()
data.info()
data.describe()

# Adding a new column called "text length" 
# which is the number of words in the text column.
data['text length'] = data['text'].apply(len)

# Creating a dataframe called data_class that contains the columns of data dataframe 
# but for only the 1 or 5 star reviews.
data_class = data[(data.stars==1)|(data.stars==5)]

# Feature and Target Variables
X = data_class['text']
y = data_class['stars']

# Creating a Count Vectorizer Object
cv = CountVectorizer()
X = cv.fit_transform(X)

# Spliting the dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

# Creating an instance of the estimator
nb = MultinomialNB()
nb.fit(X_train,y_train)

# Prediction and Evaluation
prediction = nb.predict(X_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

# Now we will try to include TF-IDF to this process using a pipline and perform text processing.
# Pipeline is a way to arrange the steps so as we don't have to perform them manually again and again.
pipeline = Pipeline([
    ('bag_of_words', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Train Test Split again
X = data_class['text']
y = data_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)


# May take some time
pipeline.fit(X_train,y_train)

# Prediction and Evaluation again
prediction = pipeline.predict(X_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))
