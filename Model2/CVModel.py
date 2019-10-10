#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:01:19 2019

@author: costa
"""

import pandas as pd


data = pd.read_csv('Modified_kindle_reviews.csv')
data.sample(5)
data = data.groupby('sentiment').head(50000).reset_index(drop = True) # Take 50000 reviews of each sentiment to train the model.
data.sentiment.value_counts()
data = data.dropna()
data['sentiment'] = data['sentiment'].astype(str)

#Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['reviewText'], data['sentiment'], test_size=0.2, random_state=5)

#Algorithms used
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

#The model

model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

model.fit(X_train, y_train) #Train the model

pred = model.predict(X_test)

model.classes_ #These are the model classes.

from sklearn.metrics import  accuracy_score
accuracy_score(y_test, pred)

#save the model
from sklearn.externals import joblib
joblib.dump(model, 'Model2/CountVectorizerModel.pkl', compress=1)


### Load the model
from sklearn.externals import joblib
model = joblib.load('Model2/CountVectorizerModel.pkl')

review1 = "This may not be a fair review of the book, as I did not finish.  I read perhaps a quarter and finally gave it up, thinking of all the other better books I had waiting.  It simply did not seem readable after fifty pages or so. "
review = "This was one of the most wonderful books I've read in months! Simply loved the characters, Stacy was sooo much real to me. I definetely recommend it! "
review2 = "Not too bad, an intro-short-story for some bigger upcoming novel. It's timothy Zahn, what did you expect from the master of Science Fiction."
pr = model.predict([review2])[0] # This is where we can predict.
if pr == 1.0:
    print("Positive");
elif pr == 0.0:
    print("Neutral")
else:
    print("Negative")
        


