#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:26:24 2019

@author: costa
"""
### Predicting Positve, Negative or Neutral Review.
### 1, 2 :- Negative.
### 3:- Neutral.
### 4, 5:- Positive Review.
import pandas as pd
import numpy as np  
from gensim.models import Word2Vec, word2vec
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import re

# This is the old dataset.
data = pd.read_csv('kindle-reviews/kindle_reviews.csv')
# data = data.sample(n = 70000, random_state = 37)
data = data.iloc[:,3:5]
data = data.dropna()
data.overall.value_counts()

positive_rating = 1
negative_rating = -1
neutral_rating = 0

# Convert the ratings to sentiments: +1, 0 or -1.
for i in range(0, len(data)):
    
    if i % 1000 == 0:
        print("Review: ", i)
    
    if data.iloc[i]['overall'] >= 4:
        data.loc[i, 'sentiment'] = positive_rating
    
    elif data.iloc[i]['overall'] == 3:
        data.loc[i, 'sentiment'] = neutral_rating
    
    else:
        data.loc[i, 'sentiment'] = negative_rating

# Save the modified dataset as csv file to save time in the future.
# data.to_csv("Modified_kindle_reviews.csv", index = False) # Don't execute this line again.



# Start here
data = pd.read_csv('Modified_kindle_reviews.csv')


data.sentiment.value_counts()

train = data.groupby('sentiment').head(40000).reset_index(drop = True)
train.sentiment.value_counts()

test =  data.groupby('sentiment').tail(10000).reset_index(drop = True)
test.sentiment.value_counts()

y_test = test.iloc[:, 2]
y_test = np.asarray(y_test)

from bs4 import BeautifulSoup

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review = str(review)
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    review = str(review)
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for review in train["reviewText"]:
    sentences += review_to_sentences(review, tokenizer)

print ("Parsing sentences from testing set")
for review in test["reviewText"]:
    sentences += review_to_sentences(review, tokenizer)

### MODEL TRAINING

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context_newWithSentimentColumn"
model.save(model_name)

model.doesnt_match("man woman child kitchen".split())

model.most_similar("man")

model.most_similar("awful")

model.most_similar("dirty")

model.similarity('woman','man')   

model.most_similar(positive=['woman','king'],negative=['man'],topn=1)

### LOADING THE Already TRAINED MODEL
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context_newWithSentimentColumn")

## WordEmbedding Model has been successfully trained!
#####################
"""
X_train = data.loc[: 80000]['reviewText'].values
y_train = data.loc[:80000]['overall'].values

X_test = data.loc[80001:]['reviewText'].values
y_test = data.loc[80001:]['overall'].values
"""

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Converting it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec 


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocating a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print ("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, 
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. 
train = train.fillna(0)
test = test.fillna(0)


clean_train_reviews = []
for review in train["reviewText"]:
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
trainDataVecs = np.nan_to_num(trainDataVecs)

print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["reviewText"]:
    clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
testDataVecs = np.nan_to_num(testDataVecs)


# Fitting a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print ("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["sentiment"] )

# saving the model
import pickle
f = open('Model/RF_W2V_Classifier.pickle', 'wb')
pickle.dump(forest, f)
f.close()
# Test & extract results 
result = forest.predict( testDataVecs )

# get accuracy score.
from sklearn.metrics import accuracy_score
accuracy   =  accuracy_score(y_test, result)




## 55% Acc.





### Predicting a single review

def prediction(review):
    from gensim.models import Word2Vec
    model = Word2Vec.load("300features_40minwords_10context_newWithSentimentColumn")

    import pickle
    f = open('Model/RF_W2V_Classifier.pickle', 'rb')
    forest = pickle.load(f)
    f.close()
    
    import sys
    sys.path.append('/home/costa/Desktop/NewFolder/Python MP/Model')

    import clean_text_vector as ctv

    review_vec = ctv.getVec(review, model, 300)
    print(forest.predict(review_vec)) 


#review1 = "This may not be a fair review of the book, as I did not finish.  I read perhaps a quarter and finally gave it up, thinking of all the other better books I had waiting.  It simply did not seem readable after fifty pages or so. "
    #review = "This was one of the most wonderful books I've read in months! Simply loved the characters, Stacy was sooo much real to me. I definetely recommend it! "
    #review2 = "Not too bad, an intro-short-story for some bigger upcoming novel. It's timothy Zahn, what did you expect from the master of Science Fiction."

# prediction("This may not be a fair review of the book, as I did not finish.  I read perhaps a quarter and finally gave it up, thinking of all the other better books I had waiting.  It simply did not seem readable after fifty pages or so.")
