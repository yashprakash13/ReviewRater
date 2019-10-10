#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 3 21:10:29 2019

@author: costa
"""

import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

data = pd.read_csv('Modified_kindle_reviews.csv') #import modified dataset.

data = data.dropna()
data = data.groupby('sentiment').head(50000).reset_index(drop=True)
data.sentiment.value_counts()


num_class = len(np.unique(data.sentiment.values))
y = data['sentiment'].values

MAX_LENGTH = 500
#The tokenizer function
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.reviewText.values)  
post_seq = tokenizer.texts_to_sequences(data.reviewText.values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)

#split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.2)

vocab_size = len(tokenizer.word_index) + 1

### MODEL 1

inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)

x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()
y_categorical = to_categorical(y_train, num_classes=3)
fitModel = model.fit([X_train], batch_size=128, y = y_categorical, verbose=1, validation_split=0.2, 
          shuffle=True, epochs=5) 

#save the model.
model.save('Model3/kerasModel.h5')

predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
accuracy_score(y_test, predicted)

# LOADING THE MODEL and predicting
review = "This was one of the most wonderful books I've read in months! Simply loved the characters, Stacy was sooo much real to me. I definetely recommend it! "
tokenizer.fit_on_texts(review)
seq = tokenizer.texts_to_sequences(review)
seq_pad = pad_sequences(seq, maxlen = 500)

model = keras.models.load_model('Model3/kerasModel.h5')
pr = model.predict(seq_pad)
pr = np.argmax(pr, axis = 1)


tc = {0:"Neutral", 1: "Positive", 2:"Negative"}
u, indices = np.unique(pr, return_inverse=True)
axis = 0
tc[u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(pr.shape),
                                None, np.max(indices) + 1), axis=axis)]]



""" MODEL 2 #This is not applied in the running model.

inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)

x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=5)


model.load_weights('weights.hdf5')
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
accuracy_score(y_test, predicted)




###                        Got accuracy about ~60% .

"""
