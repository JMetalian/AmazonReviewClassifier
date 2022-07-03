#Alisa Sinkevich

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
import gzip
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Get stopwords from nltk library
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Globals
vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 64
max_length = 240
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
N = 50000

data = []
raw_data = gzip.open('/tmp/data.json.gz', 'r')

# Grab the first N reviews for training
for i,d in enumerate(raw_data):
    if i == N:
        break
    else:
        data.append(json.loads(d))

dataFrame = pd.DataFrame.from_records(data)[['overall', 'reviewText']] #Take "overall" and "review" and create a field for them.
dataFrame.fillna("", inplace=True) #Not a number values are filled with empty blank string.

# remove all unwanted chars
dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]')
                                        .sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))

# remove stop words
dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda x: ' '.join(filter(lambda a: a not in STOPWORDS, x.split(' '))))

# Tokenize reviews and pad/truncate to max length
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(dataFrame['reviewText'])
word_index = tokenizer.word_index

tokenized_data = tokenizer.texts_to_sequences(dataFrame['reviewText'])
tokenized_data = pad_sequences(tokenized_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#split training/test sets
xtrain, xTest, yTrain, yTest = train_test_split(tokenized_data, dataFrame['overall'], test_size=0.20, random_state=0)

# Build the model
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(6, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

# Train!
m = model.fit(xtrain, yTrain, epochs=10, validation_data=(xTest, yTest), verbose=2)