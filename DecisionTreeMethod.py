import json
import gzip
import joblib
import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def DesTree(load):
    data = []
    rawData = gzip.open("Electronics_5.json.gz", 'r')
    for i in rawData:
        data.append(json.loads(i))
    N = 500000
    print("Found ", len(data), "entries. Of this dataset", N, "entries will be used in training.")

    data = data[:N]
    dataFrame = pd.DataFrame.from_records(data)[['overall', 'reviewText']] 
    dataFrame.fillna("", inplace=True)


    dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]')
                                            .sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))
    print("Preprocessing ...")

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
    joblib.dump(tfidf, 'model/tdidf.sav')
    x = tfidf.fit_transform(dataFrame['reviewText'])
    y = dataFrame['overall']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    print("training model ...")

   # nn = MLPClassifier(max_iter=10, random_state=0, early_stopping=True)
    # parameters = {
    #     "alpha": [0.0, 0.001, 1.0],
    #     "activation": ["identity", "logistic", "relu", "tanh"],
    #     "solver":["adam"],
    #     "hidden_layer_sizes":[(50, ), (100, ), (500, ), (1000, )]
    # }
    # grid_search = GridSearchCV(nn, parameters, n_jobs=-1)
    # grid_search.fit(x_train, y_train)

    # print(f'Best score {grid_search.best_score_}')
    
    # print(f'Best params: {grid_search.best_params_}')
    dtc = MLPClassifier(max_iter = 10, random_state = 0, early_stopping = True,hidden_layer_sizes=100, verbose=True)
    
    dtc.fit(x_train, y_train)

    y_pred = dtc.predict(x_test)

    print(classification_report(y_test, y_pred))
    x = 'Lemme tell ya, this has to be one of the greatest mediocre books Ive ever read. It has everything: mundane characters, adequate plot and detailed descriptions of nothing. I whole heartedly recommend this novel to anyone without anything better to do!'
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", dtc.predict(vec)[0])
    # Model testing with positive neutral and negative commentary.
    print("Model testing:\n ")
    x = "I really like this book. It is one of the best I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", dtc.predict(vec)[0])
    x = "I really hate this book. It is one of the worst I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", dtc.predict(vec)[0])
    x = "This book is ok. It is very average."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", dtc.predict(vec)[0])

    print("\n\nSaving the model.")
    joblib.dump(dtc, "model/DTC.sav")