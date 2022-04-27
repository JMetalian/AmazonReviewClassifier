import gzip
import json
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def LinearSupportVectorClassifier(load):
    data = []
    rawData = gzip.open("Electronics_5.json.gz", 'r') #Kindle review data.
    print("Loading dataset ...")
    for i in rawData:
        data.append(json.loads(i))
    N = 500000
    print("Found ", len(data), "entries. Of this dataset", N, "entries will be used in training.")
    data = data[:N] #Clip the data count

    dataFrame = pd.DataFrame.from_records(data)[['overall', 'reviewText']] #Take "overall" and "review" and create a field for them.
    dataFrame.fillna("", inplace=True) #Not a number values are filled with empty blank string.

    # remove all unwanted chars
    dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]')
                                            .sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
    print("Preprocessing ...")
    x = tfidf.fit_transform(dataFrame['reviewText'])
    y = dataFrame['overall']

    xtrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=0)

    print("training model ...")
    model = LinearSVC(C = 20, class_weight="balanced", verbose=1)

    #If a model exists, use the model.
    if load:
        print("\nLoading previous model:\n")
        model = joblib.load("model/SVC.sav")

    model.fit(xtrain, yTrain) 

    #Predictions
    yPred = model.predict(xTest)

    print(classification_report(yTest, yPred))

    # Model testing with positive neutral and negative commentary.
    print("Model testing:\n ")
    x = "I really like this book. It is one of the best I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])
    x = "I really hate this book. It is one of the worst I have read."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])
    x = "This book is ok. It is very average."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])

    # saving the model weights
    print("\n\nSaving the model.")
    joblib.dump(model, "model/SVC.sav")
