import gzip
import json
import pandas as pd
import joblib
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def LinearSupportVectorClassifier():
    data = []
    rawData = gzip.open("Electronics_5.json.gz", 'r') #Kindle review data.
    print("Loading the dataset")
    for i in rawData:
        data.append(json.loads(i))
    N = 100000 # Limit the data that will be used. Otherwise memory errors occur.
    print("Found ", len(data), "data.", N, "data will be used in training.")
    data = data[:N] #Clip the data count

    dataFrame = pd.DataFrame.from_records(data)[['overall', 'reviewText']] #Take "overall" and "review" and create a field for them.
    dataFrame.fillna("", inplace=True) #Not a number values are filled with blank string.

    #Remove all unwanted chars
    dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
    print("Preprocessing the data")
    xData = tfidf.fit_transform(dataFrame['reviewText'])
    yData = dataFrame['overall']

    xtrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.25)

    print("Training model")
    model = LinearSVC(C = 2.0, class_weight="balanced", verbose=0)

    model.fit(xtrain, yTrain) 

    #Predictions
    yPrediction = model.predict(xTest)

    print(classification_report(yTest, yPrediction))
    print("Model evaluation with test text:\n ")
    
    
    x = "I hated it. It is one of the worst I have seen."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])
    x = "It is fine. I can say it is fifty-fify."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])
    x = "I really like it. It is one of the best I have ever seen. Also, I was expecting it."
    vec = tfidf.transform([x])
    print("'" + x + "' got the rating: ", model.predict(vec)[0])

LinearSupportVectorClassifier()