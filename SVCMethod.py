#Uğur Can Kozan

import gzip
import json
import pandas as pd
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



data = []
rawData = gzip.open("Electronics_5.json.gz", 'r') #Electronics review data.
print("Loading the dataset")
for i in rawData:
    data.append(json.loads(i))
N = 100000 # Limit the data that will be used. Otherwise memory errors occur.

data = data[:N] #Clip the data count

dataFrame = pd.DataFrame.from_records(data)[['overall', 'reviewText']] #Take "overall" and "review" and create a field for them.
dataFrame.fillna("", inplace=True) #Not a number values are filled with blank string.
#Remove all unwanted chars
dataFrame['reviewText'] = dataFrame['reviewText'].map(lambda a: re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', a.lower())))

#Convert raw document to TFID matrix.
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 5), analyzer='char')
print("Preprocessing the data.")
xData = tfidf.fit_transform(dataFrame['reviewText'])
yData = dataFrame['overall']

#Train and tes split
xtrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.25)

#Training of model, adjust hyper-parameters of the SVM.
print("Model is currently in training.")
model = LinearSVC(C = 2.0, class_weight="balanced", verbose=0)
model.fit(xtrain, yTrain) 

#Predictions of provided additional input.
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