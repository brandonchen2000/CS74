import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Load training data file
rdf = pd.read_csv('Train.csv')

# Remove rows without required features
rdf = rdf[rdf['reviewText'].notna()]
rdf = rdf[rdf['summary'].notna()]
rdf = rdf[rdf['overall'].notna()]

# A review is awesome if its overall rating is 5 stars
rdf['awesome'] = rdf['overall'] == 5

# We want to analyze both text fields as one
rdf['text'] = rdf['reviewText'] + rdf['summary']

# Train and test with 10-fold split
f1s = []
kf = KFold(n_splits=10)
for train_idx, test_idx in kf.split(rdf):
    traindf = rdf.iloc[train_idx]
    testdf = rdf.iloc[test_idx]

    # Prepare sentiment analysis data
    X_train = traindf['text']
    X_test = testdf['text']
    y_train = traindf['awesome']
    y_test = testdf['awesome']

    # Transform text with CountVectorizer
    cv = CountVectorizer(ngram_range=(1,2), stop_words='english')
    X_train = cv.fit_transform(X_train, y_train)
    X_test = cv.transform(X_test)

    # Classify with logistic regression
    lr = LogisticRegression(max_iter=100, n_jobs=-1)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    f1s.append(f1_score(y_test, predictions, average='weighted'))

print(np.asarray(f1s).mean())

