import numpy as np
from numpy import mean, std
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Load training data file
rdf = pd.read_csv('Train.csv')
# Remove rows without required features
rdf = rdf[rdf['helpful'].notna()]
rdf = rdf[rdf['salesRank'].notna()]
rdf = rdf[rdf['overall'].notna()]

# A review is awesome if its overall rating is 5 stars
review_is_awesome = lambda x: 1 if x == 5 else 0
rdf['awesome'] = rdf['overall'].map(review_is_awesome)

# Number of total reviews that are helpful
total_helpful = lambda x: int(x[x.find(',')+2:-1])
rdf['totalHelpful'] = rdf['helpful'].map(total_helpful)

# Percentage of total reviews that are helpful
percent_helpful = lambda x: 0.5 if int(x[x.find(',')+2:-1]) == 0 else int(x[1:x.find(',')])/int(x[x.find(',')+2:-1])
rdf['percentHelpful'] = rdf['helpful'].map(percent_helpful)


def salesRankBooster():
    # trains based on sales rank
    f1s = []
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(rdf):
        traindf = rdf.iloc[train_idx]
        testdf = rdf.iloc[test_idx]

        X_train = traindf['salesRank'].values.reshape(-1,1)
        X_test = testdf['salesRank'].values.reshape(-1,1)
        y_train = traindf['awesome'].values.reshape(-1,1)
        y_test = testdf['awesome'].values.reshape(-1,1)

        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1s.append(f1_score(y_test, predictions, average='weighted'))
    return np.asarray(f1s).mean()

def helpfulQuantityBooster():
    # trains based on quantity of helpful reviews
    f1s = []
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(rdf):
        traindf = rdf.iloc[train_idx]
        testdf = rdf.iloc[test_idx]

        # Prepare sentiment analysis data
        X_train = traindf['totalHelpful'].values.reshape(-1,1)
        X_test = testdf['totalHelpful'].values.reshape(-1,1)
        y_train = traindf['awesome'].values.reshape(-1,1)
        y_test = testdf['awesome'].values.reshape(-1,1)

        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1s.append(f1_score(y_test, predictions, average='weighted'))
    return np.asarray(f1s).mean()

def helpfulQualityBooster():
    # trains based on percentage of helpful reviews
    f1s = []
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(rdf):
        traindf = rdf.iloc[train_idx]
        testdf = rdf.iloc[test_idx]

        # Prepare sentiment analysis data
        X_train = traindf['percentHelpful'].values.reshape(-1,1)
        X_test = testdf['percentHelpful'].values.reshape(-1,1)
        y_train = traindf['awesome'].values.reshape(-1,1)
        y_test = testdf['awesome'].values.reshape(-1,1)

        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1s.append(f1_score(y_test, predictions, average='weighted'))
    return np.asarray(f1s).mean()

def allBooster():
    #trains based on all 3 features
    f1s = []
    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(rdf):
        traindf = rdf.iloc[train_idx]
        testdf = rdf.iloc[test_idx]

        # Prepare sentiment analysis data
        X_train = traindf[['totalHelpful','percentHelpful','salesRank']]
        X_test = testdf[['totalHelpful','percentHelpful','salesRank']]
        y_train = traindf['awesome'].values.reshape(-1,1)
        y_test = testdf['awesome'].values.reshape(-1,1)

        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        f1s.append(f1_score(y_test, predictions, average='weighted'))
    return np.asarray(f1s).mean()


if __name__ == "__main__":
    print(salesRankBooster())
    print(helpfulQuantityBooster())
    print(helpfulQualityBooster())
    print(allBooster())

