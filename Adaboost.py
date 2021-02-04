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
    X = rdf[['salesRank']]
    Y = rdf[['awesome']]
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, Y.values.ravel(), cv=10)
    return (scores.mean())

def helpfulQuantityBooster():
    # trains based on quantity of helpful reviews
    X = rdf[['totalHelpful']]
    Y = rdf[['awesome']]
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, Y.values.ravel(), cv=10)
    return scores.mean()

def helpfulQualityBooster():
    # trains based on percentage of helpful reviews
    X = rdf[['percentHelpful']]
    Y = rdf[['awesome']]
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, Y.values.ravel(), cv=10)
    return scores.mean()

def allBooster():
    X = rdf[['totalHelpful','percentHelpful','salesRank']]
    Y = rdf[['awesome']]
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, Y.values.ravel(), cv=10)
    return scores.mean()

if __name__ == "__main__":
    print(salesRankBooster())
    print(helpfulQuantityBooster())
    print(helpfulQualityBooster())
    print(allBooster())

