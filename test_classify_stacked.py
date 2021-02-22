import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier

# Load training data file
rdf = pd.read_csv('Train.csv')

# Remove rows without required features
rdf = rdf[rdf['reviewText'].notna()]
rdf = rdf[rdf['summary'].notna()]
rdf = rdf[rdf['salesRank'].notna()]
rdf = rdf[rdf['overall'].notna()]
rdf = rdf[rdf['artist'].notna()]
rdf = rdf[rdf['label'].notna()]

rdf['label'] = rdf['label'].map(lambda x: hash(x) % (10 ** 8))

# A product is awesome if its average overall rating is greater than 4.5 stars
product_is_awesome = lambda x: 1 if np.mean(x) > 4.5 else 0

# Group reviews by product so that we have the overall (awesome or not awesome) rating, salesRank, 
# artist, and label for each product
proddf = rdf.groupby('amazon-id').agg({
    'overall': product_is_awesome, 
    'artist': lambda x: x.iloc[0],
    'label': lambda x: x.iloc[0],
    'salesRank': lambda x: x.iloc[0],
})

# An individual review is awesome if its overall rating is 5 stars
review_is_awesome = lambda x: 1 if x == 5 else 0
rdf['awesome'] = rdf['overall'].map(review_is_awesome)

# We want to analyze both text fields as one
rdf['text'] = rdf['reviewText'] + rdf['summary']

# Train and test with 10-fold split
f1s = []
kf = KFold(n_splits=10)
for train_idx, test_idx in kf.split(proddf):
    trainproddf = proddf.iloc[train_idx]
    testproddf = proddf.iloc[test_idx]

    # Aggregate all rows with reviews in the product dfs
    traindf = rdf[rdf['amazon-id'].isin(trainproddf.index)]
    testdf = rdf[rdf['amazon-id'].isin(testproddf.index)]

    # Prepare sentiment analysis data
    X_train = traindf['text']
    X_test = testdf['text']
    y_train = traindf['awesome']

    # Transform text with TfidfVectorizer
    tfv = TfidfVectorizer(ngram_range=(1,2))
    X_train = tfv.fit_transform(X_train, y_train)
    X_test = tfv.transform(X_test)

    # Classify with logistic regression
    lr = LogisticRegression(max_iter=10000, n_jobs=4)
    lr.fit(X_train, y_train)
    testdf['awesomePrediction'] = lr.predict(X_test)

    # Now we have actual and predicted review awesomeness scores (1: 5-star and 0: not 5-star).
    # Average the review scores to get a single value from 0 to 1.
    trainproddf['averageReviewScore'] = traindf.groupby('amazon-id').agg({'awesome': 'mean'})
    testproddf['averageReviewScore'] = testdf.groupby('amazon-id').agg({'awesome': 'mean'})
    
    # FOR BRANDON - trainproddf and testproddf have artist, label, salesRank, and averageSAScore
    # columns as features, and the overall column as ground truth

    
    # Prepare sentiment analysis data
    X_prodtrain = trainproddf[['salesRank','artist','label','averageReviewScore']]
    X_prodtest = testproddf[['salesRank','artist','label','averageReviewScore']]
    y_prodtrain = trainproddf['overall'].values.reshape(-1,1)
    y_prodtest = testproddf['overall'].values.reshape(-1,1)

    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_prodtrain, y_prodtrain)
    predictions = clf.predict(X_prodtest)
    f1s.append(f1_score(y_prodtest, predictions, average='weighted'))

    #print(trainproddf)
    #print(testproddf)

    print(classification_report(testproddf['overall'], predictions))
    f1s.append(f1_score(testproddf['overall'], predictions, average='weighted'))

print(np.asarray(f1s).mean())

    