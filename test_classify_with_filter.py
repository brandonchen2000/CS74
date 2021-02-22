import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Load training data file
rdf = pd.read_csv('Train.csv')

# Remove rows without required features
rdf = rdf[rdf['reviewText'].notna()]
rdf = rdf[rdf['summary'].notna()]
rdf = rdf[rdf['overall'].notna()]

# A product is awesome if its average overall rating is greater than 4.5 stars
product_is_awesome = lambda x: 1 if np.mean(x) > 4.5 else 0
proddf = rdf.groupby('amazon-id').agg({'overall': product_is_awesome})

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
    tfv = TfidfVectorizer(ngram_range=(1, 2))
    X_train = tfv.fit_transform(X_train, y_train)
    X_test = tfv.transform(X_test)

    # Classify with logistic regression
    lr = LogisticRegression(max_iter=100, n_jobs=4)
    lr.fit(X_train, y_train)
    testdf['prediction'] = lr.predict(X_test)

    # Products are predicted to be awesome if the average of review predictions is over 80%
    prediction_is_awesome = lambda x: 1 if np.mean(x) > 0.8 else 0
    prodpreddf = testdf.groupby('amazon-id').agg({'prediction': prediction_is_awesome})


    productavg = lambda x: np.mean(x)
    productscore = lambda x: x


    proddf = testdf.groupby('amazon-id').agg({'prediction': product_is_awesome})
    proddf2 = testdf.groupby('amazon-id').agg({'prediction': productavg})

    reviewdf = rdf.groupby('reviewerID', as_index=False).agg({'amazon-id': lambda x: list(x)})

    reviewdf2 = rdf.groupby('reviewerID', as_index=False).agg({'overall': productscore})

    length = len(reviewdf)
    deviation = 0
    count = 0
    reviewers = []

    for index, row in reviewdf.iterrows():
        if len(reviewdf['amazon-id'].iloc[index]) >= 3:
            scores = reviewdf2['prediction'].iloc[index]

            ids = reviewdf['amazon-id'].iloc[index]
            for j in range(len(reviewdf['amazon-id'].iloc[index])):
                score = scores[j]
                id = ids[j]
                avg = proddf2['prediction'].loc[id]
                deviation += ((score - avg) * (score - avg))

            deviation = deviation / len(reviewdf['amazon-id'].iloc[index])

            if (deviation > 3):
                # print(reviewdf2.iloc[i])
                reviewers.append(reviewdf['reviewerID'].iloc[index])
                count += 1
            deviation = 0

    ids = []
    for i in range(len(rdf)):
        # print(rdf['reviewerID'].iloc[i])
        if rdf['reviewerID'].iloc[i] in reviewers:
            print("Found Review")
            ids.append(i)

    print(ids)
    rdf.drop(ids)

    print(classification_report(testproddf['overall'], prodpreddf['prediction']))
    f1s.append(f1_score(testproddf['overall'], prodpreddf['prediction'], average='weighted'))

print(np.asarray(f1s).mean())


if __name__ == "__main__":
    print(ids)
    # print(len(reviewdf))
    # print(proddf)