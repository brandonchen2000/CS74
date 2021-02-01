import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer # TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Train the model
def train():
    # load the review DataFrame and clear NaNs
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'Train.csv')
    reviewDf = pd.read_csv(my_file)
    print(reviewDf.shape)
    reviewDf = reviewDf[reviewDf['reviewText'].notna()]
    print(reviewDf.shape)
    reviewDf = reviewDf[reviewDf['summary'].notna()]
    print(reviewDf.shape)


    # compute target (awesome or not awesome)
    avg = reviewDf[['amazon-id', 'overall']].groupby('amazon-id').mean()
    reviewDf['awesome'] = reviewDf['amazon-id'].map(lambda x: avg.loc[x, 'overall'] > 4.5)

    # split reviewDf into X (reviewText) and y (awesome)
    X = reviewDf['reviewText'].to_numpy()
    y = reviewDf['awesome'].to_numpy()

    # train and test with 10-fold cross validation
    results = pd.DataFrame(columns=['score', 'f1'])
    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    
        # use sklearn CountVectorizer to process reviewText
        cv = CountVectorizer()
        X_train, X_test = cv.fit_transform(X_train), cv.transform(X_test)

        # fit, predict, and score
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        score = mnb.score(X_test, y_test)
        y_pred = mnb.predict(X_test)
        f1 = f1_score(y_pred, y_test)

        results.append({'score': score, 'f1': f1}, ignore_index=True)

    print(score)
    print(f1)

# Deploy the model
def deploy():
    pass

if __name__ == "__main__":
    train()