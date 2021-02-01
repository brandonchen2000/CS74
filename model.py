import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Trains the review classifier on reviewText and summary
def train_review_classifier(train_df):
    # merge reviewText and summary into one string
    text = train_df['reviewText'] + train_df['summary']
    
    # vectorize the text with TFIDF, 1- and 2-grams and english stop words
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(text)

    # train
    y = train_df['awesome']
    mnb = MultinomialNB()
    mnb.fit(X, y)

    # save vectorizer and mnb
    pickle.dump(vectorizer, open('review_vectorizer.pickle', 'wb'))
    pickle.dump(mnb, open('review_mnb.pickle', 'wb'))

# Classifies individual reviews as awesome or not awesome
def classify_reviews(review_df):
    # merge reviewText and summary into one feature
    text = review_df['reviewText'] + review_df['summary']

    # load vectorizer and mnb
    vectorizer = pickle.load(open('review_vectorizer.pickle', 'rb'))
    mnb = pickle.load(open('review_mnb.pickle', 'rb'))

    # vectorize the text with TFIDF, 1- and 2-grams and english stop words
    X = vectorizer.transform(text)

    # classify
    classifications= mnb.predict(X)

    return classifications

# Train the model
def train():
    # load the review DataFrame and clear NaNs
    rdf = pd.read_csv('Train.csv')
    rdf = rdf.dropna()

    # compute target (awesome or not awesome)
    avg = rdf[['amazon-id', 'overall']].groupby('amazon-id', as_index=False).mean()
    avg['awesome'] = avg['overall'] > 4.5
    rdf['awesome'] = rdf['amazon-id'].map(lambda x: avg.loc[avg['amazon-id'] == x, 'awesome'].item())

    # train and test with 10-fold cross validation
    results = []
    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(avg, avg['awesome']):
        train_products = avg.iloc[train_idx]
        test_products = avg.iloc[test_idx]

        train_df = rdf.loc[rdf['amazon-id'].isin(train_products['amazon-id'])]
        test_df = rdf.loc[rdf['amazon-id'].isin(test_products['amazon-id'])]
    
        # train
        train_review_classifier(train_df)

        # test
        classifications = classify_reviews(test_df)
        test_df['prediction'] = classifications

        # score

        # need to average predictions for reviews
        classified_products = test_df[['amazon-id', 'prediction']].groupby('amazon-id').mean(numeric_only=False)
        test_products['prediction'] = test_products['amazon-id'].map(lambda x: classified_products.loc[x, 'prediction'].item())
        test_products['prediction'] = test_products['prediction'] > 0.95

        f1 = f1_score(test_products['prediction'], test_products['awesome'])

        results.append(f1)

    print(np.asarray(results).mean())

# Deploy the model
def deploy():
    pass

if __name__ == "__main__":
    train()