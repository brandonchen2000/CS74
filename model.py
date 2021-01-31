import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Classify each review's text as awesome or not awesome 
def classify_reviews(reviewDf):
    cv = CountVectorizer()

# Train the model
def train():
    # load the review DataFrame and clear NaNs
    reviewDf = pd.read_csv('Train.csv')
    reviewDf = reviewDf.dropna()

    # compute target (awesome or not awesome)
    avg = reviewDf[['amazon-id', 'overall']].groupby('amazon-id').mean()
    reviewDf['awesome'] = reviewDf['amazon-id'].map(lambda x: avg.loc[x, 'overall'] > 4.5)

    # split df into train and test sets for reviewText
    X_train, X_test, y_train, y_test = train_test_split(reviewDf['reviewText'], reviewDf['awesome'], test_size=0.2, random_state=50)
    
    # use sklearn CountVectorizer to process reviewText
    cv = CountVectorizer()
    X_train, X_test = cv.fit_transform(X_train), cv.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    score = mnb.score(X_test, y_test)
    y_pred = mnb.predict(X_test)
    f1 = f1_score(y_pred, y_test)    

    print(score)
    print(f1)

# Deploy the model
def deploy():
    pass

if __name__ == "__main__":
    train()