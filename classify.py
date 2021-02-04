import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load training, testing data files
rdf = pd.read_csv('Train.csv')
trdf = pd.read_csv('Test.csv')

# Remove rows without required features
rdf = rdf[rdf['reviewText'].notna()]
rdf = rdf[rdf['summary'].notna()]
rdf = rdf[rdf['overall'].notna()]
trdf = trdf[trdf['reviewText'].notna()]
trdf = trdf[trdf['summary'].notna()]
# `overall` not available in test data

# An individual review is awesome if its overall rating is 5 stars
review_is_awesome = lambda x: 1 if x == 5 else 0
rdf['awesome'] = rdf['overall'].map(review_is_awesome)

# We want to analyze both text fields as one
rdf['text'] = rdf['reviewText'] + rdf['summary']
trdf['text'] = trdf['reviewText'] + trdf['summary']

# Train

# Filter training data
X_train = rdf['text']
y_train = rdf['awesome']

# Transform text with TfidfVectorizer
tfv = TfidfVectorizer(ngram_range=(1,2))
X_train = tfv.fit_transform(X_train, y_train)

# Classify reviews with logistic regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

# Test

# Filter testing data
X_test = trdf['text']

# Transform and classify reviews
X_test = tfv.transform(X_test)
trdf['prediction'] = lr.predict(X_test)

# Products are predicted to be awesome if the average of review predictions is over 80%
prediction_is_awesome = lambda x: 1 if np.mean(x) > 0.8 else 0
prodpreddf = trdf.groupby('amazon-id').agg({'prediction': prediction_is_awesome})
prodpreddf['Awesome'] = prodpreddf['prediction']

# Save prediction results
output = prodpreddf.drop(columns=['prediction'])
output.to_csv('Product_Predictions.csv')