import numpy as np
import pandas as pd
import math
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

product_is_awesome = lambda x: 1 if np.mean(x) > 4.5 else 0
productavg = lambda x: np.mean(x)
productscore = lambda x: x


proddf = rdf.groupby('amazon-id').agg({'overall': product_is_awesome})
proddf2 = rdf.groupby('amazon-id').agg({'overall': productavg})






reviewdf = rdf.groupby('reviewerID').agg({'amazon-id': lambda x: list(x)})
reviewdf2 = rdf.groupby('reviewerID').agg({'overall': productscore})



length = len(reviewdf)
deviation = 0
count = 0
for i in range(length):
   if len(reviewdf['amazon-id'].iloc[i])>=3:
        scores = reviewdf2['overall'].iloc[i]
        ids = reviewdf['amazon-id'].iloc[i]
        for j in range(len(reviewdf['amazon-id'].iloc[i])):
            score = scores[j]
            id = ids[j]
            avg = proddf2['overall'].loc[id]
            deviation+= ((score-avg)*(score-avg))

        deviation = deviation/len(reviewdf['amazon-id'].iloc[i])

        if(deviation>3):
            print(reviewdf2.iloc[i])
            count +=1
        deviation = 0
print(count)


# print(reviewdf.head())

if __name__ == "__main__":
    print(reviewdf2)
    # print(len(reviewdf))
    # print(proddf)