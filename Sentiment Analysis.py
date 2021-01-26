import pandas as pd
import numpy as np
import sklearn
import nltk
import os
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split


# Do this the first time only to download nltk tools
# nltk.download()

# get training data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'Train.csv')
df = pd.read_csv(my_file)

# building boolean target column aggregating by amazon id
df['target'] = df['amazon-id']
avg = df[['amazon-id', 'overall']].groupby('amazon-id').mean()
def GetTargetBoolean(amazonID):
    # converts the target boolean given amazon ID
    return (avg.loc[amazonID, 'overall'] > 4.5)

df['target'].apply(lambda x: GetTargetBoolean(x))
#print(dataWithTargets)
print(df['target'].head())



def GetReviewStrings():
    # gets all reviews and associated dependent variables from the data
    return df[['reviewText','overall']]

#print(GetReviewStrings(10))

""" for sentence in GetReviewStrings(10):
    print(word_tokenize(sentence)) """


def RemoveStopWords(sentence):
    # remove stop words from a sentence
    stopWords = set(stopwords.words('english'))
    wordTokens = word_tokenize(sentence)
    filteredSentence = []
    for word in wordTokens:
        if word not in stopWords:
            filteredSentence.append(word)
    return filteredSentence

def Stem(sentence):
    # stem all of the words in a sentence
    wordTokens = word_tokenize(sentence)
    stemSentence = []
    for word in wordTokens:
        stemSentence.append(ps.stem(word))
    return stemSentence

""" def WordFrequency(wordTokens): #maybe more useful to do this over the 4.5+ reviews and <4.5 reviews separately
    # returns word frequency distribution in the cleaned words tokens
    for tokens in wordTokens:
        for token in tokens:
            yield token """

def ConvertTokensToDictionary(wordTokens):
    # convert word tokens to dictionary to prepare for NLTK training methods
    for token in wordTokens:
        yield dict([token,True] for token in wordTokens)

def Train():
    # train on a subset of data - currently using 70% to train
    return

""" def Sentiment(sentence):
    # get the sentiment of a string sentence
    return  """


""" def Predict(score):
    # predict star rating given the sentiment score
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

    #nltk.download('punkt')
    #nltk.download('wordnet')
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download()
    return  """


#def Split(p):
    # splits the data into a training data set with p% of the data and the remainder for testing

#def AccuracyChecker(n):
    # used to calculate accuracy of predictions running n distinct trials at a time




