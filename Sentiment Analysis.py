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
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Do this the first time only to download nltk tools
# nltk.download()

# get training data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'Train.csv')
df = pd.read_csv(my_file)
#print(df.shape)

#delete rows that have NaN fields for features used
df = df[df['reviewText'].notna()]
#print(df.shape)



def GetReviewStrings():
    # gets all reviews and associated dependent variables from the data
    return df[['reviewText', 'overall', 'target']]

#print(GetReviewStrings(10))

""" for sentence in GetReviewStrings(10):
    print(word_tokenize(sentence)) """


def RemoveStopWords(sentence):
    # remove stop words from a sentence
    stopWords = set(stopwords.words('english'))
    wordTokens = word_tokenize(sentence)
    filteredSentence = [w for w in wordTokens if not w in stopWords]
    filteredSentence = TreebankWordDetokenizer().detokenize(filteredSentence)
    return filteredSentence
"""     for word in wordTokens:
        if word not in stopWords:
            filteredSentence.append(word) """


def Stem(sentence):
    # stem all of the words in a sentence
    wordTokens = word_tokenize(sentence)
    stemSentence = []
    for word in wordTokens:
        stemSentence.append(ps.stem(word))
    stemSentence = TreebankWordDetokenizer().detokenize(stemSentence)
    return stemSentence

""" def WordFrequency(wordTokens): #maybe more useful to do this over the 4.5+ reviews and <4.5 reviews separately
    # returns word frequency distribution in the cleaned words tokens
    for tokens in wordTokens:
        for token in tokens:
            yield token """

""" def ConvertTokensToDictionary(wordTokens):
    # convert word tokens to dictionary to prepare for NLTK training methods
    for token in wordTokens:
        yield dict([token,True] for token in wordTokens) """

def CleanData():
    # adds target column, removes stop words, and lemmatizes for all review text in the dataset

    # building boolean target column aggregating by amazon id
    df['target'] = df['amazon-id']
    avg = df[['amazon-id', 'overall']].groupby('amazon-id').mean()

    """ def GetTargetBoolean(amazonID):
        # converts the target boolean given amazon ID
        return (avg.loc[amazonID, 'overall'] > 4.5)
    df['target'] = df['target'].map(lambda x: GetTargetBoolean(x)) """
    
    df['target'] = df['target'].map(lambda x: avg.loc[x, 'overall'] > 4.5)

    # remove stop words
    df['reviewText'] = df['reviewText'].map(lambda x: RemoveStopWords(x))

    # lemmatize words
    df['reviewText'] = df['reviewText'].map(lambda x: Stem(x))

    return df
    #print(df['target'].size)
    #print(df.shape)

#print(CleanData().head())


def TrainTest():
    # train on a subset of data - currently using 70% to train
    
    # pre-process data
    CleanData()

    # get only relevant columns for training and testing
    dfTrainTest = df[['reviewText','target']]
    
    # get target rows and non-target rows
    dfAwesome = dfTrainTest[dfTrainTest['target'] == True]
    dfNotAwesome = dfTrainTest[dfTrainTest['target'] == False]


    dfAwesome['reviewText'].map(lambda x: word_tokenize(x))
    dfNotAwesome['reviewText'].map(lambda x: word_tokenize(x))
    
    # build a dictionary to use in training and testing
    def buildDictForModel(dataFrame):
        for sentences in dataFrame['reviewText']:
            yield dict([token, True] for token in word_tokenize(sentences))

    awesomeTokensForModel = buildDictForModel(dfAwesome)
    notAwesomeTokensForModel = buildDictForModel(dfNotAwesome)


    awesomeDataset = [(awesomeTokenDict, 'True') for awesomeTokenDict in awesomeTokensForModel]
    notAwesomeDataset = [(notAwesomeTokenDict, 'False') for notAwesomeTokenDict in notAwesomeTokensForModel]
    dataset = awesomeDataset + notAwesomeDataset

    trainData = dataset[:70000]
    testData = dataset[70000:]

    random.shuffle(dataset)
    
    classifier = NaiveBayesClassifier.train(trainData)
    return classify.accuracy(classifier, testData)




"""     X_train, X_test, y_train, y_test = train_test_split(dfTrainTest.drop('target', axis=1), dfTrainTest['target'], test_size = 0.3)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test)) """



print(TrainTest())



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




