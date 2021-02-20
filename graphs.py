import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load training data files
rdf = pd.read_csv('Train.csv')

#Clean data
rdf = rdf[rdf['overall'].notna()]

num1 = 0
num2 = 0
num3 = 0
num4 = 0
num5 = 0

#Iterate through rows
for score in rdf['overall']:
    if score == 1:
        num1 = num1 + 1
    if score == 2:
        num2 = num2 + 1
    if score == 3:
        num3 = num3 + 1
    if score == 4:
        num4 = num4 + 1
    if score == 5:
        num5 = num5 + 1

total = num1 + num2 + num3 + num4 + num5

def RawNumberPlot():
    fig = plt.figure()
    ind = np.arange(5)
    width = 0.35
    ax = fig.add_subplot(111)
    scores = ['1', '2', '3', '4', '5']
    options = [num1,num2,num3,num4,num5]
    rects = ax.bar(ind, options, width)
    ax.bar(scores,options)
    ax.set_title('Raw Score Count')
    ax.set_xlabel('Scores')
    ax.set_ylabel('Frequency') 
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
            '%d' % int(height),
            ha='center', va='bottom')
    plt.show()

def FrequencyPlot():
    fig = plt.figure()
    ind = np.arange(5)
    width = 0.35
    ax = fig.add_subplot(111)
    scores = ['1', '2', '3', '4', '5']
    options = [num1/total,num2/total,num3/total,num4/total,num5/total]
    rects = ax.bar(ind, options, width)
    ax.bar(scores,options)
    ax.set_title('Score Frequency')
    ax.set_xlabel('Scores')
    ax.set_ylabel('Frequency') 
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
            '%.2f' % height,
            ha='center', va='bottom')
    plt.show()

if __name__ == "__main__":
    RawNumberPlot()
    FrequencyPlot()

