import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load training data and predictions files
rdf = pd.read_csv('Train.csv')

# Clean data
rdf = rdf[rdf['overall'].notna()]

# Unaggregated data graphs
def RawGraphs():
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
        plt.ylim(0,1)
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                '%.2f' % height,
                ha='center', va='bottom')
        plt.show()

    #RawNumberPlot()
    FrequencyPlot()

def AggregatedGraphs():
    # group by awesome/not awesome products

    groupedDf = rdf.groupby('amazon-id', as_index=False).agg({'overall': ['mean']})
    groupedDf['awesome'] = groupedDf['overall'] > 4.5

    awesomeNum1 = 0
    awesomeNum2 = 0
    awesomeNum3 = 0
    awesomeNum4 = 0
    awesomeNum5 = 0
    notAwesomeNum1 = 0
    notAwesomeNum2 = 0
    notAwesomeNum3 = 0
    notAwesomeNum4 = 0
    notAwesomeNum5 = 0

    for index, row in rdf.iterrows():
        groupedDfRow = groupedDf.loc[groupedDf['amazon-id'] == row['amazon-id']]
        score = row['overall']

        if groupedDfRow['awesome'].iloc[0] == True:
            if score == 1:
                awesomeNum1 = awesomeNum1 + 1
            if score == 2:
                awesomeNum2 = awesomeNum2 + 1
            if score == 3:
                awesomeNum3 = awesomeNum3 + 1
            if score == 4:
                awesomeNum4 = awesomeNum4 + 1
            if score == 5:
                awesomeNum5 = awesomeNum5 + 1
        if groupedDfRow['awesome'].iloc[0] == False:
            if score == 1:
                notAwesomeNum1 = notAwesomeNum1 + 1
            if score == 2:
                notAwesomeNum2 = notAwesomeNum2 + 1
            if score == 3:
                notAwesomeNum3 = notAwesomeNum3 + 1
            if score == 4:
                notAwesomeNum4 = notAwesomeNum4 + 1
            if score == 5:
                notAwesomeNum5 = notAwesomeNum5 + 1

    fig = plt.figure()
    ind = np.arange(5)
    width = 0.35
    ax = fig.add_subplot(111)        
    scores = ['1', '2', '3', '4', '5']
    options = [awesomeNum1,awesomeNum2,awesomeNum3,awesomeNum4,awesomeNum5]
    rects = ax.bar(ind, options, width)
    ax.bar(scores,options)
    ax.set_title('Awesome Products Raw Score Count')
    ax.set_xlabel('Scores')
    ax.set_ylabel('Frequency')
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
            '%d' % int(height),
            ha='center', va='bottom')
    plt.show()
    return

if __name__ == "__main__":
    RawGraphs()
    AggregatedGraphs()

