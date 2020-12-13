# Imports
import numpy as np
import pandas as pd
import re
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random
from keras.layers import Bidirectional
from matplotlib import pyplot
import json


'''

    SEQUENCE PARSER

'''
#cleans up sequences
def txt2seq(seq):
    s2 = []
    for s in seq:
        si = re.sub("\(\s?\)","?",s)       #replace parenthesis with question mark
        a = re.split(',|\s|-',si)      #remove any delimiters
        a = [e for e in a if e != ""]  #remove empty strings
        s2.append(a)
    return s2

#remove bad sequences from good sequences
def remBadSeq(seq):
    badSeqInd = list(findBadSeq(seq).keys())
    goodSeq = {}
    for i in range(len(seq)):
        if i not in badSeqInd:     #want to keep the indexes to get the answer for it
            goodSeq[i] = seq[i]
    return goodSeq

#converts all sequence items to float values
def floatConv(s):
    try:
        return float(s)
    except ValueError:
        try:
            num, denom = s.split('/')
            return float(num) / float(denom)
        except ValueError:
            return 'X'   #bad

#converts all the sequences to floats
def seq2Float(seq):
    floatSeq = {}
    for i in seq.keys():
        s = list(map(lambda x: floatConv(x) if x != "?" else x, seq[i]))
        if "X" in s:  #found a bad one (don't use)
            continue
        else:
            floatSeq[i] = s
    return floatSeq

#make the sequences from the data
def makeSeq(dataPath):
    #read in the raw data
    seqdataIn = pd.read_json(dataPath, orient='records')
    sequences = seqdataIn['stem']
    options = seqdataIn['options']

    split_seq = txt2seq(sequences)

    gs = remBadSeq(split_seq)

    fs = seq2Float(gs)

    return fs, options



'''

    MODEL AND SEQUENCE FORMATTING

'''

#turn sequence into index based data 
def seq2IndData(seq):
    d_input = []
    d_output = []
    test = []

    for i in range(len(seq)):
        if(seq[i] == "?"):  #skip the unknown
            test.append([[i]])
            continue

        d_input.append([[i]])
        d_output.append([[seq[i]]])

    return d_input, d_output


#turn sequence into recursive based data 
def seq2RecData(seq,look):
    d_input = []
    d_output = []
    test = []

    train_gen = TimeseriesGenerator(seq, seq, length=look, batch_size=1)
    for i in range(len(train_gen)):
        x, y = train_gen[i]

        if(y[0][1] == "?"): #add test
            test.append(x)
            continue

        d_input.append(x)
        d_output.append(y)

    return d_input, d_output

def seq2HybridData(seq):
    d_input = []
    d_output = []
    test = []

    for i in range(len(seq)-1):
        if seq[i+1] == "?":         #add test
            test.append([[seq[i]],[i]])
            continue
        elif seq[i] == "?":
            continue

        d_input.append([[seq[i]],[i]])
        d_output.append([[seq[i+1]]])

    return d_input, d_output, test



# makes a new recursive model
def makeRecursiveModel():
    # Simple LSTM - softplus Activation
    fib_model = Sequential()
    fib_model.add(LSTM(50, activation='softplus', input_shape=(fib_look, 1)))
    fib_model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.01)
    fib_model.compile(optimizer=adam, loss='mse',metrics=['mse'])
    #fib_history = fib_model.fit([X_train[:2]], [y_train[:2]], epochs=500, verbose=0)

    return fib_model

# makes a new index model
def makeIndexModel():
    # Simple LSTM - softplus Activation
    index_model = Sequential()
    index_model.add(LSTM(50, activation='softplus', input_shape=(1, 1)))
    index_model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.01)
    index_model.compile(optimizer=adam, loss='mae',metrics=['mae'])
    #index_history = index_model.fit(ind_train, ans_train, epochs=500, validation_split=0.0, verbose=0)

    return index_model

# makes a new hybrid model
def makeHybridModel():
    # simple LSTM - softplus Activation
    hybrid_model = Sequential()
    hybrid_model.add(LSTM(50, activation='softplus', input_shape=(2, 1)))
    hybrid_model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.1)
    hybrid_model.compile(optimizer=adam, loss='mae',metrics=['mae'])
    #hybrid_history = hybrid_model.fit(n_train, nplus_train, epochs=1000, validation_split=0.0, verbose=0)

    return hybrid_model


'''

    PREDICTOR

'''


#return closest index to answer in list of options
def getClosestOption(a,opt):
    d = []
    for i in opt:
        d.append(abs(a-floatConv(i)))
    return d.index(min(d))


#predict a sequence using the model options
def predictSeq(seq,options=[]):
    #classType = classifier(seq)        #get response from the classifier 


    if classType == 3:      #unknown
        m = makeHybridModel()
        x, y, test = seq2HybridData(seq)
        m.fit(x, y, epochs=1000, validation_split=0.0, verbose=0)

        a = np.squeeze(m.predict(test))[0]
        if len(options) != 0:
            return getClosestOption(a)
        else:
            return 
    else if classType == 2:
        m = makeIndexModel()
        x, y, test = seq2IndData(seq)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0)

        a = np.squeeze(m.predict(test))[0]
        if len(options) != 0:
            return getClosestOption(a)
        else:
            return a

    else if classType == 1:
        m = makeRecursiveModel()
        x, y, test = seq2RecData(seq,2)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0)

        a = np.squeeze(m.predict(test))[0]
        if len(options) != 0:
            return getClosestOption(a)
        else:
            return a

    else:       #do all 3 and take most combined or average
        m1 = makeRecursiveModel()
        m2 = makeIndexModel()
        m3 = makeHybridModel()

        x1, y1, test1 = seq2RecData(seq,2)
        x2, y2, test2 = seq2IndData(seq)
        x3, y3, test3 = seq2HybridData(seq)

        m1.fit(x1, y1, epochs=500, validation_split=0.0, verbose=0)
        m2.fit(x2, y2, epochs=500, validation_split=0.0, verbose=0)
        m3.fit(x3, y3, epochs=1000, validation_split=0.0, verbose=0)

        a1 = np.squeeze(m1.predict(test))[0]
        a2 = np.squeeze(m2.predict(test))[0]
        a3 = np.squeeze(m3.predict(test))[0]

        final_a = None

        if len(options) != 0:
            a1 = getClosestOption(a1)
            a2 = getClosestOption(a2)
            a3 = getClosestOption(a3)

        ans = [a1,a2,a3]
        final_a = max(set(ans), key = ans.count) 


# evaluate the training data
def evalTrain():
    pathTrain = "data/seq-public.json"
    pathAns = 'data/seq-public.answer.json'

    #parse the sequences
    trainSeq, ansOptions = makeSeq(pathTrain)

    #get the answers and answer options
    ansDatIn = pd.read_json(pathAns, orient='index')
    answers = ansDatIn['answer']
    options = ansOptions['options']

    #get accuracy from predictions
    correct = []
    for i,s in trainSeq.items():
        a = predictSeq(s, options[i])
        correct.append(1 if a == answers[i] else 0)

    return float(sum(correct) / len(correct))

#outputs the test predictions to json (as per the competition instructions)
def testOut():
    pathTest = "data/seq-private.json"

    #get sequences and answer options
    testSeq, ansOptions = makeSeq(pathTest)
    options = ansOptions['options']

    #get predictions
    outAns = {}
    for i,s in testSeq.items():
        a = predictSeq(s,options[i])
        outAns[i] = a

    #export to json
    j = {}
    for i,a in outAns.items():
        j[str(i)] = {"answer":[a]}

    with open('seq-private.answer.json', 'w') as outfile:
        json.dump(j,outfile)



if __name__ == "__main__":
    print(evalTrain())
    testOut()




