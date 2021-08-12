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
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import sys
import os



'''

    MODEL AND SEQUENCE FORMATTING

'''

#prep for classifier
def seq_prep(s, max_len):
    seq = s[:]

    question = [] #position of question mark
    seq_len = []
    for i in range(len(seq)):
        question.append(seq[i].index('?'))
        seq_len.append(len(seq[i]))
        seq[i][question[i]] = 0
    seq = pad_sequences(seq, padding='post', maxlen=max_len)
    seq = seq.tolist()
    mask = []
    for i in range(len(seq)):
        mask.append([0] * max_len)
    for i in range(len(mask)):
        for j in range(seq_len[i]):
            mask[i][j] = 1  #assign 1 in the mask sequence for actual values in sequence
    for i in range(len(mask)):
        mask[i][question[i]] = 2 #assign 2 in the mask sequence for question mark position
    seq2 = []
    for i in range(len(seq)):
        seq2.append(seq[i]+mask[i])
    return np.array([np.array(seq2).transpose()])

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
        d_output.append([[int(seq[i])]])

    return np.array(d_input), np.array(d_output), np.array(test)


#turn sequence into recursive based data 
def seq2RecData(seq,look):
    d_input = []
    d_output = []
    test = []

    for i in range(len(seq)-2):
        if(seq[i] == "?" and len(test) == 0):
            test.append([[0],[0]])
        elif(seq[i+1] == "?" and len(test) == 0):
            test.append([[0],[int(seq[i])]])
        elif(seq[i+2] == "?" and len(test) == 0):
            test.append([[int(seq[i])],[int(seq[i+1])]])
        elif seq[i] != "?" and seq[i+1] != "?":
            d_input.append([[int(seq[i])],[int(seq[i+1])]])
            d_output.append([[int(seq[i+2])]])

    return np.array(d_input), np.array(d_output), np.array(test)

def seq2HybridData(seq):
    d_input = []
    d_output = []
    test = []

    for i in range(len(seq)-1):
        if seq[i+1] == "?" and len(test) == 0:         #add test
            test.append([[int(seq[i])],[i]])
            continue
        elif seq[i] == "?" and len(test) == 0:
            test.append([[-int(seq[i+1])],[i]])
            continue
        elif seq[i] != "?" and seq[i+1] != "?":
            d_input.append([[int(seq[i])],[i]])
            d_output.append([[int(seq[i+1])]])

    return np.array(d_input), np.array(d_output), np.array(test)



# makes a new recursive model
def makeRecursiveModel():
    # Simple LSTM - softplus Activation
    fib_model = Sequential()
    fib_model.add(LSTM(50, activation='softplus', input_shape=(2, 1)))
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
        d.append(abs(a-float(i)))
    return d.index(min(d))


#predict a sequence using the model options
# returns answer choice index, class prediction (index, recursive, hybrid), and raw answer response
def predictSeq(classifier,seq,options=[]):
    #print(seq)

    if seq.count('?') > 1 or "?" not in seq:     #too many or too few
        return 0

    #classify sequence
    s = seq[:]
    classGuesses = classifier.predict(seq_prep([s],19))        #get response from the classifier 
    classType = np.argmax(classGuesses)

    # print(classGuesses)

    if classType == 0:      #recursive
        m = makeRecursiveModel()
        x, y, test = seq2RecData(seq,2)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m(test, training=False))
        if len(options) != 0:
            return getClosestOption(a,options), "recursive", round(float(a),4)
        else:
            return a, "recursive", a

    elif classType == 1:     #index only
        m = makeIndexModel()
        x, y, test = seq2IndData(seq)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m(test, training=False))
        if len(options) != 0:
            return getClosestOption(a,options), "index", round(float(a),4)
        else:
            return a, "index", a

    elif classType == 2:      #hybrid
        m = makeHybridModel()
        x, y, test = seq2HybridData(seq)
        m.fit(x, y, epochs=1000, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m(test, training=False))
        if len(options) != 0:
            return getClosestOption(a,options), "hybrid", round(float(a),4)
        else:
            return a, "hybrid", a


    else:       #do all 3 and take most combined or average
        m1 = makeRecursiveModel()
        m2 = makeIndexModel()
        m3 = makeHybridModel()

        x1, y1, test1 = seq2RecData(seq,2)
        x2, y2, test2 = seq2IndData(seq)
        x3, y3, test3 = seq2HybridData(seq)

        '''
        print(x1)
        print(x2)
        print(x3)
        '''

        # print(test1)
        # print(test3)

        # print(seq)
        # print(x1)
        # print(y1)

        m1.fit(x1, y1, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x1))
        m2.fit(x2, y2, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x2))
        m3.fit(x3, y3, epochs=1000, validation_split=0.0, verbose=0,steps_per_epoch=len(x3))

        a1 = np.squeeze(m1(test1, training=False))
        a2 = np.squeeze(m2(test2, training=False))
        a3 = np.squeeze(m3(test3, training=False))
        raw = np.squeeze(m3(test3, training=False))     #use hybrid for raw approx

        # print(a1)
        # print(a2)
        # print(a3)

        final_a = None

        if len(options) != 0:
            a1 = getClosestOption(a1,options)
            a2 = getClosestOption(a2,options)
            a3 = getClosestOption(a3,options)

        #return majority answer
        ans = [a1,a2,a3]
        final_a = max(set(ans), key = ans.count) 
        return final_a, "unknown", round(float(raw),4)




