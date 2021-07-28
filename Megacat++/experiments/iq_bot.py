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

    return np.array(d_input), np.array(d_output), np.array(test)


#turn sequence into recursive based data 
def seq2RecData(seq,look):
    d_input = []
    d_output = []
    test = []

    for i in range(len(seq)-2):
        if(seq[i+2] == "?"):
            test.append([[seq[i]],[seq[i+1]]])
        elif seq[i] != "?" and seq[i+1] != "?":
            d_input.append([[seq[i]],[seq[i+1]]])
            d_output.append([[seq[i+2]]])

    return np.array(d_input), np.array(d_output), np.array(test)

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
        d.append(abs(a-floatConv(i)))
    return d.index(min(d))+1


#predict a sequence using the model options
# returns answer choice index, class prediction (index, recursive, hybrid), and raw answer response
def predictSeq(classifier,seq,options=[]):
    #print(seq)

    if seq.count('?') > 1 or "?" not in seq:     #too many or too few
        return 0

    #classify sequence
    s = seq[:]
    classGuesses = classifier.predict(seq_prep([s],17))        #get response from the classifier 
    classType = np.argmax(classGuesses)


    if classType == 2:      #hybrid
        m = makeHybridModel()
        x, y, test = seq2HybridData(seq)
        m.fit(x, y, epochs=1000, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m.predict(test))
        if len(options) != 0:
            return getClosestOption(a,options), "hybrid", round(float(a),4)
        else:
            return a, "hybrid", a

    elif classType == 1:     #index only
        m = makeIndexModel()
        x, y, test = seq2IndData(seq)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m.predict(test))
        if len(options) != 0:
            return getClosestOption(a,options), "index", round(float(a),4)
        else:
            return a, "index", a

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

        m1.fit(x1, y1, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x1))
        m2.fit(x2, y2, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x2))
        m3.fit(x3, y3, epochs=1000, validation_split=0.0, verbose=0,steps_per_epoch=len(x3))

        a1 = np.squeeze(m1.predict(test1))
        a2 = np.squeeze(m2.predict(test2))
        a3 = np.squeeze(m3.predict(test3))
        raw = np.squeeze(m3.predict(test3))     #use hybrid for raw approx

        final_a = None

        if len(options) != 0:
            a1 = getClosestOption(a1,options)
            a2 = getClosestOption(a2,options)
            a3 = getClosestOption(a3,options)

        #return majority answer
        ans = [a1,a2,a3]
        final_a = max(set(ans), key = ans.count) 
        return final_a, "unknown", round(float(raw),4)




