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

    SEQUENCE PARSER

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

#cleans up sequences
def txt2seq(seq):
    s2 = []
    for s in seq:
        si = re.sub("\(\s?\)","?",s)       #replace parenthesis with question mark
        a = re.split(',|\s|-',si)      #remove any delimiters
        a = [e for e in a if e != ""]  #remove empty strings
        s2.append(a)
    return s2

#checks for valid sequences (all numbers or fractions excluding ?)
def findBadSeq(seq):
    badSeqs = {}
    for a in range(len(seq)):     #check each sequence
        s = seq[a]
        for i in s:   #check each item (assuming sequence is already split)
            if(re.match(r'[^0-9\/\?]',i)):    #check if any words or weird characters in the sequence
                badSeqs[a] = s
                break
    return badSeqs


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
    opts = seqdataIn['options']

    split_seq = txt2seq(sequences)

    gs = remBadSeq(split_seq)

    fs = seq2Float(gs)

    return fs, opts



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

    '''
    train_gen = TimeseriesGenerator(seq, seq, length=look, batch_size=1)
    for i in range(len(train_gen)):
        x, y = train_gen[i]

        if(y[0] == "?"): #add test
            u = []
            for a in x[0]:
                if(a == "?"):
                    continue
                u.append([a])
            test.append(u)
            continue

        e = []
        for b in x[0]:
            if(b == "?"):
                continue
            e.append([b])
        d_input.append(e)
        d_output.append(y)
    '''

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
def predictSeq(classifier,seq,options=[]):
    #print(seq)

    if seq.count('?') > 1 or "?" not in seq:     #too many or too few
        return 0

    #classify sequence
    s = seq[:]
    classGuesses = classifier.predict(seq_prep([s],17)).tolist()        #get response from the classifier 
    classType = classGuesses.index(max(classGuesses))


    if classType == 2:      #hybrid
        m = makeHybridModel()
        x, y, test = seq2HybridData(seq)
        m.fit(x, y, epochs=1000, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m.predict(test))[0]
        if len(options) != 0:
            return getClosestOption(a,options)
        else:
            return a

    elif classType == 1:     #index only
        m = makeIndexModel()
        x, y, test = seq2IndData(seq)
        m.fit(x, y, epochs=500, validation_split=0.0, verbose=0,steps_per_epoch=len(x))

        a = np.squeeze(m.predict(test))[0]
        if len(options) != 0:
            return getClosestOption(a,options)
        else:
            return a

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

        final_a = None

        if len(options) != 0:
            a1 = getClosestOption(a1,options)
            a2 = getClosestOption(a2,options)
            a3 = getClosestOption(a3,options)

        #return majority answer
        ans = [a1,a2,a3]
        final_a = max(set(ans), key = ans.count) 
        return final_a


# evaluate the training data
def evalTrain():
    pathTrain = "data/seq-public.json"
    pathAns = 'data/seq-public.answer.json'

    class_model = keras.models.load_model("classifier")

    #parse the sequences
    trainSeq, opts = makeSeq(pathTrain)

    #get the answers and answer options
    ansDatIn = pd.read_json(pathAns, orient='index')
    answers = ansDatIn['answer']

    #get accuracy from predictions
    correct = []
    resp = {}
    with tqdm(total=len(trainSeq)) as pbar:
        t = 0
        for i,s in trainSeq.items():
            if(not i in answers):
                continue

            try:
                a = int(predictSeq(class_model, s, opts[i]))
                print(a)
                print(answers[i][0])
                correct.append(1 if a == answers[i][0] else 0)
                resp[i] = [a,answers[i]]
            except KeyboardInterrupt:
                exit(0)
            except:
                print(sys.exc_info()[0])

            pbar.update(1)
            t+=1
            if t > 3:
                break

    tot_correct = sum(correct)
    #print accuracy
    with open('seq_train_acc.txt', 'w') as f:
        print("Problems evaluated: " + str(len(correct)),file=f)
        print("# Correct: " + str(tot_correct),file=f)
        print("Accuracy: " + str(float(tot_correct / len(correct))),file=f)

    #print raw responses
    with open("seq_train_resp.txt", 'w') as resp_file:
        print("ID : [prediction, actual]", file=resp_file)
        for k, v in resp.items():
            print(str(k) + ": " + str(v), file=resp_file)

    return float(tot_correct / len(correct)), resp

#outputs the test predictions to json (as per the competition instructions)
def testOut():
    pathTest = "data/seq-private.json"

    class_model = keras.models.load_model("classifier")

    #get sequences and answer options
    testSeq, opts = makeSeq(pathTest)

    #get predictions
    outAns = {}
    with tqdm(total=len(testSeq)) as pbar:
        t = 0
        for i,s in testSeq.items():
            try:
                a = predictSeq(class_model,s,opts[i])
                outAns[i] = a
            except KeyboardInterrupt:
                exit(0)
            except:
                print(sys.exc_info()[0])
            pbar.update(1)
            t+=1
            if t > 3:
                break

    #export to json
    j = {}
    for i,a in outAns.items():
        j[str(i)] = {"answer":[a]}

    with open('seq-private.answer.json', 'w') as outfile:
        json.dump(j,outfile)



if __name__ == "__main__":
    acc, responses = evalTrain()
    print(acc)
    for k, v in responses.items():
        print(str(k) + ":" + str(v))
    testOut()




