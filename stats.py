import numpy
import json
import pandas
import keras
from iq_bot import *


pathTrain = "data/seq-public.json"
pathAns = 'data/seq-public.answer.json'

#reimport training data
rawTrain = pd.read_json(dataPath, orient='records')
rawSeq = rawTrain['stem']

#parse the sequences
trainSeq, opts = makeSeq(pathTrain)

#get the answers and answer options
ansDatIn = pd.read_json(pathAns, orient='index')
answers = ansDatIn['answer']

#for guessing sequence class
class_model = keras.models.load_model("classifier")


#read back in the output analysis
resp = {}
f = open("seq_train_resp.txt", "r")
for l in f:
	p = l.split(":")
	i = int(p[0])

	#remove brackets
	a = p[1].replace("]","")
	a = a.replace("[","")
	pred, real = (floatConv(s) for s in a.split(", "))

	resp[i] = {"pred":pred, "real":real}	#add predicted and real answer


#### CALCULATE BAD DATA PERCENTAGE
data_ans = len(resp) / len(sequences)
badDatNum = len(rawSeq)-len(trainSeq)
print("---- BAD/GOOD DATA PERCENTAGES ----")
print("Bad data: %d / %d = %f" % (badDatNum,len(rawSeq),badDatNum/len(rawSeq)))
print("Good data: %d / %d = %f" % (len(trainSeq),len(rawSeq),len(trainSeq)/len(rawSeq)))
print("Sequences answered: %d / %d = %f" % (len(resp), len(sequences),data_ans))
print("")

#### CALCULATE THE ACCURACY PER MODEL
for i,seq in trainSeq:
	if i in resp:		#certified answered sequence
		#classify sequence
	    s = seq[:]
	    classGuesses = classifier.predict(seq_prep([s],17)).tolist()        #get response from the classifier 
	    classType = classGuesses.index(max(classGuesses))





