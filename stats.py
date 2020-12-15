import numpy
import json
import pandas
import keras
from iq_bot import *


pathTrain = "data/seq-public.json"
pathAns = 'data/seq-public.answer.json'

#reimport training data
rawTrain = pd.read_json(pathTrain, orient='records')
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
	if p[0] == "ID ":
		continue
	i = int(p[0])

	#remove brackets
	a = p[1].replace("]","")
	a = a.replace("[","")

	#get elements
	e = a.split(", ")
	pred = floatConv(e[0])
	real = floatConv(e[1])
	modelUsed = e[2]
	raw = floatConv(e[3])

	resp[i] = {"pred":pred, "real":real, "model":modelUsed, "raw":raw}	#add predicted and real answer


#### CALCULATE BAD DATA PERCENTAGE
data_ans = len(resp) / len(rawSeq)
badDatNum = len(rawSeq)-len(trainSeq)
print("---- BAD/GOOD DATA PERCENTAGES ----")
print("Bad data: %d / %d = %f" % (badDatNum,len(rawSeq),badDatNum/len(rawSeq)))
print("Good data: %d / %d = %f" % (len(trainSeq),len(rawSeq),len(trainSeq)/len(rawSeq)))
print("Sequences answered: %d / %d = %f" % (len(resp), len(rawSeq),data_ans))
print("")




models = ["Unknown","Index", "Hybrid"]
modelAcc = {}
for m in models:
	modelAcc[m] = {"total":0, "correct":0}

qType = ["Multi-choice", "Open"]
questAcc = {}
for q in qType:
	questAcc[q] = {"total":0, "correct":0}


err = []
for i,seq in trainSeq:
	if i in resp:		#certified answered sequence

		'''
		#classify sequence
	    s = seq[:]
	    classGuesses = classifier.predict(seq_prep([s],17)).tolist()        #get response from the classifier 
	    classType = classGuesses.index(max(classGuesses))
		'''

		classType = resp[i]["model"]

	    q = 1 if len(opts[i]) > 1 else 0

	    c = (1 if resp[i]["pred"] == resp[i]["real"] else 0)

	    modelAcc[models[classType]]["total"] += 1
	    modelAcc[models[classType]]["correct"] += c

	    questAcc[qType[q]]["total"] += 1
	    questAcc[qType[q]]["correct"] += c

	    if len(opts[i]) > 1:
	    	valAns = opts[i][int(resp[i]["real"])]
	    else:
	    	valAns = resp[i]["real"]

	    err.append(abs(valAns-resp[i]["raw"]))


#### CALCULATE THE ACCURACY PER MODEL
print("---- ACCURACY BY MODEL ----")
for m in models:
	p = modelAcc[m]["correct"]/modelAcc[m]["total"]
	print("%s model: %d / %d = %f" % (m, modelAcc[m]["correct"], modelAcc[m]["total"],p))



#### CALCULATE ACCURACY PER QUESTION ####
print("---- ACCURACY BY QUESTION TYPE ----")
for q in qType:
	p = questAcc[q]["correct"]/questAcc[q]["total"]
	print("%s question: %d / %d = %f" % (q, questAcc[q]["correct"], questAcc[q]["total"],p))


#### GET AVERAGE ERROR FROM TRUE ANSWER
print("---- AVERAGE ERROR FROM TRUE ANSWER ----")
avg_err = np.mean(np.array(err))
print("Average error: %f %" % (avg_err*100))










