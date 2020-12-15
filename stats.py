import numpy as np
import json
import pandas as pd
import keras
from iq_bot import makeSeq, floatConv


pathTrain = "data/seq-public.json"
pathAns = 'data/seq-public.answer.json'

#reimport training data
rawTrain = pd.read_json(pathTrain, orient='records')
seq1 = rawTrain['stem']
ids = rawTrain['id']
opt1 = rawTrain['options']

rawSeq = dict(zip(ids,seq1))
rawOpt = dict(zip(ids,opt1))




#parse the sequences
trainSeq, opts = makeSeq(pathTrain)




#get the answers and answer options
ansDatIn = pd.read_json(pathAns, orient='index')
answers = ansDatIn['answer']


'''
for i in trainSeq.keys():
	if not i in answers:
		continue
	print('')
	print("%d : %s + %s" % (i, str(trainSeq[i]),str(opts[i])))
	print("   = " + str(rawSeq[i]) + " + " + str(rawOpt[i]))
	print("answer: " + str(answers[i]))

exit(0)
'''

#for guessing sequence class
#class_model = keras.models.load_model("classifier")


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
	a = a.replace("'","")
	a = a.replace("array(", "")
	a = a.replace(", dtype=float32)","")


	#if len(opts[i]) == 0:
	#	print(a)


	#get elements
	e = a.split(", ")

	if(len(e) != 4):
		continue

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




models = ["unknown","index", "hybrid"]
modelAcc = {}
for m in models:
	modelAcc[m] = {"total":0, "correct":0}

qType = [ "Open","Multi-choice"]
questAcc = {}
for q in qType:
	questAcc[q] = {"total":0, "correct":0}


err = []
for i,seq in trainSeq.items():
	if i in resp:		#certified answered sequence

		'''
		#classify sequence
		s = seq[:]
		classGuesses = classifier.predict(seq_prep([s],17)).tolist()        #get response from the classifier 
		classType = classGuesses.index(max(classGuesses))
		'''

		classType = resp[i]["model"]

		q = (1 if (len(opts[i]) > 1) else 0)

		c = (1 if resp[i]["pred"] == resp[i]["real"] else 0)

		modelAcc[classType]["total"] += 1
		modelAcc[classType]["correct"] += c

		questAcc[qType[q]]["total"] += 1
		questAcc[qType[q]]["correct"] += c

		if len(opts[i]) > 1:
			#print(i)
			#print(opts[i])
			#print(int(resp[i]["real"])-1)
			valAns = floatConv(opts[i][int(resp[i]["real"])-1])
		else:
			valAns = floatConv(resp[i]["real"])

		#print(valAns)
		#print(floatConv(resp[i]["raw"]))

		err.append(abs(valAns-floatConv(resp[i]["raw"])))


#### CALCULATE THE ACCURACY PER MODEL
print("---- ACCURACY BY MODEL ----")
for m in models:
	if modelAcc[m]["total"] == 0:
		p = 0
	else:
		p = modelAcc[m]["correct"]/modelAcc[m]["total"]
	print("%s model: %d / %d = %f" % (m, modelAcc[m]["correct"], modelAcc[m]["total"],p))



#### CALCULATE ACCURACY PER QUESTION ####
print("---- ACCURACY BY QUESTION TYPE ----")
for q in qType:
	if questAcc[q]["total"] == 0:
		p = 0
	else:
		p = questAcc[q]["correct"]/questAcc[q]["total"]
	print("%s question: %d / %d = %f" % (q, questAcc[q]["correct"], questAcc[q]["total"],p))


#### GET AVERAGE ERROR FROM TRUE ANSWER
print("---- \% ANSWERS WITHIN 10\% ----")
w = 0
for i in err:
	if i <= 10:
		w += 1

ap = w / len(err)

#avg_err = np.mean(np.array(err))
#print("Average error: " + str((avg_err*100)) +"%")
print("10 percent error answers: " + str(ap))










