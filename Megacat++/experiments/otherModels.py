# Imports
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import sys
from keras.preprocessing.sequence import TimeseriesGenerator
import json
import random



'''

    MODEL AND SEQUENCE FORMATTING

'''

#converts sequence to the baseline format for the simple lstm model
# returns train x and y sequence subsets and test sequence
'''
def seq2BaseData(seq,look_back):
	generator = TimeseriesGenerator(seq, seq, length=look_back, batch_size=1)
	X = []
	y = []

	tX = []
	for i in range(len(generator)):
		xi, yi = generator[i]
		if yi != "?":
			X.append(xi)
			y.append(yi)
		else:
			tX.append(xi)

	return X,y, tX
'''

#reapply the answer to the missing ? in the sequence and create train and test data
def createBaseTraining(questions, look_back):
	#add answer back into sequence
	full_seqs = []
	for d in questions:
		s = d["sequence"].copy()
		s[np.where(np.array(s) == "?")[0][0]] = d["options"][int(d["answer"])]	#replace ?
		s = list(map(lambda x: int(x),s))							#convert seq to int format
		full_seqs.append(s)


	#create generator sequences from the complete sequences
	X = []
	y = []

	for seq in full_seqs:
	    train_gen = TimeseriesGenerator(seq, seq, length=look_back, batch_size=1)
	    for i in range(len(train_gen)):
	        xi, yi = train_gen[i]
	        X.append(xi)
	        y.append(yi)

	#clean up shape
	X = np.squeeze(np.asarray(X))
	y = np.squeeze(np.asarray(y))
	X = X.reshape((X.shape[0],look_back,1))

	return X, y


'''

	MODELS

'''
#import the json file that holds the domain data
# json format: "library": [rep for index numbers], "questions": [set of questions ]
# question format: "sequence": integer sequence with a single '?' character, "options": multiple choice integer options (respond according to the index of the answer), "answer": index value of the answer from the option choices, "id": id no of the question
def importJSON(jfile):
	dat = {}
	with open(jfile, "r") as f:
		dat = json.load(f)
	return dat

#baseline model with simple RNN with one LSTM layer
def makeBaseModel(input_shape):
	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(input_shape, 1)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse',metrics=['mse'])
	return model

def trainBaseModel(input_shape, domain_name, domain_questions):
	#make the model
	print("-- Creating model --")
	base_model = makeBaseModel(input_shape)

	#create training data
	print("-- Generating training data --")
	X_train, y_train = createBaseTraining(domain_questions, input_shape)

	#train
	print(f"-- Training model on {len(X_train)} samples--")
	history = base_model.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=1)

	#quick test
	print("-- TEST -- ")
	r = random.randrange(len(X_train))
	x_test = X_train[r:r+2]
	y_test = y_train[r:r+2]
	p = np.squeeze(base_model.predict(x_test))
	print(f"Actual: {y_test}")
	print(f"Prediction: {p}")

	#export it
	print("-- Exporting model --")
	base_model.save(f"base_models/base_{domain_name}_model.h5")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("MUST SPECIFY DOMAIN TO TRAIN BASE MODEL ON!")
		exit(1)

	domain = sys.argv[1]
	fib_look = int(sys.argv[2]) if len(sys.argv) > 2 else 3
	trainBaseModel(fib_look,domain, importJSON(f"question_set/{domain}_train.json")["questions"])






