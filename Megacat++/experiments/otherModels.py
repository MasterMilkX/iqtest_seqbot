# Imports
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import sys
from keras.preprocessing.sequence import TimeseriesGenerator


'''

    MODEL AND SEQUENCE FORMATTING

'''

#converts sequence to the baseline format for the simple lstm model
# returns train x and y sequence subsets and test sequence
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

	MODELS

'''


#baseline model with simple RNN with one LSTM layer
def makeBaseModel(input_shape):
	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(input_shape, 1)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse',metrics=['mse'])
	return model
