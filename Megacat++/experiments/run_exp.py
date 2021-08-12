# Imports
import numpy as np
import pandas as pd
import re
import csv
import random
import math
import json
import argparse
import os
from tqdm import tqdm
import tensorflow as tf
import keras

#import bots
from iq_bot import predictSeq


#import the json file that holds the domain data
# json format: "library": [rep for index numbers], "questions": [set of questions ]
# question format: "sequence": integer sequence with a single '?' character, "options": multiple choice integer options (respond according to the index of the answer), "answer": index value of the answer from the option choices, "id": id no of the question
def importJSON(jfile):
	dat = {}
	with open(jfile, "r") as f:
		dat = json.load(f)
	return dat

#return closest index to answer in list of options
def getClosestOption(a,opt):
    d = []
    for i in opt:
        d.append(abs(a-int(i)))
    return d.index(min(d))


#runs the experiment on the domain questions and returns accuracy, error, and other data
def run_exp(domain, test_type, bot):
	#import dataset questions
	domain_dat = importJSON(f"question_set/{domain}_{test_type}.json")
	domain_key = domain_dat['library']
	domain_quest = domain_dat['questions']

	#set up accuracies and other stored information
	correct = 0
	incorrect = 0
	raw_ans = []
	real_ans = []
	acc = 0
	qeval_ct = 0

	#specific to megacat bot (model type info)
	megacat_model_ans = {"hybrid":[], "index":[], "recursive":[], "unknown":[]}
	megacat_model_correct = {"hybrid":0, "index":0, "recursive":0, "unknown":0}

	#import the baseline bot (used a lookback = 3)
	base_bot = None
	if bot == "base":
		base_bot = keras.models.load_model(f"trained_models/base_{domain}_model.h5")

	#import the classifier model
	class_model = None
	if bot == "megacat":
		class_model = keras.models.load_model(f"trained_models/megacat_classifier.h5")

	#iterate through every questions
	with tqdm(total=len(domain_quest)) as expbar:
		for q in domain_quest:
			resp = -1
			raw_resp = None
			m_type = None

			#random bot = select random option choice
			if bot == "random":
				resp = random.randrange(len(q['options']))

			#get baseline bot response
			elif bot == "base":
				s = q["sequence"]
				m_loc = np.where(np.array(s) == "?")[0][0]
				if m_loc-3 < 0:
					resp = random.randrange(len(q['options']))
					raw_resp = 0
				else:
					x_test = [list(map(lambda x: [int(x)], s[m_loc-3:m_loc]))]

					raw_resp = np.squeeze(base_bot.predict(x_test))
					resp = getClosestOption(raw_resp, q["options"])

			#get the megacat bot with model specific info
			elif bot == "megacat":
				resp, m_type, raw_resp = predictSeq(class_model,q["sequence"],q["options"])


			#evaluate answer
			if int(resp) == int(q["answer"]):
				correct += 1
				#add megacat specific correctness
				if m_type != None:
					megacat_model_correct[m_type] += 1
			else:
				incorrect += 1

			#add raw response for error analysis later
			if raw_resp != None:
				raw_ans.append(raw_resp)
				#add megacat specific answer
				if m_type != None:
					megacat_model_ans[m_type].append(raw_resp)

			qeval_ct += 1.0
			acc = correct/qeval_ct
			real_ans.append(int(q["options"][int(q["answer"])]))

			expbar.update(1)
			expbar.set_description(f"Accuracy: {acc}")

			#print(f"guess i: {resp} | real i: {q['answer']} | options i {q['options']} ----- guess: {raw_resp} | real: {q['options'][int(q['answer'])]}")



	return acc, correct, len(domain_quest), raw_ans, real_ans, megacat_model_correct, megacat_model_ans

#error raw vs. actual responses
def calcError(guess,actual,e):
	if len(guess) == 0:
		return 0, 0

	#all error
	err = []
	for i in range(len(guess)):
		err.append(abs(actual[i]-guess[i]))

	#percentage within 10%
	w = 0
	for i in err:
		if i <= e:
			w += 1

	return err, (w / len(err))


if __name__ == "__main__":

	#parse arguments if any
	parser = argparse.ArgumentParser(description='Run bots on different domains of sequences')
	parser.add_argument('-b', '--bot',type=str,dest='b', help='Bot to use for experiment [ALL, MEGACAT, BASE, RANDOM]',required=True)
	parser.add_argument('-d', '--domain',type=str,dest='d', help='which domain to use [MARIO, SONG, CELL, ARITH]',required=True)
	parser.add_argument('-t', '--type',type=str,dest='t', help='which type of test to make [BOTH, TRAIN, TEST]',required=False)
	args = parser.parse_args()

	#run all/one of the bots
	bots = []
	if args.b.upper() == "ALL":
		bots = ["random", "base", "megacat"]
	else:
		bots = [args.b.lower()]

	#run both/either test and training set
	tt = []
	if args.t.upper() == "BOTH":
		tt = ["train", "test"]
	else:
		tt = [args.t.lower()]

	#run specific bots and test types on the domain data seqeunces
	for b in bots:
		for t in tt:
			a, c, nq, ra, rea, mmc, mma = run_exp(args.d, t, b)
			err, err_perc = calcError(ra,rea,10)
			print(f"-- [{args.d.upper()}] domain on [{t.upper()}] set with [{b.upper()}] bot --")
			print("--------------------------------------------------")
			print(f"+ Accuracy: {a}")
			print(f"+ Correct: {c} / {nq}")
			print(f"+ % Questions within 10% Error: {err_perc}")
			if b == "megacat":
				print("- - - - - - - - - - - - - - - - ")
				for m in mmc.keys():
					print(f" ( {m} MODEL )")
					if len(mma[m]) > 0:
						print(f"+ Accuracy: {mmc[m]/len(mma[m])}")
					else:
						print("+ Accuracy: n/a")
					print(f"+ Correct: {mmc[m]}")

			print("\n\n")

