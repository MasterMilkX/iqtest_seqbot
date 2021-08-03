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

#import bots
from iq_bot import predictSeq
from 


#import the json file that holds the domain data
# json format: "library": [rep for index numbers], "questions": [set of questions ]
# question format: "sequence": integer sequence with a single '?' character, "options": multiple choice integer options (respond according to the index of the answer), "answer": index value of the answer from the option choices, "id": id no of the question
def importJSON(jfile):
	dat = {}
	with open(jfile, "r") as f:
		dat = json.load(f)
	return dat


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

	#specific to megacat bot (model type info)
	megacat_model_ans = {"hybrid":0, "index":0, "recursive":0}
	megacat_model_correct = {"hybrid":0, "index":0, "recursive":0}

	#import the baseline bot
	if bot == "base":


	#iterate through every questions
	for q in domain_quest:
		resp = -1
		#random bot = select random option choice
		if bot == "random":
			resp = random.range(len(q['options']))

		#get baseline bot response
		elif bot == "base":




	return



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
	else
		tt = [args.t.lower()]

	#run specific bots and test types on the domain data seqeunces
	for b in bots:
		for t in tt:
			run_exp(args.d, t, b)

