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


#import the json file that holds the domain data
# json format: "library": [rep for index numbers], "questions": [set of questions ]
# question format: "sequence": integer sequence with a single '?' character, "options": multiple choice integer options (respond according to the index of the answer), "answer": index value of the answer from the option choices, "id": id no of the question
def importJSON(jfile):
	return


def run_exp():
	return



if __name__ == "__main__":

	#parse arguments if any
	parser = argparse.ArgumentParser(description='Run bots on different domains of sequences')
	parser.add_argument('-b', '--bot',type=str,dest='b', help='Bot to use for experiment [ALL, MEGACAT, BASE, RANDOM]',required=True)
	parser.add_argument('-d', '--domain',type=str,dest='d', help='which domain to use [MARIO, SONG, CELL]',required=True)
	parser.add_argument('-t', '--type',type=str,dest='t', help='which type of test to make [BOTH, TRAIN, TEST]',required=False)
	args = parser.parse_args()



	run_exp()
