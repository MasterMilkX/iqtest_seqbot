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
import time


# ----------------------------------     DOMAIN IMPORTER   ------------------------------------- #


#converts array of stuff (char, string, ints, other arrays, etc.) to vectorized form
#returns library key and vectorized dataset ==> {'key', 'data_vec'}
def vectorize(dset):
	
	d = np.array(dset)
	u = np.unique(sum(d,[]))    #get unique values for each
	
	#reindex based on unique value
	print(" -> Vectorizing...")
	dv = []
	with tqdm(total=len(d)) as vbar:
		for i in d:
			v = []
			for j in i:
				v.append(list(np.where(u==j))[0][0])
			dv.append(v)
			vbar.update(1)
	return {'key':u, 'data_vec':dv}


'''

  SUPER MARIO BROS LEVELS 

'''


#read in the levels as vertical slices
def importMarioLevels(loc='../Sequential Level Design/mario_ai_levels/original/'):
	level_list = [f for f in os.listdir(loc) if os.path.isfile(os.path.join(loc,f))]
	
	levels2d = []
	print(f"--> Importing {len(level_list)} levels")
	
	#open each level
	with tqdm(total=len(level_list), leave=True) as pbar:
		for level in level_list:
			with open(os.path.join(loc,level)) as f:
				l = f.readlines()
				l = np.array(list(map(lambda x: list(x.strip()), l))) #remove new line from each row
				
				#convert rows to columns (save vertical slices)
				b = l.transpose()
				m = list(map(lambda x: "".join(x),b))
				levels2d.append(m)
				pbar.update(1)
			
	return np.array(levels2d)
		
#vectorize all of the mario levels
# returns dictionary from vectorize
def vecMarioLevels():
	all_smb_levels = []
	mario_dir = '../Sequential Level Design/mario_ai_levels/'

	'''
	#all mario levels (pcg included)
	for d in os.listdir(mario_dir):
		if os.path.isdir(os.path.join(mario_dir,d)):
			print(f"---- [{d}] ----")
			all_smb_levels.extend(importMarioLevels(os.path.join(mario_dir,d)))
	'''
	#og mario levels
	all_smb_levels.extend(importMarioLevels())
		
	#vectorize
	all_smb_levels = np.array(all_smb_levels)
	#print(all_smb_levels[:10])
	vec_smb_levels = vectorize(all_smb_levels)
	return vec_smb_levels



'''

  ABC NOTATION SONGS

'''

#read in the songs (begin after the K:# line)
def vecABCSongs():
	#import the songs
	songs = []
	with open("../Music/abc_songs.txt") as f:
		l = f.read()
		full_songs = l.split('\n\n\n')
		print(f"--> Importing {len(full_songs)} Songs")
		
		#clean up songs
		with tqdm(total=len(full_songs), leave=True) as pbar:
			for s in full_songs:
				bare_song = []
				for i in s.split("\n"):
					#add measure, note length, and key
					if ":" in i and i.split(":")[0] in ['M','L','K']:
						bare_song.append(i)
							
					#add the song itself
					elif ":" not in i:
						bare_song.append(i)
					
				char_song = list("\n".join(bare_song))
				#print(len(char_song))
							
				songs.append(char_song)
				pbar.update(1)
		
	#vectorize songs
	abc_vecs = vectorize(songs)
	return abc_vecs



'''

  CELLULAR AUTOMATA

'''

#convert grid format to binary value then to an integer
def grid2Int(g):
	return int(str(np.array(g).flatten()).replace("[","").replace("]","").replace(" ",""),2)

#convert sequence of grids to int sequences
def gridSeq2Int(s):
	return [grid2Int(g) for g in s]

#convert dataset to int
def gridDat2Seq(d):
	return [gridSeq2Int(s) for s in d]


# import dataset
def vecCellAuto():
	#import grids
	gridSeqs = []
	gridSizes = []
	with open("../Cellular Automata/conway_samples_n6.txt") as f:
		l = f.read()
		cell_seq = l.split('\n---\n')
		print(f"--> Importing {len(cell_seq)} Cell Sequences")
		
		#parse the grid sets in each sequence
		with tqdm(total=len(cell_seq), leave=True) as pbar:
			for seq in cell_seq:
				gridSets = seq.split("\n\n")
				ind_grids = []

				savedGridSize = False
				
				for g in gridSets:
					gRow = g.split("\n")

					#assume square array and save nxn
					if(not savedGridSize):
						gridSizes.append(len(gRow))
						savedGridSize = True

					#turn into 2d int array
					if len(gRow) > 0:
						ind_grids.append([[int(c) for c in r] for r in gRow])
					
						
				ind_grids = ind_grids[:-1]  #remove last one (empty)
				gridSeqs.append(ind_grids)
				pbar.update(1)
			
	#already vectorized (used binary rep of whole grid)
	return {"data_vec": gridDat2Seq(gridSeqs), "key":np.array([])}



'''

  SIMPLE ARITHMETIC 

'''

#test on arithmetic sequences
# makes a sequence that starts at a random value and adds it by another value consistently 
# equation: [A_n = A_(n-1)+b; A_0, b = ?? ]
def testArith(n,k):
	d = []
	print(f"--> Creating {n} arithmetic sequences")
	with tqdm(total=n,leave=True) as pbar:
		for i in range(n):
			kr = random.randint(5,k)
			a = random.randint(0,5)
			b = random.randint(1,20)
			s = []
			for j in range(kr):
				s.append(a+j*b)      
			d.append(s)
			pbar.update(1)
	return {"data_vec":d, "key":[]}


# ---------------------------     MULTIPLE CHOICE QUESTION MAKER   ----------------------------- #


#turns a sequence into a question format
# seq - whole dataset interger sequence
# l   - length of the question sequence
# c   - number of answer choices available
# p   - probability of missing element being last in the sequence
def toMCQuestion(seq,l,c,p,lib):
	i = {}
	l = min(len(seq), l)
	
	#question
	a = random.randint(0,len(seq)-l)
	b = a+l-1
	s = seq[a:b]
	sd = math.ceil(np.std(s))

	if random.random() < p:
		m = len(s)-1              #use last element as missing
	else:
		m = random.randint(0,len(s)-1)   #random missing element
	
	#add sequence
	ans = s[m]
	s[m] = "?"    #set as missing
	i["sequence"] = list(map(lambda x: str(x), s))
	#i["sequence"] = ",".join([str(j) for j in s])
	
	#make randomly generated choices
	cs = []
	cs.append(ans)   #add right answer
	
	while len(cs) < c:       #add fake answers
		e = math.floor(ans+sd*random.uniform(-2,3))    #vary the answer choices using std and uniform randomness
		if e not in cs:
			cs.append(e)
		#in case the random value didn't work - pick directly from the key library
		elif len(lib) > 0:
			e = random.randrange(len(lib))
			if e not in cs:
				cs.append(e)
		#pick random value between the whole sequence
		else:
			e = random.randrange(min(seq),max(seq))
			if e not in cs:
				cs.append(e)

	
	random.shuffle(cs)       #shuffle order
	i["options"] = list(map(lambda x: str(x), cs))
	
	#set answer to the missing element and add
	i["answer"] = str(np.where(np.array(cs)==ans)[0][0])
	
	return i


'''
  Assume the dataset is in vectorized 2d numeric array format where the whole set is given per row/sample
  Dataset is in the form .csv
  
  ex. 
	  [ 9, 67, 14, 14, 65,  0, 23, 57, 49, 11, 22, 77, 43, 21, 60, 74, 56  .....] (length = 43)
	  
'''

# read in dataset from a csv as a 2d array
def import2DArrData(filename):
	return list(csv.reader(open(filename)))


# create multiple choice questions from all of the samples in the dataset
# INPUTS:
# minQuest = minimum number of questions to use (-1 = all samples, > len(dataset) = +random selection )
# seqRange = length range of the starting sequence
# ansRange = number range of the possible answers
# lastProb = probability of the missing element being at the last element or somewhere else

def multiChoice(vecset, minQuest = -1, seqRange=[5,15],ansRange=[3,6],lastProb=0.85):
	dataset = vecset["data_vec"]
	lib = vecset["key"]

	qdat = []
	seqstr = []                #keep track of stringified saved sequences
	shufDat = dataset.copy()
	random.shuffle(shufDat)
	
	#go over each item
	i = 0
	totalq = (len(shufDat) if minQuest == -1 else minQuest)


	print(f" --> Generating {totalq} questions\n")
	with tqdm(total=totalq, leave=True) as qbar:
		for d in shufDat:
			#empty sequence :/
			if len(d) == 0:
				continue

			#generate a question from the sample
			l = random.randint(seqRange[0],seqRange[1])
			c = random.randint(ansRange[0],ansRange[1])
			q = toMCQuestion(d,l,c,lastProb,lib)
			
			q["id"] = str(i)
			i+= 1

			seqstr.append(str(q["sequence"]))
			qdat.append(q)

			qbar.update(1)
			
			#enough questions, finish
			if minQuest > 0 and len(qdat) > minQuest:
				break
		  
		#get random extra questions to fill quota
		leftover = minQuest - len(shufDat)
		if minQuest != -1 and leftover > 0:
			for i in range(leftover):
				
				d = random.choice(shufDat)  #get random sequence from the data
				
				#generate a question from the sample
				l = random.randint(seqRange[0],seqRange[1])
				c = random.randint(ansRange[0],ansRange[1])
				q = toMCQuestion(d,l,c,lastProb,lib)

				q["id"] = str(i)
				i+= 1
				#mcbar.update(1)

				#check if already in the dataset 
				if str(q["sequence"]) not in seqstr:
					qdat.append(q)

				qbar.update(1)

				
				#enough questions, finish
				if len(qdat) > minQuest:
					break
	
	#return multiple choice set
	return qdat


#make a test per a specific domain
def testMaker(domain='arith',questions=100):
	print(f"-- Generating [{questions}] sequence questions from [{domain.upper()}] Domain --")
	vecset = {}
	if domain == "mario":
		vecset = vecMarioLevels()
	elif domain == "songs":
		vecset = vecABCSongs()
	elif domain == "cell":
		vecset = vecCellAuto()
	else:
		return {"library":[], "questions":multiChoice(testArith(questions,20),questions)}

	return {"library":vecset["key"].tolist(),"questions":multiChoice(vecset,questions)}


#export question set and key to a json file
def test2JSON(test,filename):
	with open(filename, "w") as outfile: 
		json.dump(test, outfile)
	print(f"** TEST EXPORTED TO {filename} **")



if __name__ == '__main__':

	#parse arguments if any
	parser = argparse.ArgumentParser(description='Creates tests for different domains')
	parser.add_argument('-d', '--domain',type=str,dest='d', help='which domain to use [MARIO, SONG, CELL]',required=True)
	parser.add_argument('-n', '--num_questions',type=int,dest='n', help='how many questions in the test',required=False)
	parser.add_argument('-t', '--type',type=str,dest='t', help='which type of test to make [TRAIN, TEST]',required=False)
	parser.add_argument('-f', '--filename',type=str,dest='f', help='filename to save to',required=False)

	args = parser.parse_args()

	n = 100
	if args.n:
		n = int(args.n)

	t = "train"
	if args.t:
		t = args.t

	f = f"question_set/{args.d.lower()}_{t}.json"
	if args.f:
		f = args.f

	test_set = testMaker(args.d,n)
	test2JSON(test_set,f)







