#!/usr/bin/env python
# 
# It finds the cosine similarity between each of the queries in the 
# test query and each of the question titles in stack overflow. 
# it is using features created from Word2Vec model 
#
# *************************************** #

#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
import gensim
from Word2VecUtility import Word2VecUtility
from sklearn.metrics.pairwise import cosine_similarity
import sys
import operator
import pickle
import multiprocessing as mp
import warnings
	
warnings.filterwarnings('ignore')

outputQueue = mp.Queue()


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the question and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(questioncollection, model, num_features):
    # Given a set of questioncollection (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(questioncollection),num_features),dtype="float32")
    #
    # Loop through the questioncollection
    for question in questioncollection:
       #
       # Print a status message every 1000th question
       if counter%1000. == 0.:
           print "question %d of %d" % (counter, len(questioncollection))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(question, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanTrainReviews(questioncollection):
    clean_questioncollection = []
    for question in questioncollection:#["product_title"]:
        #clean_questioncollection.append( Word2VecUtility.question_to_wordlist( question, remove_stopwords=False ))
        clean_questioncollection.append( Word2VecUtility.sku_to_wordlist( question, remove_stopwords=False ))
    return clean_questioncollection

def getCleanTestReviews(questioncollection):
    clean_questioncollection = []
    for question in questioncollection:#["query"]:
        clean_questioncollection.append( Word2VecUtility.sku_to_wordlist( question, remove_stopwords=False ))
    return clean_questioncollection


def pickleWord2VecList(avgFeatureVecs, fileName):
	with open(fileName, 'wb') as handle:
		pickle.dump(avgFeatureVecs, handle)


def unPickleWord2VecList(fileName):
	with open(fileName, 'rb') as handle:
		avgFeatureVecs = pickle.load(handle)
	return avgFeatureVecs


def getFeatureVecFromIndexSentencesInParallel(sentences, model, num_features, outputFile, cpu_index):
	featureVecs = {}
	for index, sentence in sentences.items():
		clean_sentence = Word2VecUtility.sku_to_wordlist( sentence, remove_stopwords=False )
		featureVecs[index] = makeFeatureVec(clean_sentence, model, num_features)
		if index%10000 == 0:
			print "doing %dth sentence by cpu#%d" % (index, cpu_index)
			sys.stdout.flush()
	print "pickling ", outputFile
	pickleWord2VecList(featureVecs, outputFile)
	#output.put(featureVecs)



def convertTextToWord2VecInParallel(fileName, outName, num_cpus):
	train = {}
	train_file = open(fileName,'r')
	counter=0
	for line in train_file:
		train[counter] = line.strip()
		#train.append(line.strip())
		counter += 1
	
	pickleWord2VecList(train, outName+"Text.pickle")

	num_features = 400#300    # Word vector dimensionality
	model = gensim.models.Word2Vec.load('wiki.en.word2vec.model')

	trainDictList = {}
	cpuWorkSize = int(len(train)/num_cpus) + 1
	print "cpuWorkSize", cpuWorkSize
	for i in xrange(0,num_cpus):
		endIndex = (i+1) * cpuWorkSize
		startIndex = i*cpuWorkSize
		if i==num_cpus-1:
			endIndex = len(train)
		print startIndex, endIndex, len(train.items())
		trainDictList[i] = dict(train.items()[startIndex:endIndex])
	processes = [mp.Process(target=getFeatureVecFromIndexSentencesInParallel, args=(trainDictList[i], model, num_features, outName+"FeatureVec_cpu"+str(i)+".pickle", i)) for i in range(num_cpus)]

	import time
	start_time = time.time()

	for p in processes:
		p.start()
	
	for p in processes:
		p.join()
	
	print("--- %s seconds ---" % (time.time() - start_time))

	for p in processes:
		print "procoess completed", p.pid
		#result = output.get()
		#print result.items()[0][0], result.items[len(result)-1][0]
	

def findSimilarityUsingDictAndMultiplePickles(trainTextPickle, testQ, trainFeaturePickle, num_cpus):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	num_features = 400#300    # Word vector dimensionality
	model = gensim.models.Word2Vec.load('wiki.en.word2vec.model')

	train = unPickleWord2VecList(trainTextPickle+"Text.pickle")
	trainDataVecs = {}
	for i in xrange(0, num_cpus): 
		tempVec = unPickleWord2VecList(trainFeaturePickle+"FeatureVec_cpu"+str(i)+".pickle")
		trainDataVecs.update(tempVec)
		print "tempVec length "+str(len(tempVec))
		print "trainDataVecs length "+str(len(trainDataVecs))

	print "trainDataVecs length ", len(trainDataVecs)
	test = [testQ]#[sys.argv[2]]
	#train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data_crowflower', 'train.csv'), header=0, delimiter=",", quoting=6 )

	print "Creating average feature vecs for test questioncollection"
	testDataVecs = getAvgFeatureVecs( getCleanTestReviews(test), model, num_features )
	cosines={}
	print "Query ID, question ID, Cosine"
	try:
		#query=0
		i = testDataVecs[0]
		#iquestion=0
		#print "i, len(i)=%d" % (len(i))
		for index,j in trainDataVecs.items(): 
			#print "j, len(j)=%d" % (len(j))
			cos=0.0
			try:
				cos=cosine_similarity(i,j)
			except:
				print "cos ", cos, train[index], "exception inner", sys.exc_info()[0]
				pass
			#print "%d, %d, %f"  % (j,i,cos)
			cosines[index] = cos
			#iquestion=iquestion+1
			#query=query+1
	except:
		print "exception outer", sys.exc_info()[0]
	        pass
	sorted_x = sorted(cosines.items(), key=operator.itemgetter(1),reverse=True)
	print len(cosines)#.size()           
	print len(sorted_x)
	for i in xrange(0,10):
		#print sorted_x[0][1], train[sorted_x[0][0]]
		print sorted_x[i][1], train[sorted_x[i][0]]


#if __name__ == '__main__':
# This is the old implemetation of findSimilarity which doesnt have 
# train as a dictionary
def findSimilarity(trainFile, testQ, pickleFlag=False, trainPickleFeatureFile="") :
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	num_features = 400#300    # Word vector dimensionality
	model = gensim.models.Word2Vec.load('wiki.en.word2vec.model')

	print "Creating average feature vecs for training questioncollection"
	if not pickleFlag:
		train = []
		train_file = open(trainFile,'r')
		for line in train_file:
			train.append(line.strip())
		import time
		start_time = time.time()
		trainDataVecs = getAvgFeatureVecs( getCleanTrainReviews(train), model, num_features )
		print("--- %s seconds ---" % (time.time() - start_time))
	else:
#		train = []
#		train_file = open(trainFile,'r')
#		for line in train_file:
#			train.append(line.strip())
		train = unPickleWord2VecList(trainFile)
		trainDataVecs = unPickleWord2VecList(trainPickleFeatureFile)
	test = [testQ]#[sys.argv[2]]
	#train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data_crowflower', 'train.csv'), header=0, delimiter=",", quoting=6 )
	#	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_crowflower', 'test.csv'), header=0, delimiter=",", quoting=6 )
	
	print "Read %d labeled train questioncollection, %d labeled test questioncollection" % (len(train), len(test))
	
	# Load the punkt tokenizer
	#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	# ****** Set parameters and train the word2vec model
		#
	# Import the built-in logging module and configure it so that Word2Vec
	# creates nice output messages
	
	#model = gensim.models.Word2Vec.load('300features_40minwords_10_question')
		
	print "Creating average feature vecs for test questioncollection"
	testDataVecs = getAvgFeatureVecs( getCleanTestReviews(test), model, num_features )
	
	import warnings
	warnings.filterwarnings('ignore')
	
	cosines={}
	print "Query ID, question ID, Cosine"
	try:
		query=0
		i = testDataVecs[0]
		#iquestion=0
		#print "i, len(i)=%d" % (len(i))
		for j in trainDataVecs: 
			#print "j, len(j)=%d" % (len(j))
			cos=0.0
			try:
				cos=cosine_similarity(i,j)
			except:
				print "cos ", cos, train[query], "exception inner", sys.exc_info()[0]
				pass
			#print "%d, %d, %f"  % (j,i,cos)
			cosines[query] = cos
			#iquestion=iquestion+1
			query=query+1
	except:
		print "exception outer", sys.exc_info()[0]
	        pass
	sorted_x = sorted(cosines.items(), key=operator.itemgetter(1),reverse=True)
	#print cosines.size()
	print len(cosines)#.size()           
	print len(sorted_x)
#	print len(sorted_x[0])
#	print sorted_x[0]
#	print sorted_x[1]
#	print sorted_x[2]
	for i in xrange(0,10):
		#print sorted_x[0][1], train[sorted_x[0][0]]
		print sorted_x[i][1], train[sorted_x[i][0]]
	#print sorted(cosines, reverse=True)[0:10]
