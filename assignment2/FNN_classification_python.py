#!/usr/bin/env python

####################################################################
#
# Title:	Assignment 2 - CS519_F16
# Author: 	Xi Yu (yux2@oregonstate.edu)
# Date: 	02/13/2016
#
####################################################################

import numpy as np
from numpy import exp, log2, log
import random

def unpickle(file):
	# Function: Unpack dataset
	# Paraameters
	# 	file : dataset file name
	# returns
	#	dict : dataset
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def Sigmoid(matrix):
	# Function: Sigmoid activation function
	# Paraameters
	# 	matrix : data matrix - ndarry
	# returns: sigmoid of data - ndarry
	output = []
	for layerUnits in matrix:
		output.append([1/(1 + exp(-x)) for x in layerUnits])
	return np.asarray(output,dtype='float32')

#def Softmax(matrix):
#	p_c1 = Sigmoid(matrix)
#	return np.hstack((p_c1, 1-p_c1))

def ReLu(matrix):
	# Function: ReLu activation function
	# Paraameters	
	# 	matrix : data matrix - ndarry
	# returns: ReLu of data - ndarry
	output = []
	for layerUnits in matrix:
		output.append([max(0.0, x) for x in layerUnits])
	return np.asarray(output,dtype='float32')

def nullActivation(matrix):
	return matrix

def Cross_entropy(prob_class, labels): 
	# Function: Cross entropy loss
	# Paraameters
	# 	prob_class 	: Probability of predicted class given x, (ndarry)
	#	labels 		: True labels of sample (ndarry)
	# returns:	cross entropy
	#
	return 1-np.asarray([-((y*log(1.-z)) + ((1.-y)*log(z))) for (z, y) in zip(prob_class, labels)]).mean()

def chooseSubgradient(x, subgradient):
	# Function: Compute the subgradient of ReLu
	# Paraameters
	# 	x 				: single variable
	#	subgradient 	: subgradient on zero point
	# returns:	subgradient	
	if x < 0.:
		return 0.
	elif x == 0.:
		return subgradient
	else:
		return 1.

def Gradient_relu(matrix, subgradient = 0.5):
	# Function: 	Compute the gradient of ReLu
	# Paraameters
	# 	matrix 			: data matrix - ndarry
	#	subgradient 	: subgradient on zero point
	# returns:	gradient of ReLu
	output = [] 
	for layerUnits in matrix:
		output.append([chooseSubgradient(x, subgradient) for x in layerUnits])
		#output.append([chooseSubgradient(x, random.random()) for x in layerUnits])
	return np.asarray(output, dtype='float32')

def Gradient_sigmoid(matrix):
	# Function: 	Compute the gradient of Sigmoid
	# Paraameters
	# 	matrix : 	data matrix - ndarry
	# returns:	gradient of Sigmoid
	return np.asarray([f*(1-f) for f in Sigmoid(matrix)], dtype = 'float32')

def Gradient_cross_entropy(prob_class, labels):
	# Function: Compute the gradient of cross entropy
	# Paraameters
	# 	prob_class 	: Probability of predicted class given x, (ndarry)
	#	labels 		: True labels of sample (ndarry)
	# returns:	gradient of cross entropy
	output = np.asarray([(y / (1.-z) - ((1.-y) / (z))) for (z, y) in zip(prob_class, labels)]).mean()
	if abs(output) > 5:
		return 5 * (abs(output)/output)
	else:
		return output

def Error_rate(prob_class, labels):
	# Function: 	Compute the error rate
	# Paraameters
	# 	prob_class 	: Probability of predicted class given x, (ndarry)
	#	labels 		: True labels of sample (ndarry)
	# returns:	Error rate
	tmp = []
	for i in prob_class:
		if i < 0.5:
			tmp.append(0)
		else:
			tmp.append(1)
	tmp2= []
	for i in labels:
		if i ==0.:
			tmp2.append(0)
		else:
			tmp2.append(1)
	dif = []
	for i in xrange(len(tmp)):
		if tmp[i] != tmp2[i]:
			dif.append(0)
		else:
			dif.append(1)
	return np.mean(dif)

class Layer_module:
	def __init__(self, nb_input, nb_output, batch_size = 20, activation = nullActivation, bias = True):
		self.nb_input = nb_input + 1
		if bias == True:
			self.nb_output = nb_output + 1
		else:
			self.nb_output = nb_output
		self.batch_size = batch_size
		self.activation = activation
		self.bias = bias
		# initialize weights and bias
		rng = np.random.RandomState(1234)
		self.W = np.asarray(
		rng.uniform(
			low=-np.sqrt(6. / (self.nb_input + self.nb_output)),
			high=np.sqrt(6. / (self.nb_input + self.nb_output)),
			size=(self.nb_input , self.nb_output)
		), dtype='float32')

		self.D = self.W * 0

	def SetInput(self, inputs = None):
		# Function: 	Set the input of the layer
		# Paraameters
		# 	inputs : Input data -- ndarry(shape = (batchSize, dataVectorDimension)
		self.inputs = inputs

	def Gradient_actFun(self):
		# Function: Compute the gradient of activation function
		# returns
		#	g_actFun :	gradient of activation function
		if self.activation == Sigmoid:
			self.g_actFun = Gradient_sigmoid(np.dot(self.inputs, self.W) )# dimension = batch_size * nb_output
		elif self.activation == ReLu:
			self.g_actFun = Gradient_relu(np.dot(self.inputs, self.W))
		else:
			self.g_actFun = 1
		return self.g_actFun

	def Update(self, delta_nextlayer, g_inputs_nextlayer = None, learning_rate = 0.02, momentum = 0.8):
		# Function: 	update the layer parameters
		# Paraameters
		# 	delta_nextlayer 	: gradient wrt parameters of next layer
		#	g_inputs_nextlayer	: gradient wrt inputs of next layer
		#	learning_rate		: learning rate
		#	momentum			: mementum update rate
		# returns:
		#	W 	: weights
		self.Gradient_actFun()

		if g_inputs_nextlayer != None:
			self.delta = np.multiply(np.dot(delta_nextlayer, g_inputs_nextlayer.T), self.g_actFun)
		else:
			self.delta = np.multiply(delta_nextlayer , self.g_actFun)
		
		self.g_inputs = self.W
		G = np.dot(self.inputs.T, self.delta)/ self.batch_size
		self.D = (momentum * self.D) - (learning_rate * G)
		self.W 	= self.W + self.D
		return self.W

	def Output(self):
		# Function: compute the output of the layer
		# returns:
		#	outputs: output of the layer
		self.outputs = self.activation(np.dot(self.inputs, self.W) )
		if self.bias == True:
			self.outputs[:,-1] = 1.
		return self.outputs 

if __name__ == '__main__':
	nb_epoch = 200



	# Load data
	dataset = unpickle('cifar_2class_py2.p')
	X_train = dataset["train_data"].astype('float32') / 255 
	X_train = np.hstack((X_train, np.ones((X_train.shape[0],1), 'float32')))
	Y_train = dataset['train_labels'].astype('float32')
	X_test = dataset['test_data'].astype('float32') / 255
	X_test = np.hstack((X_test, np.ones((X_test.shape[0],1), 'float32')))
	Y_test = dataset['test_labels'].astype('float32')
	# show data dimension and type
	print X_train.shape, X_train.dtype, type(X_train)
	print Y_train.shape, Y_train.dtype, type(Y_train)

	# Configure the neural network
	batch_size = 2
	nb_hiddenunit = 30
	lerning_rate = 0.1

	# construct layers
	layer1 = Layer_module(nb_input = 3072, nb_output = nb_hiddenunit, batch_size = batch_size, activation = Sigmoid, bias = True)
	layer2 = Layer_module(nb_input = nb_hiddenunit, nb_output = 1, batch_size = batch_size, activation = Sigmoid, bias = False)

	# number of mini batch 
	nb_batch_train = 10#X_train.shape[0] / batch_size
	nb_batch_test = X_test.shape[0] / batch_size

	#initialize the record variables
	lowestErrorRate_train = 1
	lowestErrorRate_test = 1
	lowestObjective_train = 10
	lowestObjective_test = 10

	# Training and testing loop
	for ephoc in xrange(nb_epoch):
		score_train = []
		errorRate_train = []
		# Minibatch loop _ trainning 
		for index in xrange(nb_batch_train):
			# Read a batch of train data and labels
			data = X_train[index*batch_size:(index+1)*batch_size]
			labels = Y_train[index*batch_size:(index+1)*batch_size]

			# configure layers
			layer1.SetInput(data)
			layer2.SetInput(layer1.Output())
			layer2.Output()

			# Compute the loss and error rate
			score_train.append(Cross_entropy(layer2.outputs, labels))
			errorRate_train.append(Error_rate(layer2.outputs, labels))

			# Update the network
			g_cross_entropy = Gradient_cross_entropy(layer2.outputs, labels)
			layer2.Update(g_cross_entropy, learning_rate = lerning_rate)
			layer1.Update(layer2.delta, layer2.g_inputs, learning_rate = lerning_rate)
			
		score_test = []
		errorRate_test = []
		# Minibatch loop _ testting 
		for index in xrange(nb_batch_train):
			# Read a batch of test data and labels
			data = X_test[index*batch_size:(index+1)*batch_size]
			labels = Y_test[index*batch_size:(index+1)*batch_size]

			# configure layers
			layer1.SetInput(data)
			layer2.SetInput(layer1.Output())
			layer2.Output()

			# Compute the loss and error rate
			score_test.append(Cross_entropy(layer2.outputs, labels))
			errorRate_test.append(Error_rate(layer2.outputs, labels))

		# Reporting objective and error rate 
		print 'ephoc:', ephoc
		# evaluate training objective
		score_train = np.mean(score_train)
		if score_train < lowestObjective_train:
			lowestObjective_train = score_train
		errorRate_train = np.mean(errorRate_train)
		if errorRate_train < lowestErrorRate_train:
			lowestErrorRate_train = errorRate_train
		print 'Training:', 'objective =', score_train, ' error rate=',errorRate_train

		# evaluate testing objective
		score_test = np.mean(score_test)
		if score_test < lowestObjective_test:
			lowestObjective_test = score_test
		errorRate_test = np.mean(errorRate_test)
		if errorRate_test < lowestErrorRate_test:
			lowestErrorRate_test = errorRate_test
		print 'Testing:',  'objective =', score_test,  ' error rate=',errorRate_test

	print 'Finished --------------------------'
	print 'Number of ephoc =', nb_epoch
	print 'Batch size =', batch_size
	print 'Hidden units =', nb_hiddenunit
	print 'Learning rate =', lerning_rate
	print 'Lowest training objective =', lowestObjective_train
	print 'Lowest testing  objective =', lowestObjective_test
	print 'Lowest training error rate =', lowestErrorRate_train
	print 'Lowest testing  error rate =', lowestErrorRate_test
 