#!/usr/bin/env python

import matplotlib.pyplot as plt

ER_batchSize = [0.2, 0.22, 0.36, 0.36, 0.315, 0.3]
batchSize = [2, 5, 10, 15, 20, 30]


ER_hiddenunit_bs5 = [0.28, 0.26, 0.22, 0.22, 0.26]
ER_hiddenunit_bs10 = [ 0.38, 0.36, 0.34, 0.34, 0.37]
hiddenunit = [10, 20,30,40,50]

ER_learningRate = [0.22, 0.24, 0.24, 0.26, 0.34, 0.42, 0.58]
learningRate = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

if __name__ == '__main__':
	plt.figure(1)
	acc = [1 - er for er in ER_batchSize]
	plt.plot(batchSize, acc,'bo-', lw=2)
	plt.title('test accuracy with different number of batch size')
	plt.xlabel('batch size')
	plt.ylabel('test accuracy')
	plt.axis([ min(batchSize), max(batchSize) ,  min(acc)* 0.99, max(acc) *1.01])

	plt.figure(2)
	acc = [1 - er for er in ER_learningRate]
	plt.plot(learningRate, acc,'bo-', lw=2)
	plt.title('test accuracy with different learning rate')
	plt.xlabel('learning rate')
	plt.ylabel('test accuracy')
	plt.axis([ min(learningRate), max(learningRate) ,  min(acc)* 0.99, max(acc) *1.01])

	plt.figure(3)
	acc_bs5 = [1 - er for er in ER_hiddenunit_bs5]
	plt.plot(hiddenunit, acc_bs5,'bo-', lw=2, label = 'batch size = 5')
	acc_bs10 = [1 - er for er in ER_hiddenunit_bs10]
	plt.plot(hiddenunit, acc_bs10,'ro-', lw=2, label = 'batch size = 10')
	plt.title('test accuracy with different number of hidden units')
	plt.xlabel('number of hidden unit')
	plt.ylabel('test accuracy')
	plt.axis([ min(hiddenunit), max(hiddenunit) ,  min(acc_bs5 + acc_bs10)* 0.99, max(acc_bs5 + acc_bs10) *1.01])
	plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=1.)

	plt.show()