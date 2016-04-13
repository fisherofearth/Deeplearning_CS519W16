#!/usr/bin/env python

########################################################
# Functionality: Plot all records to .png files
# Author: Xi Yu
# Date: 2/27/2016
########################################################

import matplotlib.pyplot as plt

# record file path
filenames = [	'./record_1/record_1.txt',
				'./record_2/record_2.txt',
				'./record_3/record_3.txt',
				'./record_4/record_4.txt',
				'./record_5_1/record_5_1.txt',
				'./record_5_2/record_5_2.txt']

# title of the records
setting_titles = [	'Setting 1: No dropout layer',
					'Setting 2: Add a 512-unit fully connected layer',
					'Setting 3: Add 2 dropout layers',
					'Setting 4: Adaptive learning rate (Adagrad)',
					'Setting 5.1: 64-hidden-unit single FCNN',
					'Setting 5.2: Sigmoid for single FCNN']

def read_record(filename):
	ephoc, loss, acc, val_loss, val_acc = [] , [], [], [], []
	with open(filename, 'r') as file:
		for line in file:
			data = line.split()
			ephoc.append(data[0])
			loss.append(data[1])
			acc.append(data[2])
			val_loss.append(data[3])
			val_acc.append(data[4])
	return ephoc, loss, acc, val_loss, val_acc

def plot(choose, show = False, save = False):

	ephoc, loss, acc, val_loss, val_acc = read_record(filenames[choose])

	plt.figure(choose, figsize=(10,15))

	plt.subplot(211)
	plt.plot(ephoc, loss,'bx-', lw=2, label = 'training loss')
	plt.plot(ephoc, val_loss,'rx-', lw=2, label = 'validation loss')
	plt.title('Loss - ' + setting_titles[choose])
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=1.)
	plt.grid(True)

	plt.subplot(212)
	error = [1.-float(a) for a in acc]
	val_error = [1.-float(a) for a in val_acc]
	plt.plot(ephoc, error,'bx-', lw=2, label = 'train error')
	plt.plot(ephoc, val_error,'rx-', lw=2, label = 'validation error')
	plt.title('Error - ' + setting_titles[choose])
	plt.xlabel('epoch')
	plt.ylabel('error')
	plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=1.)
	plt.grid(True)

	if save == True:
		plt.savefig(setting_titles[choose]+'.png')
	if show == True:
		plt.show()

if __name__ == '__main__':
	#choose = input('Record (0-5): ')
	#plot(choose, show=True, save=False)

	for r in xrange(len(filenames)):
		plot(r, show=False, save=True)
	

