import data_utils  as d_utils
import matplotlib.pyplot as plt
import numpy as np
from soft_svm import *


def main(d_format=None):
	# import data set
	train_data = np.load('./data/'+d_format+'_data.npy')
	train_labels = np.load('./data/'+d_format+'_labels.npy')

	num_data = train_data.shape[0]

	np.random.seed(2017)

	# build classifier
	classifier = soft_svm()
	classifier.train(train_data, train_labels, \
					 C=2, toler=0.001, 		\
					 kernal='gaussian', 		 \
					 gaussian_sigma=4)

	# test classifier on training set
	scores = classifier.scores(train_data)
	pred = np.ones(num_data, dtype=int)
	pred[scores < 0] = -1
	acc = np.mean(pred == train_labels)

	# visualization
	color_list = ['r', 'b', 'g']
	marker_list = ['+', 'o', '*']

	plt.figure("Data format : " + d_format)
	err_cnt = 0
	for i in range(num_data):
		if train_labels[i] * classifier.scores(train_data[i,:]) < 0:
			# Wrong classification
			c = color_list[0]
		else:
			# Correct classification
			c = color_list[train_labels[i]]
		plt.scatter(train_data[i, 0], train_data[i, 1], marker=marker_list[train_labels[i]], c=c, alpha=0.6)

	plt.title('accuracy = %.4f'%(acc))
	plt.show()



if __name__ == '__main__':
	main(d_format="nonlinear") # linear or nonlinear