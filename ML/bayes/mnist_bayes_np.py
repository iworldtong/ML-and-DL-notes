from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os

feature_s_length = 5
image_s_length   = 28
num_class = 10
data_set = "./data/mnist"
model_file = "bayes_mnist.npy"

def binarization_reverse(input_data):
	image_s_length = int(np.sqrt(input_data.shape[1]))
	num = input_data.shape[0]
	return np.ones((num, image_s_length, image_s_length)) - np.around(input_data.reshape(num, image_s_length, image_s_length))

def get_feature_map(image_set, feature_s_length = 1):
	num_train = image_set.shape[0]
	if feature_s_length > image_s_length: feature_s_length = image_s_length
	elif feature_s_length < 1 : 		  feature_s_length = 1
	anchor_list = list(range(0, image_s_length, feature_s_length))
	if anchor_list[-1] < image_s_length: anchor_list.append(image_s_length)
	num_feature = len(anchor_list) - 1
	feature_map = np.zeros((num_train, num_feature * num_feature))
	for index, img in enumerate(image_set):
		for i in range(0, num_feature):
			for j in range(0, num_feature):
				black_cnt = 0 
				white_cnt = 0
				for h in range(anchor_list[i], anchor_list[i+1]):
					for w in range(anchor_list[j], anchor_list[j+1]):
						if img[h, w] == 1: white_cnt += 1
						else: black_cnt += 1
				if black_cnt > white_cnt: feature_map[index, i * num_feature + j] = 1
	return feature_map, num_feature

def calc_error_rate(labels, labels_):
	return len(np.nonzero(np.sum(np.abs(labels - labels_), axis=1))) / labels.shape[0]

def bayes_mnist_train(data, save_model = False):
	num_train = data.train.images.shape[0]
	train_image_set = binarization_reverse(data.train.images)

	# calc feature map
	feature_map, num_feature = get_feature_map(train_image_set, feature_s_length = feature_s_length)
	
	# calc p(w_i)
	p_wi = (np.sum(data.train.labels, axis = 0) / num_train).reshape(1, num_class)
	
	# calc p(x_i|w_i)
	p_xi_wi = np.zeros((num_feature * num_feature, num_class))
	c_xi_wi = np.zeros((num_feature * num_feature, num_class))
	for index, feature in enumerate(feature_map):
		label = np.nonzero(data.train.labels[index])[0]
		for i in range(num_feature * num_feature):
			c_xi_wi[i, label] += 1
			if feature[i] == 1: 
				p_xi_wi[i, label] += 1
	p_xi_wi /= c_xi_wi

	print("Training data num :", num_train)
	
	if save_model:
		np.save(model_file, np.vstack((p_wi, p_xi_wi)))
		print("Model saved as :", model_file, "!")
	return p_wi, p_xi_wi


def bayes_mnist(data_set):
	data = input_data.read_data_sets(data_set, one_hot = True)

	if os.path.exists(model_file):
		print("Loading model from :", model_file)
		p_wi = (np.load(model_file)[0])[np.newaxis, :]
		p_xi_wi = (np.load(model_file)[1:])
	else:
		print("Perpare to train...")
		p_wi, p_xi_wi = bayes_mnist_train(data, save_model = True)

	# test calc p(w_i|X) and error rate
	num_test = data.test.labels.shape[0]
	labels = data.test.labels
	labels_  = np.zeros((num_test, num_class))
	test_image_set = binarization_reverse(data.test.images)
	test_feature_map, _ = get_feature_map(test_image_set, feature_s_length = feature_s_length)
	for index, test_feature in enumerate(test_feature_map):
		p_X_wi = np.tile(test_feature[np.newaxis, :].T, (1, num_class))
		p_X_wi = p_X_wi * p_xi_wi + (1 - p_X_wi) * (1 - p_xi_wi)
		p_X_wi = (np.cumprod(p_X_wi, axis=0)[-1]).reshape(1, num_class)  
		p_wi_X = p_X_wi * p_wi
		label_ = np.argmax(p_wi_X)
		labels_[index, label_] = 1
	error = calc_error_rate(labels, labels_)
	
	print("Testing data count: " , num_test )
	print("Error rate : ", error)

if __name__ == "__main__":
	bayes_mnist(data_set)
