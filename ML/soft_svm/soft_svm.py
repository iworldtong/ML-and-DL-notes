import numpy as np


class soft_svm(object):
	def __init__(self):
		pass

	def train(self, data, labels, **kwargs):
		'''
        Training configuration.

        Required arguments:
        - data:  size = (N, D) 
        - labels: size = (N, ) 

        Optional arguments:
        - C: Hyper parameter.
        - toler: default value - 0.0001
        - kernal: You could choose 'linear'、'gaussian'.
        - gaussian_sigma: A hyper parameter used in gaussian kernal.
        - max_stop_iters: If the parameters do not change after a certain number of iterations, exit the iteration.
		'''
		num_data = data.shape[0]
		dim = data.shape[1]

		self.train_data = data.copy()
		self.train_labels = labels.reshape(-1, 1)

		self.C = kwargs.pop('C', 1) 
		self.toler = kwargs.pop('toler', 0.0001) 
		self.max_stop_iters = int(kwargs.pop('max_stop_iters', 10))
		self.kernal_mode = kwargs.pop('kernal', 'linear') 
		self.gaussian_sigma = kwargs.pop('gaussian_sigma', 1.0) 
		
		self.K = self.calc_kernal(self.train_data, self.train_data, self.kernal_mode)
		self.err_cache = np.zeros((num_data, 2))
		self.alpha = np.zeros((num_data, 1))
		self.b = 0

		# smo
		num_iter = 1
		num_alpha_changed = 0
		entire_data_set = True
		while (num_iter <= self.max_stop_iters) and ((num_alpha_changed > 0) or (entire_data_set == True)) :
			num_alpha_changed = 0
			if entire_data_set:
				# all data
				for i in range(num_data):
					num_alpha_changed += self.inner_loop(i)
				print('Entire data set iters : ', num_iter, '  i : ', i, \
					  '  pairs changed num : ', num_alpha_changed)
			else:
				# non-boundary data
				non_boundary_i = np.nonzero((self.alpha.T > 0) * (self.alpha.T < self.C))[0]
				for i in non_boundary_i:
					num_alpha_changed += self.inner_loop(i)
					print('Non-boundary data set iters : ', num_iter, '  i : ', i, \
					  	  '  pairs changes num : ', num_alpha_changed)
			num_iter += 1

			if entire_data_set:
				entire_data_set = False
			elif num_alpha_changed > 0:
				num_iter = 0
				entire_data_set = True
			
			print("Alpha no changed iteration num : ", num_iter)

		self.W = np.dot((self.alpha * self.train_labels).reshape(1,-1), self.train_data).reshape(-1, 1)
		return self.W, self.b

	def inner_loop(self, i):
		err_i = self.calc_errs(i, self.train_labels[i, 0])

		if (self.train_labels[i,0]*err_i < -self.toler and self.alpha[i,0] < self.C) or (self.train_labels[i,0]*err_i > self.toler and self.alpha[i,0] > 0):
			#--------------#
			#  inner loop  #
			#--------------#

			# inspired search j by maximum err
			j, err_j = self.search_j(i, err_i) 

			old_alpha_i = self.alpha[i, 0] 
			old_alpha_j = self.alpha[j, 0]

			if self.train_labels[i, 0] != self.train_labels[j, 0]:
				L = np.max((0, self.alpha[j, 0] - self.alpha[i, 0]))
				H = np.min((self.C, self.C + self.alpha[j, 0] - self.alpha[i, 0]))
			else:
				L = np.max((0, self.alpha[j, 0] + self.alpha[i, 0] - self.C))
				H = np.min((self.C, self.alpha[j, 0] + self.alpha[i, 0]))
			if L == H :
				print("L == H ")
				return 0

			# n = K_11 + K_22 - 2 * K_12
			# note:
			# 	when n < 0 : kernal function doesn't satisfy Mercer's theorem, matrix K is non-positive difinite
			#				 also the objective function is a convex function, there is no minimum, the extreme value obtained at the boundary of the domain
			#   when n = 0 : sample i j  have same input characteristics
			#				 also the objective function is a monotonous function, also taking extreme values at the boundaries
			n = self.K[i, i] + self.K[j, j] - 2.0 * self.K[i, j]
			if n <= 0:   
				print('(K_11 + K_22 - K_12) <= 0')
				return 0  

			# updata alpha_j
			# 	no clip : new_a_j = old_a_j + y_2 * (err_i - err_j) / (K_11 + K_22 - 2 * K_12)
			self.alpha[j, 0] += self.train_labels[j, 0] * (err_i - err_j) / n
			self.alpha[j, 0] = np.clip(self.alpha[j, 0], L, H)  
			self.update_err_cache(j)
			if abs(self.alpha[j, 0] - old_alpha_j) < 0.00001 :  
				print('alpha_j not moving enough')   
				return 0 	

			# update a_i : new_a_i = old_a_i + y_i * y_2 * (old_a_j - new_a_j)
			self.alpha[i, 0] += self.train_labels[j, 0] * self.train_labels[i, 0] * (old_alpha_j-self.alpha[j, 0])  
			self.update_err_cache(i)

			# calc b
			b1 = self.b - err_i - self.train_labels[i,0] * (self.alpha[i, 0] - old_alpha_i) * self.K[i, i] - \
				 				  self.train_labels[j,0] * (self.alpha[j, 0] - old_alpha_j) * self.K[i, j] 
			b2 = self.b - err_j - self.train_labels[i,0] * (self.alpha[i, 0] - old_alpha_i) * self.K[i, j] - \
								  self.train_labels[j,0] * (self.alpha[j, 0] - old_alpha_j) * self.K[j, j]

			if (0 < self.alpha[i, 0]) and (self.C > self.alpha[i, 0]) :   
				self.b = b1   
			elif (0 < self.alpha[j, 0]) and (self.C > self.alpha[j, 0]) :   
				self.b = b2   
			else:   
				self.b = (b1 + b2) / 2.0  

			return 1
		else:
			return 0


	def calc_kernal(self, x1, x2, kernal_mode):
		data1 = x1.copy()
		if len(x2.shape) == 1: data2 = x2.reshape(1, -1)
		else:			       data2 = x2.copy()
		N1, D = data1.shape
		N2 = data2.shape[0]
		if kernal_mode == "linear":
			kernal = np.dot(data1, data2.T)
		elif kernal_mode == "gaussian":
			# calc L2 with no "for" --- to accelerate
			l2_1 = np.sum(data1 ** 2, 1).reshape(-1, 1)
			l2_2 = np.sum(data2 ** 2, 1).reshape(1, -1)
			l2 = l2_1 + l2_2 - 2 * np.dot(data1, data2.T)
			kernal = np.exp(- l2 / (2 * np.power(self.gaussian_sigma, 2)))
		else:
			raise NameError("No such kernal!")
		return kernal

	def calc_errs(self, index, label):
		score = np.sum((self.alpha * self.train_labels).reshape(-1) * self.K[:, int(index)]) + self.b
		err = score - float(label)
		return err

	def search_j(self, i, err_i):
		j = -1 
		max_delta_err = 0
		self.err_cache[i] = np.array([1, err_i])
		valid_err_cache = np.nonzero(self.err_cache[:, 0])
		if len(valid_err_cache) > 1:
			for k in valid_err_cache:
				if k == i: continue
				err_k = self.calc_errs(k, self.train_labels[k, 0])
				delta_err = np.abs(err_k - err_i)
				if delta_err > max_delta_err:
					j = k
					err_j = err_k
					max_delta_err = delta_err
		else:
			j = np.random.randint(0, self.train_data.shape[0])
			while i == j: j = np.random.randint(0, self.train_data.shape[0])
			err_j = self.calc_errs(j, self.train_labels[j, 0])
		return j, err_j

	def update_err_cache(self, index):
		err = self.calc_errs(index, self.train_labels[index, 0])
		self.err_cache[index] = np.array([1, err])

	def scores(self, data):
		if len(data.shape) == 1:
			data_ = data.reshape(1, -1)
		K = self.calc_kernal(self.train_data, data, self.kernal_mode)
		scores = (np.dot((self.alpha * self.train_labels).reshape(1,-1), K) + self.b)[0]
		if len(data.shape) == 1: return scores[0]
		else: 				     return scores
