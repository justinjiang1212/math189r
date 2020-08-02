"""
Starter file for nmf of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features
	3) Let k be the number of topics

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from sklearn.feature_extraction import text
import time

#########################################
#			 Helper Functions	    	#
#########################################
def nmf_cost(X, W, H):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x n
			2) W, a matrix with dimension m x k
			3) H, a matrix with dimension k x n

		This function calculates and returns the cost defined by
		|X - WH|^2.

		HINT:
			1) Use X.tocoo() to cast a matrix into sparse matrix
			2) Use S.row to access the row index of nonzero entries for 
				a sparse matrix S
			3) Use S.col to access the column index of nonzero entries for 
				a sparse matrix S
			4) Use S.data to access the value of nonzero entries for a 
				sparse matrix S
	"""

	cost = 0.
	# TODO: Calculate the cost for nmf algorithm
	"*** YOUR CODE HERE ***"
	sparse_X = X.tocoo()
	for index in range(len(sparse_X.data)):
		row = sparse_X.row[index]
		col = sparse_X.col[index]
		data = sparse_X.data[index]
		cost += (data - (W[row, :] @ H[:, col]).item(0)) ** 2

	"*** END YOUR CODE HERE ***"
	return cost

def nmf(X, k=20, max_iter=100, print_freq=5):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x n
			2) k, the number of latent factors
			3) max_iter, the maximum number of iterations
			4) print_freq, the frequency of printing the report

		This function runs the nmf algorithm and returns the following:
			1) W, a matrix with dimension m x k
			2) H, a matrix with dimension k x n
			3) cost_list, a list of costs at each iteration
	"""
	m, n = X.shape
	W = np.abs(np.random.randn(m, k) * 1e-3)
	H = np.abs(np.random.randn(k, n) * 1e-3)
	cost_list = [nmf_cost(X, W, H)]
	t_start = time.time()
	
	for iter_num in range(max_iter):
		# TODO: Update H and W according to the algorithm
		# Calculate cost and append to the list
		"*** YOUR CODE HERE ***"
		H = H * (W.T @ X) / ((W.T @ W) @ H)
		W = W * (X @ H.T) / (W @ (H @ H.T))
		cost = nmf_cost(X, W, H)
		cost_list.append(cost)
		"*** END YOUR CODE HERE ***"
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost: {:.4E}'.format(iter_num + 1, \
				cost))

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for running nmf: {t:4.2f} seconds'.format(\
			t=t_end - t_start))

	return W, H, cost_list

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')
	# NOTE: Run nltk.download() in your python shell to download
	# the reuters dataset under Corpora tab
	X = np.array([' '.join(list(reuters.words(file_id))).lower() \
		for file_id in reuters.fileids()])
	tfidf = text.TfidfVectorizer()
	X = tfidf.fit_transform(X)
	# =============STEP 1: RUNNING NMF=================
	# NOTE: Fill in code in nmf_cost(), nmf() for this step
	print('==> Running nmf algorithm on the dataset...')
	W, H, cost_list = nmf(X, k=20, print_freq=10)
	# =============STEP 2: CONVERGENCE PLOT=================
	print('==> Generating convergence plot...')
	plt.style.use('ggplot')
	plt.plot(cost_list)
	plt.xlabel('iteration')
	plt.ylabel('cost')
	plt.title('Reuters NMF Convergence Plot with {} Topics'.format(H.shape[0]))
	plt.savefig('nmf_cvg.png', format='png')
	plt.close()
	# =============STEP 3: FIND MOST FREQUENT WORDS=================
	print('==> Finding most frequent words for each topic...')
	num_top_words = 10
	'''
		NOTE: This corresponds to the largest values of each row of H
		HINT:
				1) Use np.argsort to find the index of the sorted array
				2) Use np.flip to flip the order of an array
				3) ind should have the shape of k x num_top_words
	'''
	# TODO: Find the index of the most frequent words for each topic
	"*** YOUR CODE HERE ***"
	ind = np.flip(np.argsort(H, axis=1), 1)
	
	"*** END YOUR CODE HERE ***"
	top_words = np.array(tfidf.get_feature_names())[ind]
	np.set_printoptions(threshold=np.nan)
	for topic_ind in range(H.shape[0]):
		print('-- topic {}: {}'.format(topic_ind + 1, top_words[topic_ind, :num_top_words]))
