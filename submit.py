import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
	feat=my_map(X_train)
	model = LinearSVC()
	model.fit(feat,y_train)
	w=model.coef_
	b=model.intercept_
	

################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0

	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	n = X.shape[0]
	feat1 = np.zeros((n, 15))
    
    # First 8 features
	feat1[:, :8] = 1 - 2 * X[:, :8]
    
    # 9th feature (index 8)
	feat1[:, 8] = feat1[:, 7] * feat1[:, 6]
    
    # Recursive part (features 9–14, index 9–14)
	b_indices = np.arange(5, 0, -1)  # 5,4,3,2,1
	for idx, b in enumerate(b_indices, start=9):
		feat1[:, idx] = feat1[:, idx - 1] * (1 - 2 * X[:, b])
    
    # Prepare pair indices (upper triangle, j < k)
	j_idx, k_idx = np.triu_indices(15, k=1)
    
    # Compute feature pairs
	feat_pairs = feat1[:, j_idx] * feat1[:, k_idx]
    
   
	feat = np.zeros((n, 105))
	feat[:, :feat_pairs.shape[1]] = feat_pairs

			
				 
		
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	return feat


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
	weights = w[:-1]  # First 64 elements are weights
	bias = w[-1]      # Last element is bias
	alpha = np.zeros(64)
	beta = np.zeros(64)
    
    # The last beta is the bias
	beta[63] = bias 
    # Work backwards to find all alphas and betas
	alpha = weights  # w_0 = α_0

    # Since we need non-negative values, we can set:
	p = np.zeros(64)
	q = np.zeros(64)
	r = np.zeros(64)
	s = np.zeros(64)
    
	for i in range(64):
        # For p_i - q_i = α_i + β_i
		if alpha[i] + beta[i] >= 0:
			p[i] = alpha[i] + beta[i]
		else:
			q[i] = -(alpha[i] + beta[i])
			
		if alpha[i] - beta[i] >= 0:
			r[i] = alpha[i] - beta[i]
		else:
			s[i] = -(alpha[i] - beta[i])
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
	
	return p, q, r, s

