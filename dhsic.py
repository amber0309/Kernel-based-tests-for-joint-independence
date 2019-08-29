"""
Implementation of dHSIC proposed in [1]
(Note: the method eigenvalue is still problematic)

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-08-29

[1] Pfister, Niklas, et al. "Kernel‐based tests for joint independence." 
    Journal of the Royal Statistical Society: Series B (Statistical Methodology)
    80.1 (2018): 5-31.
"""
import numpy as np
from copy import deepcopy
import math
from scipy.stats import gamma
from scipy.spatial.distance import cdist
from statsmodels.distributions.empirical_distribution import ECDF

def dhsic_test(X, alpha=0.05, method='permutation'):
	'''
	conduct joint independence test 
	If the test statitic is larger than the critical value,
	H_0 (X_1, ..., X_d are jointly independent) is rejected

	INPUT
	  X 		- matrix of all instances, (n_samples, n_vars) numpy array
	  alpha 	- level of the test
	  method 	- method of the test

	OUTPUT
	  dict 		- 'stat': test statistic
	  			- 'critical_val': critical value
	  			- 'p_val': p value
	  			- 'method': name of the test method
	'''

	n, d = X.shape
	if n < 2*d:
		print('Sample size is smaller than twice the number of variables. Test is trivial')
		return None

	K = compute_K_list(X, n ,d)
	dHSIC = compute_dHSIC(K, n, d)

	B = 1000
	if method == 'permutation':
		dHSIC_perm = np.array( [ dhsic_perm_fun(K, n, d) for _ in range(B) ])
		sortdHSIC = np.sort( dHSIC_perm*n )
		Bind = sum( n*dHSIC == sortdHSIC ) + math.ceil( (1-alpha)*(B+1) )
		if Bind < B:
			critical_val = sortdHSIC[Bind]
		else:
			critical_val = float('inf')
		p_val = ( sum( dHSIC_perm >= dHSIC )+1 ) / (B+1)
	elif method == 'bootstrap':
		dHSIC_boot = np.array( [ dhsic_boot_fun(K, n, d) for _ in range(B) ])
		sortdHSIC = np.sort( dHSIC_boot*n )
		Bind = sum( n*dHSIC == sortdHSIC ) + math.ceil( (1-alpha)*(B+1) )
		if Bind < B:
			critical_val = sortdHSIC[Bind]
		else:
			critical_val = float('inf')
		p_val = ( sum(dHSIC_boot>=dHSIC)+1 ) / (B+1)
	elif method == 'gamma':
		est_a, est_b, est_c = np.zeros((d,), dtype=float), np.zeros((d,), dtype=float), np.zeros((d,), dtype=float)
		for j in range(0, d):
			est_a[j] = np.sum(K[j]) / (n*n)
			est_b[j] = np.sum(K[j]**2) / (n*n)
			est_c[j] = np.sum( np.sum(K[j], axis=1)**2 ) / (n*n*n)
		prod_a = np.prod(est_a)
		prod_b = np.prod(est_b)
		prod_c = np.prod(est_c)
		oneoutprod_a = np.zeros( (d,), dtype=float )
		oneoutprod_b = np.zeros( (d,), dtype=float )
		oneoutprod_c = np.zeros( (d,), dtype=float )
		for j in range(0, d):
			oneoutprod_a[j] = prod_a / est_a[j]
			oneoutprod_b[j] = prod_b / est_b[j]
			oneoutprod_c[j] = prod_c / est_c[j]

		est_d = est_a**2
		prod_d = prod_a**2
		oneoutprod_d = oneoutprod_a**2
		exp_est = (1-sum(oneoutprod_a)+(d-1)*prod_a) / n
		term1 = prod_b
		term2 = (d-1)**2 * prod_d
		term3 = 2*(d-1)*prod_c
		term4, term5, term6, term7 = 0, 0, 0, 0
		for r in range(0, d-1):
			term4 = term4 + est_b[r]*oneoutprod_d[r]
			term5 = term5 + est_b[r]*oneoutprod_c[r]
			term6 = term6 + est_c[r]*oneoutprod_d[r]
			for s in range(r+1, d):
				term7 = term7 + 2*est_c[r]*est_c[s]*oneoutprod_d[r]/est_d[s]
		term4 = term4 + est_b[d-1]*oneoutprod_d[d-1]
		term5 = -2*( term5 + est_b[d-1]*oneoutprod_c[d-1] )
		term6 = -2*(d-1)*(term6 + est_c[d-1]*oneoutprod_d[d-1])

		factor1 = n-2*d
		factor2 = n*(n-1)*(n-2)
		for j in range(0, 2*d-2):
			factor1 = factor1*(n-2*d-j)
			factor2 = factor2*(n-2-j)

		var_est = 2*factor1*(term1+term2+term3+term4+term5+term6+term7)/factor2
		a = exp_est**2 / var_est
		b = n * var_est / exp_est
		critical_val = gamma.ppf(1-alpha, a, scale=b)
		p_val = 1 - gamma.cdf(n*dHSIC, a, scale=b )
	elif method == 'eigenvalue':
		est1 = np.zeros((d,), dtype=float)
		est2 = np.zeros( (n, d), dtype=float )
		for j in range(0, d):
			est1[j] =  np.sum( K[j]-np.diag(np.diag(K[j])) ) / (n*(n-1))
			est2[:,j] = np.sum( K[j], axis=1 )/n

		est1_prod = np.prod(est1)
		est2_prod = np.prod(est2, axis=1)

		a1 = np.ones((n,n), dtype=float)
		a2 = (d-1)**2 * est1_prod
		a3 = (d-1) * est2_prod
		a5 = np.zeros((n,n), dtype=float)
		a6 = np.zeros((n,n), dtype=float)
		a8 = np.zeros((n,n), dtype=float)
		a9 = np.zeros((n,1), dtype=float)
		for j in range(0, d):
			a1 = a1 * K[j]
			a5 = a5 + K[j] * est1_prod/est1[j]
			a6 = a6 + K[j] * np.tile( (est2_prod/est2[:,j]).reshape(-1,1), (1,n) )
			a9 = a9 + (est2[:,j] * est1_prod / est1[j]).reshape(-1,1)
			j2 = j+1
			while j2 < d:
				a8 = a8 + np.dot(est2[:,j].reshape(-1,1), est2[:,j2].reshape(1,-1)) * est1_prod / (est1[j]*est1[j2]) \
					+ np.dot(est2[:,j2].reshape(-1,1), est2[:,j].reshape(1,-1)) * est1_prod / (est1[j]*est1[j2])
				j2 += 1
		a3 = np.tile(a3.reshape(-1,1), (1,n))
		a4 = a3.T
		a6 = -a6
		a7 = a6.T
		a8 = a8
		a9 = np.tile(a9, (1,n))
		a10 = a9.T

		H2 = (a1+a2+a3+a4+a5+a6+a7+a8+a9+a10) / (d*(2*d-1))
		eigenvalues = np.sort( np.linalg.eigvalsh(H2)/n )[::-1].reshape(-1,1)

		M = 5000
		Z = [ np.random.randn(n,1)**2 * eigenvalues for _ in range(M) ]
		chi_dist = d*(2*d-1)*np.sum( np.concatenate(Z, axis = 1), axis = 0 )
		critical_val = np.percentile( chi_dist, (1-alpha)*100 )
		ecdf = ECDF(chi_dist)
		p_val = 1 - ecdf(dHSIC*n)

	return {'stat': dHSIC*n, 'critical_val': critical_val, 'p_val': p_val, 'method':method }

def dhsic(X):
	'''
	compute the dHSIC value 

	INPUT
	  X 		- matrix of all instances, (n_samples, n_vars) numpy array

	OUTPUT
	  dHSIC		- dHSIC value
	'''

	n, d = X.shape
	if n < 2*d:
		print('Sample size is smaller than twice the number of variables. Test is trivial')
		return None

	K = compute_K_list(X, n ,d)
	dHSIC = compute_dHSIC(K, n, d)
	return dHSIC

def median_bandwidth(dist_mat):
	n = dist_mat.shape[0]
	middle = n*(n-1)//4 
	id_tril = np.tril_indices(n, -1)
	sorted_bandwith = np.sort( dist_mat[id_tril] )
	return np.sqrt( 0.5 * sorted_bandwith[middle] )

def gaussian_grammat(dist_mat, bandwidth):
	return np.exp( -dist_mat / (2 * bandwidth**2) )

def compute_K_list(X, n, d):
	X_list = [ X[:,j].reshape(-1,1) for j in range(0, d) ]
	distmat_list = [ cdist(X_list[j], X_list[j])**2 for j in range(0, d) ]
	bandwidth_list = [ median_bandwidth( distmat_list[j] ) for j in range(0,d) ]
	K = [ gaussian_grammat( distmat_list[j], bandwidth_list[j] ) for j in range(0, d) ]
	return K

def compute_dHSIC(K, n, d):
	term1, term2, term3 = 1, 1, 2/n
	for j in range(0, d):
		term1 = term1*K[j]
		term2 = term2*np.sum(K[j])/(n*n)
		term3 = term3*np.sum( K[j], axis=0 ) / n
	term1 = np.sum(term1)
	term3 = np.sum(term3)
	dHSIC = term1/(n*n) + term2 - term3
	return dHSIC


def dhsic_perm_fun(K, n, d):
	term1 = K[1]
	term2 = np.sum(K[1])/(n*n)
	term3 = 2 * np.sum(K[1], axis=0) / (n*n)

	for j in range(1, d):
		perm = np.random.permutation(np.arange(n))
		Kperm = shuffle_grammat(K[j], perm)
		term1 = term1*Kperm
		term2 = term2*np.sum(Kperm)/(n*n)
		term3 = term3*np.sum(Kperm, axis=0) / n
	term1 = np.sum(term1)
	term3 = np.sum(term3)
	return term1/(n*n) + term2 - term3

def dhsic_boot_fun(K, n, d):
	term1 = K[1]
	term2 = np.sum(K[1])/(n*n)
	term3 = 2 * np.sum(K[1], axis=0) / (n*n)

	for j in range(1, d):
		boot = np.random.choice(n, n)
		Kboot = shuffle_grammat(K[j], boot)
		term1 = term1*Kboot
		term2 = term2*np.sum(Kboot)/(n*n)
		term3 = term3*np.sum(Kboot, axis=0) / n
	term1 = np.sum(term1)
	term3 = np.sum(term3)
	return term1/(n*n) + term2 - term3

def shuffle_grammat(K_j, perm):
	n = K_j.shape[0]
	Kperm = deepcopy(K_j)

	Kperm[:, :] = Kperm[perm, :]
	Kperm[:, :] = Kperm[:, perm]
	return Kperm
