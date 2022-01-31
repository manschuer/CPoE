import numpy as np
import GPy
from mix import inv_logDet, dot3lr, dot3rl, inv_c, diag_HtKH 

import copy, time



#### in old versions, sig2_noise could be specified, now a GPy Gaussian likelihood should be provided

class PoE:
	def __init__(self, X_trains_k, y_trains_k, kern, sig2_noise, ARD=True, CORR=None, y_test=None, X_test=None, f_fest=None, centers=None ):
		# X_trains_k, y_trains_k: list of length J with mini-batch data
		# subset_inds: array of length N with entries in 0:J-1
		# βj_version in {'1', '1/M', 'diff', 'diff_scaled'}, the latter three correspond to robust versions
		# BCMprior: Bool, True=>BCM, False=>PoE
		# CORRR: {None->independent, VS1->against exp1}


		# J: number of experts

		self.X_trains_k = X_trains_k
		self.y_trains_k = y_trains_k

		self.D = X_trains_k[0].shape[1] 	# dimension of data
		#self.N = len(y_train)		# number of total samples
		self.J = len(X_trains_k) 				# number of experts


		self.CORR = CORR

		
	
		self.centers = centers


		# create and train J experts
		#range_inds = np.arange(self.N)
		self.experts = []

		self.lik = 0

		#if not likelihood==None:
		#	sig2_noise = likelihood.variance[0]

		if CORR==None:
			for j in range(self.J):
				expert_j = Expert( self.X_trains_k[j], self.y_trains_k[j], kern.copy(), sig2_noise)
				expert_j.create_GPmodel()
				self.lik += expert_j.lik
				self.experts.append( expert_j )

				# if OPT:
				# 	self.models[j].optimize			## tain it!!!!!!!!!

		if CORR=='VS1':
			expert_0 = Expert( self.X_trains_k[0], self.y_trains_k[0], kern.copy(), sig2_noise )
			expert_0.create_GPmodel()
			self.lik += expert_0.lik
			self.experts.append( expert_0 )

			for j in range(1,self.J):
				Xj = np.concatenate([self.X_trains_k[0], self.X_trains_k[j] ])
				yj = np.concatenate([self.y_trains_k[0], self.y_trains_k[j] ])

				expert_j = Expert( Xj, yj , kern, sig2_noise)
				self.lik += expert_j.lik
				expert_j.create_GPmodel()
				self.experts.append( expert_j )

	def optimize(self):
		for j in range(self.J):
			self.experts[j].optimizeHypers()


	def extract_hypers(self):

		nHyp = 3
		Traces = np.zeros((nHyp, self.J))
		for j in range(self.J):
			Traces[0,j] = self.experts[j].GPmod.rbf.variance[0]
			Traces[1,j] = self.experts[j].GPmod.rbf.lengthscale[0]

			Traces[-1,j] = self.experts[j].GPmod.Gaussian_noise.variance[0]

		return Traces



	def predict(self, X_test):
		for j in range(self.J):
			self.experts[j].predict(X_test)

	def aggregate_minVar(self, X_test, PRED=True):		

		N_test = X_test.shape[0]
		agg = Aggregation(N_test)

		if PRED:
			self.predict(X_test)

		agg.p = np.ones(N_test)*1e-20
		agg.n = np.zeros(N_test)

		for j in range(self.J):

			exp_j = self.experts[j]

			inds_j = exp_j.v < 1/agg.p
			agg.p[inds_j] = 1/exp_j.v[inds_j]
			agg.n[inds_j] = exp_j.m[inds_j]/exp_j.v[inds_j]

		
		
		agg.v = 1/agg.p
		agg.m = agg.v * agg.n

		agg.CI[:,0] = agg.m - 1.96*np.sqrt(agg.v)
		agg.CI[:,1] = agg.m + 1.96*np.sqrt(agg.v)

		return agg

	def aggregate_NN(self, X_test, PRED=True):		

		N_test = X_test.shape[0]
		agg = Aggregation(N_test)

		#if PRED:
		#	self.predict(X_test)

		ins_J = [ self.closestInds_xt( xt, self.centers, 1)[0] for xt in X_test]
		
		#if PRED:
		MV = np.array( [self.experts[ins_J[i]].predictMV(X_test[i:(i+1),:]) for i in range(N_test) ] )


		agg.m =  MV[:,0]
		agg.v =  MV[:,1]


		#MMs = np.vstack( [self.experts[j].m for j in range(self.J)] )
		#VVs = np.vstack( [self.experts[j].v for j in range(self.J)] )


		#agg.m = np.take_along_axis(MMs, np.vstack(ins_J).T, axis=0)[0,:]
		#agg.v = np.take_along_axis(VVs, np.vstack(ins_J).T, axis=0)[0,:]

	
		agg.CI[:,0] = agg.m - 1.96*np.sqrt(agg.v)
		agg.CI[:,1] = agg.m + 1.96*np.sqrt(agg.v)

		return agg

	def closestInds_xt(self, xt, centers, NN):
		
		dist = (centers - xt)**2
		
		dist_xt = np.sqrt( np.sum(dist, axis=1) )

		return np.sort( np.argsort(dist_xt)[:NN] )



	def aggregate(self, X_test, β=1, BCM=False, PRED=True):		# β either 1 or 1/J

		N_test = X_test.shape[0]
		agg = Aggregation(N_test)

		if PRED:
			self.predict(X_test)

		for j in range(self.J):

			exp_j = self.experts[j]
			
			agg.p += β/exp_j.v
			agg.n += β*exp_j.m/exp_j.v

		
		if BCM:
			## depends on the hypers_j!!!!!!!!!!!!!
			v_prior_0 = self.experts[0].GPmod.kern.Kdiag(X_test) + self.experts[0].GPmod.Gaussian_noise.variance[0]	
			agg.p -= (self.J*β-1)/v_prior_0
		
		agg.v = 1/agg.p
		agg.m = agg.v * agg.n

		agg.CI[:,0] = agg.m - 1.96*np.sqrt(agg.v)
		agg.CI[:,1] = agg.m + 1.96*np.sqrt(agg.v)

		return agg

	def KL1(self, m1,m2,v1,v2):
		return 0.5*(np.log(v2/v1) + (v1 + (m1-m2)**2)/v2 - 1)


	def aggregate_diff(self, X_test, BCM=False, PRED=True, SCAL=False):
		# KL = 0, 1, 2

		N_test = X_test.shape[0]
		agg = Aggregation(N_test)

		if PRED:
			self.predict(X_test)

		agg.sum_j_βs_j = np.zeros(N_test)
		#agg.sum_j_βs_j = np.ones(N_test)*1e-10

		# save betas in aggregation!!
		agg.βs_J = np.zeros((self.J, N_test))

		# only temporary
		agg.βs_JT = np.zeros((self.J, N_test))

		for j in range(self.J):

			exp_j = self.experts[j]

			## depends on the hypers_j!!!!
			v_prior_j = exp_j.GPmod.kern.Kdiag(X_test) + exp_j.GPmod.Gaussian_noise.variance[0]			
			
			
			#βs_j = 0.5*( np.log(v_prior_j) -  np.log(np.minimum(1e10,exp_j.v)) )				#########MINIMUM!!
			βs_j = 0.5*( np.log(v_prior_j) -  np.log(exp_j.v) )	     + 1e-10		# 1

			#print(βs_j)

			agg.sum_j_βs_j += βs_j

			agg.βs_JT[j,:] = βs_j
			
			if not SCAL:
				agg.p += βs_j/exp_j.v
				agg.n += βs_j*exp_j.m/exp_j.v

		if SCAL:
			for j in range(self.J):

				exp_j = self.experts[j]
				agg.βs_J[j,:] = agg.βs_JT[j,:]/agg.sum_j_βs_j

				agg.p += agg.βs_J[j,:]/exp_j.v
				agg.n += agg.βs_J[j,:]*exp_j.m/exp_j.v

		if BCM:
			agg.p -= (agg.sum_j_βs_j - 1)/v_prior_j			##### v_prior_j depends on hypers!!!
			
		agg.v = 1/agg.p 
		agg.m = agg.v * agg.n

		agg.CI[:,0] = agg.m - 1.96*np.sqrt(agg.v)
		agg.CI[:,1] = agg.m + 1.96*np.sqrt(agg.v)

		return agg



	def aggregate_VS1(self, X_test, PRED=True, βs_mode='DIFF'):
		# note that the models have then be created with CORR='VS1'
		# βs_mode = {'DIFF','ONE'}

		N_test = X_test.shape[0]
		agg = Aggregation(N_test)

		if PRED:
			self.predict(X_test)

		sum_j_βs_j = np.zeros(N_test)		# sum from j=2,...,J

		# expert 0
		exp_0 = self.experts[0]
		m_0 = exp_0.m
		v_0 = exp_0.v
		p_0 = 1/v_0
		n_0 = m_0*p_0


		# starting from j=2...
		for j in range(1,self.J):

			exp_j = self.experts[j]

			if βs_mode=='DIFF':
				## depends on the hypers_j!!!!	
				if j==1:
					βs_j = 1
				else:
					βs_j = 0.5*( np.log(v_0) -  np.log(exp_j.v) )
			elif βs_mode=='ONE':
				βs_j = 1

			sum_j_βs_j += βs_j
			
			agg.p += βs_j/exp_j.v
			agg.n += βs_j*exp_j.m/exp_j.v

		# expert 0
		agg.p -= (sum_j_βs_j - 1)*p_0
		agg.n -= (sum_j_βs_j - 1)*n_0
			
		agg.v = 1/agg.p 
		agg.m = agg.v * agg.n

		agg.CI[:,0] = agg.m - 1.96*np.sqrt(agg.v)
		agg.CI[:,1] = agg.m + 1.96*np.sqrt(agg.v)

		return agg






class Expert:
	def __init__(self, X_train, y_train, kern, sig2_noise=0.1, RR=None, likelihood=None, priorN=None):
		# X_train, y_train: mini-batch of expert

		
		self.P = len(y_train)
		self.D = X_train.shape[1]

		self.X = X_train
		self.y = y_train

		self.kern = kern

		if not likelihood==None:
			self.sig2_noise = likelihood.variance[0]
		else:
			self.sig2_noise = sig2_noise

		self.likelihood = likelihood
		self.priorN = priorN

		self.RR = RR

		

		self.lik = 0

		#self.GPmod
		#self.inds


	def create_GPmodel(self):

		# define SE kernel
		#kern = GPy.kern.RBF(input_dim = self.D, variance = θ_init['σ0']**2, lengthscale = θ_init['ls'].copy(), ARD = ARD)
		#self.kern = kern
	
	
		# define full GP model operating on certain subsets of data
		self.GPmod = GPy.models.GPRegression(self.X, self.y[:,None], kernel = self.kern, noise_var=self.sig2_noise)
		#self.GPmod.Gaussian_noise.variance = self.likelihood.variance # doesnt work
		if not self.priorN==None:
			self.GPmod.Gaussian_noise.variance.set_prior(self.priorN)

		self.lik = self.GPmod.log_likelihood()

	
		#self.GPmod.Gaussian_noise = θ_init['σn']**2
		#time3 = time.time() - start

		#start = time.time()
		# if not θ_EST['σn']:
		# 	self.GPmod.Gaussian_noise.constrain_fixed()

		# if not θ_EST['σ0']:
		# 	self.GPmod.rbf.variance.constrain_fixed()

		# if not θ_EST['ls']:
		# 	self.GPmod.rbf.lengthscale.constrain_fixed()


	def optimizeHypers(self, sig2_noise_EST=True, NRAND=0):
		# might some fixed

		if not sig2_noise_EST:
			#print('noise FIXED!!')
			self.GPmod.Gaussian_noise.variance.constrain_fixed()

		#print('variance FIXED!!')
		#self.GPmod.kern.variance.constrain_fixed()

		if NRAND==0:
			self.GPmod.optimize()
		else:
			self.GPmod.optimize_restarts(NRAND)


		self.sig2_noise = self.GPmod.Gaussian_noise.variance[0]

		





	def create_GPmodel_sparse(self, MOD='VarDTC'):


		self.MOD = MOD

		# define SE kernel
		#kern = GPy.kern.RBF(input_dim = self.D, variance = θ_init['σ0']**2, lengthscale = θ_init['ls'].copy(), ARD = ARD)

		# define sparse GP model on all data
		self.GPmod = GPy.models.SparseGPRegression(self.X, self.y[:,None], kernel = self.kern, num_inducing=self.RR.shape[0])
		if MOD=='VarDTC':
			self.GPmod.inference_method = GPy.inference.latent_function_inference.VarDTC()
		if MOD=='FITC':
			self.GPmod.inference_method = GPy.inference.latent_function_inference.FITC()
		self.GPmod.Gaussian_noise = self.sig2_noise
		self.GPmod.inducing_inputs = self.RR

		if not self.priorN==None:
			self.GPmod.Gaussian_noise.variance.set_prior(self.priorN)


		if MOD=='FITC':
			self.lik = self.GPmod.log_likelihood()
		else:
			self.lik = self.GPmod.log_likelihood()[0][0]
		

		#self.GPmod.optimize(verbose=True)

	def optimize_sparse(self, OPT_R=False, OPT_TH=True, sig2_noise_EST=True):


		if not sig2_noise_EST:
			self.GPmod.Gaussian_noise.variance.constrain_fixed()
		
		if not OPT_R:
			self.GPmod.inducing_inputs.constrain_fixed()
			#self.GPmod.IND_FIX #such that gradients are not really computed!!

		if not OPT_TH:
			self.GPmod.Gaussian_noise.constrain_fixed()
			self.GPmod.kern.constrain_fixed()
			#self.GPmod.rbf.variance.constrain_fixed()
			#self.GPmod.rbf.lengthscale.constrain_fixed()

		self.GPmod.optimize()

		#print(self.GPmod.rbf.variance)

		if self.MOD=='FITC':
			self.lik = self.GPmod.log_likelihood()
		else:
			self.lik = self.GPmod.log_likelihood()[0][0]


	def predict(self, X_test):

		self.m, self.v = self.GPmod.predict( X_test ) # only diagonal predictive variances

		if len(self.m.shape)>1:
			self.m = self.m[:,0]
		if len(self.v.shape)>1:
			self.v = self.v[:,0]

		self.CI = np.zeros((len(self.m), 2))
		self.CI[:,0] = self.m - 1.96*np.sqrt(self.v)
		self.CI[:,1] = self.m + 1.96*np.sqrt(self.v)

		#return self.m, v[:,0]


	def predictMV(self, X_test):

		self.m, self.v = self.GPmod.predict( X_test ) # only diagonal predictive variances

		if len(self.m.shape)>1:
			self.m = self.m[:,0]
		if len(self.v.shape)>1:
			self.v = self.v[:,0]

		return self.m[0], self.v[0]



	def predict_CI(self, X_test):

		m, v = self.GPmod.predict( X_test ) # only diagonal predictive variances

		self.m = m[:,0]
		self.v = v[:,0]

		self.CI = np.zeros((len(self.m), 2))
		self.CI[:,0] = self.m - 1.96*np.sqrt(self.v)
		self.CI[:,1] = self.m + 1.96*np.sqrt(self.v)

		#return m, v, m - 1.96*np.sqrt(v), m + 1.96*np.sqrt(v)




class Aggregation:

	def __init__(self, N_test):

		self.n = np.zeros(N_test)		# natural means: sum_j m_j / v_j
		self.p = np.zeros(N_test)		# precisions sum_j 1/v_j

		#self.m 							# means
		#self.v 							# variances

		self.CI = np.zeros((N_test, 2))







class GP_posterior:

	def __init__(self, kern, μ, Σ, RR, σn2):

		self.jit = 1e-7
		self.M = len(μ)

		self.kern = kern 		### not copied!!!
		self.μ = copy.deepcopy(μ)
		self.Σ = copy.deepcopy(Σ)
		self.σn2 = σn2
		self.RR = copy.deepcopy(RR[:])
		self.iKrr = np.linalg.inv( kern.K(RR) + np.eye(self.M)*self.jit )

	# # cannot directly deepcopy it
	# def deep_copy(self):

	# 	new = self.copy()		# soft copy
	# 	new.μ 


	def move_to_new_Rs(self, RRnew):

		self.μ, self.Σ = self.predict_full( RRnew )

		self.M = RRnew.shape[0]
		self.iKrr = np.linalg.inv( self.kern.K( RRnew ) + np.eye(self.M)*self.jit )
		self.RR = copy.deepcopy(RRnew)


	def move_to_new_Rs_backward(self, RRnew):

		self.μ, self.Σ = self.predict_back( RRnew )

		self.M = RRnew.shape[0]
		self.iKrr = np.linalg.inv( self.kern.K( RRnew ) + np.eye(self.M)*self.jit )
		self.RR = copy.deepcopy(RRnew)



	def predict(self, Z):
		# only diag
		# including noise

		Kxr = self.kern.K(Z, self.RR)
		kzz = self.kern.Kdiag( Z ) + self.jit

		A = np.dot(Kxr, self.iKrr )
		R = kzz - np.sum( A * Kxr, 1)

		m = np.dot(A,self.μ )
		v = diag_HtKH( A.T, self.Σ )  + R

		# noise
		v += self.σn2

		return m, v



	def predict_full(self, Z):
		# full cov
		# without noise

		Kzr = self.kern.K(Z, self.RR)
		Kzz = self.kern.K( Z ) + np.eye(Z.shape[0])*self.jit

		A = np.dot(Kzr, self.iKrr )
		R = Kzz - np.dot(A, Kzr.T)

		m = np.dot(A,self.μ )
		V = np.dot(np.dot(A,self.Σ),A.T) + R

		return m, V


	def predict_back( self, Z):
	    

	    Kzr = self.kern.K(Z, self.RR)
	    Kzz = self.kern.K(Z,Z) + np.eye(Z.shape[0])*self.jit
	    
	    iV = np.linalg.inv(self.Σ)
	    n = np.dot(iV, self.μ)
	    
	    iKzz = np.linalg.inv(Kzz)
	    
	    F = np.dot(Kzr.T, iKzz )
	    Q = iKzz - np.dot( np.dot(F.T, self.iKrr), F)
	    
	    n_ = np.dot(F.T,n)
	    iV_ = np.dot(np.dot(F.T,iV),F) + Q
	    
	    V = np.linalg.inv(iV_)
	    m = np.dot(V, n_)
	    
	    return m, V

	def update_likelihood(self, y, X, fact=1.0):
		# attention when sigma_noise tends to inf

		Kxr = self.kern.K(X, self.RR)
		H = np.dot( Kxr, self.iKrr)

		iΣ = np.linalg.inv(self.Σ)
		η = np.dot( iΣ, self.μ )

		iΣ = iΣ + fact* np.dot( H.T, H/self.σn2)
		η = η + fact* np.dot( H.T, y/self.σn2)

		self.Σ = np.linalg.inv(iΣ)
		self.μ = np.dot( self.Σ, η)



