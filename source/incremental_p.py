import numpy as np
from mix import inv_logDet, dot3lr, dot3rl, inv_c, diag_HtKH 
#from scipy.optimize import  root_scalar
import math

from optim import Adam, Adagrad

import properscoring as ps

import time

from numpy_lru_cache_decorator import np_cache
from scipy.optimize import minimize
#from incremental_run import Partition

import GPy
from IPython.display import display, clear_output




def make_mini_batches(X_train, y_train, K, B):
	X_trains_k = []
	y_trains_k = []

	for k in range(K):
			X_trains_k.append( X_train[ (B*(k)):(B*(k+1)), :] )
			y_trains_k.append( y_train[(B*(k)):(B*(k+1))] )

	return X_trains_k, y_trains_k

class INCR:
	def __init__(self, X_trains_k, y_trains_k, K, P, Jp, kern, likelihood, X_test, R=np.array([]), N=None, DIAG=True, \
					MODE='sparse_global', q=0.9, shapeA=1, a=1, r=1, jit=1e-7, alpha=1,gamma=0.01, GRAD_STORE=False, PER=0, HH=1, J=1, J1=1, LINSPACE=True,\
					 KL_and_LIK=False, mvFULL=None, MINVAR_KF=True, SMOOTH=False):
		# X_trains_k, y_trains_k list of K mini-batches
		# at the moment kernel is fixed
		# ((N/K should be int and equals B)) no longer
		# X_test should be provided, at the moment only instan predictions available
		# M is the number of global inducing points, 0 <= M < ..
		# Jp is the fraction of local inducing points 0< Jp <=1.  (0 < J <= B)


		self.X_trains_k = X_trains_k
		self.y_trains_k = y_trains_k

		self.D = X_trains_k[0].shape[1] 	# dimension of data
		#self.N = len(y_train)		# number of total samples
		self.N = N		# number of total samples
		self.K = K 					# number of experts
		self.aB = int(self.N/K)		# mini-batch size (if k-means partition not exactly!!!). in AVERAGE!!
		self.P = P					# length of memory (1<=P<K-1)

		#self.U = U 					# number of mini-batches after parameter update for stochastic version

		self.R = R 					# global inducing points
		self.M = R.shape[0]			# number of global inducing points

		self.Jp = Jp 					# fraction of local inducing points

		if not MODE=='moving': 
			self.aS = np.int(np.ceil(self.Jp*self.aB)*self.P + self.M	)	# state size (approx if k-means) in AVERAGE!!
		else:
			self.aS = np.int(self.aB*(self.P-1)/J ) + np.int(self.aB/J1 )

		self.kern = kern
		#self.sig2_n = [sig2_n] ## !!
		self.likelihood = likelihood

		self.DIAG = DIAG # if using only diagonal or block in likelihood update step

		self.al = alpha #1#1e-10 #0.5			# fix alpha for sparse PEP update, it should be 1!!!!
		# in derivatives not included dak term!!!


		self.gamma = gamma		## learning rate
		#self.STOCH = STOCH

		self.GRAD_STORE = GRAD_STORE

		self.KL_and_LIK = KL_and_LIK
		self.mvFULL = mvFULL

		self.HH = HH


		self.J = J
		self.J1 = J1
		self.LINSPACE = LINSPACE

		self.MINVAR_KF = MINVAR_KF
	

		self.SMOOTH = SMOOTH
	


		#self.jit = 1e-9#1e-6#####1e-7#!!!!!!!!!!!!!!!!!!!!!!
		self.jit = jit

		print('jit ', self.jit)

		self.Ntest = X_test.shape[0]
		self.X_test = X_test

		self.MODE = MODE
		self.q = q
		self.r = r

		self.a = a

		self.PER = PER

		self.shapeA = shapeA

		if self.MODE=='gamma':
			tt = '_shapeA'+str(shapeA)
		elif self.MODE=='betaBinomial':
			tt = '_a='+str(a)
		elif self.MODE=='negbinomial':
			#tt = '_q='+str(q)
			tt = '_r='+str(r)
		else:
			tt = ''

		if self.MODE=='moving':
			tt = '_J='+str(J)+'_J1='+str(J1)

		if not self.DIAG:
			tt = 'blockLik'

		self.name = 'P'+str(P)+'_M'+str(self.M)+'_Jp'+str(self.Jp)+'_K'+str(K)+'_aB'+str(self.aB)+'_aS'+str(self.aS)+'_'+self.MODE+tt


		#self.sortData1D()
		#self.initKF()
		self.initKF_fast()
		

	def initKF_fast(self):
	
		# natural pointwise predictive mean and precision
		# self.n_agg = np.zeros(self.Ntest)
		# self.p_agg = np.zeros(self.Ntest)

		
		self.PARAMS = []

		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]


		for keri in kerns:
			
			#parVar = PARAM('variance', keri, not keri.variance.is_fixed )
			#parLen = PARAM('lengthscale', keri, not keri.lengthscale.is_fixed )			# ARD!!

			#self.PARAMS.append(parVar)
			#self.PARAMS.append(parLen)

			for par in keri.parameters[:]:

				if keri.name[:12] == 'std_periodic':
					nnam = par.name + 'P'
				else:
					nnam = par.name



				parInc = PARAM(nnam, par, keri, not par.is_fixed )
				self.PARAMS.append(parInc)
		
		parNoise = PARAM('noise', self.likelihood.variance, None, not self.likelihood.is_fixed)	

		self.PARAMS.append(parNoise)


		#if self.STOCH:
		lenPar = len(self.get_PARAMS())
		self.OPTIMIZER = Adam(self.gamma, (lenPar,) )

		if self.GRAD_STORE:
			self.PAST_VALS = np.zeros((0,lenPar))
			self.PAST_GRADS = np.zeros((0,lenPar))

		if self.KL_and_LIK:

			self.KLLs = []
			self.LIKs = [] 

	





	def get_PARAMS(self):

		params = np.zeros(0)
		for par in self.PARAMS:

			if par.EST:
				params = np.hstack([params, par.get_value()])

		return params



	def set_PARAMS(self, values):

		#### not automated
		##

		i = 0
		for par in self.PARAMS:

			if par.EST:

				par.update_value( values[i:i+par.length()] )

				i += par.length()




	# # old version when storing all
	# def initKF(self):
	# 	## init for KF
	# 	# save pointwise predictive distribution (mean, variance)
	# 	#self.M_V = np.zeros((self.K,self.Ntest,2))			# updated distribution at k
	# 	#self.M_V_trans = np.zeros((self.K,self.Ntest,2))	# translated distribution at k
	# 	#self.M_V_cumm = np.zeros((self.K,self.Ntest,2))	# cummulative distribution at k
	# 	#self.M_V_minVar_cumm = np.zeros((self.K,self.Ntest,2))	# cummulative minVar distribution at k
	# 	#self.M_V_minVar_cumm_S = np.zeros((self.K,self.Ntest,2))	# cummulative minVar smoothing distribution at k
	# 	# these aren't needed when aggregating instantanousely

	# 	# natural pointwise predictive mean and precision
	# 	self.n_agg = np.zeros(self.Ntest)
	# 	self.p_agg = np.zeros(self.Ntest)

	# 	# marginal likelihood
	# 	self.lik = 0

	# 	# select/find/decide global inducing points
	# 	# if self.M>0:
	# 	# 	indsInducing = np.random.permutation(self.N)[:self.M]
	# 	# 	self.R = self.X_train[indsInducing,:]

	# 	# for comparison also compute minVar filter
	# 	#self.m_minV = np.zeros(self.Ntest)
	# 	#self.v_minV = np.ones(self.Ntest)*1e10
	# 	# and minVar smooth
	# 	#self.m_minV_S = np.zeros(self.Ntest)
	# 	#self.v_minV_S = np.ones(self.Ntest)*1e10


	# 	# save for smoothing afterwards
	# 	#self.mks = []
	# 	#self.Pks = []
	# 	#self.Aks = []
	# 	#self.Qks = []
	# 	#self.Hpreds = []
	# 	#self.Dpreds_diag = []
	# 	#self.Rks = []
	# 	#self.Htrains = []
	# 	#self.Dtrains_diag = []

	# # def sortData1D(self):
	# # 	# sort data X_train and y_train
	# # 	ind_X_train = np.argsort(self.X_train[:,0])
	# # 	self.X_train = self.X_train[ind_X_train,:]
	# # 	self.y_train = self.y_train[ind_X_train]

	# # def randData(self, seed=123):
	# # 	# randomize data X_train and y_train
	# # 	np.random.seed(seed)

	# # 	perm = np.random.permutation(self.N)
	# # 	self.X_train = self.X_train[perm,:]
	# # 	self.y_train = self.y_train[perm]

	

	def funA(self, q, S, pps):
		return np.sum(  pps**q ) - min(S, np.sum(pps>0) )




	# not storing everything
	def apply_KF_deriv(self, PREDICT=True, GRAD=False, seed=123, PRINT=False, STORE=False, STOCH=False, U=1, PREDICT_STORE=False):
		## for batch optimization GRAD has to be True
		## for stochastic optimizatio GRAD and STOCH has to be True, and U can be set


		PREDICT = PREDICT or self.KL_and_LIK

		self.n_agg = np.zeros(self.Ntest)
		self.p_agg = np.zeros(self.Ntest)


		np.random.seed(seed)

		self.Rk = []			## store only few last? for some methods we need... change it 

		if STORE:
			self.Rks = []			## only for visualization (or smoothing?)
			self.mks = []

		if PREDICT_STORE:
			# PREDICT has to be True as well
			# overall
			#self.Ms = np.zeros((self.K, self.Ntest))
			#self.Vs = np.zeros((self.K, self.Ntest))

			#self.HH = HH	## either 1 (only after update) or 2 (also after prediction)

			# overall
			self.Ms = np.zeros((self.HH*self.K, self.Ntest))
			self.Vs = np.zeros((self.HH*self.K, self.Ntest))

			# after update k
			#self.Mks = np.zeros((self.K, self.Ntest))
			#self.Vks = np.zeros((self.K, self.Ntest))

			self.Mks = np.zeros((self.HH*self.K, self.Ntest))
			self.Vks = np.zeros((self.HH*self.K, self.Ntest))


		if self.MINVAR_KF:
			# for comparison also compute minVar filter
			self.m_minV = np.zeros(self.Ntest)
			self.v_minV = np.ones(self.Ntest)*1e10
			self.whichKs = np.zeros(self.Ntest, dtype=int)
		

		if self.SMOOTH:
			# save for smoothing afterwards
			self.mmks = []			#mks already above
			self.Pks = []
			self.Aks = []
			self.Qks = []
			self.Hpreds = []
			self.Dpreds_diag = []
			#self.Rks = [] 					#already somewhere else
			#self.Htrains = []
			#self.Dtrains_diag = []


	

		

		timeKr = 0
		timeKrr = 0
		timeTrans = 0
		timeKxr = 0 
		timeLik = 0
		timePredK = 0
		timePred = 0

		#PRINT = False



		##### make it more uniform for all methods

		Rk0 = np.zeros((0,self.D))

		#indsT = np.zeros(self.N, dtype=int)
		#lifetimes = np.zeros(self.N, dtype=int)  # empirical distribution of lifetimes
		sizes = np.zeros(self.K) # actual sizes of active sets

		lives_active_set = np.zeros(0, dtype=int)  #lives in current active set

		RlocAll1 = 0

		#Rk = [] # already above defined





		Ak = np.zeros((0,self.D))	# current active set

		# only for checking
		#Aks = []				# active sets, using Rks form above
		self.PROBS = []
		self.PROBSq = []


		for k in range(self.K):

			# get current data
			#X_k = self.X_train[ (self.B*(k)):(self.B*(k+1)), :]
			#y_k = self.y_train[(self.B*(k)):(self.B*(k+1))]

			X_k = self.X_trains_k[k]
			y_k = self.y_trains_k[k]



			Bk = X_k.shape[0]
			Jk = np.int(np.ceil(self.Jp*Bk))


	


			#print(Bk)




			# get current inducing points of size BxP (actually full GP of current and past)
			#R1 = self.X_train[np.maximum(0,self.B*(k+1-self.P)):self.B*(k+1), :]  

			if self.MODE == 'sparse_global':
				# sparse and global version	
				Rloc1 = X_k 			#+ np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!
				inds = np.random.permutation(Bk)[:Jk]
				#inds = np.arange(Jk)
				#print('no permutation!!!')
				Rloc1 = Rloc1[inds,:]

				#print(Rloc1.shape[0])

				self.Rk.append(Rloc1)

				if self.P > 1 and k > 0:
					#RlocAll1 = np.concatenate([ RlocAll0[np.maximum(0,RlocAll0.shape[0]-Jk*(self.P-1)):,:], Rloc1 ])
					RlocAll1 = np.concatenate(  self.Rk[np.maximum(0, (k+1)-self.P):(k+1)], axis=0 )

				else:
					RlocAll1 = Rloc1


				## add SHIFTed
				if self.PER!=0 and self.P>=1:
					#RlocAll1_shifted = (RlocAll1[:,0]-self.PER)[:,None]

					if X_k.shape[1]==1:
						RlocAll1_shifted = (RlocAll1[:,0]-np.random.rand(RlocAll1.shape[0])*self.PER)[:,None]
					else:
						RlocAll1_shifted = RlocAll1.copy()
						RlocAll1_shifted[:,0] = (RlocAll1_shifted[:,0]-np.random.rand(RlocAll1.shape[0])*self.PER)

					#print(RlocAll1_shifted.shape)


					RlocAll1 = np.block([[RlocAll1],[RlocAll1_shifted]])



				if self.PER!=0 and self.P==0:
					#RlocAll1_shifted = (RlocAll1[:,0]-self.PER)[:,None]


					RlocAll1_shifted = (Rloc1[:,0]-np.random.rand(Rloc1.shape[0])*self.PER)[:,None]
					RlocAll1 = RlocAll1_shifted



				if self.M>0:
					R1 = np.block([[RlocAll1],[self.R]])
				else:
					R1 = RlocAll1

				if STORE:
					self.Rks.append(R1)

			elif self.MODE=='geometric' or self.MODE=='poisson' or self.MODE=='gamma' or self.MODE=='binomial' or self.MODE=='negbinomial' or self.MODE=='betaBinomial' or self.MODE=='sparse':
				# geometric/probabilistiv markov process
				#indsK = np.arange(self.B*k, B*(k+1))
				#X_k = XX[ indsK, :]


				# #X_kn = X_k + np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!

				# indsQ = np.random.random( X_kn.shape[0] ) <= self.q
				# indsR = np.random.random( Rk0.shape[0] ) <= self.r

				# #indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
				# Rk0 = np.concatenate( [ X_kn[indsQ], Rk0[indsR] ] , axis=0 )

				# #lifetimes[indsT] += 1
				# sizes[k] = Rk0.shape[0]

				# self.Rk.append(Rk0)

				# R1 = Rk0



				Bk = X_k.shape[0]


				if self.MODE=='poisson':
					lif_k = np.random.poisson(lam=self.P, size=Bk )
				elif self.MODE=='geometric':
					lif_k = np.random.geometric(1-self.P/(1+self.P), size=Bk )-1
				elif self.MODE=='binomial':
					lif_k = np.random.binomial(self.K, self.P/self.K, size=Bk )
				elif self.MODE=='negbinomial':
					#lif_k = np.random.negative_binomial(self.q*self.P/(1-self.q), self.q, size=Bk )
					lif_k = np.random.negative_binomial(self.r, self.r/(self.P+self.r), size=Bk )
				elif self.MODE=='betaBinomial':
					#a = 1

					#b = self.a * (self.K -self.P)/self.P
					b = self.a * (self.K - self.P)/self.P

					lif1 = np.random.beta(self.a, b, size=Bk )
					lif_k = np.random.binomial(self.K, lif1)					### K left?
					#lif_k = np.random.binomial(self.K-k, lif1)

				elif self.MODE=='gamma':
					lif_k = np.floor( np.random.gamma(self.shapeA, scale=self.P/self.shapeA, size=Bk ) + 0.5 )	 #!
				elif self.MODE=='sparse':
					x = np.ceil(2*self.P*self.K / (self.K + 1))
					lif_k = np.floor( ( np.random.random(Bk)*self.K**2 - self.K**2 + self.K*x )/x )


	

				indsQ = lif_k > 0
				indsR = lives_active_set > 0


				Rk0 = np.concatenate( [ X_k[indsQ], Rk0[indsR] ] , axis=0 )
				lives_active_set = np.concatenate( [lif_k[indsQ]-1, lives_active_set[indsR]-1 ] )

				#indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
				#lifetimes[indsT] += 1
				sizes[k] = Rk0.shape[0]

				if STORE:
					self.Rks.append(Rk0)

				R1 = Rk0

			elif self.MODE=='kernel':

				S = Bk*self.P

				if k==0:
					probsCand1 = np.zeros((0))
				else:
					probsCand1 = np.mean( self.kern.K(Ak, X_k), 1)
				
				probsCand2 = np.mean( self.kern.K(X_k, X_k), 1)

				probsCon = np.concatenate([probsCand1, probsCand2])  
				probs = probsCon/np.amax(probsCon)   # normalize


				if np.sum(probs**0)<=S:
					qopt = 0
				else:
					qopt = self.q/self.P
				# elif self.funA(1, S, probs)>0:
				# 	qopt = 1
				# else:
				# 	roots = root_scalar(self.funA, args = (S, probs), method='brentq', bracket=[0,1], rtol=1e-2)
				# 	qopt = roots.root

				takes = np.random.binomial(1, probs**qopt)

				# if active set is too large, i.e. qopt==1, subsample it
				nT = np.sum(takes)
				if nT > S:
					indsOne = np.arange(len(takes))[takes==1]
					indsOne = indsOne[ np.random.permutation(len(indsOne)) ]
					takes[indsOne[:(nT-S)]] = 0
					#print('active set larger than ',S,' thus ',nT-S,'random points removed')

				self.PROBS.append(probs)
				self.PROBSq.append(probs**qopt)

				Ak = np.concatenate([ Ak[ takes[:len(probsCand1)]==True, :], X_k[ takes[len(probsCand1):]==True, :] ])

				if STORE:
					self.Rks.append(Ak)  

				R1 = Ak


			elif self.MODE=='moving':

				#X_P_past = np.vstack(self.X_trains_k[np.maximum(k-self.P+1,0):(k+1)])
				#R1 = X_P_past[np.arange(X_P_past.shape[0],0,-self.J)-1,:]


				if self.P>1 and k>0:
					X_P_past = np.vstack(self.X_trains_k[np.maximum(k-self.P+1,0):k])
					
					if self.LINSPACE:
						indPS = np.arange(X_P_past.shape[0],0,-self.J)-1
						
					else:
						NP = X_P_past.shape[0]
						indPS = NP - np.geomspace(1,NP,np.int(NP/self.J),dtype=int)
						indPS = np.unique(indPS)

					#print(indPS,len(indPS),np.int(NP/self.J),NP)
						
					Rpast = X_P_past[indPS,:]
			
					Rcur = X_k[np.arange(Bk,0,-self.J1)-1,:]

					R1 = np.vstack([Rpast, Rcur])

				else:
					R1 = X_k[np.arange(Bk,0,-self.J1)-1,:]


				#print(R1.shape)

				sizes[k] = R1.shape[0]

				if STORE:
					self.Rks.append(R1)  

			# here for all methods we have in R1 the current active set
			# R0 the previous one
			# Krr0 and iKrr0 should also be valid, computed with the old hypers!





			## change here params. NO. IN THE END.




			start = time.time()
			iKrr1, Krr1 = self.comp_kernR(R1)
			timeKr += time.time() - start

			

		



			# A: translation step
			if k==0:
				#mk = np.zeros(Jk+self.M)
				mk = np.zeros(R1.shape[0])
				Pk = Krr1

				# marginal likelihood
				self.lik = 0
				#self.lik = self.kern.log_prior()				######## set it in each mini-batch st stochastic is also working
				#self.ak = 0
				#self.low_bound = 0

				if GRAD:
					self.update_grads(R1, None, X_k)
					self.init_posterior_deriv(R1.shape[0])

			else:


				# only needed in stochastic version
				### recompute iKrr0_new, Krr0_new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				if STOCH:
					start = time.time()
					iKrr0, Krr0 = self.comp_kernR(R0)		### but with new hypers!!!! (only needed when hypers do change in between)
					timeKr += time.time() - start

				start = time.time()
				A_k, _, D_k = self.comp_kernX(R1, R0, iKrr0, DIAG=False)  
				timeKrr += time.time() - start



				if GRAD:
					# compute kernel derivatives and udpate posterior derivatives
					self.update_grads(R1, R0, X_k)
					self.update_derivs_trans(mk, Pk, A_k, iKrr0)





				start = time.time()  
				mk, Pk = self.trans(mk, Pk, A_k, D_k)
				timeTrans += time.time() - start


				if STORE and self.HH==2:
					self.mks.append(mk)









			# compute current predictive distribution after translation
			start = time.time()  
			Hpred, _, dpred = self.comp_kernX(self.X_test, R1, iKrr1)				#A)
			timePredK += time.time() - start

			if PREDICT:
				start = time.time()  
				m, v = self.pred_y_Diag(mk, Pk, Hpred, dpred, self.likelihood.variance[0])
				timePred += time.time() - start

				# aggregate cummulative predictive distribution
				if k!=0:
					self.p_agg -=  1/v
					self.n_agg -= m/v


				if PREDICT_STORE:

					if self.HH ==2:
						indHH = 2*k
						self.Vks[indHH,:] = v
						self.Mks[indHH,:] = m

						self.Vs[indHH,:] = 1/self.p_agg
						self.Ms[indHH,:] = self.Vs[indHH,:] * self.n_agg



			# if STOCH:

			# 	if (k % U) == (U-1):
			# 	##########################################
			# 	########## change here params! ###########
			# 	## update in log space

			# 		print('reset after trans')
			# 		self.reset_derivs()				##### RESET DERIVS to 0!!!!!!!!!!!
			# 		self.init_posterior_deriv(R1.shape[0]) 		# correspond to plug-in





			# B: update step
			start = time.time()  
			if self.DIAG:
				Htrain_k, _, dtrain_k = self.comp_kernX(X_k, R1, iKrr1)
				Dtrain_k = np.diag(dtrain_k)
			else:
				Htrain_k, _, Dtrain_k = self.comp_kernX(X_k, R1, iKrr1, DIAG=False)
			timeKxr += time.time() - start



			start = time.time()  
			# inlcuding derivative update!!
			mk, Pk, lik_k = self.lik_up(mk, Pk, y_k, Htrain_k, self.al*Dtrain_k+ np.eye(Bk)*self.likelihood.variance[0], DERIVS=GRAD, iK_AA=iKrr1)
			timeLik += time.time() - start




			#ak = np.trace(Dtrain_k) / (2*self.sig2_n[0])

			#print(ak)
			#ak_al = np.sum( np.log( np.diag(Dtrain_k)*self.al/self.sig2_n[0] + 1 ) ) *(1-self.al)/(2*self.al)
		
			
			#self.ak += ak_al
			#print(ak_al)
			#self.lik += lik_k #+ ak_al
			self.lik += lik_k + (self.kern.log_prior() + self.likelihood.log_prior())/self.K
			#print('log_prior ', self.kern.log_prior() )



			if PREDICT:
				# compute current predictive distribution after update
				start = time.time() 
				m, v = self.pred_y_Diag(mk, Pk, Hpred, dpred, self.likelihood.variance[0])
				timePred += time.time() - start

				# aggregate cummulative predictive distribution
				self.p_agg +=  1/v
				self.n_agg +=  m/v



				# aggregate cummulative minVar distribution
				indSmaller = v < self.v_minV
				self.v_minV[indSmaller] = v[indSmaller]
				self.m_minV[indSmaller] = m[indSmaller]
				self.whichKs[indSmaller] = k
				# # save it
				# self.M_V_minVar_cumm[k,:,0] = self.m_minV
				# self.M_V_minVar_cumm[k,:,1] = self.v_minV


				if PREDICT_STORE:

					if self.HH==1:
						indHH = k
					else:
						indHH = 2*k + 1
					
					self.Vks[indHH,:] = v
					self.Mks[indHH,:] = m

					self.Vs[indHH,:] = 1/self.p_agg
					self.Ms[indHH,:] = self.Vs[indHH,:] * self.n_agg


			if self.SMOOTH:
				if k!=0:
					self.Aks.append(A_k)
					self.Qks.append(D_k)
				else:
					self.Aks.append([])
					self.Qks.append([])

				self.Hpreds.append(Hpred)
				self.Dpreds_diag.append(dpred)
				self.mmks.append(mk)
				self.Pks.append(Pk)
				#self.Htrains.append(Htrain_k)
				#self.Dtrains_diag.append(dtrain_k)



			# update next inducing
			R0 = R1 				# all: JkxP + M
			RlocAll0 = RlocAll1 	# only JkxP
			iKrr0 = iKrr1			# of all -> compute it more clever!!!!


			if STORE:
				self.mks.append(mk)


			if STOCH:

				if (k % U) == (U-1):
				##########################################
				########## change here params! ###########
				## update in log space

					dliks = np.zeros(0)
					for par in self.PARAMS:
						if par.EST:
							pardlik = par.dlik + par.get_prior_grad()*U/self.K
							dliks = np.hstack( [dliks, - pardlik * par.get_value() ] )		## transforamtion of grads



					newValues = self.OPTIMIZER.update( np.log(self.get_PARAMS()), dliks )
					self.set_PARAMS( np.exp(newValues) )

					self.reset_derivs()				##### RESET DERIVS to 0!!!!!!!!!!!
					#self.init_posterior_deriv(R1.shape[0]) 		# correspond to plug-in


					## this with init_post_deriv gives independent gradients
					#mk = np.zeros(R1.shape[0])
					#Pk = Krr1

					if self.GRAD_STORE:

					

						self.PAST_VALS = np.vstack([self.PAST_VALS, np.exp(newValues)])
						self.PAST_GRADS = np.vstack([self.PAST_GRADS, dliks])

					#print('val :', np.exp(newValues),' grad: ', dliks)

		#if STOCH:
		#	print(' ')
			

		if PREDICT :
			# invert natural cummulative predictive distribution
			self.v_agg = 1/self.p_agg
			self.m_agg = self.v_agg*self.n_agg

			#self.m_last = m
			#self.v_last = v


			# for prediction
			self.CI = np.zeros((len(self.m_agg), 2))
			self.CI[:,0] = self.m_agg - 1.96*np.sqrt(self.v_agg)
			self.CI[:,1] = self.m_agg + 1.96*np.sqrt(self.v_agg)

			
			if self.KL_and_LIK:

				if len(self.mvFULL)==2:
					self.KLLs.append( np.mean( KL1(self.mvFULL[0], self.m_agg, self.mvFULL[1], self.v_agg) ) )
				elif len(self.mvFULL)==1:
					self.KLLs.append( np.mean( ps.crps_gaussian(self.mvFULL[0], mu=self.m_agg, sig=np.sqrt(self.v_agg) ) ) )

				self.LIKs.append(self.lik)

				








		




		# last inducings
		# just to check
		self.R1 = R1
		#self.RlocAll1 = RlocAll1
		self.mk = mk



		self.empS = np.round(np.mean(sizes),1)

		if not self.MODE=='sparse_global' and not GRAD:
			self.name += '_eS'+str(self.empS)




		if PRINT:
		#if True:
			print(self.MODE+'_emp sizes ',self.empS,'expected size ',self.aS)


			print('timeKr ', timeKr)
			print('timeKrr ',timeKrr)
			print('timeTrans ',timeTrans)
			print('timeKxr ' ,timeKxr)
			print('timeLik ',timeLik)
			print('timePredK ',timePredK)
			print('timePred ',timePred)
			#print('T1 ',T1)
			#print('T2 ',T2)
			#print('T3 ',T3)
			print(' ')




	# def generateData(self, VaryDict=None, seed=123):
			

	

	# 	np.random.seed(seed)

	# 	self.fks = []

	# 	self.Rk = []			## store only few last? for some methods we need... change it 





	# 	##### make it more uniform for all methods

	# 	Rk0 = np.zeros((0,self.D))

	# 	#indsT = np.zeros(self.N, dtype=int)
	# 	#lifetimes = np.zeros(self.N, dtype=int)  # empirical distribution of lifetimes
	# 	sizes = np.zeros(self.K) # actual sizes of active sets

	# 	lives_active_set = np.zeros(0, dtype=int)  #lives in current active set

	# 	RlocAll1 = 0



	# 	Ak = np.zeros((0,self.D))	# current active set

	# 	# only for checking
	# 	#Aks = []				# active sets, using Rks form above
	# 	self.PROBS = []
	# 	self.PROBSq = []


	# 	for k in range(self.K):

	# 		# get current data
	# 		#X_k = self.X_train[ (self.B*(k)):(self.B*(k+1)), :]
	# 		#y_k = self.y_train[(self.B*(k)):(self.B*(k+1))]

	# 		X_k = self.X_trains_k[k]
	# 		y_k = self.y_trains_k[k]



	# 		Bk = X_k.shape[0]
	# 		Jk = np.int(np.ceil(self.Jp*Bk))


	# 		#print(Bk)




	# 		# get current inducing points of size BxP (actually full GP of current and past)
	# 		#R1 = self.X_train[np.maximum(0,self.B*(k+1-self.P)):self.B*(k+1), :]  

	# 		if self.MODE == 'sparse_global':
	# 			# sparse and global version	
	# 			Rloc1 = X_k 			#+ np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!
	# 			inds = np.random.permutation(Bk)[:Jk]
	# 			#inds = np.arange(Jk)
	# 			#print('no permutation!!!')
	# 			Rloc1 = Rloc1[inds,:]

	# 			self.Rk.append(Rloc1)

	# 			if self.P > 1 and k > 0:
	# 				#RlocAll1 = np.concatenate([ RlocAll0[np.maximum(0,RlocAll0.shape[0]-Jk*(self.P-1)):,:], Rloc1 ])
	# 				RlocAll1 = np.concatenate(  self.Rk[np.maximum(0, (k+1)-self.P):(k+1)], axis=0 )

	# 			else:
	# 				RlocAll1 = Rloc1

	# 			if self.M>0:
	# 				R1 = np.block([[RlocAll1],[self.R]])
	# 			else:
	# 				R1 = RlocAll1

				

	# 		if self.MODE=='geometric' or self.MODE=='poisson' or self.MODE=='gamma' or self.MODE=='binomial' or self.MODE=='negbinomial' or self.MODE=='betaBinomial' or self.MODE=='sparse':
	# 			# geometric/probabilistiv markov process
	# 			#indsK = np.arange(self.B*k, B*(k+1))
	# 			#X_k = XX[ indsK, :]


	# 			# #X_kn = X_k + np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!

	# 			# indsQ = np.random.random( X_kn.shape[0] ) <= self.q
	# 			# indsR = np.random.random( Rk0.shape[0] ) <= self.r

	# 			# #indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
	# 			# Rk0 = np.concatenate( [ X_kn[indsQ], Rk0[indsR] ] , axis=0 )

	# 			# #lifetimes[indsT] += 1
	# 			# sizes[k] = Rk0.shape[0]

	# 			# self.Rk.append(Rk0)

	# 			# R1 = Rk0



	# 			Bk = X_k.shape[0]


	# 			if self.MODE=='poisson':
	# 				lif_k = np.random.poisson(lam=self.P, size=Bk )
	# 			elif self.MODE=='geometric':
	# 				lif_k = np.random.geometric(1-self.P/(1+self.P), size=Bk )-1
	# 			elif self.MODE=='binomial':
	# 				lif_k = np.random.binomial(self.K, self.P/self.K, size=Bk )
	# 			elif self.MODE=='negbinomial':
	# 				#lif_k = np.random.negative_binomial(self.q*self.P/(1-self.q), self.q, size=Bk )
	# 				lif_k = np.random.negative_binomial(self.r, self.r/(self.P+self.r), size=Bk )
	# 			elif self.MODE=='betaBinomial':
	# 				#a = 1

	# 				#b = self.a * (self.K -self.P)/self.P
	# 				b = self.a * (self.K - self.P)/self.P

	# 				lif1 = np.random.beta(self.a, b, size=Bk )
	# 				lif_k = np.random.binomial(self.K, lif1)					### K left?
	# 				#lif_k = np.random.binomial(self.K-k, lif1)

	# 			elif self.MODE=='gamma':
	# 				lif_k = np.floor( np.random.gamma(self.shapeA, scale=self.P/self.shapeA, size=Bk ) + 0.5 )	 #!
	# 			elif self.MODE=='sparse':
	# 				x = np.ceil(2*self.P*self.K / (self.K + 1))
	# 				lif_k = np.floor( ( np.random.random(Bk)*self.K**2 - self.K**2 + self.K*x )/x )




	# 			indsQ = lif_k > 0
	# 			indsR = lives_active_set > 0


	# 			Rk0 = np.concatenate( [ X_k[indsQ], Rk0[indsR] ] , axis=0 )
	# 			lives_active_set = np.concatenate( [lif_k[indsQ]-1, lives_active_set[indsR]-1 ] )

	# 			#indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
	# 			#lifetimes[indsT] += 1
	# 			sizes[k] = Rk0.shape[0]

		

	# 			R1 = Rk0

	# 		if self.MODE=='kernel':

	# 			S = Bk*self.P

	# 			if k==0:
	# 				probsCand1 = np.zeros((0))
	# 			else:
	# 				probsCand1 = np.mean( self.kern.K(Ak, X_k), 1)
				
	# 			probsCand2 = np.mean( self.kern.K(X_k, X_k), 1)

	# 			probsCon = np.concatenate([probsCand1, probsCand2])  
	# 			probs = probsCon/np.amax(probsCon)   # normalize


	# 			if np.sum(probs**0)<=S:
	# 				qopt = 0
	# 			else:
	# 				qopt = self.q/self.P
	# 			# elif self.funA(1, S, probs)>0:
	# 			# 	qopt = 1
	# 			# else:
	# 			# 	roots = root_scalar(self.funA, args = (S, probs), method='brentq', bracket=[0,1], rtol=1e-2)
	# 			# 	qopt = roots.root

	# 			takes = np.random.binomial(1, probs**qopt)

	# 			# if active set is too large, i.e. qopt==1, subsample it
	# 			nT = np.sum(takes)
	# 			if nT > S:
	# 				indsOne = np.arange(len(takes))[takes==1]
	# 				indsOne = indsOne[ np.random.permutation(len(indsOne)) ]
	# 				takes[indsOne[:(nT-S)]] = 0
	# 				#print('active set larger than ',S,' thus ',nT-S,'random points removed')

	# 			self.PROBS.append(probs)
	# 			self.PROBSq.append(probs**qopt)

	# 			Ak = np.concatenate([ Ak[ takes[:len(probsCand1)]==True, :], X_k[ takes[len(probsCand1):]==True, :] ])

		

	# 			R1 = Ak

	# 		# here for all methods we have in R1 the current active set
	# 		# R0 the previous one
	# 		# Krr0 and iKrr0 should also be valid, computed with the old hypers!





	# 		## change here params. NO. IN THE END.

	# 		if not VaryDict==None:

	# 			newValues = np.zeros((0,1))

	# 			for par in self.PARAMS:
	# 				if par.EST:
						
	# 					#print( par.name,' n ',VaryDict[par.name][k] )

	# 					newValues = np.vstack([newValues, VaryDict[par.name][k]])


	# 			self.set_PARAMS( newValues )







	# 		iKrr1, Krr1 = self.comp_kernR(R1)
		

			

		



	# 		# A: translation step
	# 		if k==0:
	# 			#mk = np.zeros(Jk+self.M)
	# 			mk = np.random.multivariate_normal(np.zeros(Krr1.shape[0]),Krr1,1)[0,:] 
	
		


	# 		else:


	# 			A_k, _, D_k = self.comp_kernX(R1, R0, iKrr0, DIAG=False)

	# 			ga_k = np.random.multivariate_normal(np.zeros(D_k.shape[0]),D_k,1)[0,:]
			

	# 			mk = np.dot(A_k, mk) + ga_k

			

		
	# 			#mk, Pk = self.trans(mk, Pk, A_k, D_k)
			





	# 		Htrain_k, _, dtrain_k = self.comp_kernX(X_k, R1, iKrr1)
	# 		Dtrain_k = np.diag(dtrain_k)

	# 		nu_k = np.random.multivariate_normal(np.zeros(Dtrain_k.shape[0]),Dtrain_k,1) [0,:]


	# 		eps_k = np.random.normal(0, np.sqrt( self.likelihood.variance[0]), len(nu_k) )
			 
			

	# 		f_k0 = np.dot(Htrain_k, mk) 
	# 		f_k = f_k0 + nu_k 
	# 		y_k_gen = f_k + eps_k

	# 		self.y_trains_k[k] = y_k_gen
	# 		#self.fks.append(f_k)
	# 		self.fks.append(f_k)




	# 		# update next inducing
	# 		R0 = R1 				# all: JkxP + M
	# 		RlocAll0 = RlocAll1 	# only JkxP
	# 		iKrr0 = iKrr1			# of all -> compute it more clever!!!!



		






	# not storing everything
	# def apply_KF_fast(self, seed=123, PRINT=False, STORE=False):

	# 	np.random.seed(seed)

	# 	self.Rk = []			## store only few last?

	# 	if STORE:
	# 		self.Rks = []			## only for visualizatoin
	# 		self.mks = []

	# 	timeKr = 0
	# 	timeKrr = 0
	# 	timeTrans = 0
	# 	timeKxr = 0 
	# 	timeLik = 0
	# 	timePredK = 0
	# 	timePred = 0

	# 	PRINT = False


	# 	#T1 = 0
	# 	#T2 = 0
	# 	#T3 = 0


	# 	#likterm = 0
	# 	#likterm2 = 0
	# 	#Qterm = 0
	# 	#Sigterm = 0

	

	# 	# # compute prediction kernel for global inducing points
	# 	# if self.M>0:
	# 	# 	K_Xtest_R = self.kern.K(self.X_test,self.R)
	# 	# else:
	# 	# 	K_Xtest_R = np.zeros((self.Ntest,0))

	# 	# kxxTest = self.kern.Kdiag(self.X_test)

	# 	# # prediction kernel of P past inducings
	# 	# KXR_P = np.zeros((self.Ntest,0))


	# 	Rk0 = np.zeros((0,self.D))

	# 	#indsT = np.zeros(self.N, dtype=int)
	# 	#lifetimes = np.zeros(self.N, dtype=int)  # empirical distribution of lifetimes
	# 	sizes = np.zeros(self.K) # actual sizes of active sets

	# 	lives_active_set = np.zeros(0, dtype=int)  #lives in current active set

	# 	RlocAll1 = 0

	# 	#Rk = [] # already above defined





	# 	Ak = np.zeros((0,self.D))	# current active set

	# 	# only for checking
	# 	#Aks = []				# active sets, using Rks form above
	# 	self.PROBS = []
	# 	self.PROBSq = []


	# 	for k in range(self.K):

	# 		# get current data
	# 		#X_k = self.X_train[ (self.B*(k)):(self.B*(k+1)), :]
	# 		#y_k = self.y_train[(self.B*(k)):(self.B*(k+1))]

	# 		X_k = self.X_trains_k[k]
	# 		y_k = self.y_trains_k[k]



	# 		Bk = X_k.shape[0]
	# 		Jk = np.int(np.ceil(self.Jp*Bk))


	# 		#print(Bk)




	# 		# get current inducing points of size BxP (actually full GP of current and past)
	# 		#R1 = self.X_train[np.maximum(0,self.B*(k+1-self.P)):self.B*(k+1), :]  

	# 		if self.MODE == 'sparse_global':
	# 			# sparse and global version	
	# 			Rloc1 = X_k 			#+ np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!
	# 			inds = np.random.permutation(Bk)[:Jk]
	# 			#inds = np.arange(Jk)
	# 			#print('no permutation!!!')
	# 			Rloc1 = Rloc1[inds,:]

	# 			self.Rk.append(Rloc1)

	# 			if self.P > 1 and k > 0:
	# 				#RlocAll1 = np.concatenate([ RlocAll0[np.maximum(0,RlocAll0.shape[0]-Jk*(self.P-1)):,:], Rloc1 ])
	# 				RlocAll1 = np.concatenate(  self.Rk[np.maximum(0, (k+1)-self.P):(k+1)], axis=0 )

	# 			else:
	# 				RlocAll1 = Rloc1

	# 			if self.M>0:
	# 				R1 = np.block([[RlocAll1],[self.R]])
	# 			else:
	# 				R1 = RlocAll1

	# 				if STORE:
	# 					self.Rks.append(R1)

	# 		if self.MODE=='geometric' or self.MODE=='poisson' or self.MODE=='gamma' or self.MODE=='binomial' or self.MODE=='negbinomial' or self.MODE=='betaBinomial' or self.MODE=='sparse':
	# 			# geometric/probabilistiv markov process
	# 			#indsK = np.arange(self.B*k, B*(k+1))
	# 			#X_k = XX[ indsK, :]


	# 			# #X_kn = X_k + np.random.randn(X_k.shape[0], X_k.shape[1])*1e-4   #!!!!!!!!!!!!!!!!!

	# 			# indsQ = np.random.random( X_kn.shape[0] ) <= self.q
	# 			# indsR = np.random.random( Rk0.shape[0] ) <= self.r

	# 			# #indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
	# 			# Rk0 = np.concatenate( [ X_kn[indsQ], Rk0[indsR] ] , axis=0 )

	# 			# #lifetimes[indsT] += 1
	# 			# sizes[k] = Rk0.shape[0]

	# 			# self.Rk.append(Rk0)

	# 			# R1 = Rk0



	# 			Bk = X_k.shape[0]


	# 			if self.MODE=='poisson':
	# 				lif_k = np.random.poisson(lam=self.P, size=Bk )
	# 			elif self.MODE=='geometric':
	# 				lif_k = np.random.geometric(1-self.P/(1+self.P), size=Bk )-1
	# 			elif self.MODE=='binomial':
	# 				lif_k = np.random.binomial(self.K, self.P/self.K, size=Bk )
	# 			elif self.MODE=='negbinomial':
	# 				#lif_k = np.random.negative_binomial(self.q*self.P/(1-self.q), self.q, size=Bk )
	# 				lif_k = np.random.negative_binomial(self.r, self.r/(self.P+self.r), size=Bk )
	# 			elif self.MODE=='betaBinomial':
	# 				#a = 1

	# 				#b = self.a * (self.K -self.P)/self.P
	# 				b = self.a * (self.K - self.P)/self.P

	# 				lif1 = np.random.beta(self.a, b, size=Bk )
	# 				lif_k = np.random.binomial(self.K, lif1)					### K left?
	# 				#lif_k = np.random.binomial(self.K-k, lif1)

	# 			elif self.MODE=='gamma':
	# 				lif_k = np.floor( np.random.gamma(self.shapeA, scale=self.P/self.shapeA, size=Bk ) + 0.5 )	 #!
	# 			elif self.MODE=='sparse':
	# 				x = np.ceil(2*self.P*self.K / (self.K + 1))
	# 				lif_k = np.floor( ( np.random.random(Bk)*self.K**2 - self.K**2 + self.K*x )/x )


	

	# 			indsQ = lif_k > 0
	# 			indsR = lives_active_set > 0


	# 			Rk0 = np.concatenate( [ X_k[indsQ], Rk0[indsR] ] , axis=0 )
	# 			lives_active_set = np.concatenate( [lif_k[indsQ]-1, lives_active_set[indsR]-1 ] )

	# 			#indsT = np.concatenate( [ indsK[indsQ], indsT[indsR] ] )
	# 			#lifetimes[indsT] += 1
	# 			sizes[k] = Rk0.shape[0]

	# 			if STORE:
	# 				self.Rks.append(Rk0)

	# 			R1 = Rk0

	# 		if self.MODE=='kernel':

	# 			S = Bk*self.P

	# 			if k==0:
	# 				probsCand1 = np.zeros((0))
	# 			else:
	# 				probsCand1 = np.mean( self.kern.K(Ak, X_k), 1)
				
	# 			probsCand2 = np.mean( self.kern.K(X_k, X_k), 1)

	# 			probsCon = np.concatenate([probsCand1, probsCand2])  
	# 			probs = probsCon/np.amax(probsCon)   # normalize


	# 			if np.sum(probs**0)<=S:
	# 				qopt = 0
	# 			else:
	# 				qopt = self.q/self.P
	# 			# elif self.funA(1, S, probs)>0:
	# 			# 	qopt = 1
	# 			# else:
	# 			# 	roots = root_scalar(self.funA, args = (S, probs), method='brentq', bracket=[0,1], rtol=1e-2)
	# 			# 	qopt = roots.root

	# 			takes = np.random.binomial(1, probs**qopt)

	# 			# if active set is too large, i.e. qopt==1, subsample it
	# 			nT = np.sum(takes)
	# 			if nT > S:
	# 				indsOne = np.arange(len(takes))[takes==1]
	# 				indsOne = indsOne[ np.random.permutation(len(indsOne)) ]
	# 				takes[indsOne[:(nT-S)]] = 0
	# 				#print('active set larger than ',S,' thus ',nT-S,'random points removed')

	# 			self.PROBS.append(probs)
	# 			self.PROBSq.append(probs**qopt)

	# 			Ak = np.concatenate([ Ak[ takes[:len(probsCand1)]==True, :], X_k[ takes[len(probsCand1):]==True, :] ])

	# 			if STORE:
	# 				self.Rks.append(Ak)  

	# 			R1 = Ak



	# 		start = time.time()
	# 		iKrr1, Krr1 = self.comp_kernR(R1)
	# 		timeKr += time.time() - start


	# 		# A: translation step
	# 		if k==0:
	# 			#mk = np.zeros(Jk+self.M)
	# 			mk = np.zeros(R1.shape[0])
	# 			Pk = Krr1

	# 			#P0 = Pk     # last covariance
	# 			#m0 = mk     # last mean

	# 		else:
	# 			#P0 = Pk     # last covariance
	# 			#m0 = mk     # last mean

	# 			start = time.time()
	# 			A_k, _, D_k = self.comp_kernX(R1, R0, iKrr0, DIAG=False)  
	# 			timeKrr += time.time() - start

	# 			start = time.time()  
	# 			mk, Pk = self.trans(mk, Pk, A_k, D_k)
	# 			timeTrans += time.time() - start


	# 		# compute current predictive distribution after translation
	# 		start = time.time()  
	# 		Hpred, _, dpred = self.comp_kernX(self.X_test, R1, iKrr1)				#A)
	# 		#Hpred, _, dpred, t1, t2, t3 = self.comp_kernX_TIME(self.X_test, R1, iKrr1)

	# 		# B) slightly better, but still not good
	# 		# if k-self.P < 0:
	# 		# 	numInd = 0
	# 		# else:
	# 		# 	numInd = self.Rk[k-self.P].shape[0]
		
	# 		# Kxrold = KXR_P[:,numInd:]

	# 		# Kxrnew = self.kern.K(self.X_test,Rloc1)
	# 		# KXR_P = np.concatenate([Kxrold, Kxrnew], axis=1)
	# 		# KXR = np.concatenate([KXR_P, K_Xtest_R], axis=1)
	# 		# Hpred, _, dpred = self.comp_kernX_split(KXR, iKrr1, kxxTest)


	# 		# #T1 += t1
	# 		#T2 += t2
	# 		#T3 += t3
	# 		timePredK += time.time() - start

	# 		start = time.time()  
	# 		#m, v = self.pred_y_diag(mk, Pk, Hpred, np.diag(dpred), self.sig2_n[0])
	# 		m, v = self.pred_y_Diag(mk, Pk, Hpred, dpred, self.likelihood.variance[0])
	# 		timePred += time.time() - start

	# 		# aggregate cummulative predictive distribution
	# 		if k!=0:
	# 			self.p_agg -=  1/v
	# 			self.n_agg -= m/v

	# 		# B: update step
	# 		start = time.time()  
	# 		if self.DIAG:
	# 			Htrain_k, _, dtrain_k = self.comp_kernX(X_k, R1, iKrr1)
	# 			Dtrain_k = np.diag(dtrain_k)
	# 		else:
	# 			Htrain_k, _, Dtrain_k = self.comp_kernX(X_k, R1, iKrr1, DIAG=False)
	# 		timeKxr += time.time() - start

			
	# 		# if k==0:
	# 		# 	#y_HFmu = (y_k - np.dot(Htrain_k, m0) )
	# 		# 	#diagHFSigFtHt = diag_HtKH(Htrain_k.T, P0)
	# 		# 	HF = Htrain_k
	# 		# 	D_k = np.zeros((len(y_k)))
	# 		# else:
	# 		# 	#print(A_k.shape)
	# 		# 	#print(Htrain_k.shape)
	# 		# 	#print(mk.shape)
	# 		# 	#print(y_k.shape)
	# 		# 	HF = np.dot(Htrain_k, A_k)
	# 		# 	diagHQHt = diag_HtKH(Htrain_k.T, D_k)

	# 		# HFmu = np.dot(HF, m0)
	# 		# y_HFmu = (y_k - HFmu)
	# 		# diagHFSigFtHt = diag_HtKH(HF.T, P0)


	# 		start = time.time()  
	# 		mk, Pk, lik_k = self.lik_up(mk, Pk, y_k, Htrain_k, self.al*Dtrain_k+ np.eye(Bk)*self.likelihood.variance[0])
	# 		timeLik += time.time() - start

	# 		ak = np.trace(Dtrain_k) / (2*self.likelihood.variance[0])
	# 		ak_al = np.sum( np.log( np.diag(Dtrain_k)*self.al/self.likelihood.variance[0] + 1 ) ) *(1-self.al)/(2*self.al)
		
			
	# 		self.ak += ak_al


	# 		self.lik += lik_k
			
	# 		# if k==0:
	# 		# 	self.low_bound += ( -np.sum(y_HFmu**2) - ak - np.sum(diagHFSigFtHt) ) / (2*self.sig2_n[0]) - Bk/2 * np.log(2*np.pi*self.sig2_n[0])

	# 		# else:
	# 		# 	self.low_bound += ( -np.sum(y_HFmu**2) - ak - np.sum(diagHFSigFtHt) - np.sum(diagHQHt) ) / (2*self.sig2_n[0])   - Bk/2 * np.log(2*np.pi*self.sig2_n[0])

	# 		# 	Qterm += - np.sum(diagHQHt) / (2*self.sig2_n[0])  

	# 		# likterm += -np.sum(y_HFmu**2) / (2*self.sig2_n[0]) - Bk/2 * np.log(2*np.pi*self.sig2_n[0])
	# 		# iS, lDS = inv_logDet( dot3lr( HF, P0, HF.T) + dot3rl(Htrain_k, D_k, Htrain_k.T) + self.sig2_n[0]*np.eye(HF.shape[0]))
	# 		# likterm2 += - dot3lr( y_HFmu, iS ,y_HFmu ) / (2) - Bk/2 * np.log(2*np.pi) - lDS/2
	# 		# Sigterm += - np.sum(diagHFSigFtHt)  / (2*self.sig2_n[0])
			


	# 		# compute current predictive distribution after update
	# 		start = time.time() 
	# 		#Hpred, _, dpred = self.comp_kernX(self.X_test, R1, iKrr1) # same as before!!
	# 		#m, v = self.pred_y_diag(mk, Pk, Hpred, np.diag(dpred), self.sig2_n[0])
	# 		m, v = self.pred_y_Diag(mk, Pk, Hpred, dpred, self.likelihood.variance[0])
	# 		timePred += time.time() - start

	# 		# aggregate cummulative predictive distribution
	# 		self.p_agg += 1/v
	# 		self.n_agg += m/v


	# 		# update next inducing
	# 		R0 = R1 				# all: JkxP + M
	# 		RlocAll0 = RlocAll1 	# only JkxP
	# 		iKrr0 = iKrr1			# of all -> compute it more clever!!!!

		


	# 		if STORE:
	# 			self.mks.append(mk)


	# 		#print(len(mk))


	# 	# invert natural cummulative predictive distribution
	# 	self.v_agg = 1/self.p_agg
	# 	self.m_agg = self.v_agg*self.n_agg

	# 	self.m_last = m
	# 	self.v_last = v


	# 	self.CI = np.zeros((len(self.m_agg), 2))
		
	# 	# if any( self.v_agg < 0):
	# 	# 	print('min_v is ', np.amin(self.v_agg))
	# 	# 	nn = np.sum(self.v_agg < 0)
	# 	# 	print(nn,' entries in v rounded to zero')

	# 	# 	print('inds ',np.arange(len(self.v_agg))[self.v_agg < 0])

	# 	# 	#self.v_agg = np.maximum(self.v_agg, np.ones(len(self.v_agg))*1e-10 )



	# 	self.CI[:,0] = self.m_agg - 1.96*np.sqrt(self.v_agg)
	# 	self.CI[:,1] = self.m_agg + 1.96*np.sqrt(self.v_agg)





	# 	# just to check
	# 	self.R1 = R1
	# 	#self.RlocAll1 = RlocAll1
	# 	self.mk = mk



	# 	self.empS = np.mean(sizes)

	# 	if not self.MODE=='sparse_global':
	# 		self.name += '_eS'+str(self.empS)

	# 	#print('ak ',self.ak)
	# 	#print('lik ',self.lik)
	# 	#print('low ', self.low_bound)
	# 	#print('likterm ',likterm)
	# 	#print('likterm2 ',likterm2)
	# 	#print('Qterm ',Qterm)
	# 	#print('Sigterm ',Sigterm)



	# 	if PRINT:
	# 		print(self.MODE+'_emp sizes ',self.empS,'expected size ',self.aS)


	# 		print('timeKr ', timeKr)
	# 		print('timeKrr ',timeKrr)
	# 		print('timeTrans ',timeTrans)
	# 		print('timeKxr ' ,timeKxr)
	# 		print('timeLik ',timeLik)
	# 		print('timePredK ',timePredK)
	# 		print('timePred ',timePred)
	# 		#print('T1 ',T1)
	# 		#print('T2 ',T2)
	# 		#print('T3 ',T3)
	# 		print(' ')

		

	# 	#print(self.name+' '+str(R1.shape[0]))







	# old version when storing everything (in particular for smoothing)	
	# not working with local inducing points!!!!!!!!!!!!!!!!!!!!!!!!!
	#def apply_KF(self, seed=123):

	# 	for k in range(self.K):

	# 		# get current data
	# 		X_k = self.X_train[ (self.B*(k)):(self.B*(k+1)), :]
	# 		y_k = self.y_train[(self.B*(k)):(self.B*(k+1))]

	# 		# get current inducing points of size BxP (actually full GP of current and past)
	# 		R1 = self.X_train[np.maximum(0,self.B*(k+1-self.P)):self.B*(k+1), :]   

	# 		if self.J < self.B:
	# 			np.random.seed(seed)
	# 			inds = np.random.permutation(self.B)[:self.J]
	# 			R1 = R1[inds,:]

	# 		if self.M>0:
	# 			R1 = np.block([[R1],[self.R]])
	# 		iKrr1, Krr1 = self.comp_kernR(R1)


	# 		# A: translation step
	# 		if k==0:
	# 			mk = np.zeros(self.B+self.M)
	# 			Pk = Krr1
	# 		else:
	# 			A_k, _, D_k = self.comp_kernX(R1, R0, iKrr0, DIAG=False)    
	# 			mk, Pk = self.trans(mk, Pk, A_k, D_k)


	# 		# compute current predictive distribution after translation
	# 		Hpred, _, dpred = self.comp_kernX(self.X_test, R1, iKrr1)
	# 		m, v = self.pred_y_diag(mk, Pk, Hpred, np.diag(dpred), self.sig2_n[0])
	# 		self.M_V_trans[k,:,0] = m # not needed when instant aggregation
	# 		self.M_V_trans[k,:,1] = v # not needed when instant aggregation

	# 		# aggregate cummulative predictive distribution
	# 		if k!=0:
	# 			self.p_agg -=  1/v
	# 			self.n_agg -= m/v

	# 		# B: update step
	# 		Htrain_k, _, dtrain_k = self.comp_kernX(X_k, R1, iKrr1)
	# 		mk, Pk, lik_k = self.lik_up(mk, Pk, y_k, Htrain_k, self.al*np.diag(dtrain_k)+ np.eye(self.B)*self.sig2_n[0])
	# 		self.lik += lik_k


	# 		# compute current predictive distribution after update
	# 		Hpred, _, dpred = self.comp_kernX(self.X_test, R1, iKrr1)
	# 		m, v = self.pred_y_diag(mk, Pk, Hpred, np.diag(dpred), self.sig2_n[0])
	# 		self.M_V[k,:,0] = m # not needed when instant aggregation
	# 		self.M_V[k,:,1] = v # not needed when instant aggregation

	# 		# aggregate cummulative predictive distribution
	# 		self.p_agg += 1/v
	# 		self.n_agg += m/v

	# 		# aggregate cummulative minVar distribution
	# 		indSmaller = v < self.v_minV
	# 		self.v_minV[indSmaller] = v[indSmaller]
	# 		self.m_minV[indSmaller] = m[indSmaller]
	# 		# save it
	# 		self.M_V_minVar_cumm[k,:,0] = self.m_minV
	# 		self.M_V_minVar_cumm[k,:,1] = self.v_minV


	# 		# update next inducing
	# 		R0 = R1
	# 		iKrr0 = iKrr1


	# 		# save for smoothing
	# 		self.Rks.append(R1)

	# 		if k!=0:
	# 			self.Aks.append(A_k)
	# 			self.Qks.append(D_k)
	# 		else:
	# 			self.Aks.append([])
	# 			self.Qks.append([])

	# 		self.Hpreds.append(Hpred)
	# 		self.Dpreds_diag.append(dpred)
	# 		self.mks.append(mk)
	# 		self.Pks.append(Pk)
	# 		self.Htrains.append(Htrain_k)
	# 		self.Dtrains_diag.append(dtrain_k)

	# 		# just for investigation
	# 		self.M_V_cumm[k,:,0] =  self.n_agg/self.p_agg
	# 		self.M_V_cumm[k,:,1] =  1/self.p_agg




	# 	# invert natural cummulative predictive distribution
	# 	self.v_agg = 1/self.p_agg
	# 	self.m_agg = self.v_agg*self.n_agg







	# need apply_KF and NOT apply_KF_fast
	# def apply_KS(self):
	# 	# only applicibale after apply_KF()

	# 	self.M_V_smooth = np.zeros((self.K,self.Ntest,2))

	# 	for k in np.arange(self.K-1,-1,-1):

	# 		if k==self.K-1:
	# 			mk1_S = self.mks[-1]
	# 			Pk1_S = self.Pks[-1]
	# 		else:
	# 			mk1_S, Pk1_S = self.smooth(mk1_S, self.mks[k], Pk1_S, self.Pks[k], self.Aks[k+1], self.Qks[k+1])  ## +1!!


	# 		# # compute current predictive distribution after smoothing
	# 		m, v = self.pred_y_diag(mk1_S, Pk1_S, self.Hpreds[k], np.diag(self.Dpreds_diag[k]), self.sig2_n[0])
	# 		self.M_V_smooth[k,:,0] = m
	# 		self.M_V_smooth[k,:,1] = v


	# 		# aggregate cummulative minVar distribution
	# 		indSmaller = v <= self.v_minV_S   # important
	# 		self.v_minV_S[indSmaller] = v[indSmaller]
	# 		self.m_minV_S[indSmaller] = m[indSmaller]
	# 		# save it
	# 		self.M_V_minVar_cumm_S[k,:,0] = self.m_minV_S
	# 		self.M_V_minVar_cumm_S[k,:,1] = self.v_minV_S





	# need apply_KF and NOT apply_KF_fast
	def apply_KS(self, MINVAR_KS=True):
		# only applicibale after apply_KF()

		self.M_V_smooth = np.zeros((self.K,self.Ntest,2))


		if MINVAR_KS:
			# for comparison also compute minVar filter
			self.m_minV_S = np.zeros(self.Ntest)
			self.v_minV_S = np.ones(self.Ntest)*1e10
			self.whichKs_S = np.zeros(self.Ntest, dtype=int)

		for k in np.arange(self.K-1,-1,-1):

			if k==self.K-1:
				mk1_S = self.mmks[-1]
				Pk1_S = self.Pks[-1]
			else:
				mk1_S, Pk1_S = self.smooth(mk1_S, self.mmks[k], Pk1_S, self.Pks[k], self.Aks[k+1], self.Qks[k+1])  ## +1!!


			print(mk1_S)

			#print( np.sum(mk1_S ) )


			# # compute current predictive distribution after smoothing
			m, v = self.pred_y_diag(mk1_S, Pk1_S, self.Hpreds[k], np.diag(self.Dpreds_diag[k]), self.likelihood.variance[0])
			self.M_V_smooth[k,:,0] = m
			self.M_V_smooth[k,:,1] = v


			# aggregate cummulative minVar distribution
			indSmaller = v <= self.v_minV_S   # important
			self.v_minV_S[indSmaller] = v[indSmaller]
			self.m_minV_S[indSmaller] = m[indSmaller]
			self.whichKs_S[indSmaller] = k

			# save it
			#self.M_V_minVar_cumm_S[k,:,0] = self.m_minV_S
			#self.M_V_minVar_cumm_S[k,:,1] = self.v_minV_S


	def lik_up(self,m0,P0,y,H,V, DERIVS=False, iK_AA=None):
		# update data to current prior (m0, P0)
		r = y - np.dot(H,m0)
		S = dot3lr(H,P0,H.T) + V
		#iS = inv_c(S)
		iS, logDetS = inv_logDet(S + np.eye(S.shape[0])*self.jit)
		#iS, logDetS = inv_logDet(S )
		G = dot3lr(P0, H.T, iS)

		if DERIVS:
			self.update_derivs_up(m0, P0, H, iK_AA, r, S, iS, G )



		m1 = m0 + np.dot(G,r)

		#P1 = P0 - dot3lr(G,S,G.T)
		P1 = np.dot( np.eye(P0.shape[0]) - np.dot(G,H) , P0 )	## should be more stable


		lik_k = -0.5*(len(r)*np.log(2*np.pi) + logDetS + dot3rl(r,iS,r) )


		# print('m ',m0)
		# print('P ',P0)

		# print('H ',H)
		# print('r ',r)
		# print('S ',S)
		# print('iS ',iS)
		# print('G ',G)
		# print('m+ ',m1)
		# print('P+ ',P1)


		return m1, P1, lik_k



	def trans(self,m,P,A,Q):
		# apply transition
		ms = np.dot(A,m)
		Ps = dot3lr(A,P,A.T) + Q
		return ms, Ps

	def transDiag(self,m,P,A,q):
		# apply transition
		ms = np.dot(A,m)
		#Ps = dot3lr(A,P,A.T) + q
		ps = diag_HtKH(A.T, P) + q
		return ms, ps


	def pred_f_diag(self,m,P,A,Qs):
		ms, Ps = self.trans(m,P,A,Qs)
		return ms, np.diag(Ps)

	def pred_f_Diag(self,m,P,A,qs):
		ms, ps = self.transDiag(m,P,A,qs)
		return ms, ps


	def pred_y_diag(self,m,P,A,Qs,sig2_n):
		ms, vs = self.pred_f_diag(m,P,A,Qs)
		return ms, vs + sig2_n

	def pred_y_Diag(self,m,P,A,qs,sig2_n):
		ms, vs = self.pred_f_Diag(m,P,A,qs)
		return ms, vs + sig2_n


	def comp_kernR(self, R):
		Krr = self.kern.K(R)
		iKrr = inv_c(Krr + np.eye(Krr.shape[0])*self.jit)
		return iKrr, Krr

	def comp_kernR_derivs(self, R):
		Krr = self.kern.K(R)
		iKrr = inv_c(Krr + np.eye(Krr.shape[0])*self.jit)
		return iKrr, Krr

	def comp_kernX(self, X, R, iKrr, DIAG=True):

		Kxr = self.kern.K(X,R)
		H = np.dot(Kxr, iKrr)

		#Q = dot3lr(Kxr, iKrr, Kxr.T)
		if DIAG:
			Q = diag_HtKH(Kxr.T, iKrr)
			kxx = self.kern.Kdiag(X)
			D = kxx - Q
		else:
			#Q = dot3lr(Kxr, iKrr, Kxr.T)		## it is already H
			Q = np.dot(H, Kxr.T)
			Kxx = self.kern.K(X)
			D = Kxx - Q

		return H, Q, D

	# def comp_kernX_split(self, Kxr, iKrr, kxx):
	# 	# always only diag

	# 	#Kxr = self.kern.K(X,R)
	# 	H = np.dot(Kxr, iKrr)

	# 	#Q = dot3lr(Kxr, iKrr, Kxr.T)
		
	# 	q = diag_HtKH(Kxr.T, iKrr)
	# 	#kxx = self.kern.Kdiag(X)
	# 	d = kxx - q
	

	# 	return H, q, d


	# def comp_kernX_TIME(self, X, R, iKrr, DIAG=True):

	# 	start = time.time()
	# 	Kxr = self.kern.K(X,R)
	# 	t1 =  time.time() - start

	# 	start = time.time()
	# 	H = np.dot(Kxr, iKrr)
	# 	t2 =  time.time() - start

	# 	start = time.time()
	# 	#Q = dot3lr(Kxr, iKrr, Kxr.T)
	# 	if DIAG:
	# 		Q = diag_HtKH(Kxr.T, iKrr)
	# 		kxx = self.kern.Kdiag(X)
	# 		D = kxx - Q
	# 	else:
	# 		Q = dot3lr(Kxr, iKrr, Kxr.T)
	# 		Kxx = self.kern.K(X)
	# 		D = Kxx - Q

	# 	t3 =  time.time() - start

	# 	return H, Q, D, t1, t2, t3



	def smooth(self, m1s, m0, P1s, P0, A, Q):  # you could also store m1, P1
		m1, P1 = self.trans(m0,P0,A,Q)

		iP1 = inv_c(P1 + np.eye(P1.shape[0])*self.jit)
		#iP1 = inv_c(P1 + np.eye(P1.shape[0])*1e-5 )
		L0 = dot3lr(P0,A.T,iP1)
		m0s = m0 + np.dot(L0,m1s-m1)
		P0s = P0 + dot3lr(L0, P1s-P1, L0.T)
		return m0s, P0s


	def update_grads(self, A, B=None, X=None):

		for par in self.PARAMS:

			if par.EST:

				par.update_grad(A, B, X)


	def init_posterior_deriv(self, SS):

		for par in self.PARAMS:

			if par.EST:

				shapeK = par.dKs['dK_AA'].shape

				if len(shapeK) == 2: ## only scalar parameter
					par.dlik = 0
					par.dmk = np.zeros(SS)
					par.dPk = par.dKs['dK_AA']
				elif len(shapeK) == 3:
					dim3 = shapeK[2]

					par.dlik = np.zeros(dim3)
					par.dmk = []
					par.dPk = []
					for j in range(dim3):
						par.dmk.append( np.zeros(SS) )			
						par.dPk.append(par.dKs['dK_AA'][:,:,j])

				elif len(shapeK) == 4:
					dim3 = shapeK[2]
					dim4 = shapeK[3]

					par.dlik = np.zeros(dim3*dim4)
					par.dmk = []
					par.dPk = []
					for i in range(dim3):
						for j in range(dim4):
							par.dmk.append( np.zeros(SS) )			
							par.dPk.append(par.dKs['dK_AA'][:,:,i,j])




				##### make it more automatic!!!!

				# if par.name == 'noise':
				# 	par.dlik = 0						#############
				# 	par.dmk = np.zeros(SS)
				# 	par.dPk = np.zeros((SS,SS))
				# elif par.name == 'variance':
				# 	par.dlik = 0
				# 	#par.dlik = par.get_prior_grad()			######!!!!!!
				# 	par.dmk = np.zeros(SS)
				# 	par.dPk = par.dKs['dK_AA']
				# elif par.name == 'lengthscale':
				# 	par.dlik = np.zeros(self.kern.input_dim)
				# 	#par.dlik = par.get_prior_grad()				#######
				# 	par.dmk = []
				# 	par.dPk = []
				# 	for j in range(self.kern.input_dim):
				# 		par.dmk.append( np.zeros(SS) )			### ARD
				# 		par.dPk.append(par.dKs['dK_AA'][:,:,j])
				# elif par.name == 'variances':
				# 	par.dlik = np.zeros(self.kern.input_dim)
				# 	#par.dlik = par.get_prior_grad()				#######
				# 	par.dmk = []
				# 	par.dPk = []
				# 	for j in range(self.kern.input_dim):
				# 		par.dmk.append( np.zeros(SS) )			### ARD
						# par.dPk.append(par.dKs['dK_AA'][:,:,j])



	def reset_derivs(self):

		for par in self.PARAMS:

			if par.EST:

				##### make it more automatic!!!!


				# if par.name == 'noise':
				# 	par.dlik = 0
				# elif par.name == 'variance':
				# 	par.dlik = 0
				# elif par.name == 'lengthscale':
				# 	par.dlik = np.zeros(self.kern.input_dim)
				# elif par.name == 'variances':
				# 	par.dlik = np.zeros(self.kern.input_dim)

	

				



				#if par.length()==1:
				if len(par.dKs['dK_AA'].shape) == 2:
					par.dlik = 0
				else:
					par.dlik[:] = 0#np.zeros(par.length)



	def update_derivs_trans(self, mk, Pk, Fk, iK_BB):


		for par in self.PARAMS:

			if par.EST:

				shapeK = par.dKs['dK_AA'].shape
		
				if len(shapeK) == 2: ## only scalar parameter

					par.dmk, par.dPk = self.update_deriv_trans(mk, Pk, Fk, iK_BB, par.dKs['dK_AA'], par.dKs['dK_AB'], par.dKs['dK_BB'], par.dmk, par.dPk)


				elif len(shapeK) == 3:
			
					for j in range(shapeK[2]):
						par.dmk[j], par.dPk[j] = self.update_deriv_trans(mk, Pk, Fk, iK_BB, \
														par.dKs['dK_AA'][:,:,j], par.dKs['dK_AB'][:,:,j], par.dKs['dK_BB'][:,:,j], \
														par.dmk[j], par.dPk[j])

				elif len(shapeK) == 4:	

					ij = 0
					for i in range(shapeK[2]):
						for j in range(shapeK[3]):

							par.dmk[ij], par.dPk[ij] = self.update_deriv_trans(mk, Pk, Fk, iK_BB, \
														par.dKs['dK_AA'][:,:,i,j], par.dKs['dK_AB'][:,:,i,j], par.dKs['dK_BB'][:,:,i,j], \
														par.dmk[ij], par.dPk[ij])
							ij += 1

	

	def update_deriv_trans(self, mk, Pk, Fk, iK_BB, dK_AA, dK_AB, dK_BB, dmk, dPk):

		
		dFk = np.dot( dK_AB, iK_BB) - dot3rl( Fk, dK_BB, iK_BB )

		#print(dFk)
		#dQk = dK_AA - np.dot( dK_AB, Fk.T ) + dot3rl( Fk, dK_BB ,Fk.T) - np.dot( Fk, dK_AB.T )
		dKABFkt = np.dot( dK_AB, Fk.T )
		dQk = dK_AA - dKABFkt + dot3rl( Fk, dK_BB ,Fk.T) - dKABFkt.T


		dmk1 = np.dot( dFk, mk ) + np.dot( Fk, dmk )

		dFkPkFkT = dot3rl( dFk, Pk, Fk.T) 
		#dPk1 = dot3rl( dFk, Pk, Fk.T) + dot3rl( Fk, dPk, Fk.T) + dot3rl( Fk, Pk, dFk.T) + dQk
		dPk1 = dFkPkFkT + dot3rl( Fk, dPk, Fk.T) + dFkPkFkT.T + dQk

		return dmk1, dPk1


	def update_derivs_up(self, mk, Pk, Hk, iK_AA, rk, Sk, iSk, Gk ):


		for par in self.PARAMS:

			if par.EST:
			
				shapeK = par.dKs['dK_AA'].shape

				NOISE = par.name == 'noise'

				if len(shapeK) == 2:   ## only scalar parameter
					par.dmk, par.dPk, par.dlik = self.update_deriv_up(mk, Pk, Hk, iK_AA, rk, Sk, iSk, Gk, par.dKs['dK_AA'], par.dKs['dK_XA'], par.dKs['dk_xx'], par.dmk, par.dPk, par.dlik, NOISE )
				
				elif len(shapeK) == 3:

					
					for j in range(shapeK[2]):
						par.dmk[j], par.dPk[j], par.dlik[j] = self.update_deriv_up(mk, Pk, Hk, iK_AA, rk, Sk, iSk, Gk, \
														par.dKs['dK_AA'][:,:,j], par.dKs['dK_XA'][:,:,j], par.dKs['dk_xx'][:,j], \
														par.dmk[j], par.dPk[j], par.dlik[j] )

				elif len(shapeK) == 4:
					ij = 0
					for i in range(shapeK[2]):
						for j in range(shapeK[3]):
							par.dmk[ij], par.dPk[ij], par.dlik[ij] = self.update_deriv_up(mk, Pk, Hk, iK_AA, rk, Sk, iSk, Gk, \
														par.dKs['dK_AA'][:,:,i,j], par.dKs['dK_XA'][:,:,i,j], par.dKs['dk_xx'][:,i,j], \
														par.dmk[ij], par.dPk[ij], par.dlik[ij] )

							ij += 1

	def update_deriv_up(self, mk, Pk, Hk, iK_AA, rk, Sk, iSk, Gk, dK_AA, dK_XA, dk_xx, dmk, dPk, dlik, NOISE=False):




		# transoposes: compute only once

		dHk = np.dot( dK_XA, iK_AA) - dot3rl( Hk, dK_AA, iK_AA )
		#ddiagVk = dk_xx - np.diag( np.dot( dK_XA, Hk.T ) ) + diag_HtKH( Hk.T, dK_AA  )  - np.diag( np.dot( Hk, dK_XA.T ) )
		
		ddiagVk = dk_xx - 2*np.sum( dK_XA * Hk, 1) + diag_HtKH( Hk.T, dK_AA  ) 


		drk = -np.dot( dHk, mk ) - np.dot( Hk, dmk )

		#dSk = dot3rl( dHk, Pk, Hk.T) + dot3rl( Hk, dPk, Hk.T) + dot3rl( Hk, Pk, dHk.T) + self.al * np.diag( ddiagVk )
		dHPHt = dot3rl( dHk, Pk, Hk.T)
		dSk = dHPHt + dot3rl( Hk, dPk, Hk.T) + dHPHt.T + self.al * np.diag( ddiagVk )

		if NOISE:
			dSk += np.eye(dSk.shape[0])


		#diSk = - dot3rl( iSk, dSk, iSk)
		#dGk = dot3rl( dPk, Hk.T, iSk ) + dot3rl( Pk, dHk.T, iSk ) - dot3rl( Gk, dSk, iSk)
		dGk = np.dot( np.dot( dPk, Hk.T) + np.dot( Pk, dHk.T ) - np.dot( Gk, dSk), iSk)


		dmk1 = dmk + np.dot( dGk, rk ) + np.dot( Gk, drk )
		#dPk1 = dPk - dot3rl( dGk, Sk, Gk.T) - dot3rl( Gk, dSk, Gk.T)  - dot3rl( Gk, Sk, dGk.T) 
		dGSGt = dot3rl( dGk, Sk, Gk.T)
		dPk1 = dPk - dGSGt - dot3rl( Gk, dSk, Gk.T)  - dGSGt.T

		iSr = np.dot(iSk, rk) #outside
		dlik = dlik - 0.5 * np.trace(np.dot(iSk, dSk)) - dot3rl(rk, iSk, drk) + 0.5*dot3rl(iSr, dSk, iSr) ##### -dak missing!! but not intended (also no difference)


		# print(' ')
		# print('dmk ', dmk)
		# print('dPk ', dPk)
		# print('dHk ', dHk)
		# print('drk ', drk)
		# print('dSk ', dSk)
		# #print('diSk ', diSk)
		# print('dGk ', dGk)
		# print('dmk1 ', dmk1)
		# print('dPk1 ', dPk1)
		# print(' ')

		#print(dlik)


		#### include ak!!!!!!

		return dmk1, dPk1, dlik






class PARAM:
	def __init__(self, name, par, kern, EST=False):


		self.name = name
		self.par = par
		self.kern = kern 	

		self.dKs = {}		# dictionary with all needed kernel derivatives wrt parameter
		self.EST = EST

		# in init_posterior_deriv defined
		#self.dmk = 0
		#self.dPk = 0

		#self.dlik = 0


	def get_value(self):

		return self.par.values.flatten()


		# if self.name=='noise':
		# 	val = self.kern.variance[0]  		### it is likelihood
		# elif self.name=='variance':
		# 	val = self.kern.variance[0]
		# elif self.name=='lengthscale':
		# 	val = self.kern.lengthscale

		#return val

	def get_prior_grad(self):

		if len(self.par.shape)==1:
			val = self.par._log_prior_gradients()
		else:
			val = np.zeros(0)
			for i in range(self.par.shape[0]):
				#for j in range(self.par.shape[1]):

				val = np.hstack( [val,  self.par[i]._log_prior_gradients() ] )

		return val

		# if self.name=='noise':
		# 	val = self.kern.variance._log_prior_gradients()	######################################
		# elif self.name=='variance':
		# 	val = self.kern.variance._log_prior_gradients()#[0]
		# elif self.name=='lengthscale':
		# 	val = np.reshape( self.kern.lengthscale._log_prior_gradients() , (len(self.kern.lengthscale),) )

		#return val

	def length(self):

		return len(self.get_value())

		


	def update_value(self, value):

		#self.value = value

		# maybe transformation

		# change values
		# if self.name=='noise':
		# 	self.kern.variance[0] = value 		### it is likelihood 
		# elif self.name=='variance':
		# 	self.kern.variance = value
		# elif self.name=='lengthscale':
		# 	self.kern.lengthscale = value
		
		# if len(value)==1:
		# 	self.par.fill(value[0])
		# else:
		# 	self.par.fill(value)

		#print(self.name)
		#print(value)

		self.par.param_array[:] = np.reshape( value, self.par.param_array.shape )

	def update_grad(self, A, B=None, X=None):

		
		# variance for RBF kernel
		if self.name=='variance':

			self.dKs = {'dK_AA': self.kern.dK_dσ02(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dσ02(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dσ02(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dσ02(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dσ02_diag(X)  

		# lengthscales for RBF kernel
		elif self.name=='lengthscale':

			self.dKs = {'dK_AA': self.kern.dK_dl(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dl(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dl(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dl(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dl_diag(X)   # zero anyway for ls

		# variances for linear kernel
		elif self.name=='variances':
			self.dKs = {'dK_AA': self.kern.dK_dvs(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dvs(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dvs(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dvs(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dvs_diag(X)  

		# weights for SM kernel
		elif self.name=='weights':
			self.dKs = {'dK_AA': self.kern.dK_dw(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dw(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dw(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dw(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dw_diag(X) 
		# means for SM kernel
		elif self.name=='means':
			self.dKs = {'dK_AA': self.kern.dK_dm(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dm(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dm(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dm(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dm_diag(X) 
		# variances for SM kernel
		elif self.name=='variancesSM':
			self.dKs = {'dK_AA': self.kern.dK_dv(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dv(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dv(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dv(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dv_diag(X) 

		# variances for std_periodic kernel
		elif self.name=='varianceP':
			self.dKs = {'dK_AA': self.kern.dK_dv(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dv(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dv(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dv(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dv_diag(X) 

		# lengthscales for std_periodic kernel
		elif self.name=='lengthscaleP':
			self.dKs = {'dK_AA': self.kern.dK_dl(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dl(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dl(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dl(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dl_diag(X) 

		# periods for std_periodic kernel
		elif self.name=='periodP':
			self.dKs = {'dK_AA': self.kern.dK_dp(A)}
			if B is not None:
				self.dKs['dK_AB'] = self.kern.dK_dp(A,B)
				self.dKs['dK_BB'] = self.kern.dK_dp(B)
			if X is not None:
				self.dKs['dK_XA'] = self.kern.dK_dp(X,A)
				self.dKs['dk_xx'] = self.kern.dK_dp_diag(X) 

		elif self.name=='noise':

			a = A.shape[0]
			self.dKs = {'dK_AA': np.zeros((a,a))}
			if B is not None:
				b = B.shape[0]
				self.dKs['dK_AB'] = np.zeros((a,b))
				self.dKs['dK_BB'] = np.zeros((b,b))
			if X is not None:
				n = X.shape[0]
				self.dKs['dK_XA'] = np.zeros((n,a))
				self.dKs['dk_xx'] = np.zeros(n)



def KL1(m1,m2,v1,v2):
	return 0.5*(np.log(v2/v1) + (v1 + (m1-m2)**2)/v2 - 1)  






class Independent:
	def __init__(self, DD, K, kern, likelihood, KMEANS=True, seed=0, sortDim=-1, PROJ='', bw=[None], PP=None, SPARSE=False, M_sp=None, optimizer='ADAM', ZZ=None, GLOBAL=False, priorNoise=None):

		self.DD = DD
		self.K = K
		self.kern = kern.copy()
		self.likelihood = likelihood.copy() 

		self.seed = seed

		self.SPARSE = SPARSE
		self.optimizer = optimizer

		self.ZZ = ZZ #global inducing points
		self.GLOBAL = GLOBAL


		self.priorNoise = priorNoise
		

		if PP==None:
			self.PP = Partition(DD)
			self.PP.compute_partition(K, KMEANS=KMEANS, randOrder=False, seed=seed, sortDim=sortDim, PROJ=PROJ, bw=bw)
		else:
			self.PP = PP
			self.K = PP.K
			print('K =',self.K)

		if M_sp!=None:
			self.M_sp = M_sp
		else: 
			self.M_sp = len( self.PP.y_trains_k[0])








	def newGPmod(self, k_data):

		if not self.SPARSE:
			self.GPmod = GPy.models.GPRegression(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None], kernel=self.kern, noise_var=self.likelihood.variance[0])

		else:
			#M_sp = self.M_sp
			#M_sp = len( self.PP.y_trains_k[k_data])
			if self.GLOBAL:
				ZZ = self.ZZ
			else:
				inds_BJ = np.array(  np.linspace(0,len( self.PP.y_trains_k[k_data])-1,self.M_sp, endpoint=True ) , dtype=int)
				ZZ = self.PP.X_trains_k[k_data][inds_BJ,:]


			self.GPmod = GPy.models.SparseGPRegression(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None], kernel = self.kern, Z=ZZ)

			#if inducing_fixed:
			#self.GPmod.unlink_parameter(self.GPmod.parameters[0])
			#self.GPmod.unlink_parameter(self.GPmod.inducing_inputs)
			#self.GPmod.IND_FIX = True
			self.GPmod.inducing_inputs.fix() #always fixed

			#unlink all parameters
			#while len(self.GPmod.parameters)>0:
			#	self.GPmod.unlink_parameter(self.GPmod.parameters[0])


			self.GPmod.inference_method = GPy.inference.latent_function_inference.VarDTC()
			#self.GPmod.inference_method = GPy.inference.latent_function_inference.PEP(1e-10)
			#self.GPmod.inference_method = GPy.inference.latent_function_inference.DTC()
			self.GPmod.Gaussian_noise = self.likelihood.variance[0]
			



		if self.likelihood.variance.is_fixed:
			self.GPmod.likelihood.variance.constrain_fixed()

		if self.priorNoise!=None:
			self.GPmod.likelihood.variance.set_prior(self.priorNoise)

		self.likelihood = self.GPmod.likelihood

		#compute fixed flat array
		self.fixed_flatten()


		#print('HHH')



	def updateGPmod(self, k_data):

		if not self.SPARSE:
			#self.GPmod = GPy.models.GPRegression(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None], kernel=self.kern, noise_var=self.likelihood.variance[0])

			#if self.priorNoise!=None:
			#	self.GPmod.likelihood.variance.set_prior(self.priorNoise)



			self.GPmod.set_XY(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None])

			self.likelihood = self.GPmod.likelihood

		else:
			#M_sp = self.M_sp
			#M_sp = len( self.PP.y_trains_k[k_data])
			if not self.GLOBAL:

				inds_BJ = np.array(  np.linspace(0,len( self.PP.y_trains_k[k_data])-1,self.M_sp, endpoint=True ) , dtype=int)
				#self.GPmod = GPy.models.SparseGPRegression(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None], kernel = self.kern, Z=self.PP.X_trains_k[k_data][inds_BJ,:])
				#self.GPmod.inducing_inputs = self.PP.X_trains_k[k_data][inds_BJ,:]
				self.GPmod.Z = self.PP.X_trains_k[k_data][inds_BJ,:] 


			self.GPmod.set_XY(self.PP.X_trains_k[k_data], self.PP.y_trains_k[k_data][:,None])
		
			
			#self.GPmod.set_Z(self.PP.X_trains_k[k_data][inds_BJ,:])
			

			#self.GPmod.kernel = self.kern
			#print(self.GPmod.Gaussian_noise[0]-self.likelihood.variance[0])
			#self.GPmod.Gaussian_noise = self.likelihood.variance[0]

			#print('up')


			#self.GPmod._update_gradients()
			


	def init_stoch(self, iStart=0):

		self.newGPmod(iStart) 
		pars = self.get_params()
		if self.optimizer == 'ADAM':
			self.OPTIMIZER = Adam(self.gamma, (len(pars),) )
		elif self.optimizer == 'ADAGRAD':
			self.OPTIMIZER = Adagrad(self.gamma, (len(pars),) )

		self.grads = np.zeros(len(pars))


	def stoch_gradient_step(self, UP=True):

		self.grads += self.get_gradients() #+ par.get_prior_grad()*U/self.K

		if UP:
			pars = self.get_params()
			grads = -self.grads * pars


			newValues = self.OPTIMIZER.update( np.log(pars), grads )
			self.set_params( np.exp(newValues) )
			self.grads[:] = 0
	

	def run_epochs(self, E=1, gamma=0.01, U=1, TRACE=False, PERM=True, PRINT=False, REL=1e-5):

		self.OPT_MODE = 'STOCH'

		self.gamma = gamma

		if TRACE:
			self.LIKS = []
			self.VALS = []
			self.GRADS = []

		np.random.seed(self.seed)
		self.lik0 = 1e-10
		rel = 1e10
		#for e in range(E):
		e = 0
		while e < E and rel>REL:

			self.lik = 0

			if PERM:
				perm = np.random.permutation(self.K)
			else:
				perm = range(self.K)

			for k in range(self.K):

				if k==0 and e==0:
					self.init_stoch(perm[0])
					print('init')
				else:
					#self.newGPmod(perm[k])
					self.updateGPmod(perm[k])

				#self.lik += self.GPmod.log_likelihood()
				self.lik += self.GPmod.log_likelihood()    + self.GPmod.log_prior() / self.K   ## divided by K!!


				if TRACE:
					#self.LIKS.append(self.lik)
					self.VALS.append(self.get_params())
					self.GRADS.append(self.get_gradients()*self.get_params())

				if (k % U) == (U-1) or k==self.K-1:
					self.stoch_gradient_step()
				else:
					self.stoch_gradient_step(UP=False)

			rel = np.abs((self.lik0-self.lik)/self.lik0)
			
			if PRINT:
				clear_output(wait=True)
				display('Epoch '+str(e)+' likelihood: '+str(self.lik)+ ' rel: '+str(rel)+' stop?: '+str(rel<REL) )
				#time.sleep(1)

			if TRACE:
				self.LIKS.append(self.lik)
		
			self.lik0 = self.lik
			e += 1


	def set_params(self, params):
		# update the estimated parameters in the GPmod

		self.GPmod.param_array[self.EST] = params

	def get_params(self):

		return self.GPmod.param_array[self.EST]

	def get_gradients(self):

		#return self.GPmod.gradient_full[self.EST]
		res =  self.GPmod._log_likelihood_gradients()[self.EST] 

		if hasattr(self.GPmod._log_prior_gradients(), "__len__"): 
			lpg = self.GPmod._log_prior_gradients()[self.EST]
			if self.OPT_MODE == 'STOCH':
				lpg /= self.K
			res += lpg
			#print(self.GPmod._log_prior_gradients())

		return res


	def fixed_flatten(self):
		self.is_fixed_flat = []
		self.find_is_fix(self.GPmod.parameters)
		self.EST = np.array(self.is_fixed_flat)==False

	def find_is_fix(self, pars):
	#e.g. lengthscales together fixed

		for pp in pars:
			if len(pp.parameters) == 0:
				#for i in range(len(pp)):
				for i in range(np.prod(pp.shape)):
					self.is_fixed_flat.append(pp.is_fixed)
			else:
				self.find_is_fix(pp.parameters)


	def opt_batch(self, GTOL=1e-2, maxF=200, TRACE=False):


		self.OPT_MODE = 'BATCH'

		self.newGPmod(0) #dummy data


		#print(self.GPmod)

		x0 = np.log( self.get_params())

		method = 'L-BFGS-B'
		#method = 'BFGS'

		self.TRACE = TRACE
		if TRACE:
			self.LIKS = []
			self.VALS = []
			self.GRADS = []
		    

		self.res = minimize(f, x0, method=method, jac=df, args=self, options={'disp': True, 'gtol':GTOL, 'maxfun': maxF} )

		self.set_params(np.exp(self.res.x))
		self.lik =     self.res.fun
        
            
@np_cache(maxsize=1)    
def f_df(log_params, OBJ):

	t0 = time.time()


	params = np.exp(log_params)
	OBJ.set_params(params)

	lik = 0
	grads = np.zeros(len(params))

	for k in range(OBJ.K):

		#OBJ.newGPmod(k)
		OBJ.updateGPmod(k)
		#lik += -OBJ.GPmod.log_likelihood()
		lik += -OBJ.GPmod.log_likelihood() - OBJ.GPmod.log_prior()
		grads += -OBJ.get_gradients()*OBJ.get_params()

	if OBJ.TRACE:
		OBJ.LIKS.append(-lik)
		OBJ.VALS.append(OBJ.get_params())
		OBJ.GRADS.append(grads)

	#print('timed_df',time.time()-t0)
	return lik, grads

def f(log_params, OBJ):

	f, _ = f_df(log_params, OBJ)

	return f


def df(log_params, OBJ):

	_, df = f_df(log_params, OBJ)

	return df 





from GPy.core.sparse_gp import SparseGP
from GPy.core.parameterization.variational import VariationalPosterior





def _update_gradients(self):
    self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
    if self.mean_function is not None:
        self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)

    if isinstance(self.X, VariationalPosterior):
        #gradients wrt kernel
        dL_dKmm = self.grad_dict['dL_dKmm']
        self.kern.update_gradients_full(dL_dKmm, self.Z, None)
        kerngrad = self.kern.gradient.copy()
        self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                Z=self.Z,
                                                dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                dL_dpsi2=self.grad_dict['dL_dpsi2'])
        self.kern.gradient += kerngrad

        #gradients wrt Z
        self.Z.gradient = self.kern.gradients_X(dL_dKmm, self.Z)
        self.Z.gradient += self.kern.gradients_Z_expectations(
                           self.grad_dict['dL_dpsi0'],
                           self.grad_dict['dL_dpsi1'],
                           self.grad_dict['dL_dpsi2'],
                           Z=self.Z,
                           variational_posterior=self.X)
    else:
        #gradients wrt kernel
        self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
        kerngrad = self.kern.gradient.copy()
        self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
        kerngrad += self.kern.gradient
        self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z, None)
        self.kern.gradient += kerngrad
        #gradients wrt Z
        #if not self.IND_FIX:
        if not self.inducing_inputs.is_fixed:
            
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
        #else:
        #    print('no Z grads!')
    #if not self.IND_FIX:
    if not self.inducing_inputs.is_fixed:
        self._Zgrad = self.Z.gradient.copy()


setattr(SparseGP, "_update_gradients", _update_gradients)
#print('new attribute for sparse GP!!')



# from GPy.inference.latent_function_inference import PEP, DTC
# from GPy.util import diag
# from GPy.util.linalg import jitchol, tdot, dtrtrs, dtrtri, pdinv
# from GPy.inference.latent_function_inference.posterior import Posterior



# def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
#     assert mean_function is None, "inference with a mean function not implemented"


#     print('inference')
#     t0 = time.time()

#     num_inducing, _ = Z.shape
#     num_data, output_dim = Y.shape

#     #make sure the noise is not hetero
#     sigma_n = likelihood.gaussian_variance(Y_metadata)
#     if sigma_n.size >1:
#         raise NotImplementedError("no hetero noise with this implementation of PEP")

#     Kmm = kern.K(Z)
#     Knn = kern.Kdiag(X)
#     Knm = kern.K(X, Z)
#     U = Knm

#     #factor Kmm
#     diag.add(Kmm, self.const_jitter)
#     Kmmi, L, Li, _ = pdinv(Kmm)

#     #compute beta_star, the effective noise precision
#     LiUT = np.dot(Li, U.T)
#     sigma_star = sigma_n + self.alpha * (Knn - np.sum(np.square(LiUT),0))
#     beta_star = 1./sigma_star

#     # Compute and factor A
#     A = tdot(LiUT*np.sqrt(beta_star)) + np.eye(num_inducing)
#     LA = jitchol(A)

#     # back substitute to get b, P, v
#     URiy = np.dot(U.T*beta_star,Y)
#     tmp, _ = dtrtrs(L, URiy, lower=1)
#     b, _ = dtrtrs(LA, tmp, lower=1)
#     tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
#     v, _ = dtrtrs(L, tmp, lower=1, trans=1)
#     tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
#     P = tdot(tmp.T)

#     alpha_const_term = (1.0-self.alpha) / self.alpha

#     #compute log marginal
#     log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
#                    -np.sum(np.log(np.diag(LA)))*output_dim + \
#                    0.5*output_dim*(1+alpha_const_term)*np.sum(np.log(beta_star)) + \
#                    -0.5*np.sum(np.square(Y.T*np.sqrt(beta_star))) + \
#                    0.5*np.sum(np.square(b)) + 0.5*alpha_const_term*num_data*np.log(sigma_n)
#     #compute dL_dR
#     Uv = np.dot(U, v)
#     dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - (1.0+alpha_const_term)/beta_star + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) \
#         + np.sum(np.square(Uv), 1))*beta_star**2 

#     # Compute dL_dKmm
#     vvT_P = tdot(v.reshape(-1,1)) + P
#     dL_dK = 0.5*(Kmmi - vvT_P)
#     KiU = np.dot(Kmmi, U.T)
#     dL_dK += self.alpha * np.dot(KiU*dL_dR, KiU.T)

#     # Compute dL_dU
#     vY = np.dot(v.reshape(-1,1),Y.T)
#     dL_dU = vY - np.dot(vvT_P, U.T)
#     dL_dU *= beta_star
#     dL_dU -= self.alpha * 2.*KiU*dL_dR

#     dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)
#     dL_dthetaL += 0.5*alpha_const_term*num_data / sigma_n
#     grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':dL_dR * self.alpha, 'dL_dKnm':dL_dU.T, 'dL_dthetaL':dL_dthetaL}

#     #construct a posterior object
#     post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)


#     print(time.time()-t0)

#     return post, log_marginal, grad_dict



# setattr(PEP, "inference", inference)




# def inferenceDTC(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
#     assert mean_function is None, "inference with a mean function not implemented"
#     #assert X_variance is None, "cannot use X_variance with DTC. Try varDTC."


#     print('inferenceDTC')
#     t0 = time.time()

#     num_inducing, _ = Z.shape
#     num_data, output_dim = Y.shape

#     #make sure the noise is not hetero
#     precision = 1./likelihood.gaussian_variance(Y_metadata)
#     if precision.size > 1:
#         raise NotImplementedError("no hetero noise with this implementation of DTC")

#     Kmm = kern.K(Z)
#     Knn = kern.Kdiag(X)
#     Knm = kern.K(X, Z)
#     U = Knm
#     Uy = np.dot(U.T,Y)

#     #factor Kmm
#     Kmmi, L, Li, _ = pdinv(Kmm)

#     # Compute A
#     LiUTbeta = np.dot(Li, U.T)*np.sqrt(precision)
#     A = tdot(LiUTbeta) + np.eye(num_inducing)

#     # factor A
#     LA = jitchol(A)

#     # back substutue to get b, P, v
#     tmp, _ = dtrtrs(L, Uy, lower=1)
#     b, _ = dtrtrs(LA, tmp*precision, lower=1)
#     tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
#     v, _ = dtrtrs(L, tmp, lower=1, trans=1)
#     tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
#     P = tdot(tmp.T)

#     #compute log marginal
#     log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
#                    -np.sum(np.log(np.diag(LA)))*output_dim + \
#                    0.5*num_data*output_dim*np.log(precision) + \
#                    -0.5*precision*np.sum(np.square(Y)) + \
#                    0.5*np.sum(np.square(b))

#     # Compute dL_dKmm
#     vvT_P = tdot(v.reshape(-1,1)) + P
#     dL_dK = 0.5*(Kmmi - vvT_P)

#     # Compute dL_dU
#     vY = np.dot(v.reshape(-1,1),Y.T)
#     dL_dU = vY - np.dot(vvT_P, U.T)
#     dL_dU *= precision

#     #compute dL_dR
#     Uv = np.dot(U, v)
#     dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - 1./precision + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) + np.sum(np.square(Uv), 1))*precision**2

#     dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)

#     grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':np.zeros_like(Knn), 'dL_dKnm':dL_dU.T, 'dL_dthetaL':dL_dthetaL}

#     #construct a posterior object
#     post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)


#     print(time.time()-t0)

#     return post, log_marginal, grad_dict



# setattr(DTC, "inference", inferenceDTC)
