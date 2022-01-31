import GPy
from matplotlib import pyplot as plt
import numpy as np
from PoE_2 import PoE
from PoE_2 import Expert
import pandas as pd
from incremental_p import INCR
from incremental_p import Independent


from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import properscoring as ps

import imageio
#from IPython.display import Image
import matplotlib.cm as cm


import time, pickle

import functools
from numpy_lru_cache_decorator import np_cache

from scipy.optimize import minimize


#### correspond to notebook incremental_run 


from operator import itemgetter 






class RunFullGP:
	def __init__(self, DD, kern, likelihood, priorN=None, seed=0):

		self.DD = DD
		self.kern = kern.copy()
		self.likelihood = likelihood.copy()
		self.priorN = priorN
		#self.sig2_noise = sig2_noise

	def run(self, OPT=False, NRAND=0):

		start = time.time()
	
		EXfull = Expert(self.DD.X_train, self.DD.y_train, self.kern, likelihood=self.likelihood, priorN=self.priorN)
		EXfull.create_GPmodel()

		if OPT:
			EXfull.optimizeHypers(not self.likelihood.is_fixed, NRAND=NRAND)

		EXfull.predict_CI(self.DD.X_test)
		timeFull = time.time() - start

		return ResultRun(EXfull.m, EXfull.v, EXfull.CI, 'fullGP', time=timeFull, obj=EXfull, OBJ=self, lik=EXfull.GPmod.log_likelihood())


class RunSparseGP:
	def __init__(self, DD, kern, M, likelihood, priorN=None, seed=0, RR=[[None]]):
		#if RR provided then M is ignored

		start = time.time()

		self.DD = DD
		self.kern = kern.copy()
		self.likelihood = likelihood.copy()
		self.priorN = priorN
		#self.sig2_noise = sig2_noise
		#self.M = M
		


		np.random.seed(seed)

		if RR[0][0]==None: 
			self.M = M
			indsRand_M = np.random.permutation(len(DD.y_train))[:M]
		
			RR = DD.X_train[indsRand_M,:].copy()	
			np.random.seed(seed)
			RR += np.random.randn(RR.shape[0], RR.shape[1] )*1e-4   #!!!
			self.RR = RR
		else:
			self.M = RR.shape[0]
			self.RR = RR



		self.timeInit = time.time() - start

		

	def run(self, OPT_R=False, OPT_TH=False, MOD='VarDTC'):

		start = time.time()
		EXsparse = Expert(self.DD.X_train, self.DD.y_train, self.kern, likelihood=self.likelihood, priorN=self.priorN, RR=self.RR)
		EXsparse.create_GPmodel_sparse(MOD=MOD)
		
		EXsparse.optimize_sparse(OPT_R, OPT_TH, sig2_noise_EST=not self.likelihood.is_fixed)

		EXsparse.predict_CI(self.DD.X_test)
		timeSparse = time.time() - start


		if MOD=='FITC':
			ll = EXsparse.GPmod.log_likelihood()
		else:
			ll = EXsparse.GPmod.log_likelihood()[0][0]


		return ResultRun(EXsparse.m, EXsparse.v, EXsparse.CI, 'sparse'+str(self.M), time=self.timeInit+timeSparse, obj=EXsparse, OBJ=self, lik=ll)



class RunPoE:
	def __init__(self, DD, K, kern, likelihood, KMEANS=True, seed=0, sortDim=-1, PROJ='', bw=[None], clusterInds=[None], KDTREE=False, B_stop=100, timeOpt=0, randOrder=False):

		start = time.time()


		self.DD = DD
		self.K = K
		self.kern = kern.copy()
		self.likelihood = likelihood.copy()
		#self.sig2_noise = sig2_noise
		#self.sig2_noise_EST = sig2_noise_EST

		PP = Partition(DD)

		startPar = time.time()
		if KDTREE:
			PP.compute_partition2(B_stop=B_stop, seed=seed)
			self.K = PP.K
			print('new K is', self.K)
		else:
			PP.compute_partition(K, KMEANS=KMEANS, randOrder=randOrder, seed=seed, sortDim=sortDim, PROJ=PROJ, bw=bw, clusterInds=clusterInds)
		
		self.timePar = time.time() - startPar

		print(self.timePar)

		self.PP = PP
		
		self.PoE1 = PoE(PP.X_trains_k, PP.y_trains_k, kern, self.likelihood.variance[0], centers= PP.centers )
		self.timePoE = time.time() - start

		self.timeOpt = timeOpt


	def optimize_run(self, INDEP=False, GTOL=5e-2, maxF=30):

		start = time.time()

		if INDEP:
			# independent optimization
			self.PoE1.optimize()
		else:


			########## not autom

			#x0 = np.log( np.array([self.kern.variance[0], self.kern.lengthscale[0], self.sig2_noise]) )
			#x0 = np.log ( self.PoE1.experts[0].GPmod.param_array )
			x0 = np.log( self.get_params())

			self.ESTs =  self.get_ESTs()
			
			# print(self.ESTs)


			# print(x0[self.ESTs==True])


			method = 'L-BFGS-B'
			#method = 'BFGS'
			self.res = minimize(f_PoE, x0[self.ESTs==True], method=method, jac=df_PoE, args=self, options={'disp': True, 'gtol':GTOL, 'maxfun': maxF} )


		self.timeOpt = time.time() - start


	def get_ESTs(self):

		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]

		ESTs = np.zeros(0)
		for keri in kerns:
			#ESTs = np.hstack([ESTs, 1-keri.variance.is_fixed, np.ones(len(keri.lengthscale), dtype=int)*(1-keri.lengthscale.is_fixed) ])

			for par in keri.parameters[:]:
				ESTs = np.hstack([ESTs, np.ones(np.prod(par.values.shape), dtype=int)*(1-par.is_fixed) ])

			
		ESTs = np.hstack([ESTs, 1-self.likelihood.variance.is_fixed ])

		return ESTs



	def get_params(self):

		#params = np.hstack([self.kern.variance, self.kern.lengthscale, self.sig2_noise])

		
		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]
		
		params = np.zeros(0)
		for keri in kerns:
			#params = np.hstack([params, keri.variance, keri.lengthscale])
			for par in keri.parameters[:]:
				params = np.hstack([params, par.values.flatten()])

		params = np.hstack([params, self.likelihood.variance[0]])

		return params


	def run(self, method='minVar'):

		

		start = time.time()

		if method=='minVar':
			agg = self.PoE1.aggregate_minVar(self.DD.X_test)

		elif method=='NN':
			agg = self.PoE1.aggregate_NN(self.DD.X_test)
			
		elif method=='PoE-1':
			agg = self.PoE1.aggregate(self.DD.X_test)

		elif method=='PoE-1/J':
			agg = self.PoE1.aggregate(self.DD.X_test, 1/self.K)

		elif method=='GPoE':
			agg = self.PoE1.aggregate_diff(self.DD.X_test)

		elif method=='GPoE-scaled' or method=='GPoEs':
			agg = self.PoE1.aggregate_diff(self.DD.X_test, SCAL=True)

		elif method=='BCM':
			agg = self.PoE1.aggregate(self.DD.X_test, BCM=True)

		elif method=='RBCM':
			agg = self.PoE1.aggregate_diff(self.DD.X_test, BCM=True)


		timePred = time.time() - start

		timeA = self.timePoE + self.timeOpt  + timePred

		#agg.method = method

		return ResultRun(agg.m, agg.v, agg.CI, method, time=timeA, obj=agg, OBJ=self, lik=self.PoE1.lik)


@np_cache(maxsize=1)
def f_df_PoE( params, runPoE ):



	# set new params
	### not automated yet!!!!!!!!!!!!!!
	expParams = np.exp(params)

	allParams = runPoE.get_params()

	allParams[runPoE.ESTs==True] = expParams


	if runPoE.kern.name == 'sum':
		kerns = runPoE.kern.parts
	else:
		kerns = [runPoE.kern]

	i = 0
	for keri in kerns:

		#keri.variance = allParams[i]
		#keri.lengthscale = allParams[i+1:i+1+len(keri.lengthscale) ]

		for par in keri.parameters[:]:

			#print(par.values)

			shapeV = par.values.shape
			lenV = np.prod(shapeV)
			value = allParams[i:i+lenV]

			#print(shapeV)
			#print(lenV)
			
			par.param_array[:] = np.reshape(value,shapeV)
			
			# if lenV==1:
			# 	par.param_array[:](value[0])
			# else:
			# 	par.fill( np.reshape(value,shapeV) )

			i += lenV

		#i += 1+len(keri.lengthscale)


	runPoE.likelihood.variance = allParams[-1]


	# run method
	runPoE.PoE1 = PoE(runPoE.PP.X_trains_k, runPoE.PP.y_trains_k, runPoE.kern, runPoE.likelihood.variance[0])




	# sum derivatives together, over all experts
	for i,exp in enumerate(runPoE.PoE1.experts):
		if i==0:
			dliks = exp.GPmod.gradient_full 
		else:
			dliks += exp.GPmod.gradient_full 


	likVal = runPoE.PoE1.lik + runPoE.kern.log_prior() + runPoE.likelihood.log_prior()

	dliksTot = dliks
	dliksTot[:-1] += runPoE.kern._log_prior_gradients()
	dliksTot[-1] += runPoE.likelihood._log_prior_gradients()

	return likVal, dliksTot[runPoE.ESTs==True]






def f_PoE( params, runPoE ):

	f, _ = f_df_PoE(params, runPoE)

	#print('fff')
	#print(np.exp(params))
	#print(-f)


	return -f

def df_PoE( params, runPoE ):

	_, df = f_df_PoE(params, runPoE)

	#print('dfff')
	#print(-df * np.exp(params))

	return -df * np.exp(params)



@np_cache(maxsize=1)
def df_stoch_PoE( params, runPoE ):



	# set new params
	### not automated yet!!!!!!!!!!!!!!
	expParams = np.exp(params)

	allParams = runPoE.get_params()

	allParams[runPoE.ESTs==True] = expParams


	if runPoE.kern.name == 'sum':
		kerns = runPoE.kern.parts
	else:
		kerns = [runPoE.kern]

	i = 0
	for keri in kerns:

		#keri.variance = allParams[i]
		#keri.lengthscale = allParams[i+1:i+1+len(keri.lengthscale) ]

		for par in keri.parameters[:]:

			#print(par.values)

			shapeV = par.values.shape
			lenV = np.prod(shapeV)
			value = allParams[i:i+lenV]

			#print(shapeV)
			#print(lenV)
			
			par.param_array[:] = np.reshape(value,shapeV)
			
			# if lenV==1:
			# 	par.param_array[:](value[0])
			# else:
			# 	par.fill( np.reshape(value,shapeV) )

			i += lenV

		#i += 1+len(keri.lengthscale)


	runPoE.likelihood.variance = allParams[-1]


	# run method
	runPoE.PoE1 = PoE(runPoE.PP.X_trains_k, runPoE.PP.y_trains_k, runPoE.kern, runPoE.likelihood.variance[0])




	# sum derivatives together, over all experts
	for i,exp in enumerate(runPoE.PoE1.experts):
		if i==0:
			dliks = exp.GPmod.gradient_full 
		else:
			dliks += exp.GPmod.gradient_full 


	likVal = runPoE.PoE1.lik + runPoE.kern.log_prior() + runPoE.likelihood.log_prior()

	dliksTot = dliks
	dliksTot[:-1] += runPoE.kern._log_prior_gradients()
	dliksTot[-1] += runPoE.likelihood._log_prior_gradients()

	return likVal, dliksTot[runPoE.ESTs==True]



	
class RunGRBCM:
	def __init__(self, DD, K, kern, likelihood, KMEANS=True, seed=0, KDTREE=False, B_stop=100, timeOpt=0, randOrder=False):

		start = time.time()


		self.DD = DD
		self.K = K
		self.kern = kern.copy()
		self.likelihood = likelihood.copy()
		#self.sig2_noise = sig2_noise
		#self.sig2_noise_EST = sig2_noise_EST

		PP = Partition(DD)

		self.PP = PP

		if KDTREE:
			PP.compute_partition2_GRBCM(B_stop=B_stop, seed=seed)
			self.K = PP.K
			print('new K is', self.K)
		else:
		
			PP.compute_partitionGRBCM(K, KMEANS=KMEANS, randOrder=randOrder, seed=seed)

		

		#print(time.time() - start)

		self.PoE1 = PoE(PP.X_trains_k, PP.y_trains_k, kern, self.likelihood.variance[0], CORR='VS1')
		self.timePoE = time.time() - start

		#print(self.timePoE)

		self.timeOpt = timeOpt
		self.lik = 0


	def optimize_run(self, GTOL=5e-2, maxF=30):

		start = time.time()




		#x0 = np.log( np.array([self.kern.variance[0], self.kern.lengthscale[0], self.sig2_noise]) )
		#x0 = np.log ( self.PoE1.experts[0].GPmod.param_array )
		x0 = np.log( self.get_params())

		self.ESTs =  self.get_ESTs()



		method = 'L-BFGS-B'
		#method = 'BFGS'
		self.res = minimize(f_PoE, x0[self.ESTs==True], method=method, jac=df_PoE, args=self, options={'disp': True, 'gtol':GTOL, 'maxfun': maxF} )


		#for exp in self.PoE1.experts:
		#	print(exp.lik)


		self.lik = self.PoE1.lik



		self.PoE1 = PoE(self.PP.X_trains_k, self.PP.y_trains_k, self.kern, self.likelihood.variance[0], CORR='VS1')


		self.timeOpt = time.time() - start



	def get_ESTs(self):

		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]

		ESTs = np.zeros(0)
		for keri in kerns:
			#ESTs = np.hstack([ESTs, 1-keri.variance.is_fixed, np.ones(len(keri.lengthscale), dtype=int)*(1-keri.lengthscale.is_fixed) ])

			for par in keri.parameters[:]:
				ESTs = np.hstack([ESTs, np.ones( np.prod(par.values.shape), dtype=int)*(1-par.is_fixed) ])

			
		ESTs = np.hstack([ESTs, 1-self.likelihood.variance.is_fixed ])

		return ESTs



	def get_params(self):

		#params = np.hstack([self.kern.variance, self.kern.lengthscale, self.sig2_noise])

		
		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]
		
		params = np.zeros(0)
		for keri in kerns:
			#params = np.hstack([params, keri.variance, keri.lengthscale])
			for par in keri.parameters[:]:
				params = np.hstack([params, par.values.flatten()])

		params = np.hstack([params, self.likelihood.variance[0]])

		return params



	def run(self):

		

		start = time.time()
		agg = self.PoE1.aggregate_VS1(self.DD.X_test)
		timePred = time.time() - start

		timeA = self.timePoE + timePred + self.timeOpt 

		#agg.method = 'GRBCM'

		#lik = self.PoE1.lik
		lik = 0

		return ResultRun(agg.m, agg.v, agg.CI, 'GRBCM', time=timeA  , obj=agg, OBJ=self, lik=self.lik)
		#return ResultRun(agg.m, agg.v, agg.CI, 'GRBCM_K'+str(self.K), time=timeA, obj=agg, OBJ=self, lik=lik)




# class RunBGP:
# 	def __init__(self, DD, kern, likelihood):

# 		self.DD = DD
# 		self.kern = kern.copy()
# 		self.likelihood = likelihood.copy()



# 	def run(self, K, P, sp=1, jit=1e-15, EE_EE0 = (False, False))


# 		B = int(DD.Ntrain/K)
# 		J = int(np.ceil(sp*B))

# 		BGP = BandGP(self.kern, self.DD, self.lik)




class RunINC:
	def __init__(self, DD, K, kern, likelihood, KMEANS=True, seed=0, randOrder=False, sortDim=-1, PROJ='', bw=[None], clusterInds=[None]):

		self.DD = DD
		self.K = K
		self.kern = kern.copy()
		self.likelihood = likelihood.copy()
		#self.sig2_noise = sig2_noise

		#self.sig2_noise_EST = sig2_noise_EST

	

		self.seed = seed

		start = time.time()

		self.PP = Partition(DD)

		self.PP.compute_partition(K, KMEANS=KMEANS, randOrder=randOrder, seed=seed, sortDim=sortDim, PROJ=PROJ, bw=bw, clusterInds=clusterInds)

		
		self.timeInit = time.time() - start


	def run(self, P=1, MODE='sparse_global', a=None, q=0.5, Jp=1, M=0, PRINT=False, r=1, DIAG=True, \
						STORE=False, jit=1e-7, PREDICT=True, GRAD=False, OPT=False, GTOL=5e-2, alpha=1,\
						U=1, Uinit=1, STOCH=False, gamma=0.01, E=10, LAST_EPOCH_FIX=True, GRAD_STORE=False, FIRST_DEEP=False, GTOL0=1e-1, maxF=30,\
						PREDICT_STORE=False, PER=0, HH=1, J=1, J1=1, LINSPACE=True, KL_and_LIK=False, mvFULL=None,\
						INDEP=True, PERM=True, REL=1e-5, MINVAR_KF=True, SMOOTH=False):
										#GTOL = 5e-2		#1e-1, 1e-2





		if a==None:
			#a = 1
			if self.K-P==0:
				a = 1
			else:
				a = np.round(np.sqrt(P/(self.K-P)),2)


		start = time.time()
		if M>0:
			# select randomly
			#np.random.seed(self.seed)
			#indsRand_M = np.random.permutation(self.DD.Ntrain)[:M]

			#select ordered
			indsOrd_M = np.array(np.floor(np.linspace(0,self.DD.Ntrain-1,M)),dtype=int)
			RR = self.DD.X_train[indsOrd_M,:].copy()	
			RR += np.random.randn(RR.shape[0], RR.shape[1] )*1e-4   #!!!
		else:
			RR = np.zeros((0,self.DD.X_train.shape[1]))


		incModP = INCR(self.PP.X_trains_k, self.PP.y_trains_k, self.K, P, Jp, self.kern, self.likelihood, \
						self.DD.X_test, N=self.DD.Ntrain, MODE=MODE, a=a, R=RR, q=q, r=r, DIAG=DIAG, jit=jit, \
						alpha=alpha, gamma=gamma, GRAD_STORE=GRAD_STORE, PER=PER, HH=HH, J=J, J1=J1, LINSPACE=LINSPACE, KL_and_LIK=KL_and_LIK, mvFULL=mvFULL,\
						MINVAR_KF=MINVAR_KF, SMOOTH=SMOOTH)

		#self.incModP = incModP



		### here kern gets changed!!
		if OPT:
			#f_df_INCR(0, incModP)
			#f_INCR(0, incModP)
			#df_INCR(0, incModP)


			# in log space!! 
			# not automatic!!!

			#x0 = np.log( np.array([self.kern.variance[0], self.kern.lengthscale[0], self.sig2_noise]) )

			if STOCH:

				if FIRST_DEEP:

					method = 'L-BFGS-B'

					incModP.K = Uinit 		#### 1 		#U or 1?	#######!!!!!
			
					x0 = np.log( incModP.get_PARAMS() )	

					# deterministic
					self.res0 = minimize(f_INCR, x0, method=method, jac=df_INCR, args=incModP, options={'disp': True, 'gtol':GTOL0, 'maxfun': maxF} )

					# or stochastic
					#for e in range(10):
					#	incModP.apply_KF_deriv(PRINT=False, STORE=False, PREDICT=False, GRAD=True, STOCH=True, U=U)			## U or Uinit??

					incModP.K = self.K


				for e in range(E):
					
					print(E)
					incModP.apply_KF_deriv(PRINT=False, STORE=False, PREDICT=False, GRAD=True, STOCH=True, U=U)


			else:

				method = 'L-BFGS-B'
				#method = 'BFGS'



				x0 = np.log( incModP.get_PARAMS() )	
				self.res = minimize(f_INCR, x0, method=method, jac=df_INCR, args=incModP, options={'disp': True, 'gtol':GTOL, 'maxfun': maxF} )




		if LAST_EPOCH_FIX:
			incModP.apply_KF_deriv(PRINT=PRINT, STORE=STORE, PREDICT=PREDICT, GRAD=GRAD, PREDICT_STORE=PREDICT_STORE)
		else:
			incModP.apply_KF_deriv(PRINT=PRINT, STORE=STORE, PREDICT=True, GRAD=True, STOCH=True, U=U, PREDICT_STORE=PREDICT_STORE)








		timePred = time.time() - start


		self.timeA = self.timeInit + timePred


		self.incModP = incModP
		return ResultRun(incModP.m_agg, incModP.v_agg, incModP.CI, incModP.name, time=self.timeA, obj=incModP, OBJ=self, lik=incModP.lik)


	def runKS(self):

		start = time.time() 

		MINVAR_KS=True

		self.incModP.apply_KS(MINVAR_KS=MINVAR_KS)


		print( np.sum( self.incModP.v_minV_S < 0 ) )
		print( np.min(self.incModP.v_minV_S) )
	


		self.incModP.CI_S = np.zeros((len(self.incModP.m_minV_S), 2))
		self.incModP.CI_S[:,0] = self.incModP.m_minV_S - 1.96*np.sqrt(self.incModP.v_minV_S)		
		self.incModP.CI_S[:,1] = self.incModP.m_minV_S + 1.96*np.sqrt(self.incModP.v_minV_S)

		timeSmooth = time.time() - start

		return ResultRun(self.incModP.m_minV_S, self.incModP.v_minV_S, self.incModP.CI_S, self.incModP.name, time=self.timeA+timeSmooth, obj=self.incModP, OBJ=self, lik=self.incModP.lik)




	def runIndep(self, STOCH=True, E=10, gamma=0.01, U=1, GRAD_STORE=False, PERM=True, PRINT=True, REL=1e-5, maxF=100, PP=None):

		start = time.time()

		if PP==None:
			PPi = self.PP
		else:
			PPi = PP


		self.II = Independent(self.DD, self.K, self.kern, self.likelihood, PP=PPi)

		if STOCH:
			self.II.run_epochs(E=E, gamma=gamma, U=U, TRACE=GRAD_STORE, PERM=PERM, PRINT=PRINT, REL=REL)

		else:
			self.II.opt_batch(TRACE=GRAD_STORE, GTOL=REL, maxF=maxF)

		self.timeOpt = time.time() - start


	def runKF(self, P, J=1, J1=1, MODE='moving', PRINT=False, STORE=False, PREDICT_STORE=False, jit=1e-7, HH=1, timePP=0):

		start = time.time()

		incModP = INCR(self.PP.X_trains_k, self.PP.y_trains_k, self.K, P, 1,  self.II.kern, self.II.likelihood, \
					self.DD.X_test, N=self.DD.Ntrain, MODE=MODE, jit=jit, J=J, J1=J1, HH=HH)

		incModP.apply_KF_deriv(PRINT=PRINT, STORE=STORE, PREDICT=True, GRAD=False, PREDICT_STORE=PREDICT_STORE)

		self.timeKF = time.time() - start

		timeA = timePP + self.timeInit + self.timeOpt + self.timeKF

		return ResultRun(incModP.m_agg, incModP.v_agg, incModP.CI, incModP.name, time=timeA, obj=incModP, OBJ=self, lik=incModP.lik)

			
		



	# only possible for Independet
	def plot_II_VALS(self, figsize=(10,5)):

		plt.figure(figsize=figsize)
		VALS = np.array(self.II.VALS)
		nIter, nPar = VALS.shape
		for i in range(nPar):
			plt.plot( VALS[:,i], '.-' )

	def plot_II_GRADS(self, figsize=(10,5)):

		plt.figure(figsize=figsize)
		GRADS = np.array(self.II.GRADS)
		nIter, nPar = GRADS.shape
		for i in range(nPar):
			plt.plot( GRADS[:,i], '.-' )

	def plot_II_LIKS(self, figsize=(10,5)):

		plt.figure(figsize=figsize)

		LIKS = np.array(self.II.LIKS)
		plt.plot( LIKS, '.-' )


#### it is outside
#@functools.lru_cache(maxsize=1)
@np_cache(maxsize=1)
def f_df_INCR( params, incModP):

	#print('logpars ', params)
	#print('pars ', np.exp(params))

	incModP.set_PARAMS(np.exp(params))

	incModP.apply_KF_deriv(PRINT=False, STORE=False, PREDICT=False, GRAD=True)

	dliks = np.zeros(0)
	for par in incModP.PARAMS:

		
		if par.EST:
			dliks = np.hstack( [dliks, par.dlik + par.get_prior_grad() ] )
			

	#print('return ')
	#print(incModP.lik, dliks)

	#fvals.append(incModP.lik)
	
	return incModP.lik, dliks


def f_INCR( params, incModP ):

	f, _ = f_df_INCR(params, incModP)

	#print('fff')
	#print(np.exp(params))
	#print(-f)


	return -f

def df_INCR( params, incModP):

	_, df = f_df_INCR(params, incModP)

	#print('dfff')
	#print(-df * np.exp(params))



	return -df * np.exp(params)



	# def run2(self, P=1, MODE='sparse_global', a=None, q=0.5, Jp=1, M=0, PRINT=False, r=1, DIAG=True, STORE=False, jit=1e-7):


	# 	if a==None:
	# 		#a = 1
	# 		if self.K-P==0:
	# 			a = 1
	# 		else:
	# 			a = np.round(np.sqrt(P/(self.K-P)),2)


	# 	start = time.time()
	# 	if M>0:
	# 		np.random.seed(self.seed)
	# 		indsRand_M = np.random.permutation(self.DD.Ntrain)[:M]
	
	# 		RR = self.DD.X_train[indsRand_M,:].copy()	
	# 		RR += np.random.randn(RR.shape[0], RR.shape[1] )*1e-4   #!!!
	# 	else:
	# 		RR = np.zeros((0,self.DD.X_train.shape[1]))


	# 	incModP = INCR(self.PP.X_trains_k, self.PP.y_trains_k, self.K, P, Jp, self.kern, self.sig2_noise, \
	# 					self.DD.X_test, N=self.DD.Ntrain, MODE=MODE, a=a, R=RR, q=q, r=r, DIAG=DIAG, jit=jit)
	# 	incModP.apply_KF_deriv(PRINT=PRINT, STORE=STORE)
	# 	timePred = time.time() - start


	# 	timeA = self.timeInit + timePred


	# 	return ResultRun(incModP.m_agg, incModP.v_agg, incModP.CI, incModP.name, time=timeA, obj=incModP, OBJ=self, lik=incModP.lik)



class ResultRun:
	def __init__(self, m, v, CI, name, obj=None, OBJ=None, time=0, lik=0 ):

		self.m = m
		self.v = v
		self.CI = CI
		self.obj = obj
		self.OBJ = OBJ
		self.time = time
		self.lik = lik
		self.name = name

		# compute stats
		# plot

	def compute_stats(self, mFull=[None], vFull=[None]):
		DD = self.OBJ.DD
		if hasattr(DD, 'Xt_IN'):
			self.stats = statsDF(self.m, self.v, DD.y_test, DD.f_test,  mFull, vFull,  self.lik, self.time, name=self.name, IN=DD.Xt_IN)
		else:
			self.stats = statsDF(self.m, self.v, DD.y_test, DD.f_test,  mFull, vFull,  self.lik, self.time, name=self.name)
		return self.stats

	def plot_full(self, figsize=(30,4), DATA=True , lev=20, SCAT=True , ax=None, LEG=True, name='', P1=False):
		DD = self.OBJ.DD

		if DD.X_train.shape[1] == 1 or P1:
			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]

			if ax==None:
				PLT=plt
				#PLT.title(self.name)
			else:
				PLT=ax
				#PLT.set_title(self.name, pad=-15)

			if DD.f_test[0]==None:
				ff = DD.y_test
			else:
				ff = DD.f_test

			plot_common(Xts,  inds, self.m, self.CI, DATA=DATA, X_train=DD.X_train, y_train=DD.y_train, Ftest=ff, fg=figsize, ax=ax, LEG=LEG)
			plot_mean_CI(Xts, self.m, self.CI, inds, label=name, col='b-', ax=ax, LEG=LEG)
			
			if LEG:
				PLT.legend()

			if name=='':
				plt.title(self.name)
			else:
				plt.title(name)

		elif DD.X_train.shape[1] == 2:
			plt.figure(figsize=(5,5))

			if name=='':
				plt.title(self.name)
			else:
				plt.title(name)
			#plt.tricontour(X_test[:,0], X_test[:,1], mFull, lev)
			#plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.');

			if DD.GRID:
				plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, lev, cmap=DD.cmap)
			
			if SCAT:
				plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.', cmap=DD.cmap)

			plt.show();
		else:
			print('not supported for D = ',DD.X_train.shape[1])

		



	def plot_poe(self, mFull=[None], CI_Full=None, figsize=(30,4), DATA=True , lev=20, SCAT=True  , ax=None, LEG=True, P1=False):

		DD = self.OBJ.DD

		if ax==None:
			PLT=plt

		else:
			PLT=ax


		if DD.X_train.shape[1] == 1 or P1:
			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]

			if DD.f_test[0]==None:
				ff = DD.y_test
			else:
				ff = DD.f_test

			plot_common(Xts,  inds, mFull, CI_Full, DATA=DATA, X_train=DD.X_train, Ftest=ff, y_train=DD.y_train, fg=figsize, ax=ax, LEG=LEG)
			plot_mean_CI(Xts, self.m, self.CI, inds, label=self.name, col='b-', ax=ax, LEG=LEG)
			exj = self.OBJ.PoE1.experts[0]
			if self.name[:5]=='GRBCM':
				PLT.plot( exj.X[:,0],  exj.y, 'k.',label='first expert')
			else:
				PLT.plot( exj.X[:,0],  exj.y, 'g.',label='first expert')
			if LEG:
				PLT.legend()


		elif DD.X_train.shape[1] == 2:
			plt.figure(figsize=(5,5))
			plt.title(self.name)
			#plt.tricontour(X_test[:,0], X_test[:,1], mFull, lev)
			#plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.')

			if DD.GRID:
				plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, lev, cmap=DD.cmap)
			
			if SCAT:
				plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.', cmap=DD.cmap)

			exj = self.OBJ.PoE1.experts[0]
			plt.plot(exj.X[:,0], exj.X[:,1], 'ko', label='first expert')
			plt.legend()
			plt.show();
		else:
			print('not supported for D = ',DD.X_train.shape[1])



	def plot_sparse(self, mFull=[None], CI_Full=None, figsize=(30,4), DATA=True , lev=20, SCAT=True , ax=None, LEG=True, name='', P1=False, F_True=True):

		DD = self.OBJ.DD

		if ax==None:
			PLT=plt

		else:
			PLT=ax

		if name=='':
			nam = self.name
		else:
			nam=name

		if DD.X_train.shape[1] == 1 or P1:
			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]

			if DD.f_test[0]==None:
				ff = DD.y_test
			else:
				ff = DD.f_test

			if F_True==False:
				ff = [None]

			plot_common(Xts,  inds, mFull, CI_Full, DATA=DATA, X_train=DD.X_train, y_train=DD.y_train, Ftest=ff, fg=figsize, ax=ax, LEG=LEG)
			PLT.plot( self.obj.GPmod.inducing_inputs[:,0], self.obj.GPmod.posterior.mean, 'k.', label='inducing')
			plot_mean_CI(Xts, self.m, self.CI, inds, label=nam, col='b-', ax=ax, LEG=LEG)
			if LEG:
				PLT.legend()

		elif DD.X_train.shape[1] == 2:
			plt.figure(figsize=(5,5))
			plt.title(self.name)
			#plt.tricontour(X_test[:,0], X_test[:,1], mFull, lev)
			#plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.')

			if DD.GRID:
				plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, lev, cmap=DD.cmap)
			
			if SCAT:
				plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.', cmap=DD.cmap)

			plt.plot( self.obj.GPmod.inducing_inputs[:,0], self.obj.GPmod.inducing_inputs[:,1], 'ko', label='inducing')
			if LEG:
				plt.legend()
			plt.show();
		else:
			print('not supported for D = ',DD.X_train.shape[1])



	def plot_inc(self, mFull=[None], CI_Full=None, figsize=(30,4), DATA=True , lev=20, SCAT=True , ax=None, LEG=True, name='', P1=False):

		DD = self.OBJ.DD

		

		if DD.X_train.shape[1] == 1 or P1:

			if ax==None:
				PLT=plt
			else:
				PLT=ax

			if name=='':
				nam = self.name
			else:
				nam=name


			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]

			if DD.f_test[0]==None:
				ff = DD.y_test
			else:
				ff = DD.f_test

			plot_common(Xts,  inds, mFull, CI_Full, DATA=DATA, X_train=DD.X_train, y_train=DD.y_train, Ftest=ff, fg=figsize, ax=ax, LEG=LEG)
			plot_mean_CI(Xts, self.m, self.CI, inds, label=nam, col='b-', ax=ax, LEG=LEG)

			
			PLT.plot( self.obj.R1[:,0],  self.obj.mk, 'k.', label='last inducings')
			PLT.plot( self.obj.X_trains_k[0][:,0],  self.obj.y_trains_k[0], 'g.', label='first data')
			if LEG:
				PLT.legend()



		elif DD.X_train.shape[1] == 2:
			plt.figure(figsize=(5,5))
			plt.title(self.name)
			#plt.tricontour(X_test[:,0], X_test[:,1], mFull, lev)
			#plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.')
			if DD.GRID:
				plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, lev, cmap=DD.cmap)
			
			if SCAT:
				plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.', cmap=DD.cmap)


			plt.plot( self.obj.R1[:,0],  self.obj.R1[:,1], 'ko', label='last inducings')	
			plt.plot( self.obj.X_trains_k[-1][:,0],  self.obj.X_trains_k[-1][:,1], 'r.', label='last data')
			plt.legend()
			plt.show();
		else:
			print('not supported for D = ',DD.X_train.shape[1])

	def plot_bgp(self, mFull=[None], CI_Full=None, figsize=(30,4), DATA=True , lev=20, SCAT=True , ax=None, LEG=True, name='', P1=False, F_True=True):

		DD = self.OBJ.DD

		

		if DD.X_train.shape[1] == 1 or P1:

			if ax==None:
				PLT=plt
			else:
				PLT=ax

			if name=='':
				nam = self.name
			else:
				nam=name


			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]

			if DD.f_test[0]==None:
				ff = DD.y_test
			else:
				ff = DD.f_test

			if F_True==False:
				ff = [None]

			plot_common(Xts,  inds, mFull, CI_Full, DATA=DATA, X_train=DD.X_train, y_train=DD.y_train, Ftest=ff, fg=figsize, ax=ax, LEG=LEG)
			plot_mean_CI(Xts, self.m, self.CI, inds, label=nam, col='b-', ax=ax, LEG=LEG)

			
			PLT.plot( np.vstack(self.obj.Aks),  self.obj.m_post, 'k.', label=' inducings')
			PLT.plot( self.obj.Xks[0][:,0],  self.obj.yks[0], 'g.', label='first data')
			if LEG:
				PLT.legend()



		elif DD.X_train.shape[1] == 2:
			plt.figure(figsize=(5,5))
			plt.title(self.name)
			#plt.tricontour(X_test[:,0], X_test[:,1], mFull, lev)
			#plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.')
			if DD.GRID:
				plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, lev, cmap=DD.cmap)
			
			if SCAT:
				plt.scatter(DD.X_test[:,0], DD.X_test[:,1], c=self.m, marker='.', cmap=DD.cmap)


			plt.plot( self.obj.Aks[-1][:,0],  self.obj.Aks[-1][:,1], 'ko', label='last inducings')	
			plt.plot( self.obj.Xks[-1][:,0],  self.obj.Xks[-1][:,1], 'r.', label='last data')
			plt.legend()
			plt.show();
		else:
			print('not supported for D = ',DD.X_train.shape[1])


	def plot_2D_1D(self,  mFull, CI_Full, dim=0, name='', ylim=(-6,6)):
		nx = self.OBJ.DD.NtestSQRT

		mm = np.reshape( self.m, (nx, nx) )
		CI1 = np.reshape( self.CI[:,0], (nx, nx) )
		CI2 = np.reshape( self.CI[:,1], (nx, nx) )

		mmFull = np.reshape( mFull, (nx, nx) )
		CI1Full = np.reshape( CI_Full[:,0], (nx, nx) )
		CI2Full = np.reshape( CI_Full[:,1], (nx, nx) )

		if dim==0:
			numS = len(self.OBJ.DD.x)
			for i,val in enumerate(self.OBJ.DD.x):
				plt.figure(figsize=(10,3))
				plt.ylim(ylim)
				plt.plot(self.OBJ.DD.y, mm[i,:], 'g', label=self.name)
				plt.plot(self.OBJ.DD.y, CI1[i,:], 'g:')
				plt.plot(self.OBJ.DD.y, CI2[i,:], 'g:')

				plt.plot(self.OBJ.DD.y, mmFull[i,:], 'b', label='fullGP')
				plt.fill_between(self.OBJ.DD.y, CI1Full[i,:], CI2Full[i,:], alpha= 0.2)
				plt.legend(loc=1)

				plt.savefig('2DATA/'+name+'_plotSlice_'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close()
			

	

		if dim==1:
			numS = len(self.OBJ.DD.y)
			for i,val in enumerate(self.OBJ.DD.y):
				plt.figure(figsize=(10,3))
				plt.ylim(ylim)
				plt.plot(self.OBJ.DD.x, mm[:,i], 'g', label=self.name)
				plt.plot(self.OBJ.DD.x, CI1[:,i], 'g:')
				plt.plot(self.OBJ.DD.x, CI2[:,i], 'g:')

				plt.plot(self.OBJ.DD.x, mmFull[:,i], 'b', label='fullGP')
				plt.fill_between(self.OBJ.DD.x, CI1Full[:,i], CI2Full[:,i], alpha= 0.2)
				plt.legend(loc=1)

				plt.savefig('2DATA/'+name+'_plotSlice_'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close()


		fps = 5
		range01 = range(0,numS)

		imageio.mimsave('2DATA/slice_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotSlice_'+str(k)+'.png') for k in range01], fps=fps)
		print('Image(''2DATA/slice_'+name+'.gif)')

	# only for bgp and D=1
	def plot_predictive_evolution2(self, name='', fps=5, figsize=(30,4), ylimA=4, mFull=[None], CI_Full=[None], Plot_True=False ):

		DD = self.OBJ.DD

		K = self.OBJ.K

		Aks = self.obj.Aks

	
		for k in range(K):
			f, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(figsize[0],2*figsize[1]))
			

			inds = np.argsort(DD.X_test[:,0])
			Xts = DD.X_test[inds,0]


			# CIk = np.zeros(self.CI.shape)
			# CIk[:,0] = self.obj.Mks[k,:] - 1.96*np.sqrt(self.obj.Vks[k,:])
			# CIk[:,1] = self.obj.Mks[k,:] + 1.96*np.sqrt(self.obj.Vks[k,:])
			# plot_mean_CI(Xts, self.obj.Mks[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[0])
			
			# if not mFull[0]==None:
			# 	axs[0].plot(Xts, mFull[inds], 'b')
			# 	axs[0].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

			# axs[0].plot( Rks[k][:,0], self.obj.mks[k], 'ko', label='active set')	;
			# axs[0].plot( self.obj.X_trains_k[k][:,0],  self.obj.y_trains_k[k], 'r.', label='data');
			# axs[0].legend(loc=1);
			# axs[0].set_title('individual');


			# CIk = np.zeros(self.CI.shape)
			# CIk[:,0] = self.obj.Ms[k,:] - 1.96*np.sqrt(self.obj.Vs[k,:])
			# CIk[:,1] = self.obj.Ms[k,:] + 1.96*np.sqrt(self.obj.Vs[k,:])
			# plot_mean_CI(Xts, self.obj.Ms[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[1])
			

			# if not mFull[0]==None:
			# 	axs[1].plot(Xts, mFull[inds], 'b')
			# 	axs[1].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

			# axs[1].plot( Rks[k][:,0], self.obj.mks[k], 'ko', label='active set')	;
			# for j in range(k+1):
			# 	axs[1].plot( self.obj.X_trains_k[j][:,0],  self.obj.y_trains_k[j], 'r.');

			# if Plot_True:
			# 	axs[1].plot( Xts, DD.y_test[inds], 'g:', label='y_test')	;

			# axs[1].legend(loc=1);
			# axs[1].set_title('cummulative');



	



			plt.ylim(-ylimA,ylimA)
			f.subplots_adjust(hspace=0.)
			
			plt.savefig('2DATA/'+name+'_plotEvo_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
			plt.close();

		#fps = 5
		range01 = range(0,K)

		imageio.mimsave('2DATA/evoBGP_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotEvo_'+str(k)+'.png') for k in range01], fps=fps)
		print('Image(''2DATA/evoBGP_'+name+'.gif)')

	# only for incs and D=1
	def plot_predictive_evolution(self, name='', fps=5, figsize=(30,4), ylimA=4, mFull=[None], CI_Full=[None], Plot_True=False ):

		DD = self.OBJ.DD

		K = self.OBJ.K

		Rks = self.obj.Rks

		if self.obj.HH == 1:
			for k in range(K):
				f, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(figsize[0],2*figsize[1]))
				

				inds = np.argsort(DD.X_test[:,0])
				Xts = DD.X_test[inds,0]


				CIk = np.zeros(self.CI.shape)
				CIk[:,0] = self.obj.Mks[k,:] - 1.96*np.sqrt(self.obj.Vks[k,:])
				CIk[:,1] = self.obj.Mks[k,:] + 1.96*np.sqrt(self.obj.Vks[k,:])
				plot_mean_CI(Xts, self.obj.Mks[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[0])
				
				if not mFull[0]==None:
					axs[0].plot(Xts, mFull[inds], 'b')
					axs[0].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

				axs[0].plot( Rks[k][:,0], self.obj.mks[k], 'ko', label='active set')	;
				axs[0].plot( self.obj.X_trains_k[k][:,0],  self.obj.y_trains_k[k], 'r.', label='data');
				axs[0].legend(loc=1);
				axs[0].set_title('individual');


				CIk = np.zeros(self.CI.shape)
				CIk[:,0] = self.obj.Ms[k,:] - 1.96*np.sqrt(self.obj.Vs[k,:])
				CIk[:,1] = self.obj.Ms[k,:] + 1.96*np.sqrt(self.obj.Vs[k,:])
				plot_mean_CI(Xts, self.obj.Ms[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[1])
				

				if not mFull[0]==None:
					axs[1].plot(Xts, mFull[inds], 'b')
					axs[1].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

				axs[1].plot( Rks[k][:,0], self.obj.mks[k], 'ko', label='active set')	;
				for j in range(k+1):
					axs[1].plot( self.obj.X_trains_k[j][:,0],  self.obj.y_trains_k[j], 'r.');

				if Plot_True:
					axs[1].plot( Xts, DD.y_test[inds], 'g:', label='y_test')	;

				axs[1].legend(loc=1);
				axs[1].set_title('cummulative');



				plt.ylim(-ylimA,ylimA)
				f.subplots_adjust(hspace=0.)
				
				plt.savefig('2DATA/'+name+'_plotEvo_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close();


		# self.HH == 2
		else:


			for k in range(2*K):
				f, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(figsize[0],2*figsize[1]))
					

				k_half = int(np.floor(k/2))

				inds = np.argsort(DD.X_test[:,0])
				Xts = DD.X_test[inds,0]


				CIk = np.zeros(self.CI.shape)
				CIk[:,0] = self.obj.Mks[k,:] - 1.96*np.sqrt(self.obj.Vks[k,:])
				CIk[:,1] = self.obj.Mks[k,:] + 1.96*np.sqrt(self.obj.Vks[k,:])
				plot_mean_CI(Xts, self.obj.Mks[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[0])
				
				if not mFull[0]==None:
					axs[0].plot(Xts, mFull[inds], 'b')
					axs[0].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

				#axs[0].plot( Rks[k_half][:,0], self.obj.mks[k], 'ko', label='active set')	;
				axs[0].plot( self.obj.X_trains_k[k_half][:,0],  self.obj.y_trains_k[k_half], 'r.', label='data');
				axs[0].legend(loc=1);
				axs[0].set_title('individual');

				CIk = np.zeros(self.CI.shape)
				CIk[:,0] = self.obj.Ms[k,:] - 1.96*np.sqrt(self.obj.Vs[k,:])
				CIk[:,1] = self.obj.Ms[k,:] + 1.96*np.sqrt(self.obj.Vs[k,:])
				plot_mean_CI(Xts, self.obj.Ms[k,:], CIk, inds, label=self.name, col='b:', lw=1, ax=axs[1])
				

				if not mFull[0]==None:
					axs[1].plot(Xts, mFull[inds], 'b')
					axs[1].fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)

				#axs[1].plot( Rks[k_half][:,0], self.obj.mks[k], 'ko', label='active set')	;
				for j in range(k_half+1):
					axs[1].plot( self.obj.X_trains_k[j][:,0],  self.obj.y_trains_k[j], 'r.');
				axs[1].legend(loc=1);
				axs[1].set_title('cummulative');







				plt.ylim(-ylimA,ylimA)
				f.subplots_adjust(hspace=0.)
				
				plt.savefig('2DATA/'+name+'_plotEvo_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close();

		#fps = 5
		range01 = range(0,self.obj.HH*K)

		imageio.mimsave('2DATA/evo_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotEvo_'+str(k)+'.png') for k in range01], fps=fps)
		print('Image(''2DATA/evo_'+name+'.gif)')


	
	# only for incs
	def plot_activeSet(self, name='', fps=5, mFull=[None], CI_Full=None, figsize=(30,4)):

		DD = self.OBJ.DD

		K = self.OBJ.K

		Rks = self.obj.Rks

		if DD.D==1:

			for k in range(K):
				plt.figure(figsize=figsize);

				

				inds = np.argsort(DD.X_test[:,0])
				Xts = DD.X_test[inds,0]

				#plt.plot(Xts, self.m[inds], 'g')
				#plt.fill_between(Xts, self.CI[inds,0], self.CI[inds,1], alpha= 0.1)
				if mFull[0]!=None:
					plt.plot(Xts, mFull[inds],'c',label='full')
					plt.fill_between(Xts, CI_Full[inds,0], CI_Full[inds,1], alpha= 0.1)
				plot_mean_CI(Xts, self.m, self.CI, inds, label=self.name, col='b:')
				

				plt.plot( Rks[k][:,0], self.obj.mks[k], 'ko', label='active set')	;
				plt.plot( self.obj.X_trains_k[k][:,0],  self.obj.y_trains_k[k], 'r.', label='data');
				plt.legend(loc=1);
				plt.title(self.name);
				
				plt.savefig('2DATA/'+name+'_plotAct_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close();

			#fps = 5
			range01 = range(0,K)

			imageio.mimsave('2DATA/active_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotAct_'+str(k)+'.png') for k in range01], fps=fps)
			print('Image(''2DATA/active_'+name+'.gif)')


		if DD.D==2:

			for k in range(K):
				plt.figure(figsize=(5,5))
				plt.title(self.name)
			
				if DD.GRID:
					plt.tricontour(DD.X_test[:,0], DD.X_test[:,1], self.m, 20, cmap=DD.cmap)
				

				plt.plot( Rks[k][:,0],  Rks[k][:,1], 'ko', label='active set')	
				plt.plot( self.obj.X_trains_k[k][:,0],  self.obj.X_trains_k[k][:,1], 'r.', label='data')
				plt.legend(loc=1)
				
				plt.savefig('2DATA/'+name+'_plotAct_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close()

			fps = 5
			range01 = range(0,K)

			imageio.mimsave('2DATA/active_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotAct_'+str(k)+'.png') for k in range01], fps=fps)
			print('Image(''2DATA/active_'+name+'.gif)')




# def plotFirstExpert(PoE, ind=0):
# 	exj = PoE.experts[ind]
# 	plt.plot( exj.X[:,0],  exj.y, 'go')



def plot_common(Xts, inds, mFull, CIfull, DATA=True, X_train=None, y_train=None, Ftest=[None], FULL=True, fg=(30,4), ax=None, LEG=True):

	if ax==None:
		PLT=plt
		plt.figure(figsize=fg)
	else:
		PLT=ax



	
	#plt.figure(figsize=(20,4))

	#plt.plot(Xts, Ftest[inds], label='true');

	if not Ftest[0]==None:
		PLT.plot(Xts, Ftest[inds],'g',label='true')


	if FULL:
		# full

		if not mFull[0]==None:
			PLT.plot(Xts, mFull[inds],'c',label='full')
			PLT.fill_between(Xts, CIfull[inds,0], CIfull[inds,1], alpha= 0.2)


	if DATA:
		PLT.plot(X_train[:,0],y_train,'r.',label='train', markersize=3)

	

	#plt.ylim(-6.5,6.5)
	if LEG:
		PLT.legend();

# def plot_mean_CI(x, obj, CI=True, inds=None, label=' ', col='k'):
# 	# x already sorted

# 	plt.title(label)

# 	plt.plot(x, obj.m[inds],col, label=label)
# 	plt.plot(x, obj.CI[inds,0],col)
# 	plt.plot(x, obj.CI[inds,1],col)

	# plt.legend()


def plot_mean_CI(x, m, CI, inds, label=' ', col='k', DOTT=False, ax=None, LEG=True, lw=0.4):

	if ax==None:
		PLT=plt
		PLT.title(label)

	else:
		PLT=ax
		PLT.set_title(label, loc='center', pad=-15)


	# x already sorted
	
	if DOTT:
		col +=':'

	#lw = 0.4

	PLT.plot(x, m[inds], col, label=label, linewidth=lw)
	PLT.plot(x, CI[inds,0], col, linewidth=lw)
	PLT.plot(x, CI[inds,1], col, linewidth=lw)

	if LEG:
		PLT.legend();



#####################################################
############# data generation #######################
#####################################################

class Data:
	def __init__(self, Ntrain=100, Ntest=1000, kern=None, sig2_noise = 0.05, seed=1223, a = 0.1, SAVE=False, \
					MIDDLE_FILLED=False, FUTURE=False, FUT_s=1, FUT_a=1.1, Ntest_IN=0, UNIF=True, name='', GRID=False, scal=1, kernsAdd=None, SPLITS=False, ARGSsplit=[0.9,0.1,100,5]):

		# a ### should decrease with higher D


		self.Ntrain = Ntrain
		self.Ntest = Ntest
		self.kern = kern
		self.sig2_noise = sig2_noise
		self.seed = seed
		self.a = a
		self.MIDDLE_FILLED = MIDDLE_FILLED		## only in 1D

		self.FUTURE = FUTURE					## only in 1D
		self.FUT_s = FUT_s			## where to stop trainign data
		self.FUT_a = FUT_a			## where to stop testing data
		self.Ntest_IN = Ntest_IN	## number of testing point inside training data

		self.GRID	= GRID						# only in 2D
		self.UNIF = UNIF

		self.kernsAdd = kernsAdd # only in 1D

		self.f_test = [None]

		self.SPLITS = SPLITS
		self.ARGSsplit = ARGSsplit


		self.SAVE = SAVE

		self.cmap = cm.get_cmap(name='viridis')
		#self.cmap = cm.get_cmap(name='hsv')


		
		if not kern == None:
			self.D = kern.input_dim

			if kern.name=='rbf':
				self.sig2_0 = kern.variance[0]
				self.lengthscale = kern.lengthscale[0]
				self.name = 'Ntrain'+str(Ntrain)+'_Ntest'+str(Ntest)+'_D'+str(self.D)+'_sig2_noise'+str(sig2_noise)+'_sig2_0'+str(self.sig2_0)+'_l'+str(self.lengthscale)+'_seed'+str(seed)+name
			else:
				self.name = 'Ntrain'+str(Ntrain)+'_Ntest'+str(Ntest)+'_D'+str(self.D)+'_seed'+str(seed)+name

			start = time.time()
			# generate training and testing data

			if self.D==1:
				self.X_train, self.X_test, self.y_train, self.f_test, self.y_test, self.Xt_IN, self.f_train = self.genData1D(scal=scal)
			else:
				self.X_train, self.X_test, self.y_train, self.f_test, self.y_test, self.Xt_IN, self.f_train = self.genDataD()

			self.time = time.time() - start

			if True:
				print(str(np.mean(self.Xt_IN)*100)+'% test data points IN' )
				print('timeData ', self.time)
				if self.SAVE: print('saved in '+self.name)

			if self.SAVE: 

				pickle.dump([self.X_train, self.X_test, self.y_train, self.f_test, self.y_test, self.Xt_IN, self.name], open(self.name, 'wb'))



	def returnSubset(self, Ntrain, Ntest, SORT=False):

		DD = Data(Ntrain, Ntest, sig2_noise = self.sig2_noise, seed=self.seed, a = self.a, MIDDLE_FILLED=self.MIDDLE_FILLED)
		DD.kern = self.kern

		if not SORT:
			# randomize
			np.random.seed(self.seed)
			perm = np.random.permutation(len(self.y_train))
			X_train = self.X_train[perm,:]
			y_train = self.y_train[perm]
		else:
			inds = np.argsort(self.X_train[:,0])
			X_train = self.X_train[inds,:]
			y_train = self.y_train[inds]

		DD.X_train = X_train[:Ntrain,:]
		DD.y_train = y_train[:Ntrain]
		DD.X_test = self.X_test[:Ntest,:]
		DD.f_test = self.f_test[:Ntest]
		DD.y_test = self.y_test[:Ntest]
		DD.Xt_IN = self.Xt_IN[:Ntest]
		DD.name = self.name

		return DD





	def genData1D(self, scal=1):

		# only works for D=1

		np.random.seed(self.seed)

	

		if self.MIDDLE_FILLED:
			a = 0
		else:
			a = self.a

		self.Zks = []

		#scal = 1
		#scal = 5


		if not self.FUTURE and not self.SPLITS:
			if self.UNIF:
				# uniform
				X1 = np.linspace(-1,-a,int(self.Ntrain/2))
				X2 = np.linspace(a,1,int(self.Ntrain/2))
			
			else:
				# random
				X1 = np.random.rand(int(self.Ntrain/2))*(1-a) - 1.
				X2 = np.random.rand(int(self.Ntrain/2))*(1-a) + a

				inds1 = np.argsort(X1)
				X1 = X1[inds1]
				inds2 = np.argsort(X2)
				X2 = X2[inds2]

			Xtrain = np.concatenate((X1, X2))

		else: # future mode or splitMode
			if self.UNIF:
				# uniform
				Xtrain = np.linspace(0,self.FUT_s,self.Ntrain)
			else:
			# random
			
				Xtrain = np.random.rand(self.Ntrain)*self.FUT_s

				inds1 = np.argsort(Xtrain)
				Xtrain = Xtrain[inds1]


		Xtrain = Xtrain[:,None]


		if not self.FUTURE and not self.SPLITS:
			Xtest = np.random.rand(self.Ntest,self.D)*(2+4*self.a)-(1+2*self.a)

		elif self.SPLITS:

			Xtest = np.random.rand(self.Ntest,self.D)*self.FUT_s

			pIn = self.ARGSsplit[0]
			pOut = self.ARGSsplit[1]
			NN = self.ARGSsplit[2]
			Nsec = self.ARGSsplit[3]
			starts = np.random.binomial(NN,pIn, Nsec)
			ends = np.random.binomial(NN, pOut, Nsec)
			inter = np.reshape(np.vstack([starts,ends]).T, Nsec*2)
			inter2 = np.hstack([0,inter])
			startEnds = np.cumsum(inter2[:-1]) / np.sum(inter2[:-1])
			SE = np.reshape(startEnds, (Nsec,2))

			## construct testIN
			INs = np.zeros(self.Ntest)
			for i in range(Nsec):
				inI = (Xtest[:,0]>SE[i,0]) & (Xtest[:,0]<SE[i,1])
				INs[inI] = 1
			INs = INs==1
			

			## construct shifted Xtrains
			startsNorm = starts/np.sum(inter2[:-1])
			endsNorm = ends[:-1]/np.sum(inter2[:-1])
			startsNormCum = np.cumsum(startsNorm)
			endsNormCum = np.cumsum(endsNorm)
			startsNormCum0 = np.hstack([0,startsNormCum])

			Xtrain *= startsNormCum0[-1]
			XtrainC = Xtrain.copy()
			for i in range(Nsec-1):
				indT = (Xtrain[:,0] > startsNormCum0[i+1]) & (Xtrain[:,0] <= startsNormCum0[i+2])
				indT = indT==1
				XtrainC[indT,0] += endsNormCum[i]

			Xtrain = XtrainC


		else:	# future

			Xtest1 = np.random.rand(self.Ntest-self.Ntest_IN,self.D)*(self.FUT_a-self.FUT_s)+self.FUT_s
			Xtest2 = np.random.rand(self.Ntest_IN,self.D)*self.FUT_s

			Xtest = np.concatenate((Xtest1, Xtest2))


		X = np.vstack((Xtrain,Xtest))*scal
		mu = np.zeros((self.Ntrain+self.Ntest)) # vector of the means

		if self.kernsAdd==None:
			C = self.kern.K(X,X) # covariance matrix
			Z = np.random.multivariate_normal(mu,C,1).transpose()
		else:
			Z = np.zeros((self.Ntrain+self.Ntest,1))

			for k, kernK in enumerate(self.kernsAdd):
				Ck = kernK.K(X,X) 
				Zk = np.random.multivariate_normal(mu,Ck,1).transpose()

				Z += Zk
				self.Zks.append(Zk)


		Ytrain = Z[0:self.Ntrain] + np.random.randn(self.Ntrain,1)*np.sqrt(self.sig2_noise)

		Ytest = Z[self.Ntrain:] + np.random.randn(self.Ntest,1)*np.sqrt(self.sig2_noise)

		# only D=1
		if not self.FUTURE and not self.SPLITS:
			Xtest_IN = (Xtest <= 1) & (Xtest >= a) | (Xtest >= -1) & (Xtest <= -a)
		elif self.SPLITS:
			Xtest_IN = INs[:,None]
			# already defined above
		else:
			Xtest_IN = Xtest < self.FUT_s

		return X[0:self.Ntrain,:], X[self.Ntrain:,:], Ytrain[:,0], Z[self.Ntrain:,0], Ytest[:,0], Xtest_IN[:,0], Z[:self.Ntrain,0]




	# D dimensional case
	def genDataD(self):

		# a    ### should decrease with higher D

		np.random.seed(self.seed)
		#D = kern.input_dim

		self.Zks = []

		
		Xtrain = np.random.rand(self.Ntrain, self.D)*2-1

		if self.D==2 and self.GRID:
			NtestSQRT = np.int(np.sqrt(self.Ntest) )
			if  NtestSQRT * NtestSQRT != self.Ntest :
				raise Exception('Ntest has to be SQUARE')

			self.NtestSQRT = NtestSQRT
			
			
			self.x = np.linspace(-(1+self.a), 1+self.a, NtestSQRT)
			self.y = np.linspace(-(1+self.a), 1+self.a, NtestSQRT)
			xv, yv = np.meshgrid(self.x, self.y)

			self.X1 = np.reshape(xv, NtestSQRT*NtestSQRT)
			self.X2 = np.reshape(yv, NtestSQRT*NtestSQRT)

			Xtest = np.vstack((self.X1,self.X2)).T
		else:
			Xtest = np.random.rand(self.Ntest, self.D)*(2+2*self.a)-(1+self.a)

		X = np.vstack((Xtrain,Xtest))
		mu = np.zeros((self.Ntrain+self.Ntest)) # vector of the means


		#C = self.kern.K(X,X) # covariance matrix
		#Z = np.random.multivariate_normal(mu,C,1).transpose()

		if self.kernsAdd==None:
			C = self.kern.K(X,X) # covariance matrix
			Z = np.random.multivariate_normal(mu,C,1).transpose()
		else:
			Z = np.zeros((self.Ntrain+self.Ntest,1))

			for k, kernK in enumerate(self.kernsAdd):
				Ck = kernK.K(X,X) 
				Zk = np.random.multivariate_normal(mu,Ck,1).transpose()

				Z += Zk
				self.Zks.append(Zk)



		Ytrain = Z[0:self.Ntrain] + np.random.randn(self.Ntrain,1)*np.sqrt(self.sig2_noise)

		Ytest = Z[self.Ntrain:] + np.random.randn(self.Ntest,1)*np.sqrt(self.sig2_noise)

		Xtest_OUT = np.zeros(self.Ntest, dtype=bool)
		for d in range(self.D):
			Xtest_OUT = Xtest_OUT | (Xtest[:,d] > 1) | (Xtest[:,d] < -1)

		Xtest_IN = ~Xtest_OUT


		return X[0:self.Ntrain,:], X[self.Ntrain:,:], Ytrain[:,0], Z[self.Ntrain:,0], Ytest[:,0], Xtest_IN[:], Z[:self.Ntrain,0]


	def plot_data(self, lev=10, SCAT=True, figsize=(30,4), P1=False):
		if self.X_train.shape[1]==1 or P1:
			self.plot_1D_data(figsize)
		elif self.X_train.shape[1]==2:
			self.plot_2D_data(lev=lev, SCAT=SCAT)
		else:
			print('not supported for D = ',self.X_train.shape[1])


	def plot_1D_data(self, figsize=(30,4)):
		plt.figure(figsize=figsize)

		if not self.f_test[0]==None:
			inds = np.argsort(self.X_test[:,0])
			plt.plot(self.X_test[inds,0],self.f_test[inds]);

			XXin = self.X_test[self.Xt_IN,0]
			inds = np.argsort(XXin)
			#plt.plot(XXin[inds],self.f_test[self.Xt_IN][inds], 'y-');
		else:
			plt.plot(self.X_test[:,0],self.y_test, 'b-');


		plt.plot(self.X_train[:,0],self.y_train[:],"r.")
		plt.show();


	def plot_1D_addKernel(self, figsize=(30,4), DATA=False, ax=None):

		if ax!=None:
			PLT = ax
		else:
			PLT = plt
			plt.figure(figsize=figsize)

		inds = np.argsort(self.X_test[:,0])
	

		for k in range(len(self.Zks)):
			fs = self.Zks[k][self.Ntrain:,0]
			PLT.plot(self.X_test[inds,0],fs[inds], label=self.kernsAdd[k].name);
		PLT.plot(self.X_test[inds,0],self.f_test[inds]);
		PLT.legend()
		
		if DATA:
			PLT.plot(self.X_train[:,0],self.y_train[:],"r.")

		#PLT.show();


	def plot_2D_addKernel(self, figsize=(4,4), lev=20):

		n_kern = len(self.kernsAdd)
		
		f, axs = plt.subplots(1,n_kern+1, sharex=True, sharey=True, figsize=((n_kern+1)*figsize[0],figsize[1]))

		for k in range(n_kern):
			fs = self.Zks[k][self.Ntrain:,0]
			

			axs[k].tricontour(self.X_test[:,0], self.X_test[:,1], fs, lev, cmap=self.cmap)


		axs[-1].tricontour(self.X_test[:,0], self.X_test[:,1], self.f_test, lev, cmap=self.cmap)
		



	def plot_2D_data(self, lev=20, SCAT=True, figsize=(5,5)):
		plt.figure(figsize=figsize)
		plt.plot(self.X_train[:,0], self.X_train[:,1], 'ro', label='train')
		plt.plot(self.X_test[:,0], self.X_test[:,1], 'b.', label='test_OUT');
		#if len(self.Xt_IN)>0:
		plt.plot(self.X_test[self.Xt_IN,0], self.X_test[self.Xt_IN,1], 'g.', label='test_IN');
		plt.legend()
		plt.show();


		plt.figure(figsize=figsize)
		plt.title('f_test')
		
		if self.GRID:
			plt.tricontour(self.X_test[:,0], self.X_test[:,1], self.f_test, lev, cmap=self.cmap)
			#plt.imshow(self.f_test,  cmap=self.cmap)
			#plt.plot_surface(self.X1, self.X2, np.reshape(self.f_test, self.NtestSQRT, self.NtestSQRT) )
		if SCAT:
			plt.scatter(self.X_test[:,0], self.X_test[:,1], c=self.f_test, marker='.', cmap=self.cmap);
		plt.show();

	def plot_2D_1D(self, dim = 0, name='', ylim=(-6,6)):

		if self.D != 2:
			raise Exception('only provided for D=2')

		nx = self.NtestSQRT

		fv = np.reshape( self.f_test, (nx, nx) )

		if dim==0:
			numS = len(self.x)
			for i,val in enumerate(self.x):
				plt.figure(figsize=(10,3))
				plt.ylim(ylim)
				plt.plot(self.y, fv[i,:])


				plt.savefig('2DATA/'+name+'_plotSlice_'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close()

		if dim==1:
			numS = len(self.y)
			for i,val in enumerate(self.y):
				plt.figure(figsize=(10,3))
				plt.ylim(ylim)
				plt.plot(self.x, fv[:,i])

				plt.savefig('2DATA/'+name+'_plotSlice_'+str(i)+'.png', bbox_inches = 'tight', pad_inches = 0)
				plt.close()


		fps = 5
		range01 = range(0,numS)

		imageio.mimsave('2DATA/slice_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plotSlice_'+str(k)+'.png') for k in range01], fps=fps)
		print('Image(''2DATA/slice_'+name+'.gif)')








def loadData(Datname):


	X_train, X_test, y_train, f_test, y_test, Xt_IN, name  = pickle.load(open(Datname, 'rb'))
	DD = Data(X_train.shape[0], X_test.shape[0], sig2_noise = 0, seed=0, a = 0)

	DD.X_train = X_train
	DD.X_test = X_test
	DD.y_train = y_train
	DD.f_test = f_test
	DD.y_test = y_test
	DD.Xt_IN = Xt_IN
	DD.name = name
	DD.D = X_train.shape[1]

	return DD

def plot_kernels_sample(kerns, Ntest=1000, num=2, seed=0, scal=1, GRID=True, SCAT=True, FIGnew=False, figsize=(30,4)):

	D = kerns[0].input_dim
	if D == 1:
		plot_kernels_sample_1d(kerns, Ntest=Ntest, num=num, seed=seed, scal=scal, FIGnew=FIGnew, figsize=figsize)
	elif D == 2:
		plot_kernels_sample_2d(kerns, Ntest=Ntest, seed=seed, scal=scal, GRID=GRID, SCAT=SCAT)
	else:
		raise Exception('only provided for D=1,2')


def kernel_sample_1d(kern, Ntest=1000, num=2, seed=0, scal=1):

	np.random.seed(seed)
	D = kern.input_dim

	
	X_test = ( np.random.rand(Ntest,D)*2.4-1.2 )*scal


	mu = np.zeros(Ntest) # vector of the means
	C = kern.K(X_test,X_test) # covariance matrix
	Z = np.random.multivariate_normal(mu,C,num)

	return Z.T, X_test
	

	

def plot_kernels_sample_1d(kerns, Ntest=1000, num=2, seed=0, scal=1, FIGnew=False, figsize=(30,4)):

	plt.figure(figsize=figsize)

	
	for k in kerns:

		if FIGnew:
			plt.figure(figsize=(30,4))

		ZT, X_test = kernel_sample_1d(k, Ntest, num, seed, scal)
		inds = np.argsort(X_test[:,0])
		for i in range(num):
			plt.plot(X_test[inds,0],ZT[inds,i], label=k.name);

	plt.legend()
	plt.show();


def kernel_sample_2d(kern, Ntest=1000, seed=0, scal=1, GRID=True):

	np.random.seed(seed)
	D = kern.input_dim

	NtestSQRT = np.int(np.sqrt(Ntest) )
	if  NtestSQRT * NtestSQRT != Ntest :
		raise Exception('Ntest has to be SQUARE')

	if GRID:
		a = 0.2	
		
		x = np.linspace(-(1+a)*scal, (1+a)*scal, NtestSQRT)
		y = np.linspace(-(1+a)*scal, (1+a)*scal, NtestSQRT)
		xv, yv = np.meshgrid(x, y)

		X1 = np.reshape(xv, NtestSQRT*NtestSQRT)
		X2 = np.reshape(yv, NtestSQRT*NtestSQRT)

		X_test = np.vstack((X1,X2)).T
	else:
		X_test = ( np.random.rand(Ntest,D)*2.4-1.2 )*scal


	mu = np.zeros(Ntest) # vector of the means
	C = kern.K(X_test,X_test) # covariance matrix
	Z = np.random.multivariate_normal(mu,C)

	return Z.T, X_test

def plot_kernels_sample_2d(kerns, Ntest=1000, seed=0, scal=1, GRID=True, SCAT=True):


	cmap = cm.get_cmap(name='viridis')
	
	for k in kerns:

		ZT, X_test = kernel_sample_2d(k, Ntest, seed, scal, GRID)

		plt.figure(figsize=(5,5))
		plt.title(k.name)

		if GRID:
			plt.tricontour(X_test[:,0], X_test[:,1], ZT, 20, cmap=cmap)
			
		if SCAT:
			plt.scatter(X_test[:,0], X_test[:,1], c=ZT, marker='.', cmap=cmap)

		#plt.legend()
		plt.show();


#####################################################
################### partition #######################
#####################################################


class Partition:
	def __init__(self, DD):

		self.DD = DD
		self.centers = None

	def compute_partition2(self, B_stop, col_start=0, Nrestart_partition=1, seed=0, randOrder=False):


		self.X_trains_k = []
		self.y_trains_k = []
		self.split(self.DD.X_train, self.DD.y_train, col_start, self.X_trains_k, self.y_trains_k, B_stop)
		self.K = len(self.y_trains_k)

		#print('self.K',self.K)

		self.compute_centers_and_distances()

		if self.K>1:
			if randOrder:
				np.random.seed(seed)
				indOrder = np.random.permutation(self.K)
			else:
				indOrder = self.compute_order(Nrestart_partition=Nrestart_partition, seed=seed)
			self.reorder(indOrder)


	def compute_partition2_GRBCM(self, B_stop, col_start=0, Nrestart_partition=1, seed=0):

		## usual KDTREE
		self.X_trains_k = []
		self.y_trains_k = []
		self.split(self.DD.X_train, self.DD.y_train, col_start, self.X_trains_k, self.y_trains_k, B_stop)
		self.K = len(self.y_trains_k)

		self.compute_centers_and_distances()

		sizeCluster0 = np.int(np.ceil(np.sum(self.length_of_clusters)/(self.K+1)))
		indParts = np.random.random_integers(self.K,size=sizeCluster0)-1

		self.X_trains_k = list(self.X_trains_k)
		self.y_trains_k = list(self.y_trains_k)
		self.X_trains_k.append( np.zeros( (sizeCluster0,self.DD.D) ) )
		self.y_trains_k.append( np.zeros( sizeCluster0 ) )
		for i,iP in enumerate(indParts):
			Nk =  self.X_trains_k[iP].shape[0]
			ind = np.random.random_integers(Nk)-1
			self.X_trains_k[-1][i,:] = self.X_trains_k[iP][ind,:]
			self.y_trains_k[-1][i] = self.y_trains_k[iP][ind]

			self.y_trains_k[iP] = np.delete( self.y_trains_k[iP], ind )
			self.X_trains_k[iP] = np.delete( self.X_trains_k[iP], ind, axis=0 )


		self.K += 1

		print('self.K',self.K)

		self.compute_centers_and_distances()
		
		indOrder = self.compute_order(Nrestart_partition=Nrestart_partition, seed=seed, ind0=self.K-1)
		self.reorder(indOrder)




	def reorder(self, indOrder):
		self.y_trains_k = itemgetter(*indOrder)(self.y_trains_k)
		self.X_trains_k = itemgetter(*indOrder)(self.X_trains_k)

		self.compute_centers_and_distances()



	def split(self, X, y, col, partition_listX, partition_listY, B_stop):
		B, D = X.shape

	
		if B>B_stop:
		
			median = B // 2

			args_col = np.argsort(X[:,col])    
			arg1 = args_col[:median]
			arg2 = args_col[median:]

			# half of the data
			X1 = X[arg1,:]
			X2 = X[arg2,:]

			y1 = y[arg1]
			y2 = y[arg2]

			# store it as batch if smaller than B_stop
			if X1.shape[0] <= B_stop:
				partition_listX.append(X1)
				partition_listY.append(y1)
			else: # split it recursively in the parts
				self.split(X1, y1, (col +1)%D, partition_listX, partition_listY, B_stop)

			if X2.shape[0] <= B_stop:
				partition_listX.append(X2)
				partition_listY.append(y2)
			else:
				self.split(X2, y2, (col +1)%D, partition_listX, partition_listY,  B_stop)

		else:
			partition_listX.append(X)
			partition_listY.append(y)

    

	# compute K partitions 
	def compute_partition(self, K, seed=0, KMEANS=False, randOrder=False, Nrestart_partition=1, sortDim=-1, PROJ='', bw=[None], clusterInds=[None]):
		# if kmeans false  then just compute list of mini-batches 
		# other wise apply k-mean clustering
		# if randOrder then just random cluster mean order
		# otherwise greedy minimial cluster order is computed
		# Nrestart_partition number of restarts for computing greedy ordering (very fast)

		# PROJ = '', 'PCA', 'wb', in the last case, specify wb (otherwise random), the first entry is bias

		B = np.int(len(self.DD.y_train)/K)

		np.random.seed(seed)

		self.X_trains_k = []
		self.y_trains_k = []
		self.K = K

		dist_tot = 0

		if KMEANS:
			kmeans = KMeans(n_clusters=K, random_state=seed).fit(self.DD.X_train)
			centers = kmeans.cluster_centers_

			if randOrder:
				inds_sorted = kmeans.labels_		# new inds of all data

			else:
				if self.DD.X_train.shape[1] == 1:
					indOrder = np.argsort( kmeans.cluster_centers_[:,0] )

				else:

					dist_min = 1e20

					

					for nr in range(Nrestart_partition):
						dist_tot = 0

						# make greedy shortes center 
						MM = squareform(pdist(kmeans.cluster_centers_))		# distance matrix of cluster means
						MM += np.eye(K)*1e10

						self.MM = MM.copy()

						#indIJ = np.unravel_index(np.argmin(MM, axis=None), MM.shape)

						ind0 = np.random.randint(K)

						indOrder_temp = np.zeros(K, dtype=int)		# order of cluster inds
						indOrder_temp[0] = ind0   #indIJ[0]

						#dist_tot += MM[ind0,indOrder_temp[1]]
						#MM[indIJ[0],:] = 1e10
						#MM[:,indOrder_temp[0]] = 1e10

						#indOrder_temp[1] = indIJ[1]

						for k in range(0,K-1):
							indOrder_temp[k+1] = np.argmin(MM[indOrder_temp[k] , :])
							dist_tot += MM[indOrder_temp[k],indOrder_temp[k+1]]
							MM[indOrder_temp[k],:] = 1e10
							MM[:,indOrder_temp[k]] = 1e10


						#print('min dist ',dist_min,' current dist', dist_tot)

						if dist_tot < dist_min:
							dist_min = dist_tot
							indOrder = indOrder_temp





				inds_sorted = np.zeros(self.DD.X_train.shape[0], dtype=int)		# new inds of all data
				for k in range(K):
					indsk = kmeans.labels_ == indOrder[k]
					inds_sorted[indsk] = k

				#print('total dist ',dist_tot)

			# store ordered centers and distances
			#self.indOrder = indOrder
			self.centers = centers[indOrder,:]
			self.MM = self.MM[indOrder,:][:,indOrder]
			
			# construct list of sorted clusters
			self.length_of_clusters = []
			for k in range(K):

				inds0 = (inds_sorted == k)

				self.length_of_clusters.append( sum(inds0) )

				self.X_trains_k.append( self.DD.X_train[inds0,:] )
				self.y_trains_k.append( self.DD.y_train[inds0]	)



		else:

			if sortDim > -1:
				dataOrder = np.argsort( self.DD.X_train[:,sortDim] )
				XX = self.DD.X_train[dataOrder, :]
				yy = self.DD.y_train[dataOrder]

			elif PROJ!='':
				if PROJ == 'PCA':
					XTX = np.dot(self.DD.X_train.T, self.DD.X_train)
					eig = np.linalg.eigh(XTX)
					indsDim = np.argsort(-eig[0])
					i = indsDim[0]
					#print(i)
					w = eig[1][:,i]
					bw = np.concatenate([[0], w])
					#print(bw.shape)
					#print(eig[0])


				elif PROJ == 'bw':
					if bw[0]==None:
						bw = np.random.random(self.DD.D+1)
		
				proj = np.dot( self.DD.X_train, bw[1:] ) + bw[0]
				wnorm = np.sqrt(np.dot(bw[1:], bw[1:]))
				distances = proj / wnorm

					#print(distances.shape)

				dataOrder = np.argsort( distances )
				self.dataOrder = dataOrder
				XX = self.DD.X_train[dataOrder, :]
				yy = self.DD.y_train[dataOrder]

			elif clusterInds[0]!=None:
				for k in range(K):

					inds_ks = (clusterInds==k)
					self.X_trains_k.append( self.DD.X_train[ inds_ks, :] )
					self.y_trains_k.append( self.DD.y_train[inds_ks] )

			else:
				XX = self.DD.X_train
				yy = self.DD.y_train


			


			if clusterInds[0]==None:
				# just construt list of mini-batches
				
				for k in range(K):

					

					if k==K-1:
						self.X_trains_k.append( XX[ (B*(k)):, :] )
						self.y_trains_k.append( yy[(B*(k)):] )
					else:
						self.X_trains_k.append( XX[ (B*(k)):(B*(k+1)), :] )
						self.y_trains_k.append( yy[(B*(k)):(B*(k+1))] )


			centers = None
			dist_tot = 0


		


		return centers, dist_tot


	def compute_centers_and_distances(self):
		# for KMEANS already done




		centersL = []
		self.length_of_clusters = []
		for k in range(self.K):
			centersL.append( np.mean( self.X_trains_k[k] , 0)[:,None].T )

			self.length_of_clusters.append( len(self.y_trains_k[k]) )


		self.centers = np.concatenate(centersL)

		self.MM = squareform(pdist(self.centers))	
		self.MM += np.eye(self.K)*1e10






	def compute_order(self, Nrestart_partition=1, seed=0, ind0=None):
		dist_min = 1e20


		for nr in range(Nrestart_partition):
			dist_tot = 0

			MM = self.MM.copy()

			np.random.seed(nr+seed)
			if ind0 == None:
				ind0 = np.random.randint(self.K)

			#print('--------------------------ind0', ind0)

			indOrder_temp = np.zeros(self.K, dtype=int)  # order of cluster inds
			indOrder_temp[0] = ind0  

			for k in range(0,self.K-1):
				indOrder_temp[k+1] = np.argmin(MM[indOrder_temp[k] , :])
				dist_tot += MM[indOrder_temp[k],indOrder_temp[k+1]]
				MM[indOrder_temp[k],:] = 1e10
				MM[:,indOrder_temp[k]] = 1e10


			#print('min dist ',dist_min,' current dist', dist_tot)

			if dist_tot < dist_min:
				dist_min = dist_tot
				indOrder = indOrder_temp

		return indOrder




	def compute_partitionGRBCM(self, K, seed=0, KMEANS=False, randOrder=True):
		# first set is always selected randomly
		# if kmeans false then just compute list of mini-batches (spezify B, should be int)
		# other wise apply (k-1)-mean clustering for remaining data
		# if randOrder then just random cluster mean order (makes no difference in GRBCM)
		# otherwise greedy minimial cluster order is computed

		B = np.int(len(self.DD.y_train)/K)

		np.random.seed(seed)

		self.X_trains_k = []
		self.y_trains_k = []
		self.K = K

		Ntrain = len(self.DD.y_train)

		# select randomly B points for the first expert
		indsGRBCM = np.random.permutation(Ntrain)
		indYes = indsGRBCM[:B]
		indNo = indsGRBCM[B:]


		if KMEANS:
			kmeans = KMeans(n_clusters=K-1, random_state=seed).fit(self.DD.X_train[indNo,:])

			aug_labels = np.zeros(Ntrain) # new inds of all data
			aug_labels[indYes] = 0
			aug_labels[indNo] = kmeans.labels_ + 1	

			if randOrder:
				inds_sorted = aug_labels # new inds of all data

			else:
				
				if self.DD.X_train.shape[1] == 1:
					indOrderM = np.argsort( kmeans.cluster_centers_[:,0] )		#k-1
					indOrder = np.concatenate([np.array([0]), indOrderM+1])

				else:

					centers = np.concatenate( [ (self.DD.X_train[indYes,:])[0,:][None,:], kmeans.cluster_centers_] ) # with dummy center of 1. expert 

					# make greedy shortes center 
					MM = squareform(pdist(centers))		# distance matrix of cluster means
					MM += np.eye(K)*1e10

					indOrder = np.zeros(K, dtype=int)		# order of cluster inds
					indOrder[0] = 0 	# always first


					for k in range(0,K-1):
						indOrder[k+1] = np.argmin(MM[indOrder[k] , :])
						MM[indOrder[k],:] = 1e10
						MM[:,indOrder[k]] = 1e10

				inds_sorted = np.zeros(self.DD.X_train.shape[0], dtype=int)		# new inds of all data
				for k in range(K):
					indsk = aug_labels == indOrder[k]
					inds_sorted[indsk] = k

			
			# construct list of sorted clusters
			for k in range(K):

				inds0 = (inds_sorted == k)

				self.X_trains_k.append( self.DD.X_train[inds0,:] )
				self.y_trains_k.append( self.DD.y_train[inds0]	)



		else:
			# just construt list of mini-batches
			# create the indeces for the other experts
			subset_indsR = self.compute_inds_for_GRBCM(indYes, Ntrain, B, K)
			
			for k in range(K):
				indsK = (subset_indsR == k)
				self.X_trains_k.append( self.DD.X_train[ indsK, :] )
				self.y_trains_k.append( self.DD.y_train[ indsK ] )



		return indYes


	def compute_inds_for_GRBCM(self, inds0, N, B, K):
		subset_indsR = np.zeros(N)
		subset_indsR[inds0] = 0

		i = 0
		for j in np.arange(1,K):
		    nj = 0
		    while nj < B:
		        if i not in inds0:
		            subset_indsR[i] = j
		            nj += 1
		            
		        i += 1

		return subset_indsR


	def plot_partitions(self, name='', GIF=False, lev=20, GRID=False, figsize=(30,4)):
		if self.DD.X_train.shape[1]==1:
			self.plot_partitions1D(name='', GIF=GIF, figsize=figsize)
		elif self.DD.X_train.shape[1]==2:
			self.plot_partitions2D(name='', GIF=GIF, lev=lev, GRID=GRID)
		else:
			print('not supported for D = ',self.DD.X_train.shape[1])



	def plot_partitions1D(self, name='', GIF=False, figsize=(30,4)):

		plt.figure(figsize=figsize)
		plt.ylim(-6,6)
		#plt.xlim(-1.1,1.1) #below


		inds = np.argsort(self.DD.X_test[:,0])
		plt.plot(self.DD.X_test[inds,0],self.DD.f_test[inds]);

		plt.xlim( self.DD.X_test[inds,0][0], self.DD.X_test[inds,0][-1])

		for k in range(self.K):
			plt.plot(self.X_trains_k[k][:,0], self.y_trains_k[k],'.');
			if GIF:
				plt.savefig('2DATA/'+name+'_plot_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)
			

		plt.show()

		if GIF:
			fps = 5
			range01 = range(0,self.K)

			imageio.mimsave('2DATA/run1_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plot_'+str(k)+'.png') for k in range01], fps=fps)
			print('Image(''2DATA/run1_'+name+'.gif)')


	def plot_partitions2D(self, name='', GIF=False, lev=20, GRID=True):

		plt.figure(figsize=(5,5))
		plt.ylim(-1.1,1.1)
		plt.xlim(-1.1,1.1)

		for k in range(self.K):
			plt.plot(self.X_trains_k[k][:,0], self.X_trains_k[k][:,1],'.');

			if self.DD.GRID and GRID:
				cmap = cm.get_cmap(name='Greys')
				plt.tricontour(self.DD.X_test[:,0], self.DD.X_test[:,1], self.DD.f_test, lev, cmap=cmap)

			if GIF:
				plt.savefig('2DATA/'+name+'_plot_'+str(k)+'.png', bbox_inches = 'tight', pad_inches = 0)

		if not GIF:
			plt.show()

		if GIF:
			fps = 5
			range01 = range(0,self.K)

			imageio.mimsave('2DATA/run1_'+name+'.gif', [imageio.imread('2DATA/'+name+'_plot_'+str(k)+'.png') for k in range01], fps=fps)

			print('Image(''2DATA/run1_'+name+'.gif)')


#####################################################
################### evaluation #######################
#####################################################

def KL1(m1,m2,v1,v2):
	return 0.5*(np.log(v2/v1) + (v1 + (m1-m2)**2)/v2 - 1)     

def stats(fTest, m, σy2, yTest ):
	quant = 1.96
	σy = np.sqrt(σy2)

	errF = np.sqrt(np.mean( (fTest-m)**2 ))
	#errFull = np.sqrt(np.mean( (mFull-m)**2 ))
	negLogP = np.mean( (fTest-m)**2 / σy2 + np.log(σy2) + np.log(np.pi*2))			### with ftest or ytest??
	cov = np.mean( (yTest <= m+quant*σy ) * (yTest >= m-quant*σy) )

		
	CRPS = np.mean( ps.crps_gaussian(yTest, mu=m, sig=σy ) )

	errAbs = np.mean( np.abs(fTest-m) )


	return errF, negLogP, cov, CRPS, errAbs

def stats2full(m, mFull, σy2, vFull):

	errFull = np.sqrt(np.mean( (mFull-m)**2 ))
	KL = np.mean( KL1(mFull, m, vFull, σy2) )

	return errFull, KL

def statsDF(m, σy2, yTest, fTest=[None], mFull=[None], vFull=[None],  lik=0, time=0, IN=None, name=' '):

	FACT = 1000

	if fTest[0]==None:
		fTest = yTest

	errF, negLogP, cov, CRPS, errAbs = stats(fTest, m, σy2, yTest)
	
	if not mFull[0]==None or not vFull[0]==None:
		errFull, KL = stats2full(m, mFull, σy2, vFull)
		df1 = pd.DataFrame({'time': [time], 'lik': [lik], 'KLx'+str(FACT): [KL*FACT], 'errFull': [errFull], 'CRPS': [CRPS], 'errF': [errF], 'errAbs': [errAbs],  'negLogP': [negLogP], 'cov':[cov]})
	else:
		df1 = pd.DataFrame({'time': [time], 'lik': [lik], 'CRPS': [CRPS], 'errF': [errF], 'errAbs': [errAbs],  'negLogP': [negLogP], 'cov':[cov]})

	

	
	df1.rename({0:name}, inplace=True)

	if not IN is None:
		OUT = ~IN
		errF_IN, negLogP_IN, cov_IN, CRPS_IN, errAbs_IN = stats(fTest[IN], m[IN], σy2[IN], yTest[IN])
		errF_OUT, negLogP_OUT, cov_OUT, CRPS_OUT, errAbs_OUT = stats(fTest[OUT], m[OUT], σy2[OUT], yTest[OUT])

		if not mFull[0]==None or not vFull[0]==None:
			errFull_IN, KL_IN = stats2full(m[IN], mFull[IN], σy2[IN], vFull[IN])
			errFull_OUT, KL_OUT = stats2full(m[OUT], mFull[OUT], σy2[OUT], vFull[OUT])
		

			df1_IN = pd.DataFrame({'KL_INx'+str(FACT): [KL_IN*FACT], 'errFull_IN': [errFull_IN], 'CRPS_IN': [CRPS_IN], 'errF_IN': [errF_IN], 'errAbs_IN': [errAbs_IN], 'negLogP_IN': [negLogP_IN], 'cov_IN':[cov_IN]})
			df1_OUT = pd.DataFrame({'KL_OUTx'+str(FACT): [KL_OUT*FACT], 'errFull_OUT': [errFull_OUT], 'CRPS_OUT': [CRPS_OUT], 'errF_OUT': [errF_OUT], 'errAbs_OUT': [errAbs_OUT], 'negLogP_OUT': [negLogP_OUT], 'cov_OUT':[cov_OUT]})
			
		else:

			df1_IN = pd.DataFrame({'CRPS_IN': [CRPS_IN], 'errF_IN': [errF_IN], 'errAbs_IN': [errAbs_IN], 'negLogP_IN': [negLogP_IN], 'cov_IN':[cov_IN]})
			df1_OUT = pd.DataFrame({'CRPS_OUT': [CRPS_OUT], 'errF_OUT': [errF_OUT], 'errAbs_OUT': [errAbs_OUT], 'negLogP_OUT': [negLogP_OUT], 'cov_OUT':[cov_OUT]})
		
		df1_IN.rename({0:name}, inplace=True)
		df1_OUT.rename({0:name}, inplace=True)

		df1 = pd.concat([df1,df1_IN, df1_OUT],1)

	return df1.round(3)





def compute_several_stats(OBJ, mFull=[None], vFull=[None]):

    for i in range(len(OBJ)):
        OBJ[i].compute_stats(mFull, vFull)
        if i==0:
            st = OBJ[i].stats
        else:
            st = pd.concat([st, OBJ[i].stats])
    return st




