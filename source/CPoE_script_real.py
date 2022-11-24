from matplotlib import pyplot as plt
import numpy as np
import GPy, pickle
import pandas as pd
import time

import sys
sys.path.insert(0, '../../source')

from incremental_run import Data, loadData, plot_kernels_sample_1d, plot_kernels_sample, compute_several_stats
from incremental_run import Partition, RunFullGP, RunSparseGP, RunPoE, RunGRBCM, RunINC, statsDF, ResultRun
from utils.incremental_p import Independent
from CPoE import BlockGP
import gc
from utils.SRGP.RECC import REC
from utils.SRGP.optim import Adam
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb




"""
Script for running experiments for real world datasets for CPoE, full GP, PoE, sparse GP and non-GP regression methods.
"""
class SCRIPT1:
	def __init__(self, loc_csv, Nreps=1, seed0=0, name='', Ntestmax=1000, NtestFIX=False, FULL=True, kernelMODE=1, path_results = ''):

		self.Nreps = Nreps
		self.loc_csv = loc_csv

		self.seeds = np.arange(Nreps)+seed0
		self.name = name

		self.NtestFIX = NtestFIX
		self.Ntestmax = Ntestmax

		self.FULL = FULL
		self.kernelMODE = kernelMODE
		self.path_results = path_results

		DD = self.makeData(self.loc_csv)
		print(loc_csv)
		print(name)
		print('D=',DD.D)
		print('Ntrain=',DD.Ntrain)
		print('Ntest=',DD.Ntest)

	"""
	Compute training/test data for given data set.
	"""
	def makeData(self, loc_csv, Ntestfrac=0.1, seed=0):

		data = pd.read_csv(loc_csv, index_col=0)
		data = np.array(data)

		Ntot = data.shape[0]

		if self.NtestFIX:
			Ntest = self.Ntestmax
		else:
			Ntest = np.minimum( self.Ntestmax, np.int(Ntot*Ntestfrac) )
		Ntrain = Ntot-Ntest

		np.random.seed(seed)

		DD = Data(Ntrain=Ntrain, Ntest=Ntest)
		permN = np.random.permutation(Ntot)
		indsTrain = permN[:Ntrain]
		indsTest = permN[Ntrain:]

		DD.X_train = data[indsTrain,:-1]
		DD.y_train = data[indsTrain,-1]
		DD.X_test = data[indsTest,:-1]
		DD.y_test = data[indsTest,-1]
		DD.f_test = data[indsTest,-1]

		DD.Xt_IN = None #np.zeros(Ntest)==False
		DD.D = DD.X_train.shape[1]

		return DD

	"""
	Compute 3 different kernels. SE and more flexibel kernel.
	"""
	def makeKernel(self, D):

		if self.kernelMODE==1:
			kern = GPy.kern.RBF(input_dim = D, variance = 1, lengthscale = np.ones(D), ARD = True)
		elif self.kernelMODE==2:
			kernRBF2 = GPy.kern.RBF(input_dim = D, variance = 1, lengthscale = np.ones(D)*10, ARD = True)
			kernCOS2 = GPy.kern.Cosine(input_dim = D, lengthscale=10)
			kernMLP = GPy.kern.MLP(input_dim = D)
			kernLIN = GPy.kern.Linear(input_dim = D)
			kern = kernRBF2 * kernCOS2 + kernMLP + kernLIN
		elif self.kernelMODE==3:
			kernRBF = GPy.kern.RBF(input_dim = D, variance = 1, lengthscale = np.ones(D), ARD = True)
			kernCOS = GPy.kern.Cosine(input_dim = D)
			kernRBF2 = GPy.kern.RBF(input_dim = D, variance = 1, lengthscale = np.ones(D)*10, ARD = True)
			kernCOS2 = GPy.kern.Cosine(input_dim = D, lengthscale=10)
			kernMLP = GPy.kern.MLP(input_dim = D)
			kernLIN = GPy.kern.Linear(input_dim = D)
			kern = kernRBF * kernCOS + kernRBF2 * kernCOS2 + kernMLP + kernLIN


		lik = GPy.likelihoods.Gaussian(variance=1)

		return kern, lik


	def runfullGP(self):

		RESs = []
		MVs = []

		for i in range(self.Nreps):
			print(i, 'rep')

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])
			kern, lik = self.makeKernel(DD.D)
			rFullGP = RunFullGP(DD, kern, likelihood=lik, seed=self.seeds[i]).run(True)

			RESs.append( rFullGP.compute_stats(rFullGP.m, rFullGP.v) )

			pickle.dump( RESs, open( self.path_results+self.name+'_fullGP', 'wb' ) )

			MVs.append( np.vstack([ rFullGP.m, rFullGP.v ]).T )
			pickle.dump( MVs, open( self.path_results+self.name+'_fullGP_MV', 'wb' ) )

			gc.collect()


		return RESs, self.name+'_fullGP'




	def runCPoE(self, K0, Ps, p, HYPERS='BATCH', TRACE=False, gamma=0.01, E=5, REL=1e-10):

		if not hasattr(Ps, '__len__'):
			Ps = [Ps]


		if self.FULL:
			MVs = pickle.load( open( self.path_results+self.name+'_fullGP_MV', 'rb' ) )

		RESs = [ [] for i in range(len(Ps)) ]
		for i in range(self.Nreps):
			print(i, 'rep')

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])
			kern, lik = self.makeKernel(DD.D)


			RES0 = CPoE(DD, kern, lik, K0, Ps=Ps, p=p, HYPERS=HYPERS, seed=self.seeds[i], TRACE=TRACE, gamma=gamma, E=E, B_increase=False,  REL=REL)

			for j in range(len(Ps)):
				if self.FULL:
					RESs[j].append( compute_several_stats([RES0[j]], MVs[i][:,0], MVs[i][:,1]) )
				else:
					RESs[j].append( compute_several_stats([RES0[j]]) )

				nam = '_K'+str(K0)+'_P'+str(Ps[j])+'_p'+str(p)
				if HYPERS=='STOCH':
					nam += 'stoch'

				namT = self.path_results+self.name+'_CPoE'+nam
				pickle.dump( RESs[j], open( namT, 'wb' ) )

				gc.collect()
			gc.collect()

		return RESs, namT







	def runPoE(self, K0, HYPERS='BATCH', TRACE=False, gamma=0.01, E=5, REL=1e-10):

		nam = '_K'+str(K0)

		if HYPERS=='STOCH':
			nam += 'stoch'
		# else:
		# 	nam = ''


		if self.FULL:
			MVs = pickle.load( open( self.path_results+self.name+'_fullGP_MV', 'rb' ) )
		namsePoE = np.array(['minVar','GPoE-scaled','BCM','RBCM'])

		RESs = []
		for i in range(self.Nreps):

			print(i, 'rep')

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])
			argsPartPoE = {'KDTREE':True, 'B_stop':np.ceil(DD.Ntrain/K0)}
			kern, lik = self.makeKernel(DD.D)

			runsPoE = PoE_temp(DD, kern, lik, K0,  HYPERS=HYPERS, seed=self.seeds[i], TRACE=TRACE, gamma=gamma, E=E, REL=REL, argsPartPoE=argsPartPoE, namsePoE=namsePoE)


			if self.FULL:
				RESs.append( compute_several_stats(runsPoE, MVs[i][:,0], MVs[i][:,1]) )
			else:
				RESs.append( compute_several_stats(runsPoE) )

			namT = self.path_results+self.name+'_PoE'+nam
			pickle.dump( RESs, open( namT, 'wb' ) )

			gc.collect()

		return RESs, namT




	def runSparseGP(self, MMs, OPT_R=False):

		if self.FULL:
			MVs = pickle.load( open( self.path_results+self.name+'_fullGP_MV', 'rb' ) )

		if OPT_R:
			nam = '_R'
		else:
			nam = ''


		RESs = []
		for i in range(self.Nreps):
			print(i, 'rep')

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])
			kern, lik = self.makeKernel(DD.D)

			runsSparse = [ RunSparseGP(DD, kern, m, likelihood=lik, seed=self.seeds[i]).run(OPT_TH=True, OPT_R=OPT_R) for m in MMs]

			if self.FULL:
				RESs.append( compute_several_stats(runsSparse, MVs[i][:,0], MVs[i][:,1]) )
			else:
				RESs.append( compute_several_stats(runsSparse) )

			namT = self.path_results+self.name+'_SGP'+nam
			pickle.dump( RESs, open( namT, 'wb' ) )

			gc.collect()

		return RESs, namT


	def runSparseGPfact(self, MMs, K, HYPERS='BATCH', TRACE=False, gamma=0.01, E=5, REL=1e-10, rec=False):

		if self.FULL:
			MVs = pickle.load( open( self.path_results+self.name+'_fullGP_MV', 'rb' ) )

		#nam = ''

		if HYPERS=='STOCH':
			nam = 'stoch'
		else:
			nam = ''


		RESs = []
		for i in range(self.Nreps):
			print(i, 'rep')

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])
			kern, lik = self.makeKernel(DD.D)


			runsSparse = [ runSparseIndep(DD, K, m, kern, lik, self.seeds[i], HYPERS=HYPERS, TRACE=TRACE, gamma=gamma, E=E, REL=REL, rec=rec) for m in MMs]


			#runsSparse = [ RunSparseGP(DD, kern, m, likelihood=lik, seed=self.seeds[i]).run(OPT_TH=True, OPT_R=OPT_R) for m in MMs]

			if self.FULL:
				RESs.append( compute_several_stats(runsSparse, MVs[i][:,0], MVs[i][:,1]) )
			else:
				RESs.append( compute_several_stats(runsSparse) )

			namT = self.path_results+self.name+'_SGPstoch'+nam
			pickle.dump( RESs, open( namT, 'wb' ) )

			gc.collect()

		return RESs, namT



	def runNON_GP(self, ALG='MLP', hidden_layer_sizes=(500,500), nam=''):

		RESs = []
		MVs = []

		for i in range(self.Nreps):
			print(i, 'rep')

			ts = time.time()

			DD = self.makeData(self.loc_csv, seed=self.seeds[i])

			if ALG=='MLP':
				reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=self.seeds[i], max_iter=500).fit(DD.X_train, DD.y_train)
			elif ALG=='LinReg':
				reg = LinearRegression().fit(DD.X_train, DD.y_train)
			elif ALG=='XGboost':
				reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=self.seeds[i], max_depth=3, n_estimators=100, learning_rate=0.1).fit(DD.X_train, DD.y_train)

			ypred = reg.predict(DD.X_test)

			ts2 = time.time() - ts
			errF = np.sqrt( np.mean( (ypred - DD.y_test)**2 ) )
			errAbs = np.mean( np.abs( ypred - DD.y_test ))

			res = pd.DataFrame(np.array([[ts2,0,0,0,0,errF,errAbs,0,0]]), columns=['time','lik','KLx1000','errFull','CRPS','errF','errAbs','negLogP','cov'])

			RESs.append( res )

			pickle.dump( RESs, open( self.path_results+self.name+'_'+ALG+nam, 'wb' ) )



			gc.collect()


		return RESs, self.path_results+self.name+'_'+ALG












def CPoE(DD, kern, lik, K0, Ps, p=1, HYPERS='FIX', seed=0, TRACE=False, gamma=0.01, E=5, jit=1e-3, B_increase=False, REL=1e-10):
	#HYPERS: FIX, BATCH, STOCH
	#Ps should be list, also in the scalar case
	#if B_increase, then obviously only 1 P in Ps!!


	ST = time.time()
	if not HYPERS=='FIX':

		if B_increase:
			B_STOP = np.int( np.ceil( DD.Ntrain* (Ps[0]+1) / K0  ) )
		else:
			B_STOP = int(np.ceil( DD.Ntrain / K0  ))

		BGP_IND = BlockGP(kern, DD, lik)
		BGP_IND.run0(K0, 0, KDTREE=True, B_stop=B_STOP, seed=seed)

		IND = Independent( DD, K0, kern, lik,  PP=BGP_IND.PART, SPARSE=False)

		if HYPERS=='BATCH':
			IND.opt_batch(GTOL=5e-1, maxF=300, TRACE=TRACE)
		elif HYPERS=='STOCH':
			IND.run_epochs(E=E, gamma=gamma, U=1, TRACE=TRACE, PERM=True, PRINT=True, REL=REL)

		kern = IND.kern
		lik = IND.likelihood

	timeOPT= time.time()-ST


	if p==1:
		J_MODE = 'MAX'
	else:
		J_MODE = 'FIX'

	BGP = BlockGP(kern, DD, lik)

	RESS = []
	for i,P in enumerate(Ps):

		BGP.run0(K0, P, sp=p, jit=jit, KDTREE=True, B_stop= B_STOP, seed=seed, TIMEOPT=timeOPT, J_MODE=J_MODE)

		RES = BGP.run_opt(OPT=False)

		RESS.append(RES)

	# if SCAL:
	# 	RESS = RESS[0]

	return RESS




def PoE_temp(DD, kern, lik, K0,  HYPERS='FIX', seed=0, TRACE=False, gamma=0.01, E=5, REL=1e-10, argsPartPoE=None, namsePoE=None):
	#HYPERS: FIX, BATCH, STOCH

	B_STOP = int(np.ceil( DD.Ntrain / K0  ))

	ST = time.time()
	if HYPERS=='STOCH':


		BGP_IND = BlockGP(kern, DD, lik)
		BGP_IND.run0(K0, 0, KDTREE=True, B_stop=B_STOP, seed=seed)

		IND = Independent( DD, K0, kern, lik,  PP=BGP_IND.PART, SPARSE=False)


		IND.run_epochs(E=E, gamma=gamma, U=1, TRACE=TRACE, PERM=True, PRINT=True, REL=REL)

		kern = IND.kern
		lik = IND.likelihood

	timeOPT= time.time()-ST

	rPoE = RunPoE(DD, K0, kern, lik,  seed=seed, **argsPartPoE, timeOpt=timeOPT)
	runGRBCM = RunGRBCM(DD, K0, kern, lik,  seed=seed, **argsPartPoE, timeOpt=timeOPT)

	if HYPERS=='BATCH':
		rPoE.optimize_run(GTOL=5e-1, maxF=300)
		runGRBCM.optimize_run(GTOL=5e-1, maxF=300)


	runsPoE = [rPoE.run(nam) for nam in namsePoE]
	rGRBCM = runGRBCM.run()
	runsPoE.append(rGRBCM)

	return runsPoE



def runSparseIndep(DD, K, M, kern, lik, seed=0, HYPERS='FIX', TRACE=False, gamma=0.01, E=5, REL=1e-10, rec=False):
    #HYPERS: FIX, BATCH, STOCH)

    np.random.seed(seed)
    ZZ = DD.X_train[ np.random.permutation(DD.Ntrain)[:M], :]

    PART = Partition(DD)
    PART.compute_partition(K, randOrder=True)

    ST = time.time()
    if not HYPERS=='FIX':


        IND = Independent( DD, K, kern, lik,  PP=PART, SPARSE=True, ZZ=ZZ, GLOBAL=True)

        if HYPERS=='BATCH':
            IND.opt_batch(GTOL=5e-1, maxF=300, TRACE=TRACE)
        elif HYPERS=='STOCH':
            IND.run_epochs(E=E, gamma=gamma, U=1, TRACE=TRACE, PERM=True, PRINT=True, REL=REL)

    if not rec:
        GPmod = GPy.models.SparseGPRegression(DD.X_train, DD.y_train[:,None], kernel = IND.kern, Z=ZZ)
        GPmod.Gaussian_noise = IND.likelihood.variance[0]
        GPmod.inducing_inputs.fix() #always fixed
        GPmod.inference_method = GPy.inference.latent_function_inference.VarDTC()

        mm, vv = GPmod.predict(DD.X_test)
        m = mm[:,0]
        v = vv[:,0]

        lml = GPmod.log_likelihood()[0][0]
        obj = GPmod

    else:
        θopt = {'σ0': Adam(0.01, (1,) ) , 'ls': Adam(0.01, (1,) ) , 'σn': Adam(0.01, (1,) ) , 'R': Adam(0.001, ZZ.shape )  }
        sizeMiniBatch = int(np.ceil( DD.Ntrain / K  ))
        θ = {'σ0': np.sqrt(IND.kern.variance[0]), 'ls': IND.kern.lengthscale.param_array, 'σn': np.sqrt(IND.likelihood.variance[0]), 'R': ZZ}
        params_EST = {'σ0': False, 'ls': False, 'σn': False, 'R': False}
        modSRGP = REC(DD.X_train, DD.y_train[:,None], IND.kern, 1, sizeMiniBatch, θ, θopt, α=0.001, params_EST=params_EST)

        modSRGP.run()

        m, v = modSRGP.predict_diag(DD.X_test)

        lml = modSRGP._log_marginal_likelihood[0]
        obj = modSRGP



    CI = np.zeros((len(m),2))
    sqv= np.sqrt(v)
    CI[:,0] = m - 1.96*sqv
    CI[:,1] = m + 1.96*sqv

    if HYPERS=='STOCH':
    	st = 'stoch'
    else:
    	st = ''
    return ResultRun(m, v, CI, 'sparseFact'+st, time=time.time()-ST, \
                    obj=obj, OBJ=PART, lik=lml)



def meanPD(DFs):
    return sum( DFs )/len(DFs)

def sdPD(DFs, M=None):
	if M==None:
	    M = meanPD(DFs)

	SD = np.sqrt(  sum([ (X  - M)**2 for X in DFs]) / len(DFs)  )
	SDM = SD/np.sqrt(len(DFs)) #sd of mean
    
	return M, SD, SDM

def sdmPD(DFs, M=None):
	if M==None:
	    M = meanPD(DFs)

	SD = np.sqrt(  sum([ (X  - M)**2 for X in DFs]) / len(DFs)  )
	SDM = SD/np.sqrt(len(DFs)) #sd of mean
	    
	return SDM