import numpy as np
import GPy, pickle, time
import pandas as pd
from incremental_run import Partition, ResultRun
from mix import diag_AtKB, inv_logDet, inv_logDet_jit, dot3lr, dot3rl, diag_HtKH 
from numpy.linalg import cholesky as cholesky_np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg import inv
from scipy.sparse import bsr_matrix, spdiags
from sksparse.cholmod import cholesky, analyze
from sksparse.cholmod import cholesky_AAt
from scipy.sparse import csc_matrix, csr_matrix, triu, tril, find, isspmatrix_csc, lil_matrix
import imageio
from matplotlib import pyplot as plt
from utils.incremental_p import PARAM
from utils.incremental_p import Independent
from incremental_run import Partition, constructData
from utils.numpy_lru_cache_decorator import np_cache
from scipy.optimize import minimize
from invTak import invTak_11, preTak



"""
CPoE implements the Correlated Product of Experts algorithm based on the paper 

Correlated Product of Experts
for Sparse Gaussian Process Regression
Manuel Schuerch,Dario Azzimonti1, Alessio Benavoli, Marco Zaffalon
(https://arxiv.org/pdf/2112.09519.pdf)

"""

def CPoE(X_train, y_train, kern, lik, J, C, p=1, HYPERS='FIX', X_test=None, y_test=None, f_test=None, priorN=None, seed=0, TRACE=False, gamma=0.01, E=5, jit=1e-3, B_increase=False, REL=1e-10, **kwargs):
	"""
	Computes the CPoE GP approximation.

	X_train: input training data, 2D numpy array, samples per columns
	y_train: output training data, 1D numpy array
	kern: kernel object from GPy
	lik: likelihood object from GPy
	J: number of experts, ideally power of 2
	C: degree of correlation between the experts, between 1 and J
	p: sparsity parameter, between 0 and 1,
	HYPERS: either 'FIX', 'BATCH', 'STOCH' for fixed hyperparameters, deterministic optimization (BATCH) or stochastic optimization (STOCH)
	X_test, y_test, f_test: (2D, 1D, 1D) numpy array for predictions for which prediction are done, if missing, the prediction for the training data is computed
	priorN: prior GPy object for the Gaussian noise
	gamma: learning rate for stochastic optimization
	E: maximal number of epochs for stochastic optimization
	REL: relative tolerance for stochastic optimization stopping criteria
	jit: jiiter to add on the diagonal for stability reason
	"""


	###################################
	## DATA
	###################################

	# if no test data is provided, the training data is set to it
	if (X_test is None) or (y_test is None):
		X_test = X_train
		y_test = y_train

	# the data is wrapped into a Data object
	DD = constructData(X_train, y_train, X_test, y_test, f_test)


	###################################
	## HYPERPARAMETER ESTIMATION
	###################################

	ST = time.time()
	if not HYPERS=='FIX':

		###################################
		## COMPUTE PARTITION WITH KDTREE
		###################################
		PP = Partition(DD)

		if B_increase:
			Kinc = np.int( np.ceil( J/C  ) )
		else:
			Kinc = J
		PP.compute_partition2(int(DD.Ntrain/Kinc), col_start=0, seed=seed, randOrder=False)
		IND = Independent( DD, Kinc, kern, lik,  PP=PP, SPARSE=False, priorNoise=priorN)


		###################################
		## HYPERPARAMETER ESTIMATION
		###################################

		# compute either batch or stochastic optimization
		if HYPERS=='BATCH':
			IND.opt_batch(GTOL=5e-1, maxF=300, TRACE=TRACE)
		elif HYPERS=='STOCH':
			IND.run_epochs(E=E, gamma=gamma, U=1, TRACE=TRACE, PERM=True, PRINT=True, REL=REL)

		# use the optimized kernel and likelihood
		kern = IND.kern
		lik = IND.likelihood

	timeOPT= time.time()-ST

	###################################
	## RUN CPOE for fixed hyperparameters
	###################################
	BGP = BlockGP(kern, DD, lik)
	BGP.run0(J, C-1, sp=p, jit=jit, KMEANS=False, KDTREE=True, B_stop=int(DD.Ntrain/J), seed=seed, TIMEOPT=timeOPT, J_MODE='FIX')
	RES = BGP.run_opt(OPT=False)


	return RES, BGP









"""
Main class for computing marginal posteriors of correlated blocks.
"""
class BlockGP:
	def __init__(self, kern, DD, lik):

		self.kern = kern.copy()
		self.lik = lik.copy()
		self.DD = DD


	def run0(self, K, P, sp=1, jit=1e-10, KMEANS=True, seed=0, randOrder=False, sortDim=-1, PROJ='', bw=[None], KDTREE=False, B_stop=100, \
				BAND=False, J_MODE='FIX', DUMMY=False, TIMEOPT=0, MIDDLE_CENTER=False, alpha=1):


		START = time.time()
		start = time.time()

		if not P<K:
			print('P has to be smaller than K')


		self.K = K
		self.P = P
		self.B = int(self.DD.Ntrain/self.K)
		self.J = int(np.ceil(sp*self.B))
		self.jit = jit
		self.alpha = alpha # alpha=1: FITC, alpha=1e-10: VFE
		self.J_MODE = J_MODE
		self.seed = seed
		self.DUMMY = DUMMY
		self.MIDDLE_CENTER = MIDDLE_CENTER


		# make partition
		timePart = self.makePartition(KMEANS=KMEANS, randOrder=randOrder, seed=seed, sortDim=sortDim, PROJ=PROJ, bw=bw, KDTREE=KDTREE, B_stop=B_stop)  ##time

		# compute predecessor structure
		self.compute_predecessors(BAND) ## always blockwise
		self.compute_predecessorPuk()
		self.init_params()


		self.time0 = time.time() - START + TIMEOPT


	# 
	def run1(self):


		timesPrec = self.computePrecisionBlock( )

		START = time.time()
		self.compute_factor_and_mpost()
		timeFac = time.time() - START

		START = time.time()
		self.compute_Sig_sparse3()
		timeSig = time.time() - START

		self.lml, timeLML = self.log_marg_lik2()



	def run1b(self):

		self.compute_K_sparse() 
		self.update_parms()

	def run2(self):

		START = time.time()

		tS = time.time()
		self.compute_Sigs22sp()

		tS = time.time()
		self.predict_Xt2b()

		self.time2 = time.time() - START
		self.timeA = self.time0 + self.time1 + self.time2


		start = time.time()
		mss1, vss1, self.weights = self.predict_aggregatePoE_f(self.PREDSm, self.PREDSv, pow=np.int( np.log(self.DD.Ntrain) )*(self.P +1) )
		RES = ResultRun(mss1, vss1, self.makeCI(mss1,vss1), 'CPoE('+str(self.P+1)+')', time=self.timeA+time.time()-start, obj=self, OBJ=self, lik=self.lml)


		return RES


	# run the hyperparameter optimization
	def run_opt(self, OPT=True, GTOL=1e-2, maxF=100):

		START = time.time()


		if OPT:
			method = 'L-BFGS-B'
			#method = 'BFGS'

			x0 = np.log( self.get_PARAMS() )
			self.res = minimize(f_BGP, x0, method=method, jac=df_BGP, args=self, options={'disp': True, 'gtol':GTOL, 'maxfun': maxF} )

		self.run1()


		self.time1 = time.time() - START


		return self.run2()



	def compute_predecessorPuk(self):
		self.predecessors_and_k = []
		self.predecessors_and_k_C = []
		for k in range(self.K):
			self.predecessors_and_k.append(np.array( np.hstack([self.predecessors[k], k]) ,dtype=int)     )
			self.predecessors_and_k_C.append(np.array( np.hstack([self.predecessors[k], k, np.arange(k+1,self.P+1)]) ,dtype=int)     )


		return self.predecessors_and_k


	def compute_predecessors(self, BAND=False, PRINT_PR=False):

		if not BAND:
			lowerOrdered =  np.tril( self.PART.MM, -1)
			upFill = np.triu(np.ones_like(lowerOrdered)*1e20,0)
			lowerPlusFill = lowerOrdered + upFill
			#np.sum( np.triu(lowerOrdered,-1) )  #total distance of current order

			argL = [np.array([], dtype=int)] # list of array with P closest center indeces, predecessorsP
			argL_sorted = [np.array([], dtype=int)] ####sorted!!!!
			#argM = [np.array([], dtype=int)]
			backNeig = [[]] #list of list with backwards neighbours backNeig[k] = list([bwn1,bwn2,...])
			for pk in np.arange(1,self.K):
				backNeig.append([])
				args, _ = self.takePmins(lowerPlusFill[pk,:(pk)], self.P)
				argL.append(args)
				argL_sorted.append(np.sort(args))
				#argM.append(mins)
				for ll in args:
					backNeig[ll].append(pk)

			self.predecessors = argL_sorted
			self.successors = backNeig


		else:

			#make band argL
			argC = [[]]
			backNeigC = []
			for k in np.arange(1,self.K):
				argC.append(np.arange(np.maximum(0,k-self.P),k))
				backNeigC.append(np.arange(k,np.minimum(self.K,k+self.P)))
			backNeigC.append([])

			self.predecessors = argC
			self.successors = backNeigC


		if False:
			self.print_predecessors()



	def takePmins(self, arr, P):

		args = np.argsort(arr)
		return args[:P], arr[args][:P]

	def print_predecessors(self):
		for k in range(self.K):
			a00 = np.repeat('0 ',self.K)
			a00[k] = '1 '
			a00[self.predecessors[k]] = '* '
			a0 = ''.join(a00)
			print(a0)



	def plot_predecessors(self, SUCCESSORS=True, Ak=False, Ck=False, Pos=(None, None), name=''):
		# Ak==True only after posterior computitona


		GIF = True
		#name = ''

		centers = self.PART.centers
		if Ak:
			Aks = self.Aks
			#if self.CK_MODE!='FULL':
			#	Aks = Aks[]
		else:
			Aks = self.PART.X_trains_k# Aks

		for kj in range(self.K):

			plt.figure(figsize=(5,5))
			plt.plot(centers[:,0], centers[:,1], 'r.-');
			plt.plot(centers[kj,0], centers[kj,1], 'ro', markersize=10, label='current');
			plt.plot(Aks[kj][:,0], Aks[kj][:,1], 'r.')
			if kj>0:
				argLabel1 = [ {'label':'P predecessors'}, {}]
				argLL = self.predecessors[kj]
				for il, ll in enumerate(argLL):
					plt.plot(centers[ll,0], centers[ll,1], 'bo', markersize=10,**argLabel1[np.minimum(il,1)]);
					if not Ck:
						plt.plot(Aks[ll][:,0],Aks[ll][:,1], 'b.')

				if Ck:
					A_B0 = np.vstack( [self.Aks[ kjj ] for kjj in  self.predecessors[kj] ])
					A_B2 = np.vstack( [A_B0[self.Cks[kj],:]] )
					plt.plot(A_B2[:,0],A_B2[:,1], 'b.')




			if SUCCESSORS:
				argLabel = [ {'label':'successors'}, {}]
				#backward neighbours
				for il, ll in enumerate(self.successors[kj]):

					plt.plot(centers[ll,0], centers[ll,1], 'go', markersize=10, **argLabel[np.minimum(il,1)]);

					plt.plot(Aks[ll][:,0], Aks[ll][:,1], 'g.')



			plt.ylim(-1,1)
			plt.xlim(-1,1);
			if Pos[0]!=None:
				plt.legend(loc=8, bbox_to_anchor=Pos)
			else:
				plt.legend(loc=8)

			if GIF:
				plt.savefig('GIF_PART/'+name+'_plot_'+str(kj)+'.png', bbox_inches = 'tight', pad_inches = 0)


		if GIF:
			fps = 2
			range01 = range(0,self.K)

			imageio.mimsave('GIF_PART/part_'+name+'.gif', [imageio.imread('GIF_PART/'+name+'_plot_'+str(k)+'.png') for k in range01], fps=fps)
			print('Image(''GIF_PART/part_'+name+'.gif)')




	def makeCI(self, m, v):


		CI = np.zeros((len(m),2))
		sqv= np.sqrt(v)
		CI[:,0] = m - 1.96*sqv
		CI[:,1] = m + 1.96*sqv

		return CI

	def compute_KL_at_inducing_f(self, GPmodfull):
		# at f values, not y!
		mAA, pAA = GPmodfull.predict(np.vstack(self.Aks))
		mA = mAA[:,0]
		pA = pAA[:,0] - self.lik.variance[0]  ######### f values!!

		self.kls_Aks = KL1(mA, self.m_post, pA, self.p_post)

		print('KLx1000 at Aks = ',np.mean(self.kls_Aks)*1e3)

		return mA, pA, np.mean(self.kls_Aks)*1e3


	def makePartition(self, KMEANS=True, randOrder=False, seed=0, sortDim=-1, PROJ='', bw=[None], KDTREE=False, B_stop=100, col_start=0):

		start = time.time()
		self.PART = Partition(self.DD)
		if KDTREE:
			self.PART.compute_partition2(B_stop, col_start=col_start, seed=seed, randOrder=randOrder)
		else:
			self.PART.compute_partition(self.K, KMEANS=KMEANS, randOrder=randOrder, seed=seed, sortDim=sortDim, PROJ=PROJ, bw=bw)
		self.Xks = self.PART.X_trains_k
		self.yks = self.PART.y_trains_k

		if KMEANS or KDTREE:
			#print('J ',self.J,' minL ',min(self.PART.length_of_clusters),' maxL ',max(self.PART.length_of_clusters))


			#print('J_MODE',self.J_MODE)

			if self.J_MODE=='MAX':
				self.J = max(self.PART.length_of_clusters)
				#print('new J is maxL = ',self.J)



			if self.J_MODE=='MIN':
				self.J = min(self.PART.length_of_clusters)
				#print('new J is minL = ',self.J)



		if KDTREE:
			self.K = self.PART.K
			#print('new K is ',self.K)
			self.B = int(self.DD.Ntrain/self.K)



		if not KMEANS and not KDTREE:
			self.PART.compute_centers_and_distances()


		timePart = time.time() - start

		return timePart



	def compMat_lik(self):

		ran = range(self.K)

		K_As = [ self.kern.K(self.Aks[k]) for k in ran]
		if self.P>0:
			iK_As = [invEE(K_As[k], self.jit) for k in ran]
		else:
			iK_As_logdets = [ inv_logDet_jit(K_As[k], self.jit) for k in ran]
			iK_As = [iK_As_logdets[k][0] for k in ran]
			logdets = [iK_As_logdets[k][1] for k in ran]

			self.logdetS = -np.sum(logdets)



		K_XAs = [ self.kern.K(self.Xks[k], self.Aks[k]) for k in ran]
		Hs = [ np.dot(K_XAs[k], iK_As[k]) for k in ran]
		#qs = [ self.kern.Kdiag(self.Xks[k]) - diag_HtKH(K_XAs[k].T, iK_As[k]) for k in ran]
		kxxs = [ self.kern.Kdiag(self.Xks[k])  for k in ran]
		ds = [  diag_HtKH(K_XAs[k].T, iK_As[k]) for k in ran]
		qs = [ kxxs[k] - ds[k] for k in ran]

		return Hs, qs, iK_As


	def compMat_trans(self):
		ran = np.arange(1,self.K)

		K_Bs = [ self.kern.K(self.A_Bs[k-1]) for k in ran]
		iK_Bs = [invEE(K_Bs[k-1], self.jit) for k in ran]

		K_ABs = [ self.kern.K( self.Aks[k], self.A_Bs[k-1]) for k in ran]

		Fs = [ np.dot(K_ABs[k-1], iK_Bs[k-1]) for k in ran]

		Ds = [ np.dot(Fs[k-1], K_ABs[k-1].T) for k in ran]
		Kxxs = [ self.kern.K(self.Aks[k]) for k in ran]
		Qs = [ Kxxs[k-1] -  Ds[k-1] for k in ran]
		# Qs = [ BGP.kern.K(self.Aks[k]) - np.dot(Fs[k-1], K_ABs[k-1].T) for k in ran]


		return Fs, Qs, iK_Bs


	def compFQF(self):


		iSig_order = np.zeros((self.K, self.K), dtype=int)
		for k in range(self.K):
			iSig_order[k,self.predecessors_and_k[k]] = 1
		iSig_order = np.dot(iSig_order.T, iSig_order)
		iSig_order[iSig_order!=0] = np.arange(1, np.sum(iSig_order!=0)+1 )

		csr_shape = csr_matrix(iSig_order) #for #csr_shape.indices, .indptr
		FQF_data = np.zeros((len(csr_shape.indices), self.J, self.J))


		for k in range(self.K):
			inds = self.predecessors_and_k[k][:,np.newaxis]
			fqf_inds = iSig_order[inds,inds.T].flatten() - 1
			FQF_data[fqf_inds] += bsr_matrix( self.FQFks[k], blocksize=(self.J, self.J) ).data

		FQF = bsr_matrix( (FQF_data, csr_shape.indices, csr_shape.indptr), (self.J*self.K,self.J*self.K) )
		return FQF

	def compFQF_HVH(self):


		iSig_order = np.zeros((self.K, self.K), dtype=int)
		for k in range(self.K):
			iSig_order[k,self.predecessors_and_k[k]] = 1
		iSig_order = np.dot(iSig_order.T, iSig_order)
		iSig_order[iSig_order!=0] = np.arange(1, np.sum(iSig_order!=0)+1 )

		csr_shape = csr_matrix(iSig_order) #for #csr_shape.indices, .indptr
		FQF_data = np.zeros((len(csr_shape.indices), self.J, self.J))
		HVH_data = np.zeros((len(csr_shape.indices), self.J, self.J))


		for k in range(self.K):
			inds = self.predecessors_and_k[k][:,np.newaxis]
			fqf_inds = iSig_order[inds,inds.T].flatten() - 1
			inds2 = self.predecessors_and_k_C[k][:,np.newaxis]
			hvh_inds = iSig_order[inds2,inds2.T].flatten() - 1
			FQF_data[fqf_inds] += bsr_matrix( self.FQFks[k], blocksize=(self.J, self.J) ).data
			HVH_data[hvh_inds] += bsr_matrix( self.HVHks[k], blocksize=(self.J, self.J) ).data

		FQF = bsr_matrix( (FQF_data, csr_shape.indices, csr_shape.indptr), (self.J*self.K,self.J*self.K) )
		HVH = bsr_matrix( (HVH_data, csr_shape.indices, csr_shape.indptr), (self.J*self.K,self.J*self.K) )
		return FQF, HVH


	def compute_H(self):
		# since it is not used, it is not efficient, ie dense

		self.H = np.zeros((self.DD.Ntrain, self.K*self.J))


		for k in range(self.K):
			ind_psi = self.index_block( self.predecessors_and_k_C[k], self.J )
			ind_k = self.index_block( np.array([k]), self.B )

			self.H[ind_k[:,None], ind_psi[None,:]] = self.Hs[k]

	def compute_F(self):
		# since it is not used, it is not efficient, ie dense

		self.F = np.eye(self.K*self.J)

		if self.P !=0:

			for k in range(1,self.K):
				ind_pi = self.index_block( self.predecessors[k], self.J )
				ind_k = self.index_block( np.array([k]), self.J )

				self.F[ind_k[:,None], ind_pi[None,:]] = self.Fs[k-1]






	def compPredSets(self, k):
		## could be done much cleaner jsut with blocks

		a_b = []
		a_b_inds = []
		a_b_inds_local = []
		for i, kjj in enumerate( self.predecessors[k]):

			indsLJ = np.array(  np.linspace(0,self.J-1,self.J, endpoint=True ) , dtype=int) ##could also vary!!!


			a_b.append( self.Aks[ kjj ][indsLJ,:] )
			a_b_inds.append( kjj*self.J + indsLJ )
			a_b_inds_local.append( i*self.J + indsLJ )

		A_B = np.vstack( a_b )
		self.Cks.append(np.hstack(a_b_inds_local))
		A_B_cinds = np.hstack(a_b_inds)



		self.colInds.append(A_B_cinds)
		self.colInds_j.append( np.tile(A_B_cinds, self.J) ) 	##repeated for each row

		self.A_Bs.append(A_B)#############################################################3storage???



	# compute all block indices for inds with block size M
	def index_block(self, ind, M):


		return np.repeat( ind*M, M )   +  np.tile( np.arange(M), len(ind) )




	# with block sparse, cleaner
	# nor direct callable, use run
	def computePrecisionBlock(self):




		startTot = time.time()


		self.Aks = []
		self.colInds = [[]] ##individual column indeces, not repeated
		self.Cks = [[]]  #sparse local subsets inds of neighbourhood,
		#self.iKbig = []
		self.iKsmall = []
		self.iKmedium = [[]]

		HVHs = []
		rks = []

		self.Vs = []
		self.Hs = []

		self.Fs = [[]]

		Fks2 = []
		#self.iQks2 = []

		self.A_Bs = []

		#colInds2 = []   #### block columns
		self.colInds_j = []  #### individual columnd indeces, repeated


		timeFQlik = 0
		timeFQtrans = 0
		timeInvQ = 0
		timeFQF = 0



		self.Aks = [ self.Xks[k][np.array(  np.linspace(0,len(self.yks[k])-1,self.J, endpoint=True ) , dtype=int),:]  for k in range(self.K)]


		if self.P==0:
			#if not self.HVH_CORRECT:
				#self.iQks2 = iK_As
				#self.iKbig = iK_As
			ran = range(self.K)
			K_As = [ self.kern.K(self.Aks[k]) for k in ran]
			iK_As_logdets = [ inv_logDet_jit(K_As[k], self.jit) for k in ran]

			logdets = [iK_As_logdets[k][1] for k in ran]
			self.logdetS = -np.sum(logdets)

			self.iKbig = [iK_As_logdets[k][0] for k in ran]
			self.iQks2  = self.iKbig



		if self.P>0:

			start = time.time()
			# k=0
			iKrr0, Krr0 = comp_kernR(self.kern, self.Aks[0], jit=self.jit)

			[ self.compPredSets(k) for k in range(1,self.K) ]

			self.Fs, Qs, self.iKmedium = self.compMat_trans()
			timeFQtrans += time.time() -start


			self.Qks = [ Krr0 ] +  Qs




			start = time.time()

			#iQks = [ invEE(Qs[k-1], self.jit) for k in np.arange(1, self.K)]
			iQks_logdets = [ inv_logDet_jit(Qs[k-1], self.jit) for k in np.arange(1, self.K)]

			#print('LEN',len(iQks_logdets))
			#print('SHAPE',iQks_logdets[0][1])

			iQks = [iQks_logdets[k][0] for k in np.arange(0, self.K-1)]
			logdets = [iQks_logdets[k][1] for k in np.arange(0, self.K-1)]



			self.logdetS = -np.sum(logdets) + 2*np.sum(np.log(np.diag(cholesky_np(iKrr0))))
			#self.iQks2 = self.iQks2 + iQks
			self.iQks2 = [ iKrr0 ] + iQks


			timeInvQ += time.time() -start



			iQFs = [np.dot(self.iQks2[k], self.Fs[k-1])  for k in range(1,self.K) ]
			AAs = [self.iKmedium[k-1] + np.dot(self.Fs[k-1].T, iQFs[k-1])   for k in range(1,self.K) ] # why not directly iKmedium?

			#self.iKbig = self.iKbig + [ np.block([[AAs[k-1],-iQFs[k-1].T],[-iQFs[k-1],self.iQks2[k]]])  for k in range(1,self.K) ]
			self.iKbig = [ iKrr0 ] + [ np.block([[AAs[k-1],-iQFs[k-1].T],[-iQFs[k-1],self.iQks2[k]]])  for k in range(1,self.K) ]




		# latent likelihood
		ran = range(self.K)
		iKAApsi =   [ self.iKbig[self.P] for k in np.arange(self.P) ]  \
				  + [ self.iKbig[k] for k in np.arange(self.P,self.K) ]

		if self.P==0:
			Apsi = self.Aks
		else:
			Apsi =   [ np.vstack([ self.A_Bs[self.P-1], self.Aks[self.P] ]) for k in np.arange(self.P) ] \
				+ [ np.vstack([ self.A_Bs[k-1], self.Aks[k] ]) for k in np.arange(self.P,self.K) ]
		# reduce additional storage

		#self.Apsi2 = Apsi


		K_XAs = [ self.kern.K(self.Xks[k], Apsi[k]) for k in ran]
		self.Hs = [ np.dot(K_XAs[k], iKAApsi[k]) for k in ran]
		#qs = [ self.kern.Kdiag(self.Xks[k]) - diag_HtKH(K_XAs[k].T, iK_As[k]) for k in ran]
		kxxs = [ self.kern.Kdiag(self.Xks[k])  for k in ran]
		ds = [  diag_HtKH(K_XAs[k].T, iKAApsi[k]) for k in ran]
		self.Vs = [ self.alpha*(kxxs[k] - ds[k]) + self.lik.variance[0] for k in ran]


		gks = [ self.Hs[k].T/self.Vs[k]  for k in ran]
		self.HVHks = [ np.dot( gks[k], self.Hs[k]) for k in ran]
		rks = [ np.dot( gks[k], self.yks[k]) for k in ran]

		#self.b2 = np.hstack(rks) # not correct




		self.b = np.zeros(self.J*self.K)
		for k in ran:
			#print(rks[k].shape)
			#print(index_block ( self.predecessors_and_k_C[k], self.J).shape )
			self.b[self.index_block( self.predecessors_and_k_C[k], self.J)] += rks[k]




		start1 = time.time()

		sizeM = self.J*self.K

		if self.P>0:

			ttt = time.time()
			FsI = [np.eye(self.J)] + [np.hstack( [ -self.Fs[k-1], np.eye(self.J) ]) for k in np.arange(1,self.K)]
			#print('++++stack',time.time() - ttt)


			ttt = time.time()
			self.FQFks = [dot3lr(FsI[k].T, self.iQks2[k], FsI[k]) for k in np.arange(0,self.K)]
			#print('++++FQFks',time.time() - ttt)
			#print('self.FQFks')


			ttt = time.time()
			#self.FQF = self.compFQF(
			self.FQF, self.HVH = self.compFQF_HVH()
			#print('++++FQF_new',time.time() - ttt)

			tf = time.time()
			self.FQF = bsr_matrix( self.FQF, blocksize=(self.J, self.J))  ####why??
			#print('++++BSR',time.time() - tf)
		else:
			#self.FQF = csc_matrix( bsr_matrix((np.array( self.iQks2 ),range(self.K),range(self.K+1)), shape=(sizeM, sizeM))  )
			self.FQF =  bsr_matrix((np.array( self.iQks2 ),range(self.K),range(self.K+1)), shape=(sizeM, sizeM))
			self.FQFks = self.iQks2

			self.HVH = bsr_matrix((np.array( self.HVHks ),range(self.K),range(self.K+1)), shape=(sizeM, sizeM))


		timeFQF += time.time() -start1
		self.iSig =  self.FQF + self.HVH

		timeTot = time.time() - startTot


		return  timeTot


	def dKs_(self, PAR, which_dK, which_pos=0):
	# auxilary function to get the rigth dKs
	# assume eiter scalar or vector (no matrix!)

		dK = PAR.dKs[which_dK]

		if PAR.length()==1:
			return dK
		else:
			if not which_dK=='dk_xx':
				return dK[:,:,which_pos]
			else:
				return dK[:,which_pos]

	def compute_derivatives(self, PAR):
	# assuming only scalar parameters scalar


		NOISE = (PAR.name == 'noise')



		dHVHs = [[] for wp in range(PAR.length())]
		dQs = [[] for wp in range(PAR.length())]
		dFs = [[] for wp in range(PAR.length())] ## already in column format
		dbs = [[] for wp in range(PAR.length())]

		dyVy = np.zeros(PAR.length())



		for k in range(self.K):


			if k>0 and self.P>0:
				# compute conditioning set
				A_B0 = np.vstack( [self.Aks[ kjj ] for kjj in  self.predecessors[k] ]) #independent of theta
				BB = np.vstack( [A_B0[self.Cks[k],:]] ) #independent of theta
			else:
				BB = None

			# update the kernel derivatives
			PAR.update_grad(self.Aks[k], BB, self.Xks[k])



			for wp in range(PAR.length()):


				dK_AA = self.dKs_(PAR, 'dK_AA', wp)
				if k>0 and self.P>0:
					dK_BB = self.dKs_(PAR, 'dK_BB', wp)
					dK_AB = self.dKs_(PAR, 'dK_AB', wp)
				dK_XA = self.dKs_(PAR, 'dK_XA', wp)
				dk_xx = self.dKs_(PAR, 'dk_xx', wp)


				# likelihood update
				H_dKaa = np.dot( self.Hs[k], dK_AA )
				dH = np.dot( dK_XA - H_dKaa, self.iKsmall[k] )

				#dv = dk_xx - np.sum( ( 2*dK_XA.T - H_dKaa  )  * self.Hs[k] , 0)

				dv = dk_xx - 2*np.sum(self.Hs[k].T * dK_XA.T,0) + np.sum(H_dKaa.T * self.Hs[k].T, 0)
				if NOISE:
					dv = np.ones(len(dv))



				gk = self.Hs[k].T/self.Vs[k] #indep theta
				HtiVdH = np.dot(gk, dH)
				HtiVdViVH = np.dot(gk*dv, gk.T)

				dHVH = HtiVdH.T - HtiVdViVH + HtiVdH

				dHVHs[wp].append(dHVH)

				yiv = self.yks[k]/self.Vs[k]
				dbs[wp].append( np.dot( dH.T - gk*dv, yiv)  )


				dyVy[wp] += 0.5* np.sum( yiv**2 * dv )    -  0.5*np.sum(dv/self.Vs[k])

				if not NOISE:
					if k>0 and self.P>0:
						# transition
						F_dKbb = np.dot( self.Fs[k], dK_BB )
						dF = np.dot( dK_AB - F_dKbb , self.iKmedium[k] )
						dKab_Ft = np.dot( dK_AB, self.Fs[k].T )
						dQ = dK_AA - dKab_Ft - dKab_Ft.T + np.dot(F_dKbb,self.Fs[k].T)

						dQs[wp].append(dQ)
						for pj in range(dF.shape[0]):
							dFs[wp].append( -dF[pj,:] )
					else:
						#dQs.append( -dot3lr(self.iKsmall[k], PAR.dKs['dK_AA'], self.iKsmall[k]) )
						dQs[wp].append(  dK_AA)

		PAR.dliks = np.zeros(PAR.length())
		for wp in range(PAR.length()):

			## outside loop
			JK = self.J * self.K

			db = np.hstack(dbs[wp])



			if NOISE:
				dFQF = csc_matrix((JK, JK))
			else:
				dQ = bsr_matrix((np.array( dQs[wp] ),range(self.K),range(self.K+1)), shape=(JK, JK))
				if self.P>0:
					dF = csr_matrix(( np.hstack(dFs[wp]), self.colInds2, self.indptr2), shape=(JK, JK))

					dF_QF = np.dot(dF.T, self.QF)
					dFQF = dF_QF + dF_QF.T - dot3lr(self.QF.T, dQ, self.QF)


				elif self.P==0:
					dFQF = - dot3lr(self.iQ, dQ, self.iQ)




			dHVH = bsr_matrix((np.array( dHVHs[wp] ),range(self.K),range(self.K+1)), shape=(JK, JK))
			diSig =  dFQF + dHVH

			bmiSig = np.dot(db - 0.5*np.dot(diSig, csc_matrix(self.m_post).T).toarray()[:,0] , self.m_post)


			#sumSigdiSig = -0.5*np.sum( diSig.multiply(self.factor.inv()))
			#print('full SIG INVERSE!!!!!!!')
			sumSigdiSig = -0.5*np.sum( self.Sig.multiply(diSig))

			self.diSig = diSig
			self.dFQF = dFQF

			sumKS = 0.5*np.sum(dFQF.multiply(self.K_sp )) ### also with K???


			PAR.dliks[wp] = dyVy[wp]  + bmiSig + sumSigdiSig + sumKS




		########################################################### functions for block implementation
	def compute_factor_and_mpost(self):

		if not isspmatrix_csc( self.iSig):
			tf = time.time()
			iSig_csc = csc_matrix(self.iSig)   # explicit conversion to csc (needed for cholesky)
			#print('conv',time.time()-tf)
		else:
			iSig_csc = self.iSig

		shape = csr_matrix( (np.arange(1,len(self.iSig.indices)+1), self.iSig.indices, self.iSig.indptr), shape=(self.K, self.K) )  #+1 beacuase otherwise 0 is sparse
		fac0 = analyze(csc_matrix(shape))

		self.ppi0 = fac0.P()
		self.ipp0 = np.argsort(self.ppi0)
		ppi = fac0.P()[:,np.newaxis]

		IJK = np.reshape( np.arange(self.iSig.shape[0]), (self.K, self.J) )
		self.ppi0_long = IJK[ self.ppi0].flatten()
		self.ipp0_long = np.argsort(self.ppi0_long)



		shape_perm = shape[ppi,ppi.T] #efficient? otherwise make full and afterwards csr again

		self.factor = cholesky(csc_matrix(bsr_matrix( (self.iSig.data[shape_perm.data-1], shape_perm.indices, shape_perm.indptr) )), ordering_method='natural' )

		if self.DUMMY:
			shape = csr_matrix( (np.random.rand(len(self.iSig.indices)), self.iSig.indices, self.iSig.indptr), shape=(self.K, self.K) )
			shape = shape + shape.T
			self.facDummy = cholesky(csc_matrix(csr_matrix( (shape.data[shape_perm.data-1], shape_perm.indices, shape_perm.indptr) )), ordering_method='natural' )


		self.m_post = self.factor(self.b[self.ppi0_long])[self.ipp0_long] # compute posterior mean

		return self.factor, self.m_post


	def compute_condSet(self):
		self.condSets = [[]]
		ti0 = time.time()
		for k in range(1,self.K):
			kj = k*self.J
			allInds = find(self.FQF[kj,:])[1]
			self.condSets.append( allInds[allInds < kj] )


	def compute_Sig_sparse3(self):

		timeTot = time.time()


		stT = time.time()
		Lb, self.LOOKUP, indices, indptr = preTak(self.factor.L(), self.K, self.J)
		tiPre = time.time() - stT


		stT = time.time()
		self.Xdata_block = invTak_11(Lb.data, self.LOOKUP, indices, indptr,  self.K, self.J)
		tiInv = time.time() - stT
		#print('xdata')



		stT = time.time()
		self.iP = self.ipp0_long

		self.Sig_bsr_half = bsr_matrix( (self.Xdata_block, Lb.indices, Lb.indptr) )
		#self.Sig_unordered = ( tril(XX,-1).T + tril(XX,0) ).toarray()
		self.Sig_unordered =  tril(self.Sig_bsr_half,-1).T + tril(self.Sig_bsr_half,0)

		self.p_post = self.Sig_unordered.diagonal()[self.iP]

		tiPost = time.time() - stT



		timeTot = time.time() - timeTot





	def closestInds_xt(self, xt, centers, NN):
		#dist = (self.PART.centers - xt)**2
		dist = (centers - xt)**2

		dist_xt = np.sqrt( np.sum(dist, axis=1) )

		return np.sort( np.argsort(dist_xt)[:NN] )




	def make_shape(self, M_sp):
		#return csc_matrix(( np.ones(M_sp.nnz, dtype=int), M_sp.nonzero() ), shape=M_sp.shape)
		return csc_matrix(( np.ones(len(M_sp.nonzero()[0]), dtype=int), M_sp.nonzero() ), shape=M_sp.shape)

	def compute_shape(self, M_sp, PL=True):


		SH_sp = self.make_shape(M_sp)
		if PL:
			print('#nonzeros ',M_sp.nnz, 'filled ',M_sp.nnz/np.prod(M_sp.shape))
			plt.imshow(SH_sp.toarray())

		return SH_sp.toarray()



	# with Sig with sparse and unordered!
	def compute_Sigs22sp(self):
		#self.Sigs = []
		self.mus = []


		self.ipp_m = self.ipp0[:,np.newaxis]
		self.LOOKUPtranspose_perm = np.triu( np.ones_like(self.LOOKUP) )[self.ipp_m,self.ipp_m.T]
		self.LOOKUP_perm = self.LOOKUP[self.ipp_m,self.ipp_m.T]

		sT = time.time()
		self.Sigs = [self.accessXdata_MULTI(self.predecessors_and_k[k], self.predecessors_and_k[k]) for k in range(self.K)]
		#print('.Sigs',time.time()-sT)

		sT = time.time()
		for k in range(self.K):

			#print(k)

			indsK = (k)* int(self.J) + np.arange(self.J)
			if self.P>0 and k!=0:
				iis_non = np.hstack([self.colInds[k], indsK])
			else:
				iis_non = indsK

			self.mus.append( self.m_post[iis_non]  )
		#print('.Mus',time.time()-sT)


	def accessXdata(self,i,j): ##it has to be there!! (we could check if LOOKUP is 0, but first entry is also 0)
		X = self.Xdata_block[self.LOOKUP[self.ipp_m,self.ipp_m.T][i,j]]

		if self.LOOKUPtranspose_perm[i,j]==1:
			X = X.T
		return X


	def accessXdata_MULTI(self,iis,jjs):
		## only working if we access values which are there!!!!

		#i0,j0 blockindex in new/small array XX
		#i,j blockindex in big SIg

		XX = np.zeros((len(iis)*self.J,len(jjs)*self.J))
		for i0,i in enumerate(iis):
			for j0,j in enumerate(jjs):

				XX[i0*self.J:(i0+1)*self.J,j0*self.J:(j0+1)*self.J] = self.accessXdata(i,j)

		return XX








	def compute_K_sparse(self, COND_SET=False):

		timeTot = time.time()

		JK = self.J*self.K
		iis = np.arange(JK)
		lli = len(iis)
		ei_sp = csc_matrix((np.ones(lli), (iis,range(lli) )), shape=(JK,lli)) #sparse Ei matrix
		ei_p_sp = self.facS.apply_P(ei_sp) # apply permutation since L!

		timeY = time.time()
		Y = self.facS.solve_L(ei_p_sp, use_LDLt_decomposition=False) #it is sparse!!!!, it is L⁻1
		timeY = time.time() -timeY

		self.Y = Y

		timeDat = timegh = 0

		if COND_SET:
			self.compute_condSet()  ###compute condSet!!!!!!!!!

		self.K_sp = csc_matrix((JK,JK)) # K_sp is only uper triangular with diagonal blocks ranging over the main diag
		for k in range(self.K):


			indsK = k*self.J + np.arange(self.J)
			if self.P>0:
				#iis_non = np.hstack([self.colInds[k], indsK])
				iis_non = np.hstack([self.condSets[k], indsK]) ######new, condSet!!
			else:
				iis_non = indsK

			ll_iis = len(iis_non)

			timeD = time.time()
			data_S =  np.dot(Y[:,indsK].T, Y[:,iis_non] )  #it is transposed
			data_flat = data_S.toarray().flatten()
			timeDat += time.time() -timeD




			timeg = time.time()

			indPtr0 = np.zeros(JK+1,dtype=int)
			indPtr0[indsK+1] = ll_iis
			indPtrC = np.cumsum(indPtr0)
			indRow = np.tile(iis_non, self.J)

			ghij = csc_matrix( (data_flat,indRow, indPtrC), shape=(JK,JK))
			self.K_sp += ghij

			timegh +=  time.time() - timeg

		## attention, K is  upper triangular with extension
		# make it symmetric

		self.K_sp = triu(self.K_sp)
		self.K_sp = self.K_sp + triu(self.K_sp,1).T


		timeTot = time.time() - timeTot




	# after compute_Sig_ks, more effiecient variant of predict_Xt2
	def predict_Xt2b(self, Xt=[[None]]):


		if Xt[0][0]==None:
			Xt = self.DD.X_test

		A_Bs = []
		ran = range(self.K-self.P)

		stT = time.time()
		KAx_s = self.kern.K(np.vstack(self.Aks), Xt)

		KAx_s_list = np.vsplit(KAx_s, self.K)
		#print('timePred_KxsAs3',time.time()-stT)
		KAx_s_new = []

		stT = time.time()
		for kp in ran:

			if self.P>0:
				A_B0 = np.vstack( [self.Aks[ kjj ] for kjj in  self.predecessors[kp+self.P] ])
				A_B2 = np.vstack( [A_B0[self.Cks[kp+self.P],:], self.Aks[kp+self.P]] )


				KAx_0 = np.vstack( [KAx_s_list[ kjj  ] for kjj in  self.predecessors[kp+self.P] ])
				KAx = np.vstack( [KAx_0[ self.Cks[kp+self.P],: ], KAx_s_list[kp+self.P]] )


			else:
				A_B2 = self.Aks[kp+self.P]
				KAx = KAx_s_list[kp+self.P]

			A_Bs.append(A_B2)
			KAx_s_new.append(KAx)

		#print('timePred0',time.time()-stT)



		stTT = time.time()




		stT = time.time()
		#Hxss = [ np.dot(KAx_s_new[kp].T, self.iKbig[kp+self.P]) for kp in ran ]
		HAx = [ np.dot(self.iKbig[kp+self.P], KAx_s_new[kp]) for kp in ran ]
		#print('timePred_Hxss',time.time()-stT)

		mks = [np.dot(self.mus[kp+self.P], HAx[kp] ) for kp in ran ]

		kxks = self.kern.Kdiag(Xt)


		stT = time.time()
		qkss = [ kxks  -diag_HtKH(KAx_s_new[kp], self.iKbig[kp+self.P])  for kp in ran ]
		#print('timePred_qkss',time.time()-stT)

		stT = time.time()
		#vks = [ diag_HtKH( Hxss[kp].T, self.Sigs[kp+self.P]) + qkss[kp] for kp in ran ]
		vks = [ diag_HtKH( HAx[kp], self.Sigs[kp+self.P]) + qkss[kp] for kp in ran ]
		#print('timePred_vks',time.time()-stT)

		#print('timePred1',time.time()-stTT)


		self.PREDSm = np.array(mks)
		self.PREDSv = np.array(vks) ##without noise
		self.PREDSvn = self.PREDSv + self.lik.variance[0]
		self.Qs_Ks = np.array(qkss)





	def log_det_FQF2(self):


		if not isspmatrix_csc( self.FQF ):
			FQF_csc = csc_matrix(self.FQF)   # explicit conversion to csc (needed for cholesky)
		else:
			FQF_csc = self.FQF

		self.facS = cholesky(FQF_csc)

		return self.facS.logdet()



	def log_marg_lik2(self):

		start = time.time()

		vdiag = np.hstack(self.Vs)

		#detS = self.log_det_FQF2()
		detS = self.logdetS
		detPost = -self.factor.logdet()
		detV = -np.sum(np.log( vdiag ))

		sqY = -np.sum(  self.DD.y_train**2 / vdiag )
		sqB = np.dot(self.b, self.m_post)

		logPi = -len(self.DD.y_train)*np.log(np.pi*2)

		timeTot = time.time() - start

		return 0.5*(detS + detPost + detV + sqY + sqB + logPi), timeTot






	#with Q, no longer used
	def predict_aggregate(self, pow=1):
		#self.Qs_Ks[self.Qs_Ks < 1e-100] = 1e-100 # becuase there are -0.0
		#dVQ2 = -0.5*np.log(self.PREDSv/self.Qs_Ks)
		dVQ2 = 0.5**pow*np.log(self.Qs_Ks/self.PREDSv)**pow  + 1e-100


		#print(self.PREDSv)

		#dVQ2[self.Qs_Ks <= self.jit] = 1e100 ########!!!

		sumVQ = np.sum(dVQ2,0)
		wei = dVQ2/sumVQ


		vs = 1/ np.sum( wei/self.PREDSvn, 0)
		ms = np.sum( wei*self.PREDSm/self.PREDSvn, 0) * vs

		return ms, vs, wei

	#with Q, no longer used
	def predict_aggregate_f(self, pow=1):
		#self.Qs_Ks[self.Qs_Ks < 1e-100] = 1e-100 # becuase there are -0.0

		#print('Q',self.Qs_Ks)

		#dVQ2 = -0.5*np.log(self.PREDSv/self.Qs_Ks)
		dVQ2 = -0.5**pow*np.log(self.Qs_Ks/self.PREDSv)**pow  + 1e-100

		#dVQ2[self.Qs_Ks <= self.jit] = 1e100 ########!!!

		#print('dVQ2',dVQ2)


		sumVQ = np.sum(dVQ2,0)
		wei = dVQ2/sumVQ


		vs = 1/ np.sum( wei/self.PREDSv, 0)
		ms = np.sum( wei*self.PREDSm/self.PREDSv, 0) * vs

		return ms, vs+self.lik.variance[0], wei




	def predict_aggregatePoE(self, M, V, pow=1):


		#dVQ2 = (-0.5*np.log(V/(self.kern.variance[0]+self.lik.variance[0])) )**pow   + 1e-100
		dVQ2 = (-0.5*np.log(V/(self.kern.Kdiag(np.zeros((1,self.DD.D)))+self.lik.variance[0])) )**pow   + 1e-100
		sumVQ = np.sum(dVQ2,0)
		wei = dVQ2/sumVQ


		vs = 1/ np.sum( wei/V, 0)
		ms = np.sum( wei*M/V, 0) * vs

		return ms, vs, wei



	def predict_aggregatePoE_f(self, M,V,pow=1):

		#dVQ2 = (-0.5*np.log(V/(self.kern.variance[0])) )**pow   + 1e-100
		dVQ2 = (-0.5*np.log(V/(self.kern.Kdiag(np.zeros((1,self.DD.D))) ) ) )**pow   + 1e-100
		sumVQ = np.sum(dVQ2,0)
		wei = dVQ2/sumVQ


		vs = 1/ np.sum( wei/V, 0)
		ms = np.sum( wei*M/V, 0) * vs

		return ms, vs+self.lik.variance[0], wei

	def predict_aggregateMV(self):
		ai = np.expand_dims(np.argmin(self.PREDSvn, axis=0), axis=0)

		vsMV = np.take_along_axis(self.PREDSvn, ai, axis=0)
		msMV = np.take_along_axis(self.PREDSm, ai, axis=0)

		return msMV[0,:], vsMV[0,:]





	def update_parms(self):

		start = time.time()

		all_derivs = []
		for par in self.PARAMS:

				if par.EST:

					self.compute_derivatives(par)
					all_derivs.append(par.dliks)

		self.dliks = np.hstack(all_derivs)

		timeGrad = time.time() - start
		print(timeGrad)





	## parameters #########################################################
	def init_params(self):


		self.PARAMS = []

		if self.kern.name == 'sum':
			kerns = self.kern.parts
		else:
			kerns = [self.kern]


		for keri in kerns:

			for par in keri.parameters[:]:

				if keri.name[:12] == 'std_periodic':
					nnam = par.name + 'P'
				else:
					nnam = par.name


				parInc = PARAM(nnam, par, keri, not par.is_fixed )
				self.PARAMS.append(parInc)

		parNoise = PARAM('noise', self.lik.variance, None, not self.lik.is_fixed)

		self.PARAMS.append(parNoise)



	def get_PARAMS(self):

		params = np.zeros(0)
		for par in self.PARAMS:

			if par.EST:
				params = np.hstack([params, par.get_value()])

		return params


	def set_PARAMS(self, values):


		i = 0
		for par in self.PARAMS:

			if par.EST:

				par.update_value( values[i:i+par.length()] )

				i += par.length()






   

def invEE(M, jit=1e-15):

    return inv(M + np.eye(M.shape[0])*jit)

def comp_kernR(kern, R, jit=1e-15):

    Krr = kern.K(R)
    iKrr = invEE(Krr, jit)

    return iKrr, Krr

def comp_kernX(kern, X, R, iKrr, DIAG=True):

    Kxr = kern.K(X,R)
    H = np.dot(Kxr, iKrr)


    #Q = dot3lr(Kxr, iKrr, Kxr.T)
    if DIAG:
        Q = diag_HtKH(Kxr.T, iKrr)
        kxx = kern.Kdiag(X)
        D = kxx - Q
    else:
        #Q = dot3lr(Kxr, iKrr, Kxr.T)       ## it is already H
        Q = np.dot(H, Kxr.T)
        Kxx = kern.K(X)
        D = Kxx - Q

    return H, Q, D



def comp_FQ(kern, A, B , DIAG=False, jit=1e-15):
    # for data A given data B

    iK_BB, K_BB = comp_kernR(kern, B, jit=jit)
    F_AB, D_AB, Q_AB = comp_kernX(kern, A, B, iK_BB, DIAG=DIAG)


    return F_AB, Q_AB, K_BB, iK_BB, D_AB


def pred_y_diag(m,P,Fs,qs,sig2n):
    ms, vs = transDiag(m,P,Fs,sig2n)
    return ms, vs + sig2n

def transDiag(m,P,Fs,qs):
    ms = np.dot(Fs,m)
    vs = diag_HtKH(Fs.T,P) + qs
    return ms, vs


def trans(m,P,F,Q):
    m1 = np.dot(F,m)
    P1 = dot3lr(F,P,F.T) + Q
    return m1, P1


def KL1(m1,m2,v1,v2):
    return 0.5*(np.log(v2/v1) + (v1 + (m1-m2)**2)/v2 - 1)

def SE(m1,m2):
    return (m1-m2)**2

def KL1m(m1,m2,P1,P2):
    iP1, lD1 = inv_logDet(P1)
    iP2, lD2 = inv_logDet(P2)
    #return 0.5*(np.trace(np.dot(iP2,P1)) + lD2 - lD1 + dot3lr( (m2-m1).T,iP2,(m2-m1) )  - len(m1))

    tr = 0.5*np.trace(np.dot(iP2,P1))
    lDs = 0.5*(lD2 - lD1)
    sq = 0.5*dot3lr( (m2-m1).T,iP2,(m2-m1) )
    C = -0.5*len(m1)

    return tr+lDs+sq+C, (tr,lDs,sq,C)
    #return tr+lDs+sq+C, (tr+C,lDs,sq)
    #return tr+lDs+sq+C, (tr+lDs,sq,C)




#### it is outside
#@functools.lru_cache(maxsize=1)
@np_cache(maxsize=1)
def f_df_BGP( params, BGP):


    BGP.set_PARAMS(np.exp(params))

    BGP.run1()
    BGP.run1b()


    print(np.exp(params))
    print(BGP.lml, BGP.dliks)
    return BGP.lml, BGP.dliks


def f_BGP( params, BGP ):

    f, _ = f_df_BGP(params, BGP)

    return -f

def df_BGP( params, BGP):

    _, df = f_df_BGP(params, BGP)

    return -df * np.exp(params)




class Perm():
    def __init__(self, p):
        self.p = p
        self.ip = np.argsort(p)

        self.p_ = self.p[:,np.newaxis]
        self.ip_ = self.ip[:,np.newaxis]

    def Pvec(self, vec):

        return vec[self.p]

    def Pmat(self, Mat):

        return Mat[self.p_, self.p_.T]

    def iPvec(self, vec):

        return vec[self.ip]

    def iPmat(self, Mat):

        return Mat[self.ip_, self.ip_.T]
