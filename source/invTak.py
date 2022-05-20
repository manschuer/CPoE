import numpy as np

from mix import invEE
from numpy.linalg import inv

from scipy.sparse import bsr_matrix, csr_matrix


def invTak_full0(Lfull, Xfull, K, B ):
    for ik in np.arange(K-1,-1,-1):
        inds_k = ik*B + np.arange(B)
        inds_k_m = inds_k[:,np.newaxis]
        inds_rest = np.arange((ik+1)*B,K*B)
        inds_rest_m = inds_rest[:,np.newaxis]
        #print(inds_k, inds_rest)

        La = Lfull[inds_k_m, inds_k_m.T]
        #iLa = invEE(La, jit=0)
        iLa = inv(La, jit=0)


        if ik<(K-1):
            Lb = Lfull[inds_rest_m, inds_k_m.T]
            Tb = np.dot(Lb, iLa)
            Zc = Xfull[inds_rest_m, inds_rest_m.T]
            Zb = - np.dot(Zc, Tb)
            Za = - np.dot(Zb.T, Tb)

            Xfull[inds_rest_m, inds_k_m.T] = Zb
            Xfull[inds_k_m, inds_rest_m.T] = Zb.T

        else:
                Za = 0


        Xfull[inds_k_m, inds_k_m.T] = np.dot(iLa.T,iLa) + Za



def preTak(L, K, B):

    stT = time.time()
    Lb = bsr_matrix(L, blocksize=(B,B))
    #print('Lb', time.time() - stT )

    stT = time.time()
    INDS = csr_matrix( (np.arange(len(Lb.data)), Lb.indices, Lb.indptr), shape=(K,K) ).toarray()
    LOOKUP = INDS + np.tril(INDS,-1).T
    #print('LOOKUP',time.time() - stT)


    stT = time.time()
    Lbt = Lb.T
    #print('Lbt',time.time() - stT)

    return Lb, LOOKUP, Lbt.indices, Lbt.indptr



def invTak_1(Ldata_block, LOOKUP, indices, indptr, K, B):
    ## Ldata_block and Xdata_block is in block-sparse-row (bsr) format but index given by full matrix LOOKUP
    ## looping over data has to be done in csc ordering 
    

    Xdata_block = np.zeros_like(Ldata_block)

  
    for i in range(K-1,-1,-1):

        #iLa = invEE( Ldata_block[LOOKUP[i,i]])
        iLa = inv( Ldata_block[LOOKUP[i,i]])
        Xdata_block[ LOOKUP[i,i] ] = np.dot(iLa.T, iLa)
        

        for j_i in range(indptr[i+1]-1, indptr[i]-1, -1):

            j = indices[j_i]



            #print('')
            #print('i',i,'j',j,LOOKUP[i,j])

            Q = np.zeros((B,B))
            for lc in range(indptr[i+1] -1, indptr[i], -1):
                l = indices[lc]


                ind_block = LOOKUP[ j, l ]

                if l<=j:
                    X =  Xdata_block[ ind_block  ]
                else:
                    X =  Xdata_block[ ind_block  ].T


                Q += np.dot( X, np.dot( Ldata_block[ LOOKUP[l,i] ], iLa ) )



            Xdata_block[ LOOKUP[i,j] ] -= Q
        
 
    return Xdata_block





def invTak_11(Ldata_block, LOOKUP, indices, indptr, K, B):
    ## Ldata_block and Xdata_block is in block-sparse-row (bsr) format but index given by full matrix LOOKUP
    ## looping over data has to be done in csc ordering 
    

    Xdata_block = np.zeros_like(Ldata_block)


    #iLas = [invEE(Ldata_block[i_d]) for i_d in np.diag(LOOKUP)]
    iLas = [inv(Ldata_block[i_d]) for i_d in np.diag(LOOKUP)]
    LLas = [ np.dot(iLas[i].T, iLas[i])  for i in range(K) ]

  
    for i in range(K-1,-1,-1):

        Xdata_block[ LOOKUP[i,i] ]  = LLas[i]
        
        for j in  np.flip( indices[ indptr[i]:indptr[i+1] ] ):

            Q = np.zeros((B,B))
            for l in  indices[ indptr[i]+1:indptr[i+1] ] :
                
                X =  Xdata_block[ LOOKUP[ j, l ]  ]

                if l>j:
                    X =  X.T

                Q += np.dot( X, np.dot( Ldata_block[ LOOKUP[l,i] ], iLas[i] ) )


            Xdata_block[ LOOKUP[i,j] ] -= Q
        
 
    return Xdata_block










def invTak_2(Ldata_block, LOOKUP, indices, indptr, K, B):
    ## Ldata_block and Xdata_block is in block-sparse-row (bsr) format but index given by full matrix LOOKUP
    ## looping over data has to be done in csc ordering 
    

    Xdata_block = np.zeros_like(Ldata_block)

    #stT = time.time()
    #iLas = [invEE(Ldata_block[i_d]) for i_d in np.diag(LOOKUP)]
    iLas = [inv(Ldata_block[i_d]) for i_d in np.diag(LOOKUP)]
    LLas = [ np.dot(iLas[i].T, iLas[i])  for i in range(K) ]
    #print('iLas',time.time() - stT)

  
    for i in range(K-1,-1,-1):

  

        #iLa = invEE( Ldata_block[LOOKUP[i,i]])
        #Xdata_block[ LOOKUP[i,i] ] = np.dot(iLa.T, iLa)

        #Xdata_block[ LOOKUP[i,i] ] = np.dot(iLas[i].T, iLas[i])

        Xdata_block[ LOOKUP[i,i] ]  = LLas[i]

        
        for j in  np.flip( indices[ indptr[i]:indptr[i+1] ] ):
            
         
            #print('i',i,'j',j,LOOKUP[i,j])

            # Q = np.zeros((B,B))
         
            # XX = []
            # TT = []
            # Qs = []

            # for l in  indices[ indptr[i]+1:indptr[i+1] ] :

            #     X =  Xdata_block[ LOOKUP[ j, l ]  ]

            #     if l>j:
            #         X = X.T

            #     XX.append(X)
            #     TT.append(np.dot( Ldata_block[ LOOKUP[l,i] ], iLas[i] ))
            #     Qs.append(np.dot( X, np.dot( Ldata_block[ LOOKUP[l,i] ], iLas[i] ) ))

            #     Q += np.dot( X, np.dot( Ldata_block[ LOOKUP[l,i] ], iLas[i] ) )



            #if len(TT)>0:


            #AA =  np.array(TT)
            #YY = np.array(XX)
            #YY = np.reshape( YY, (-1,YY.shape[-1]) )
            #AA = np.reshape(A, (A.shape[0]*A.shape[1],A.shape[2]) )


            ls =  indices[ indptr[i]+1:indptr[i+1] ] 
            BB = np.dot( Ldata_block[ LOOKUP[ls,i] ], iLas[i] )
           

            Y1 =  Xdata_block[ LOOKUP[ j, ls[ls<=j] ]  ]
            Y2 =  Xdata_block[ LOOKUP[ j, ls[ls>j] ]  ]
            Y2 = np.transpose(Y2, (0,2,1) )

            #YYY =  np.vstack([ np.reshape(Y1, (-1,Y1.shape[-1]) ), np.reshape(Y2, (-1,Y1.shape[-1]) ) ] )
                

            





            #AA = np.reshape(AA, (AA.shape[0]*AA.shape[1],AA.shape[2]) )
            #BB = np.reshape(BB, (BB.shape[0]*BB.shape[1],BB.shape[2]) )
            #if np.sum(np.abs(YY-YYY))>1e-15:
            #print(np.sum(np.abs(YY-YYY)))

            #print(np.sum(np.abs(AA-BB)))


            #print(AA-BB)
            #print(BB)


            #QQ = np.dot(YYY.T, np.reshape(BB, (-1,BB.shape[-1])) )

            
            #print('Y1',Y1.shape)
            #print('Y2',Y2.shape)
            Y12 = np.concatenate([Y1, Y2])



            #YY12 = np.reshape( Y12, (-1, Y12.shape[-1]) )
            #print(YY12 - YYY)



            #print( YYY[:32,:] -Y12[0] )

            #print( np.dot(Y12, np.reshape(BB, (-1,BB.shape[-1]))).shape ) 


            #print(np.dot(Y12, BB))


            #print(Y12.shape)
            #print('BB',BB.shape)
            
            #if Y1.shape[0]>0:
            #    Q1 = np.dot(Y1[0], BB[0])
            #else:
            #    Q1 = np.dot(Y2[0], BB[0])

            GG = np.array([np.dot(Y12[t], BB[t]) for t in range(Y12.shape[0]) ])
            QQ = np.sum( GG, 0)


            #print(GG.shape)
            #print(len(Qs))

            

            #print( sum( Qs ) - Q  )
            #print( QQ - Q)

          


            #for t in range(Y12.shape[0]):
            #    Q1 = np.dot(Y12[t], BB[t])
            #    if np.sum(np.abs(Q1-Qs[t])) > 1e-15:
            #        print(np.sum(np.abs(Q1-Qs[t])))


            #for t in range(len(Qs)):
            #    if np.sum(np.abs(GG[t]-Qs[t])) > 1e-15:
            #        print(np.sum(np.abs(Q1-Qs[t])))

            #print(QQ.shape)
            #print('QQ',QQ)
            #print('Q',Q)

            

            #print(np.sum(QQ+Q))


              


            Xdata_block[ LOOKUP[i,j] ] -= QQ
        
 
    return Xdata_block















###############################################OLD, without  blocks...



def invTak(L_full, d_diag):
    
    n = L_full.shape[1]

    

    X = np.diag( 1/d_diag )


    for j in np.arange(n-1,-1,-1):
        for i in np.arange(j,-1,-1):

            
            #print(L_full[(i+1):n,i])
         
            #q = np.dot(U_full[i,(i+1):n], X[(i+1):n, j])
            q = np.dot(L_full[(i+1):n,i], X[(i+1):n, j])


       
     
       

            X[i,j] -=  q
            #X[j,i] -=  (i!=j)*q
            
            if i!=j:
                X[j,i] -=  q
                
            
    return X


def invTak2(L_full, d_diag):
    
    n = L_full.shape[1]

    

    X = np.diag( 1/d_diag )


    
    for i in np.arange(n-2,-1,-1):


     


        for j in np.arange(n-1,i-1,-1): 


            # print('i ',i,' j ',j)

            # print(L_full[(i+1):n,i])
         
            #q = np.dot(U_full[i,(i+1):n], X[(i+1):n, j])
            q = np.dot(L_full[(i+1):n,i], X[(i+1):n, j])


            #print('q ',q)


  
       

            X[i,j] -=  q
            #X[j,i] -=  (i!=j)*q
            
            if i!=j:
                X[j,i] -=  q

       
            
                
            
    return X


def invTak2m(L_full, d_diag):
    
    n = L_full.shape[1]

    

    #X = np.diag( 1/d_diag )
    X2 = np.diag( 1/d_diag )


    
    for i in np.arange(n-2,-1,-1):

        li = L_full[(i+1):n,i]

        xxi = np.dot(li, X2[(i+1):n, i+1:n ] )

        qii = np.dot(li, -xxi)


        # Xuse = np.zeros_like(X2)
        # Xuse[(i+1):n, i+1:n] = 1
        # Luse = np.zeros_like(X2)
        # Luse[(i+1):n,i] = 2

        #print(np.round(X[i,i]-qii,2))

        #print(np.round( XXi, 2) )

        X2[i, i+1:n ] = -xxi
        X2[i+1:n,i ] = -xxi
        X2[i,i] -=  qii

        # print('i ',i)
        # print(Xuse)
        # print(Luse)


       
                     
            
    return X2


from scipy.sparse import csc_matrix
import time

def invTak2ms(L_sparse, d_diag):
    
    n = L_sparse.shape[1]

    t1 = t2 = t3 = t4 = t5 = 0

    #X = np.diag( 1/d_diag )
    #X2 = np.diag( 1/d_diag )
    X2 = np.diag( 1/d_diag )
    #X3 = lil_matrix(  spdiags( 1/d_diag, 0, n, n ) )


    
    for i in np.arange(n-2,-1,-1):

        #print(i)


        ss = time.time()
        li_s = L_sparse[(i+1):n,i]
        t1 += time.time()-ss



        ss = time.time()
        #inds_ax = li.nonzero()+i+1
        inds_ax = (li_s.nonzero()[0]+i+1)[np.newaxis,:]
        t2 += time.time()-ss
        

        ss = time.time()
        #print(X2[inds_ax.T, inds_ax]-X2[(i+1):n, i+1:n ])
        xxi2 = np.dot(li_s.data, X2[inds_ax.T, inds_ax] )
        t3 += time.time()-ss


        #print(np.round( X2[inds_ax.T, inds_ax], 2) )


        Xuse = np.zeros_like(X2)
        Xuse[inds_ax.T, inds_ax] = 1
        Luse = np.zeros_like(X2)
        Luse[inds_ax,i] = 2


        ss = time.time()
        qii2 = np.dot(li_s.data, -xxi2)
        t4 += time.time()-ss

        ss = time.time()
        X2[i, inds_ax[0,:] ] = -xxi2
        X2[inds_ax[0,:],i ] = -xxi2
        X2[i,i] -=  qii2
        t5 += time.time()-ss


        #print('i ',i)
        #print(Luse)
        #print(Xuse)


        #print(np.sum(np.abs(X3-X2)))

    print('t1 ',t1)
    print('t2 ',t2)
    print('t3 ',t3)
    print('t4 ',t4)
    print('t5 ',t5)

                     
            
    return X2


from scipy.sparse import spdiags, lil_matrix






def invTak_s2(L_sparse, d_diag):

    n = L_sparse.shape[1]
    
    print(n)

    #X = np.diag( 1/d_diag )

    X = lil_matrix( spdiags( 1/d_diag, 0, n, n ) )


    for i in np.arange(n-2,-1,-1):
        for j in np.arange(n-1,i-1,-1): 

            #print('i ',i,'j ',j)

          
         
            #q = np.dot(U_full[i,(i+1):n], X[(i+1):n, j])
            #print('L ',L_sparse[(i+1):n,i].toarray())
            #print(L_sparse[(i+1):n,i].shape)
            #print('X ',X[(i+1):n, j].toarray())
            #print(X[(i+1):n, j].T.shape)
            #q = np.dot(L_sparse[(i+1):n,i].T, X[(i+1):n, j])
            #q = np.dot( L_sparse.getrow(i)[(i+1):n].T, X[(i+1):n, j])
            #q2 = L_sparse.getrow(i)[(i+1):n].multiply( X.getrow(j)[(i+1):n]).sum()  ##not working??

            q = L_sparse[(i+1):n,i].multiply(X[(i+1):n, j]).sum()

            


            #print(L_sparse.getrow(i)[(i+1):n].nnz)

            #print(q.toarray())

            #q = q.toarray()[[0]]
            #q = q[[0]].toarray()
     
            #print(q-q2)

            #print(q.shape)

            X[i,j] -=  q
            #X[j,i] -=  (i!=j)*q
            
            if i!=j:
                X[j,i] -=  q
                
            
    return X




def invTak_s3(L_sparse, d_diag):

    n = L_sparse.shape[1]
    
    iis, jjs = L_sparse.T.nonzero()


    X = lil_matrix( spdiags( 1/d_diag, 0, n, n ) )

    nnz = L_sparse.nnz
    for counter in range(nnz):
        i = iis[nnz-counter-1]
        j = jjs[nnz-counter-1]


        #print(i, j)


    #for i in np.arange(n-2,-1,-1):
        #for j in np.arange(n-1,i-1,-1): 

           

        q = L_sparse[(i+1):n,i].multiply(X[(i+1):n, j]).sum()

        


        X[i,j] -=  q
     
        
        if i!=j:
            X[j,i] -=  q
                
            
    return X



def invTak_s4(L_sparse, d_diag):

    n = L_sparse.shape[1]
    
    iis, jjs = L_sparse.T.nonzero()


    X = np.diag( 1/d_diag )
    L_full = L_sparse.toarray()

    nnz = L_sparse.nnz
    for counter in range(nnz):
        i = iis[nnz-counter-1]
        j = jjs[nnz-counter-1]


        #print(i, j)


        q = np.dot(L_full[(i+1):n,i], X[(i+1):n, j])

        
        #print(X[(i+1):n, j])

        #q = L_sparse[(i+1):n,i].multiply(X[(i+1):n, j]).sum()

        


        X[i,j] -=  q
     
        
        if i!=j:
            X[j,i] -=  q
                
            
    return X


def invTak_s5(L_sparse, d_diag):

    n = L_sparse.shape[1]
    
    iis, jjs = L_sparse.T.nonzero()


    X = np.diag( 1/d_diag )
    L_full = L_sparse.toarray()

    nnz = L_sparse.nnz
    iold = -1
    for counter in range(nnz):
        i = iis[nnz-counter-1]
        j = jjs[nnz-counter-1]


        #print(i, j)

        if i!=iold:
            iold = i
            li = L_full[(i+1):n,i]
        #else:
         #   print('s')
        q = np.dot(li, X[(i+1):n, j])

           

        #q = L_sparse[(i+1):n,i].multiply(X[(i+1):n, j]).sum()

        


        X[i,j] -=  q
     
        
        if i!=j:
            X[j,i] -=  q
                
            
    return X


def invTak_s6(L_sparse, d_diag):

    n = L_sparse.shape[1]
    
    iis, jjs = L_sparse.T.nonzero()


    X = lil_matrix( spdiags( 1/d_diag, 0, n, n ) )

    nnz = L_sparse.nnz
    iold = -1

    for counter in range(nnz):
        i = iis[nnz-counter-1]
        j = jjs[nnz-counter-1]


        if i!=iold:
            iold = i
            li = L_sparse[(i+1):n,i]
        #else:
         #   print('s')

           

        q = li.multiply(X[(i+1):n, j]).sum()

        


        X[i,j] -=  q
     
        
        if i!=j:
            X[j,i] -=  q
                
            
    return X