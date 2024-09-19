# Generated with SMOP  0.41
from scipy.sparse import spdiags
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components

    # min_{S>=0, S'*1=1, S*1=1, F'*F=I}  ||S - A||^2 + 2*lambda*trace(F'*Ln*F)    

def EProjSimplex_new(v,k=None):
    #  min  1/2 || x - v||^2   s.t. x>=0, 1'x=1
    if k is None:
        k = 1
    ft = 1
    n = v.shape[1]
    if (n != 0):
        v0 = v - np.mean(v) + k / n
        vmin = np.min(v0)
        if vmin < 0:
            f=1
            lambda_m=0
            while abs(f) > 10 ** - 10:
                v1=v0 - lambda_m
                posidx=v1 > 0
                npos=np.sum(posidx)
                g=- npos
                f=np.sum(v1[posidx]) - k
                lambda_m=lambda_m - f / g
                ft=ft + 1
                if ft> 100:
                    x = np.maximum(v1, 0)
                    break
            x = np.maximum(v1, 0)

        else:
            x = v0
    else:
        x=0
    return x

def L2_distance_11(a,b):
    # a,b: two matrices. each column is a data
# d:   distance matrix of a and b
    if a.shape[0] == 1:
        a = np.concatenate((a, np.zeros((1, a.shape[1]))), axis=0)
        b = np.concatenate((b, np.zeros((1, b.shape[1]))), axis=0)
    m=a.shape[0]
    aa = np.empty((m,m))
    bb = np.empty((m,m))
    ab = np.empty((m,m))
    d= np.empty((m,m))
    for i in range(a.shape[0]):
        a[i] = [sparse_matrix.toarray() for sparse_matrix in a[i]]  #(m,)
        b[i] = [sparse_matrix.toarray() for sparse_matrix in b[i]]  #(m,)
        for j in range(len(a[i])):
            aa[i][j] =np.sum(np.multiply(a[i][j], a[i][j]))# a[i][j] (n,m)
            bb[i][j] = np.sum(np.multiply(b[i][j], b[i][j]))#b[i][j]  (m,m)
            ab[i][j] = np.sum(np.dot(a[i][j], b[i][j]))
            d[i][j] = aa[i][j]+bb[i][j]-2*ab[i][j]
    d = np.real(d)
    d = np.maximum(d, 0)
    return d



def L2_distance_1(a,b):#c*n  c*m
    # a,b: two matrices. each column is a data   #a  7*1474   b 7*7
# d:   distance matrix of a and b
    if a.shape[0] == 1:
        a = np.concatenate((a, np.zeros((1, a.shape[1]))), axis=0)
        b = np.concatenate((b, np.zeros((1, b.shape[1]))), axis=0)
    aa =np.sum(np.multiply(a, a), axis=0)#(,numsample)
    bb = np.sum(np.multiply(b, b), axis=0)#(,anchor)
    ab = a.T@b  #n*m
    cc=aa.reshape(-1,1)
    d =np.tile(cc,(1,bb.shape[0])) + np.tile(bb, (aa.shape[0], 1))- 2 * ab
    d = np.nan_to_num(d,nan=0)
    return d

def eig1(A, c, isMax=True, isSym=True):
    if c is None:
        c = A.shape[0]
        isMax = True
        isSym = True
    elif c >= A.shape[0]:
        c = A.shape[0]

    if isMax is None:
        isMax = True
        isSym = True

    if isSym is None:
        isSym = True

    if isSym:
        A = np.maximum(A.toarray(), (A.toarray()).T)
    #if np.any(np.isinf(A)) or np.any(np.isnan(A)):
        # Handle the presence of infinite or NaN values
        # For example, replace them with zeros or remove them depending on the context of your problem
        #A = np.nan_to_num(A)  # Replace NaN with zero and inf with finite numbers
    d, v = np.linalg.eig(A)
    idx = np.argsort(-d if isMax else d)
    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]
    eigval_full = d[idx]

    return eigvec, eigval, eigval_full

def graphconncomp(G):
    components = list(nx.connected_components_subgraphs(G))
    S = len(components)
    C = [-1] * G.number_of_nodes()
    for i, comp in enumerate(components):
        for node in comp:
            C[node] = i
    return S, C



def coclustering_bipartite_fast1(A,G,c,sum_alpha,NITER,islocal=1):# n*m n*m  c
    if NITER is None:
        NITER = 30
    lambda_=0.1
    n,m=G.shape #n*m
    G=csr_matrix(G)    #这里输出和matlab相同   即G是对的
    a1=np.sum(G,axis=1)   #1*m
    D1a = spdiags(1.0/np.sqrt(a1.T), 0, n, n).T #n*n  列向量变行向量直接.T
    a2=G.sum(axis=0)
    D2a=spdiags(1.0/np.sqrt(a2),0,m,m).T  #m*m
    A1=D1a@G@D2a #n*m
    SS2=(A1.T)@A1  #m*m
    V,ev0,_=eig1(SS2,m)
    V=V[:,0:c]  #m*c
    U = A1 @ V  #n*m m*c
    U = U / np.sqrt(ev0[0:c])  #n*c
    U = np.sqrt(2) / 2 * U  #n*c
    V = np.sqrt(2) / 2 * V  #m*c

    A = np.array(A)  #n*m
    idxa=[None]*n
    for i in range(n):
        if islocal == 1:
            idxa0=np.where(A[i,:] > 0)
        else:
            idxa0=np.arange(m)
        idxa[i]=idxa0
    
    idxam=[None]*m
    for i in range(m):
        if islocal == 1:
            idxa0=np.where(A[:,i] > 0)
        else:
            idxa0=np.arange(n)
        idxam[i]=idxa0
    D1=D1a.copy()
    D2=D2a.copy()
    for iter in range(NITER):
        U1=D1@U  #n*n  n*c
        V1=D2@V #m*m m*c
        dist=L2_distance_1(U1.T,V1.T)  #c*n   c*m=n*m
        S=np.zeros((n,m))
        for i in range(n):
            idxa0=idxa[i]
            ai=A[i,idxa0]
            di=dist[i,idxa0]
            ad=(ai - 0.5*lambda_*di) / sum_alpha
            S[i,idxa0]= EProjSimplex_new(ad)
        Sm=np.zeros((m,n))
        for i in range(m):
            idxa0=idxam[i]
            ai=A[idxa0,i]
            di=dist[idxa0,i]
            ad=(ai - 0.5*lambda_*di)/ sum_alpha
            Sm[i,idxa0]=EProjSimplex_new(ad)
        S=csr_matrix(S)  #n*m
        Sm=csr_matrix(Sm) #m*n
        SS=(S + Sm.T) / 2
        d1 = np.sum(SS, axis=1)
        D1=spdiags(1.0/np.sqrt(d1.T), 0, n, n).T
        d2 = np.sum(SS, axis=0)
        D2=spdiags(1.0 /np.sqrt(d2),0,m,m).T
        SS1=D1@SS@D2  #n*m
        SS2=SS1.T@SS1  #m*m
        V,ev0,ev=eig1(SS2,c)   #m*c
        U=(SS1@V) / np.sqrt(ev0)
        U = np.sqrt(2) / 2 * U #n*c
        V = np.sqrt(2) / 2 * V
        U_old=U.copy()
        V_old=V.copy()
        if ev.shape[0] > c:
            fn1=sum(ev[0:c])
            fn2=sum(ev[0:c + 1])
            if fn1 < c - 1e-07:
                lambda_=2*lambda_
            else:
                if fn2 > c + 1 - 1e-07:
                    lambda_=lambda_ / 2
                    U=U_old.copy()
                    V=V_old.copy()
                else:
                    break
        else:
            fn1=sum(ev[0:c])
            if fn1 < c - 1e-07:
                lambda_=2*lambda_
            else:
                break
    term2=0.1*(np.trace((U.T)@U)+ np.trace((V.T)@V) - np.trace((U.T)@SS1@V))
    SS0=csr_matrix((n + m,n + m))
    SS0[:n,n:]=SS
    SS0[n:,:n]=SS.T

    _, y = connected_components(SS0, directed=False)
    y1=y[:n].T
    y2=y[n:].T
    return y1,y2,SS,U,V,term2