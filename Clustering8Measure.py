# Generated with SMOP  0.41
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import networkx as nx

from sklearn.metrics import  silhouette_samples



def calculate_cell_type_asw(data, labels):
    unique_labels = np.unique(labels)
    silhouette_values = silhouette_samples(data, labels, metric='euclidean')
    asw = np.mean(silhouette_values)
    scaled_asw = (asw + 1) / 2
    return asw

def graph_connectivity(data, labels):

    unique_labels = np.unique(labels)
    gc_scores = []
    labels = labels.flatten()
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        subset_data = data[indices]

        graph = kneighbors_graph(subset_data.T, n_neighbors=5, mode='connectivity')
        lcc_size = len(max(nx.connected_components(nx.Graph(graph)), key=len))
        gc_score = lcc_size / len(indices)
        gc_scores.append(gc_score)

    final_gc_score = np.mean(gc_scores)

    return final_gc_score

def contingency(c1, c2):
    labels = np.unique(c1)
    clusters = np.unique(c2)
    n_labels = len(labels)
    n_clusters = len(clusters)
    contingency_matrix = np.zeros((n_labels, n_clusters), dtype=np.int) 
    for i in range(n_labels):
        for j in range(n_clusters):
            contingency_matrix[i, j] = np.sum((c1 == labels[i]) & (c2 == clusters[j]))
    return contingency_matrix


def compute_f(T, H):
    if len(T) != len(H):
        raise ValueError("Length of T and H must be equal")

    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT += np.sum(Tn)
        numH += np.sum(Hn)
        numI += np.sum(Tn & Hn)

    p = 1
    r = 1
    f = 1
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return f, p, r

import numpy as np

def compute_nmi(labels_true, labels_pred):
    labels_pred=labels_pred.T
    labels_true=labels_true.T
    n = len(labels_true)
    # Calculate the mutual information
    result = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true), np.unique(labels_pred)))
    contingency_matrix=result[0]
    contingency_matrix = contingency_matrix / n
    sum_rows = np.sum(contingency_matrix, axis=1)
    sum_cols = np.sum(contingency_matrix, axis=0)
    outer_product = np.outer(sum_rows, sum_cols)
    outer_product[outer_product == 0] = 1
    mi = np.sum(contingency_matrix * np.log((contingency_matrix + 1e-15) / (outer_product)))

    # Calculate the entropy of the true labels
    entropy_true = -np.sum(sum_rows * np.log(sum_rows + 1e-15))

    # Calculate the entropy of the predicted labels
    entropy_pred = -np.sum(sum_cols * np.log(sum_cols + 1e-15))

    # Calculate the normalized mutual information
    nmi = mi / np.sqrt(entropy_true * entropy_pred)

    return nmi

from munkres import Munkres
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    #f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    #precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    #recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro,precision_macro,recall_macro

def eva(data,label):
    scaled_asw=calculate_cell_type_asw(data,label)
    final_gc_score = graph_connectivity(data,label)
    return scaled_asw,final_gc_score


def Clustering8Measure(Y,predY):
    predY=predY.reshape(-1,1)   #转换成一维列向量
    n=Y.shape[0]
    uY=np.unique(Y)#[1,2,3,4,5,6,7]
    nclass=uY.shape[0]
    Y0=np.zeros((n,1))
    if nclass != max(Y):
        for i in range(nclass):
            Y0[np.where(Y == uY[i])]=i
        Y=Y0
    uY=np.unique(predY)
    nclass=uY.shape[0]   #7
    predY0=np.zeros((n,1))
    if nclass != max(predY):
        for i in range(nclass):
            predY0[np.where(predY == uY[i])]=i
        predY=predY0
    Lidx=np.unique(Y)
    classnum=Lidx.shape[0]
    predLidx=np.unique(predY)
    pred_classnum=predLidx.shape[0]
    # purity
    correnum=0
    for ci in range (pred_classnum):
        incluster=Y[np.where(predY == predLidx[ci])]
        inclunub,_=np.histogram(incluster,bins=np.arange(0,max(incluster)))
        if inclunub.size==0:
            inclunub=0
        correnum=correnum + np.max(inclunub)
    Purity=correnum / predY.shape[0]
    res=bestMap(Y,predY)
    #res = res.reshape(-1, 1)
    #sum=0
    #for i in range(len(Y)):
        #if(Y[i][0]==res[i][0]):
        #if(Y[i][0]==predY[i][0]):
            #sum=sum+1
    #ACC=sum/len(Y)
    #MIhat=MutualInfo(Y,res)
    #Fscore,Precision,Recall=compute_f(Y,predY)
    #nmi=compute_nmi(Y,predY)
    #AR=RandIndex(Y,predY)
    labels_true = Y.flatten()  # or labels_true.flatten()
    labels_pred=predY.flatten()
    res=res.flatten()
    nmi=nmi_score(labels_true,res,average_method='arithmetic')
    ARI=ari_score(labels_true,res)
    ACC, Fscore, Precision, Recall=cluster_acc(labels_true,res)
    return ACC,nmi,Purity,Fscore,Precision,Recall,ARI


def bestMap(L1,L2):
    L1 = np.array(L1).flatten()
    L2 = np.array(L2).flatten()
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')

    L1 = L1 - np.min(L1) + 1
    L2 = L2 - np.min(L2) + 1

    nClass = max(np.max(L1), np.max(L2))
    nClass=int(nClass)
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == i + 1) & (L2 == j + 1))
    row_ind, col_ind = linear_sum_assignment(-G)
    newL2 = np.zeros_like(L2)
    for i in range(nClass):
        newL2[L2 == i + 1] = col_ind[i] + 1

    return newL2

def MutualInfo(L1, L2):
    L1 = np.ravel(L1)
    L2 = np.ravel(L2)
    if L1.size != L2.size:
        raise ValueError('size(L1) must == size(L2)')

    Label = np.unique(L1)
    nClass = len(Label)

    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    if nClass2 < nClass:
        L1 = np.concatenate([L1, Label])
        L2 = np.concatenate([L2, Label])
    elif nClass2 > nClass:
        L1 = np.concatenate([L1, Label2])
        L2 = np.concatenate([L2, Label2])

    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == Label[i]) & (L2 == Label[j]))
    sumG = np.sum(G)

    P1 = np.sum(G, axis=1) / sumG
    P2 = np.sum(G, axis=0) / sumG

    if np.sum(P1 == 0) > 0 or np.sum(P2 == 0) > 0:
        raise ValueError('Smooth fail!')
    else:
        H1 = np.sum(-P1 * np.log2(P1))
        H2 = np.sum(-P2 * np.log2(P2))
        P12 = G / sumG
        PPP = P12 / P2[:, None] / P1[None, :]
        PPP[np.abs(PPP) < 1e-12] = 1
        MI = np.sum(P12 * np.log2(PPP))
        MIhat = MI / max(H1, H2)
        MIhat = np.real(MIhat)

    return MIhat


def hungarian(A):
    m, n = A.shape
    if m != n:
        raise ValueError('HUNGARIAN: Cost matrix must be square!')
    # Save original cost matrix.
    orig = A.copy()
    # Reduce matrix.
    A = hminired(A)
    # Do an initial assignment.
    A, C, U = hminiass(A)
    # Repeat while we have unassigned rows.
    while U[n]:
        # Start with no path, no unchecked zeros, and no unexplored rows.
        LR = np.zeros(n)
        LC = np.zeros(n)
        CH = np.zeros(n)
        RH = np.concatenate([np.zeros(n), -1])
        SLC = []
        r = int(U[n])
        LR[r] = -1
        SLR = [r]
        while True:
            # If there are free zeros in row r
            if A[r, n] != 0:
                # ...get column of first free zero.
                l = int(-A[r, n])
                # yet marked as unexplored..
                if A[r, l] != 0 and not (0 and RH[r] == 0):
                    # Insert row r first in unexplored list.
                    RH[r] = RH[n]
                    RH[n] = r
                    # is.
                    CH[r] = -A[r, l]
            else:
                # If all rows are explored..
                if RH[n] <= 0:
                    # Reduce matrix.
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                # Re-start with first unexplored row.
                r = int(RH[n])
                l = int(CH[r])
                CH[r] = -A[r, l]
                if A[r, l] == 0:
                    # ...remove row r from unexplored list.
                    RH[n] = RH[r]
                    RH[r] = 0
            # While the column l is labelled, i.e. in path.
            while LC[l] != 0:
                # If row r is explored..
                if RH[r] == 0:
                    # If all rows are explored..
                    if RH[n] <= 0:
                        # Reduce cost matrix.
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                    # Re-start with first unexplored row.
                    r = int(RH[n])
                # Get column of next free zero in row r.
                l = int(CH[r])
                CH[r] = -A[r, l]
                if A[r, l] == 0:
                    # ...remove row r from unexplored list.
                    RH[n] = RH[r]
                    RH[r] = 0
            # If the column found is unassigned..
            if C[l] == 0:
                # Flip all zeros along the path in LR,LC.
                A, C, U = hmflip(A, C, LC, LR, U, l, r)
                break
            else:
                # ...else add zero to path.
                # Label column l with row r.
                LC[l] = r
                SLC.append(l)
                r = C[l]
                LR[r] = l
                SLR.append(r)

    # Calculate the total cost.
    T = np.sum(orig[C, np.arange(n)])
    return T

    

def hminired(A):
    m, n = A.shape
    # Subtract column-minimum values from each column.
    colMin = np.min(A, axis=0)
    A = A - np.tile(colMin, (m, 1))
    # Subtract row-minimum values from each row.
    rowMin = np.min(A, axis=1)
    A = A - np.tile(rowMin, (n, 1)).T
    # Get positions of all zeros.
    i, j = np.where(A == 0)
    # Extend A to give room for row zero list header column.
    A = np.vstack((A, np.zeros((1, n))))
    # Add row zero list header column.
    A[:, n] = 0
    for k in range(n):
        # Get all columns in this row.
        cols = j[i == k]
        A[k, [n] + cols.tolist()] = [-cols.size, 0]
    return A
    
    

def hminiass(A):
        n, np1 = A.shape
        # Initialize return vectors.
        C = np.zeros(n)
        U = np.zeros(n + 1)
        # Initialize last/next zero "pointers".
        LZ = np.zeros(n)
        NZ = np.zeros(n)
        for i in range(n):
            # Set j to first unassigned zero in row i.
            lj = n + 1
            j = int(-A[i, lj])
            # in an unassigned column (C[j]==0).
            while C[j] != 0:
                # Advance lj and j in zero list.
                lj = j
                j = int(-A[i, lj])
                if j == 0:
                    break
            if j != 0:
                # We found a zero in an unassigned column.
                # Assign row i to column j.
                C[j] = i
                A[i, lj] = A[i, j]
                NZ[i] = -A[i, j]
                LZ[i] = lj
                A[i, j] = 0
            else:
                # We found no zero in an unassigned column.
                # Check all zeros in this row.
                lj = n + 1
                j = int(-A[i, lj])
                while j != 0:
                    # Check the row assigned to this column.
                    r = int(C[j])
                    lm = int(LZ[r])
                    m = int(NZ[r])
                    while m != 0:
                        # Stop if we find an unassigned column.
                        if C[m] == 0:
                            break
                        # Advance one step in list.
                        lm = m
                        m = int(-A[r, lm])
                    if m == 0:
                        # We failed on row r. Continue with next zero on row i.
                        lj = j
                        j = int(-A[i, lj])
                    else:
                        # We found a zero in an unassigned column.
                        # Replace zero at (r, m) in unassigned list with zero at (r, j)
                        A[r, lm] = -j
                        A[r, j] = A[r, m]
                        NZ[r] = -A[r, m]
                        LZ[r] = j
                        A[r, m] = 0
                        C[m] = r
                        A[i, lj] = A[i, j]
                        NZ[i] = -A[i, j]
                        LZ[i] = lj
                        A[i, j] = 0
                        C[j] = i
                        break
        # Create vector with list of unassigned rows.
        # Mark all rows have assignment.
        r = np.zeros(n)
        rows = C[C != 0]
        r[rows] = rows
        empty = np.where(r == 0)[0]
        # Create vector with linked list of unassigned rows.
        U[np.concatenate([n + 1, empty])] = np.concatenate([empty, 0])
        return A, C, U



def hmflip(A=None,C=None,LC=None,LR=None,U=None,l=None,r=None):

    #HMFLIP Flip assignment state of all zeros along a path.
    
    #[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
#Input:
#A   - the cost matrix.
#C   - the assignment vector.
#LC  - the column label vector.
#LR  - the row label vector.
#U   - the 
#r,l - position of last zero in path.
#Output:
#A   - updated cost matrix.
#C   - updated assignment vector.
#U   - updated unassigned row list vector.
    
    # v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
    
    m,n=A.shape
    while (1):
        # Move assignment in column l to row r.
        C[l]=r
        # Find zero before this.
        m=np.where(A(r,np.arange()) == - l)
        A[r,m]=A(r,l)
        A[r,l]=0
        if (LR(r) < 0):
            U[n + 1]=U(r)
            U[r]=0

            return A,C,U
        else:
            # Move back in this row along the path and get column of next zero.
            l=LR(r)

            A[r,l]=A(r,n + 1)

            A[r,n + 1]=- l

            r=LC(l)


    

def hmreduce(A,CH,RH,LC,LR,SLC,SLR):
    #HMREDUCE Reduce parts of cost matrix in the Hungerian method.
    
    #[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
#Input:
#A   - Cost matrix.
#CH  - vector of column of 'next zeros' in each row.
#RH  - vector with list of unexplored rows.
#LC  - column labels.
#RC  - row labels.
#SLC - set of column labels.
#SLR - set of row labels.
    
    #Output:
#A   - Reduced cost matrix.
#CH  - Updated vector of 'next zeros' in each row.
#RH  - Updated vector of unexplored rows.
    
    # v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    m,n=A.shape
    # Find which rows are covered, i.e. unlabelled.
    coveredRows=LR == 0
    # Find which columns are covered, i.e. labelled.
    coveredCols=LC != 0
    r = np.where(~coveredRows)[0]
    c = np.where(~coveredCols)[0]
    # Get minimum of uncovered elements.
    m = np.min(A[r[:, np.newaxis], c])

    # Subtract minimum from all uncovered elements.
    A[r[:, np.newaxis], c] -= m
    # Check all uncovered columns..
    for j in c:
        # ...and uncovered rows in path order..
        for i in SLR:
            # If this is a (new) zero..
            if (A[i,j] == 0):
                # If the row is not in unexplored list..
                if (RH[i] == 0):
                    # ...insert it first in unexplored list.
                    RH[i]=RH[n]
                    RH[n]=i
                    CH[i]=j
                # Find last unassigned zero on row I.
                row=A[i,:]
                colsInList=- row[row < 0]
                if (len(colsInList) == 0):
                    # No zeros in the list.
                    l=n
                else:
                    l=colsInList[row[colsInList] == 0]
                # Append this zero to end of list.
                A[i,l]=- j

    # Add minimum to all doubly covered elements.
    r=np.where(coveredRows)[0]
    c=np.where(coveredCols)[0]
    # Take care of the zeros we will remove.
    i, j = np.where(A[r[:, np.newaxis], c] <= 0)
    i=r(i)
    j=c(j)
    for k in range(len(i)):
        # Find zero before this in this row.
        lj = np.where(A[i[k], :] == -j[k])[0]
        A[i[k], lj] = A[i[k], j[k]]
        A[i[k], j[k]] = 0
    A[r[:, np.newaxis], c] += m
    return A,CH,RH