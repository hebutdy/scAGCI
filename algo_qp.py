# Generated with SMOP  0.41

import numpy as np
from sklearn.preprocessing import StandardScaler
from coclustering_bipartite_fast1 import coclustering_bipartite_fast1
from Clustering8Measure import Clustering8Measure
import warnings
from GCN_test1 import GCN0,GCN1,GCN2
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from torch.optim import Adam
import torch.nn as nn
from sklearn.cluster import KMeans
from GCN_test1 import GAT

class MultiCrossEntropyLoss():
    def __init__(self):
        self.predicts=None
        self.labels=None
        self.num=None
    def __call__(self, predicts,labels):
        return self.forward(predicts,labels)
    def forward(self,predicts,labels):
        self.predicts=predicts
        self.labels=labels
        self.num=self.predicts.shape[0]
        loss=0

        for i in range(0,self.num):
            index=int(self.labels[i])
            log_preds = torch.nn.functional.log_softmax(self.predicts[i], dim=0)
            loss-=log_preds[index]
        return loss/self.num



def createadj(X):
    N, M = X.T.shape
    distances = euclidean_distances(X.T)
    # 根据距离构建邻接矩阵
    k = 5  # 设置k值为5，表示每个样本与其最近的5个样本相连
    adjacency_matrix = kneighbors_graph(X.T, k, mode='connectivity', include_self=True)
    # 将稀疏矩阵转换为密集矩阵
    adjacency_matrix = adjacency_matrix.toarray()
    return adjacency_matrix

def train_GAT(tensor,Z,d,Y):
    adj_common=createadj(Z)
    Y_tensor = torch.FloatTensor(Y)
    k = np.unique(Y).shape[0]
    GAT_common = GAT(d, 128, k, 0.3, 0.5, 1)  # 图注意力初始化
    epochs=200
    criterion = MultiCrossEntropyLoss()
    adj_tensors = torch.from_numpy(adj_common)
    optimizer = Adam(GAT_common.parameters(), lr=0.1)
    for epoch in range(epochs):
        common=GAT_common(tensor,adj_tensors)  #pbmc:2874*240   2874*2874
        loss=criterion(common, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return common








def train_gcn(X,Y,d):
    epochs=100
    numview = X.shape[0]
    adj = [None] * numview
    feat_dims = [None] * numview
    numsample = Y.shape[0]
    numclass = np.unique(Y).shape[0]
    Y_tensor=torch.FloatTensor(Y)
    for i in range(numview):
        di = X[i][0].shape[0]
        adj[i] = np.zeros((numsample, numsample))
        adj[i] = createadj(X[i][0])
        feat_dims[i] = di
    for j in range(numview):
        for i in range(len(adj[j])):
            if adj[j][i].dtype != np.double:
                adj[j][i] = adj[j][i].astype(np.double)
    adj_tensors = [torch.from_numpy(a) for a in adj]
    # 图卷积初始化
    gcn_model0 = GCN0(feat_dims[0], nhid1=500,nhid2=128,nclass=numclass)

    criterion=MultiCrossEntropyLoss()
    gcn_model0.train()
    loss=0
    output=[None]*numview
    for j in range(numview):
        for i in range(len(X[j][0])):
            if X[j][0][i].dtype != np.double:
                X[j][0][i] = X[j][0][i].astype(np.double)
    #loaded_model0 = GCN0(feat_dims[0], nhid1=1500,nhid2=128,nclass=numclass)  # 替换为您的GCN模型类和参数
    #loaded_model0.load_state_dict(torch.load('gcn_model02.pth'))
    #loaded_model0.eval()
    #gcn_model0 = GCN0(feat_dims[0], nhid1=1500, nhid2=128, nclass=numclass)
    optimizer0 = Adam(gcn_model0.parameters(), lr=0.1)
    for epoch in range(5):
        X_tensors = [torch.from_numpy(arr) for sublist in X for arr in sublist]
        output[0] = gcn_model0(X_tensors[0], adj_tensors)  # numsample*d 相当与AZ
        loss0 = criterion(output[0], Y_tensor)
        #loss=criterion(output[0], Y_tensor) + criterion(output[2], Y_tensor)
        optimizer0.zero_grad()
        loss0.backward()
        optimizer0.step()
        print(loss0)
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[0].data.cpu().numpy() )
    ACC, nmi, Purity, Fscore, Precision, Recall, ARI = Clustering8Measure(Y, y_pred)
    print('Kmeans result')
    print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI)
    #torch.save(gcn_model0.state_dict(), 'gcn_model0.pth')
    gcn_model1 = GCN1(feat_dims[1], nhid1=500, nhid2=128, nclass=numclass)
    #loaded_model1 = GCN1(feat_dims[1], nhid1=1500, nhid2=128, nclass=numclass)  # 替换为您的GCN模型类和参数
    #loaded_model1.load_state_dict(torch.load('gcn_model11.pth'))
    #loaded_model1.eval()
    optimizer1 = Adam(gcn_model1.parameters(), lr=0.1)
    #optimizer1 = Adam(gcn_model1.parameters(), lr=0.0001)
    for epoch in range(500):
        output[1] = gcn_model1(X_tensors[1], adj_tensors)  # numsample*d 相当与AZ
        loss1 = criterion(output[1], Y_tensor)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        print(loss1)
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[1].data.cpu().numpy())
    ACC,nmi,Purity,Fscore,Precision,Recall,ARI=Clustering8Measure(Y,y_pred)
    print('Kmeans result')
    print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI)
    torch.save(gcn_model1.state_dict(), 'gcn_model1.pth')
    gcn_model2 = GCN2(feat_dims[2], nhid1=500, nhid2=128, nclass=numclass)
    #loaded_model2 = GCN2(feat_dims[2], nhid1=1500, nhid2=128, nclass=numclass)  # 替换为您的GCN模型类和参数
    #loaded_model2.load_state_dict(torch.load('gcn_model21.pth'))
    #loaded_model2.eval()
    optimizer2 = Adam(gcn_model2.parameters(), lr=0.0001)
    for epoch in range(500):
        output[2] = gcn_model2(X_tensors[2], adj_tensors)  # numsample*d 相当与AZ
        loss2 = criterion(output[2], Y_tensor)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        print(loss2)
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[2].data.cpu().numpy())
    ACC, nmi, Purity, Fscore, Precision, Recall, ARI = Clustering8Measure(Y, y_pred)
    print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI)
    torch.save(gcn_model2.state_dict(), 'gcn_model2.pth')




def algo_qp(X,Y,d,numanchor):
    maxIter=100
    IterMax=100
    #参数定义
    m=numanchor
    numclass=np.unique(Y).shape[0]  #c
    numview=X.shape[0]   #视图数量
    numsample =Y.shape[0]   #样本数
    adj = [None] * numview
    W=[None]*numview    #每个锚点图的权重矩阵W[i]  #di*d
    A=np.zeros((d,m))   #d,m
    Z=np.eye(m,numsample)  #m,n 锚点与原始cell关系矩阵
    feat_dims = [None] * numview
    Y_tensor = torch.FloatTensor(Y)
    for i in range(numview):
        di = X[i][0].shape[0]
        adj[i] = np.zeros((numsample, numsample))
        adj[i] = createadj(X[i][0])
        feat_dims[i] = di
    for j in range(numview):
        for i in range(len(adj[j])):
            if adj[j][i].dtype != np.double:
                adj[j][i] = adj[j][i].astype(np.double)
    adj_tensors = [torch.from_numpy(a) for a in adj]
    # 图卷积初始化

    for j in range(numview):
        for i in range(len(X[j][0])):
            if X[j][0][i].dtype != np.double:
                X[j][0][i] = X[j][0][i].astype(np.double)
    X_tensors = [torch.from_numpy(arr) for sublist in X for arr in sublist]
    for i in range(numview):
        di = X[i][0].shape[0]
        W[i] = np.zeros((di, d))
    alpha=(np.ones((1,numview)) / numview).reshape(-1,1)
    flag=1
    iter=0
    output=[None]*numview
    obj=[None]*(maxIter+1)
    #print("train start")
    #train_gcn(X,Y,d)
    print("train end")
    while flag:
        iter=iter + 1
        sumAlpha=0

        for ia in range(numview):
            al2=alpha[ia][0]**2
            sumAlpha=sumAlpha + al2
        # 使用图卷积网络将聚到节点，得到output  大小numsample*d
        loaded_model1 = GCN1(feat_dims[1], nhid1=1500, nhid2=128, nclass=numclass)  # 替换为您的GCN模型类和参数
        loaded_model1.load_state_dict(torch.load('./trained_model/cellmix/gcn_model11.pth'))
        loaded_model1.eval()
        loaded_model0 = GCN0(feat_dims[0], nhid1=1500, nhid2=128, nclass=numclass)  # 替换为您的GCN模型类和参数
        loaded_model0.load_state_dict(torch.load('./trained_model/cellmix/gcn_model02.pth'))
        loaded_model0.eval()
        loaded_model2 = GCN2(feat_dims[2], nhid1=1500, nhid2=128, nclass=numclass)  # 替换为您的GCN模型类和参数
        loaded_model2.load_state_dict(torch.load('./trained_model/cellmix/gcn_model2.pth'))
        loaded_model2.eval()
        with torch.no_grad():
            output[0] = loaded_model0(X_tensors[0], adj_tensors)
            output[1] = loaded_model1(X_tensors[1], adj_tensors)
            output[2] = loaded_model2(X_tensors[2], adj_tensors)

        part1 =   (output[0].T.detach().numpy() @ Z.T) + (
                    output[1].T.detach().numpy() @ Z.T) +  (
                            output[2].T.detach().numpy() @ Z.T)  # (numsample*d).T  (m*numsample).T
        if np.any(np.isinf(part1)) or np.any(np.isnan(part1)):
            part1 = np.nan_to_num(part1)
        Unew, __, Vnew = np.linalg.svd(part1, full_matrices=False)  # d*k k*m
        A = Unew @ Vnew  # d*m   #输出无误
        #更新Z
        B = np.zeros((numsample, m))  # n*m
        # 更 新 Z
        for iv in range(numview):
            B = B + (alpha[iv][0] ** 2) * output[iv].detach().numpy() @ A  # numsample*d  d*m
        G = np.zeros((numsample, m))  # n*m
        G[:m, :] = np.eye(m)
        label, _, P, _, _, term2 = coclustering_bipartite_fast1(B, G, numclass, sumAlpha, IterMax)  # n*m n*m  c
        Z = P.T  # m*n

        AZ = A @ Z  # d*m m*n

        # 更新W
        for iv in range(numview):
            C = 1 / numview * X[iv][0] @ AZ.T  # di*n   n*d
            U, s, V = np.linalg.svd(C, full_matrices=False)  # matlab的输出和python输出互为转置   di*k  k*d
            W[iv] = U @ V  # di d

        M=np.zeros((numview,1))
        for iv in range(numview):
            M[iv][0]=np.linalg.norm(X[iv][0] - W[iv]@A@Z,'fro')  #di*d  d*m  m*numsample
        Mfra=M ** - 1
        Q=1 / np.sum(Mfra)  #1*1
        alpha=Q*Mfra
        term1=0
        for iv in range(numview):
            term1=term1 +(alpha[iv][0] ** 2)*(np.linalg.norm(X[iv][0] -W[iv]@A@Z ,'fro'))**2
        obj[iter]=term1 + term2
        if (iter > 9) and  (abs((obj[iter - 1] - obj[iter]) / (obj[iter - 1])) < 0.000001):
            flag=0
        if (iter > 9) and (obj[iter] < 1e-10) :
            flag=0
        if (iter >= maxIter):
            flag=0

    return  A,W,Z,iter,obj,alpha,label,output,adj_tensors

    