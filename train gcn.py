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


def createadj(X):
    N, M = X.T.shape
    distances = euclidean_distances(X.T)
    # 根据距离构建邻接矩阵
    k = 5  # 设置k值为5，表示每个样本与其最近的5个样本相连
    adjacency_matrix = kneighbors_graph(X.T, k, mode='connectivity', include_self=True)
    # 将稀疏矩阵转换为密集矩阵
    adjacency_matrix = adjacency_matrix.toarray()
    return adjacency_matrix

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
    gcn_model0 = GCN0(feat_dims[0], nhid1=1500,nhid2=128,nclass=numclass)
    gcn_model1=GCN1(feat_dims[1],nhid1=1500,nhid2=128,nclass=numclass)
    gcn_model2=GCN2(feat_dims[2],nhid1=1500,nhid2=128,nclass=numclass)
    criterion=MultiCrossEntropyLoss()
    optimizer0 = Adam(gcn_model0.parameters(), lr=0.01)
    optimizer1 = Adam(gcn_model1.parameters(), lr=0.01)
    optimizer2 = Adam(gcn_model2.parameters(), lr=0.01)
    gcn_model0.train()
    loss=0
    output=[None]*numview
    for j in range(numview):
        for i in range(len(X[j][0])):
            if X[j][0][i].dtype != np.double:
                X[j][0][i] = X[j][0][i].astype(np.double)
    gcn_model1.train()
    for epoch in range(epochs):
        X_tensors = [torch.from_numpy(arr) for sublist in X for arr in sublist]
        output[0] = gcn_model0(X_tensors[0], adj_tensors)  # numsample*d 相当与AZ
        loss0 = criterion(output[0], Y_tensor)
        #loss=criterion(output[0], Y_tensor) + criterion(output[2], Y_tensor)
        optimizer0.zero_grad()
        loss0.backward()
        optimizer0.step()
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[0].data.cpu().numpy() )
    ACC, nmi, Purity, Fscore, Precision, Recall, ARI = Clustering8Measure(Y, y_pred)
    print('Kmeans result')
    print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI)

    gcn_model1.train()
    for epoch in range(3*epochs):
        output[1] = gcn_model1(X_tensors[1], adj_tensors)  # numsample*d 相当与AZ
        loss1 = criterion(output[1], Y_tensor)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[1].data.cpu().numpy())
    ACC,nmi,Purity,Fscore,Precision,Recall,ARI=Clustering8Measure(Y,y_pred)
    print('Kmeans result')
    print(ACC, nmi, Purity, Fscore, Precision, Recall, ARI)

    gcn_model2.train()
    for epoch in range(epochs):
        output[2] = gcn_model2(X_tensors[2], adj_tensors)  # numsample*d 相当与AZ
        loss2 = criterion(output[2], Y_tensor)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(output[2].data.cpu().numpy())
    ACC, nmi, Purity, Fscore, Precision, Recall, ARI = Clustering8Measure(Y, y_pred)

    print('Kmeans result')
    print(ACC,nmi,Purity,Fscore,Precision,Recall,ARI)
    print("train gcn")

def algo_qp(X, Y, d, numanchor):
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
    gcn_model0 = GCN0(feat_dims[0], nhid1=1500,nhid2=128, nclass=numclass)
    gcn_model1 = GCN1(feat_dims[1], nhid1=1500, nhid2=128, nclass=numclass)
    gcn_model2 = GCN2(feat_dims[2], nhid1=1500, nhid2=128, nclass=numclass)

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
    train_gcn(X,Y,d)
