import pandas as pd
import numpy as np

def adj_list_to_matrix(filename, matrix_len):
    import scipy.sparse as sp
    metapath = pd.read_csv(filename,sep='\t',header=None)
    metapath = metapath.fillna(0).astype(int)
    # Create an empty adjacency matrix
    adj_matrix_size = pd.DataFrame(metapath).max().max()
    adj_matrix = np.zeros((matrix_len, matrix_len))
    # Fill in the adjacency matrix
    for i in metapath[0]:
        for j in metapath.loc[i,1:]:
                adj_matrix[i][j] = 1
    adj_matrix = sp.csr_matrix(adj_matrix, dtype=np.int64)
    return adj_matrix

def constrct_data(path1, path2):
    features = pd.read_csv(path1, header = None)
    features = torch.tensor(features.iloc[:,1:].values.tolist())  
    adj_matrix = adj_list_to_matrix(path2,len(features))
    return adj_matrix, features

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    

import torch
import torch.nn.functional as F
from src.utils import *
from src.model import *
from src.train import *

dataset = 'B-Dataset'

# one-hot enconding
AllNode = pd.read_csv('./data/'+dataset+'/Allnode.csv',index_col='index')
node_label = torch.tensor(AllNode.index.values.tolist())
features = F.one_hot(node_label, num_classes=len(node_label)).float()
print(features.shape)


if dataset=='B-Dataset':
    Attr_path = './data/'+dataset+'/AllNodeAttribute.csv'
else:
    Attr_path = './data/'+dataset+'/AllNodeAttribute_ChemS_PhS.csv'


for num in range(3):
    adj_matrix = pd.read_csv('./data/'+dataset+'/Generating_subgraphs_adj_matrix_'+str(num+1)+'.csv',header=None)
    adj_matrix = sp.csr_matrix(adj_matrix, dtype=np.int64)
    adj = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

    # set parameter
    class item:
        def __init__(self):
            self.epochs = 200
            self.lr = 1e-1
            self.k1 = 200  
            self.k2 = 10
            self.epsilon1 = 0.03 
            self.epsilon2 = 0.05
            self.hidden = 128# 64
            self.dropout = 0.5
            self.runs = 1

    args = item()

    node_sum = adj.shape[0]
    edge_sum = adj.sum()/2
    row_sum = (adj.sum(1) + 1)
    norm_a_inf = row_sum/ (2*edge_sum+node_sum)

    adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))

    features = F.normalize(features, p=1)
    feature_list = []
    feature_list.append(features)
    for i in range(1, args.k1):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

    norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)
    norm_fea_inf = torch.mm(norm_a_inf, features)

    hops = torch.Tensor([0]*(adj.shape[0]))
    mask_before = torch.Tensor([False]*(adj.shape[0])).bool()

    for i in range(args.k1):
        dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
        mask = (dist<args.epsilon1).masked_fill_(mask_before, False)
        mask_before.masked_fill_(mask, True)
        hops.masked_fill_(mask, i)
    mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
    mask_final.masked_fill_(mask_before, False)
    hops.masked_fill_(mask_final, args.k1-1)
    print("Local Smoothing Iteration calculation is done.")

    input_feature = aver(hops, adj, feature_list)

    print("Local Smoothing is done.")

    Emdebding_GCN = pd.DataFrame(input_feature.detach().cpu().numpy())
    Emdebding_GCN.to_csv('./data/'+dataset+'/Emdebding_GCN_input_ChatGPT_'+str(num+1)+'.csv', header=None,index=False)
    
from src.fusion import Attention

dataset = 'B-Dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = pd.read_csv('./data/'+dataset+'/Emdebding_GCN_input_ChatGPT_1.csv',header=None)
B = pd.read_csv('./data/'+dataset+'/Emdebding_GCN_input_ChatGPT_2.csv',header=None)
C = pd.read_csv('./data/'+dataset+'/Emdebding_GCN_input_ChatGPT_3.csv',header=None)

A = torch.tensor(A.values).to(device)
B = torch.tensor(B.values).to(device)
C = torch.tensor(C.values).to(device)
attention_embedding = torch.stack((A, B, C), dim=0).type(torch.FloatTensor).to(device)
print(attention_embedding.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
attention = Attention(1,metapath_embeddings.shape[1],64)
attention = attention.to(device)
attention_embedding = attention(metapath_embeddings)
pd.DataFrame(attention_embedding.detach().cpu().numpy()).to_csv('./data/'+dataset+'/Embedding_Attention_ChatGPT_.csv',header=0,index=0)

dataset = 'B-Dataset'
Positive = pd.read_csv('./data/'+dataset+'/DrDiNum.csv', header = None)# DrDiNum.csv
Negative = pd.read_csv('./data/'+dataset+'/AllNegativeSample.csv', header = None)#NegativeNum  AllNegativeSample
Negative = Negative.sample(n=len(Positive), random_state=520,replace=True)
Attribute = pd.read_csv('./data/'+dataset+'/Embedding_Attention_ChatGPT_.csv', header = None)
Positive[2] = Positive.apply(lambda x:1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x:0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X = pd.concat([Attribute.loc[result[0].values.tolist()].reset_index(drop=True),Attribute.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = result[2]
print(X.shape)

from xgboost import XGBClassifier
k_fold = 10
print("%d fold CV"% k_fold)
i=0

AllResult = []

skf = StratifiedKFold(n_splits=k_fold,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
  
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = XGBClassifier(n_estimators=999,n_jobs=-1)#subsample=0.8
    model.fit(np.array(X_train), np.array(Y_train))
    y_score = model.predict_proba(np.array(X_test))
    RandomF_data = pd.DataFrame(np.vstack([Y_test, y_score[:,1]]).T)
    RandomF_data.to_csv('./results/'+dataset+'/predict/XGBoost_'+ str(i)+ 'Prob.csv', header = False, index = False)
    fpr,tpr,thresholds=roc_curve(Y_test,y_score[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    average_precision = average_precision_score(Y_test, y_score[:,1])
    print("---------------------------------------------")
    print("AUC=%0.4f, AUPR=%0.4f"%(roc_auc,average_precision))
    print("---------------------------------------------\n")
    i+=1
 