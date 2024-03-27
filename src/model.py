import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
# import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.layers import Embedding, Dense, Dropout

class DNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DNN, self).__init__()
        self.fcn1 = nn.Linear(nfeat, nhid)
        self.fcn2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn1(x)
        x1=x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        return F.log_softmax(x, dim=1), x

# class Attention_layer(Layer):
    # '''
    # input shape:  [None, n, k]
    # output shape: [None, k]
    # '''
    # def __init__(self):
        # super().__init__()

    # def build(self, input_shape): # [None, field, k]
        # self.attention_w = Dense(input_shape[1], activation='relu')
        # self.attention_h = Dense(1, activation=None)

    # def call(self, inputs, **kwargs): # [None, field, k]
        # if K.ndim(inputs) != 3:
           # raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        # print(K.ndim(inputs))
        # x = self.attention_w(inputs)  # [None, field, field]
        # x = self.attention_h(x)       # [None, field, 1]
        # a_score = tf.nn.softmax(x)
        # a_score = tf.transpose(a_score, [0, 2, 1]) # [None, 1, field]
        # output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # (None, k)
        # return output
        
# class GraphAttentionLayer(nn.Module):
    # """
    # Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    # 图注意力层
    # """
    # def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        # super(GraphAttentionLayer, self).__init__()
        # self.in_features = in_features   # 节点表示向量的输入特征维度
        # self.out_features = out_features   # 节点表示向量的输出特征维度
        # self.dropout = dropout    # dropout参数
        # self.alpha = alpha     # leakyrelu激活的参数
        # self.concat = concat   # 如果为true, 再进行elu激活
        
        # # 定义可训练参数，即论文中的W和a
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # # 定义leakyrelu激活函数
        # self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    # def forward(self, inp, adj):
        # """
        # inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        # adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        # """
        # h = torch.mm(inp, self.W)   # [N, out_features]
        # N = h.size()[0]    # N 图的节点数
        
        # a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # # [N, N, 2*out_features]
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        # zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        # attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        # attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        # attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        # h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # # 得到由周围节点通过注意力权重进行更新的表示
        # if self.concat:
            # return F.elu(h_prime)
        # else:
            # return h_prime 
    
    # def __repr__(self):
        # return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'