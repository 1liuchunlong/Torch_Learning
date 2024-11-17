import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model 词嵌入的维度 vocab 词表的大小
        super(Embeddings, self).__init__()
        # 定义Embedding 层
        self.lut = nn.Embedding(vocab, d_model)
        # 将参数传入类中
        self.d_model = d_model
    def forward(self, x):
        # x表示 输入进模型的文本 通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)

d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embeddings(d_model, vocab)
embres = emb(x)
print(f'embres :  {embres}')
print(f'shape == {embres.shape}')
