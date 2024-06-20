# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import networkx as nx
from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.gcn_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, adj_matrix, node_features):
        gcn_output = torch.matmul(adj_matrix, self.gcn_layer(node_features))
        return gcn_output


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, input_seq):
        lstm_output, _ = self.lstm(input_seq)
        return lstm_output


fmp = {'比利时': 0, '法国': 1, '希腊': 2, '意大利': 3, '西班牙': 4, '土耳其': 5, '日本': 6, '韩国': 7, '中国台湾': 8,
       '美国': 9, '特立尼达和多巴哥': 10, '阿曼': 11, '卡塔尔': 12, '阿联酋': 13, '阿尔及利亚': 14, '利比亚': 15,
       '尼日利亚': 16, '澳大利亚': 17, '文莱': 18, '印度尼西亚': 19, '马来西亚': 20, '葡萄牙': 21, '印度': 22,
       '英国': 23, '埃及': 24, '墨西哥': 25, '中国': 26, '挪威': 27, '阿根廷': 28, '加拿大': 29, '巴西': 30, '智利': 31,
       '乌克兰': 32, '科威特': 33, '泰国': 34, '俄罗斯联邦': 35, '也门': 36, '缅甸': 37, '秘鲁': 38, '新加坡': 39,
       '巴基斯坦': 40}
mp = {0: '比利时', 1: '法国', 2: '希腊', 3: '意大利', 4: '西班牙', 5: '土耳其', 6: '日本', 7: '韩国', 8: '中国台湾',
      9: '美国', 10: '特立尼达和多巴哥', 11: '阿曼', 12: '卡塔尔', 13: '阿联酋', 14: '阿尔及利亚', 15: '利比亚',
      16: '尼日利亚', 17: '澳大利亚', 18: '文莱', 19: '印度尼西亚', 20: '马来西亚', 21: '葡萄牙', 22: '印度',
      23: '英国', 24: '埃及', 25: '墨西哥', 26: '中国', 27: '挪威', 28: '阿根廷', 29: '加拿大', 30: '巴西', 31: '智利',
      32: '乌克兰', 33: '科威特', 34: '泰国', 35: '俄罗斯联邦', 36: '也门', 37: '缅甸', 38: '秘鲁', 39: '新加坡',
      40: '巴基斯坦'}
nfmp = {'北美洲': 0, '美国': 1, '特立尼达和多巴哥': 2, '阿曼': 3, '卡塔尔': 4, '阿联酋': 5, '阿尔及利亚': 6,
        '利比亚': 7,
        '尼日利亚': 8, '澳大利亚': 9, '文莱': 10, '印度尼西亚': 11, '马来西亚': 12, '比利时': 13, '法国': 14,
        '希腊': 15,
        '意大利': 16, '西班牙': 17, '土耳其': 18, '欧洲': 19, '日本': 20, '韩国': 21, '中国台湾': 22, '亚太地区': 23,
        '中南美洲': 24, '葡萄牙': 25, '印度': 26, '埃及': 27, '英国': 28, '加拿大': 29, '秘鲁': 30, '挪威': 31,
        '俄罗斯联邦': 32, '也门': 33, '墨西哥': 34, '阿根廷': 35, '巴西': 36, '智利': 37, '科威特': 38, '中东地区': 39,
        '中国': 40, '巴基斯坦': 41, '新加坡': 42, '泰国': 43, '缅甸': 44, '乌克兰': 45, '非洲': 46}


df11 = pd.read_excel("Trade movements as LNG/2000.xlsx")
labels11 = list(df11.columns.values)
G1 = nx.DiGraph()

for i in range(0, 41):
    G1.add_node(i)

for i in range(df11.shape[0]):
    for j in range(df11.shape[1]):
        if j != 0 and float(df11.iloc[i, j]) > 0.0:
            G1.add_edge(fmp[df11.iloc[i, 0]], fmp[labels11[j]])

df21 = pd.read_excel("Trade movements as LNG/2001.xlsx")
labels21 = list(df21.columns.values)
G2 = nx.DiGraph()

for i in range(0, 41):
    G2.add_node(i)

for i in range(df21.shape[0]):
    for j in range(df21.shape[1]):
        if j != 0 and float(df21.iloc[i, j]) > 0.0:
            G2.add_edge(fmp[df21.iloc[i, 0]], fmp[labels21[j]])

df31 = pd.read_excel("Trade movements as LNG/2002.xlsx")
labels31 = list(df31.columns.values)
G3 = nx.DiGraph()

for i in range(0, 41):
    G3.add_node(i)

for i in range(df31.shape[0]):
    for j in range(df31.shape[1]):
        if j != 0 and float(df31.iloc[i, j]) > 0.0:
            G3.add_edge(fmp[df31.iloc[i, 0]], fmp[labels31[j]])

df41 = pd.read_excel("Trade movements as LNG/2003.xlsx")
labels41 = list(df41.columns.values)
G4 = nx.DiGraph()
for i in range(0, 41):
    G4.add_node(i)

for i in range(df41.shape[0]):
    for j in range(df41.shape[1]):
        if j != 0 and float(df41.iloc[i, j]) > 0.0:
            G4.add_edge(fmp[df41.iloc[i, 0]], fmp[labels41[j]])

df51 = pd.read_excel("Trade movements as LNG/2004.xlsx")
labels51 = list(df51.columns.values)
G5 = nx.DiGraph()
for i in range(0, 41):
    G5.add_node(i)

for i in range(df51.shape[0]):
    for j in range(df51.shape[1]):
        if j != 0 and float(df51.iloc[i, j]) > 0.0:
            G5.add_edge(fmp[df51.iloc[i, 0]], fmp[labels51[j]])

df61 = pd.read_excel("Trade movements as LNG/2005.xlsx")
labels61 = list(df61.columns.values)
G6 = nx.DiGraph()
for i in range(0, 41):
    G6.add_node(i)

for i in range(df61.shape[0]):
    for j in range(df61.shape[1]):
        if j != 0 and float(df61.iloc[i, j]) > 0.0:
            G6.add_edge(fmp[df61.iloc[i, 0]], fmp[labels61[j]])

df71 = pd.read_excel("Trade movements as LNG/2006.xlsx")
labels71 = list(df71.columns.values)
G7 = nx.DiGraph()
for i in range(0, 41):
    G7.add_node(i)

for i in range(df71.shape[0]):
    for j in range(df71.shape[1]):
        if j != 0 and float(df71.iloc[i, j]) > 0.0:
            G7.add_edge(fmp[df71.iloc[i, 0]], fmp[labels71[j]])

df81 = pd.read_excel("Trade movements as LNG/2007.xlsx")
labels81 = list(df81.columns.values)
G8 = nx.DiGraph()
for i in range(0, 41):
    G8.add_node(i)

for i in range(df81.shape[0]):
    for j in range(df81.shape[1]):
        if j != 0 and float(df81.iloc[i, j]) > 0.0:
            G8.add_edge(fmp[df81.iloc[i, 0]], fmp[labels81[j]])

df91 = pd.read_excel("Trade movements as LNG/2008.xlsx")
labels91 = list(df91.columns.values)
G9 = nx.DiGraph()
for i in range(0, 41):
    G9.add_node(i)

for i in range(df91.shape[0]):
    for j in range(df91.shape[1]):
        if j != 0 and float(df91.iloc[i, j]) > 0.0:
            G9.add_edge(fmp[df91.iloc[i, 0]], fmp[labels91[j]])

df101 = pd.read_excel("Trade movements as LNG/2009.xlsx")
labels101 = list(df101.columns.values)
G10 = nx.DiGraph()
for i in range(0, 41):
    G10.add_node(i)

for i in range(df101.shape[0]):
    for j in range(df101.shape[1]):
        if j != 0 and float(df101.iloc[i, j]) > 0.0:
            G10.add_edge(fmp[df101.iloc[i, 0]], fmp[labels101[j]])

df111 = pd.read_excel("Trade movements as LNG/2010.xlsx")
labels111 = list(df111.columns.values)
G11 = nx.DiGraph()
for i in range(0, 41):
    G11.add_node(i)
for i in range(df111.shape[0]):
    for j in range(df111.shape[1]):
        if j != 0 and float(df111.iloc[i, j]) > 0.0:
            G11.add_edge(fmp[df111.iloc[i, 0]], fmp[labels111[j]])

df121 = pd.read_excel("Trade movements as LNG/2011.xlsx")
labels121 = list(df121.columns.values)
G12 = nx.DiGraph()
for i in range(0, 41):
    G12.add_node(i)

for i in range(df121.shape[0]):
    for j in range(df121.shape[1]):
        if j != 0 and float(df121.iloc[i, j]) > 0.0:
            G12.add_edge(fmp[df121.iloc[i, 0]], fmp[labels121[j]])

df131 = pd.read_excel("Trade movements as LNG/2012.xlsx")
labels131 = list(df131.columns.values)
G13 = nx.DiGraph()
for i in range(0, 41):
    G13.add_node(i)

for i in range(df131.shape[0]):
    for j in range(df131.shape[1]):
        if j != 0 and float(df131.iloc[i, j]) > 0.0:
            G13.add_edge(fmp[df131.iloc[i, 0]], fmp[labels131[j]])

df141 = pd.read_excel("Trade movements as LNG/2013.xlsx")
labels141 = list(df141.columns.values)
G14 = nx.DiGraph()
for i in range(0, 41):
    G14.add_node(i)

for i in range(df141.shape[0]):
    for j in range(df141.shape[1]):
        if j != 0 and float(df141.iloc[i, j]) > 0.0:
            G14.add_edge(fmp[df141.iloc[i, 0]], fmp[labels141[j]])

df151 = pd.read_excel("Trade movements as LNG/2014.xlsx")
labels151 = list(df151.columns.values)
G15 = nx.DiGraph()
for i in range(0, 41):
    G15.add_node(i)

for i in range(df151.shape[0]):
    for j in range(df151.shape[1]):
        if j != 0 and float(df151.iloc[i, j]) > 0.0:
            G15.add_edge(fmp[df151.iloc[i, 0]], fmp[labels151[j]])
b1 = {}
b2 = {}
b3 = {}
b4 = {}
b5 = {}
b6 = {}
b7 = {}
b8 = {}

df161 = pd.read_excel("Trade movements as LNG/2015.xlsx")
labels161 = list(df161.columns.values)
G16 = nx.DiGraph()
for i in range(0, 41):
    G16.add_node(i)

for i in range(df161.shape[0]):
    for j in range(df161.shape[1]):
        if j != 0 and float(df161.iloc[i, j]) > 0.0:
            G16.add_edge(fmp[df161.iloc[i, 0]], fmp[labels161[j]])
            b1[(fmp[df161.iloc[i, 0]], fmp[labels161[j]])] = df161.iloc[i, j]

df171 = pd.read_excel("Trade movements as LNG/2016.xlsx")
labels171 = list(df171.columns.values)
G17 = nx.DiGraph()
for i in range(0, 41):
    G17.add_node(i)

for i in range(df171.shape[0]):
    for j in range(df171.shape[1]):
        if j != 0 and float(df171.iloc[i, j]) > 0.0:
            G17.add_edge(fmp[df171.iloc[i, 0]], fmp[labels171[j]])
            b2[(fmp[df171.iloc[i, 0]], fmp[labels171[j]])] = df171.iloc[i, j]

df181 = pd.read_excel("Trade movements as LNG/2017.xlsx")
labels181 = list(df181.columns.values)
G18 = nx.DiGraph()
for i in range(0, 41):
    G18.add_node(i)

for i in range(df181.shape[0]):
    for j in range(df181.shape[1]):
        if j != 0 and float(df181.iloc[i, j]) > 0.0:
            G18.add_edge(fmp[df181.iloc[i, 0]], fmp[labels181[j]])
            b3[(fmp[df181.iloc[i, 0]], fmp[labels181[j]])] = df181.iloc[i, j]

df191 = pd.read_excel("Trade movements as LNG/2018.xlsx")
labels191 = list(df191.columns.values)
G19 = nx.DiGraph()
for i in range(0, 41):
    G19.add_node(i)

for i in range(df191.shape[0]):
    for j in range(df191.shape[1]):
        if j != 0 and float(df191.iloc[i, j]) > 0.0:
            G19.add_edge(fmp[df191.iloc[i, 0]], fmp[labels191[j]])
            b4[(fmp[df191.iloc[i, 0]], fmp[labels191[j]])] = df191.iloc[i, j]

df201 = pd.read_excel("Trade movements as LNG/2019.xlsx")
labels201 = list(df201.columns.values)
G20 = nx.DiGraph()
for i in range(0, 41):
    G20.add_node(i)

for i in range(df201.shape[0]):
    for j in range(df201.shape[1]):
        if j != 0 and float(df201.iloc[i, j]) > 0.0:
            G20.add_edge(fmp[df201.iloc[i, 0]], fmp[labels201[j]])
            b5[(fmp[df201.iloc[i, 0]], fmp[labels201[j]])] = df201.iloc[i, j]

df211 = pd.read_excel("Trade movements as LNG/2020.xlsx")
labels211 = list(df211.columns.values)
G21 = nx.DiGraph()
for i in range(0, 41):
    G21.add_node(i)

for i in range(df211.shape[0]):
    for j in range(df211.shape[1]):
        if j != 0 and float(df211.iloc[i, j]) > 0.0:
            G21.add_edge(fmp[df211.iloc[i, 0]], fmp[labels211[j]])
            b6[(fmp[df211.iloc[i, 0]], fmp[labels211[j]])] = df211.iloc[i, j]

df221 = pd.read_excel("Trade movements as LNG/2021.xlsx")
labels221 = list(df221.columns.values)

G22 = nx.DiGraph()
for i in range(0, 41):
    G22.add_node(i)

for i in range(df221.shape[0]):
    for j in range(df221.shape[1]):
        if j != 0 and float(df221.iloc[i, j]) > 0.0:
            G22.add_edge(fmp[df221.iloc[i, 0]], fmp[labels221[j]])
            b7[(fmp[df221.iloc[i, 0]], fmp[labels221[j]])] = df221.iloc[i, j]

df231 = pd.read_excel("Trade movements as LNG/2022.xlsx")
labels231 = list(df231.columns.values)
G23 = nx.DiGraph()
for i in range(0, 41):
    G23.add_node(i)

for i in range(df231.shape[0]):
    for j in range(df231.shape[1]):
        if j != 0 and float(df231.iloc[i, j]) > 0.0:
            G23.add_edge(fmp[df231.iloc[i, 0]], fmp[labels231[j]])

a = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23]

rg = [[2, 17, 24, 24, 25, 27, 31, 31, 32],
      [2, 18, 23, 24, 24, 24, 26, 28, 30],
      [2, 6, 9, 12, 12, 14, 16, 16, 18],
      [2, 7, 9, 9, 11, 13, 13, 13, 13],
      [2, 7, 12, 12, 12, 12, 12, 14, 17],
      [2, 7, 9, 10, 10, 11, 16, 16, 20],
      [2, 2, 2, 2, 2, 3, 10, 12, 22],
      [2, 2, 2, 4, 4, 5, 5, 5, 5],
      [2, 3, 3, 6, 6, 6, 6, 6, 6],
      [2, 12, 16, 22, 22, 22, 23, 24, 29],
      [2, 2, 3, 3, 3, 5, 5, 6, 6],
      [2, 2, 2, 7, 11, 17, 18, 23, 28],
      [2, 8, 11, 11, 17, 17, 21, 23, 23],
      [2, 2, 2, 2, 2, 3, 4, 6, 23],
      [2, 2, 2, 2, 2, 6, 16, 19, 24],
      [2, 6, 12, 14, 17, 19, 19, 19, 21],
      [2, 10, 18, 21, 22, 22, 22, 24, 26],
      [2, 3, 3, 3, 3, 8, 14, 21, 28],
      [2, 7, 8, 17, 19, 22, 22, 23, 23],
      [2, 14, 22, 27, 30, 30, 30, 32, 32],
      [2, 5, 7, 14, 15, 18, 23, 23, 26],
      [2, 10, 14, 21, 24, 25, 25, 25, 27],
      [2, 7, 11, 16, 19, 19, 20, 20, 22],
      [2, 21, 24, 29, 29, 29, 29, 29, 31],
      [2, 11, 20, 21, 25, 25, 28, 29, 33],
      [2, 2, 9, 14, 16, 16, 16, 16, 20],
      [2, 2, 2, 2, 2, 2, 4, 10, 32],
      [2, 2, 6, 10, 13, 17, 19, 19, 25],
      [2, 9, 19, 26, 29, 30, 30, 32, 32],
      [2, 2, 3, 4, 4, 6, 11, 16, 33],
      [2, 10, 14, 14, 20, 20, 22, 22, 23],
      [2, 2, 4, 5, 7, 12, 15, 18, 24],
      [2, 6, 12, 26, 28, 29, 29, 29, 29],
      [2, 3, 3, 3, 3, 3, 3, 4, 6],
      [2, 2, 3, 5, 9, 15, 23, 25, 33],
      [2, 2, 2, 2, 2, 2, 4, 7, 29],
      [2, 11, 18, 21, 28, 28, 31, 31, 33],
      [2, 2, 2, 3, 3, 5, 7, 10, 20],
      [2, 5, 8, 10, 17, 18, 18, 20, 20],
      [2, 13, 16, 22, 26, 26, 26, 28, 32],
      [2, 22, 31, 32, 32, 32, 32, 32, 32],
      [2, 2, 4, 5, 8, 9, 13, 14, 31],
      [2, 6, 8, 9, 11, 11, 11, 11, 11],
      [2, 8, 12, 15, 20, 24, 25, 29, 30],
      [2, 2, 2, 2, 2, 4, 5, 5, 5],
      [2, 2, 2, 2, 4, 4, 6, 7, 30],
      [2, 9, 17, 26, 27, 27, 30, 30, 32]]

mx = [2, 22, 31, 32, 32, 32, 32, 32, 33]

ans = []
pos = pd.read_excel("xlabel.xlsx")
for epoch in range(0, 10):
    for i in range(2, 3):
        for j in range(0, 23):
            for k in range(0, 41):
                for l in range(0, mx[i]):
                    a[j].nodes[k][pos.iloc[l, 0]] = [0]

        for key, value in fmp.items():
            s = "country/" + key + ".xlsx"
            df = pd.read_excel(s)
            for j in range(0, rg[nfmp[key]][i]):
                for k in range(0, 24):
                    if k != 0:
                        a[k - 1].nodes[value][df.iloc[j, 0]] = [df.iloc[j, k]]

        # 数据准备
        num_nodes = 41
        input_dim = mx[i]
        hidden_dim = 64
        noise_dim = 20
        seq_length = 21
        feature_dim = 64
        # 创建模型实例
        gcn_model = GCN(num_nodes, input_dim, hidden_dim)
        lstm_model = LSTMModel(hidden_dim, hidden_dim)

        matrix_list = []
        for ii in range(0, 23):
            edge_index = torch.tensor(list(a[ii].edges)).t().contiguous()
            x = torch.tensor([a[ii].nodes[node]['feat1'] for node in a[ii].nodes], dtype=torch.float)
            for j in range(1, mx[i]):
                feat = torch.tensor([a[ii].nodes[node][pos.iloc[j, 0]] for node in a[ii].nodes],
                                    dtype=torch.float)
                x = torch.cat((x, feat), dim=1)  # 将特征连接在一起

            matrix_list.append(x.numpy())

        matrix_array = np.array(matrix_list)

        X = []
        y = []
        # 在时序上进行迭代训练
        for time_step in range(0, 22):
            # 获取当前时间步的数据
            input_data = torch.FloatTensor(matrix_array[time_step])
            # GCN前向传播
            adj_matrix = nx.adjacency_matrix(a[time_step]).todense()

            gcn_output = gcn_model(torch.FloatTensor(adj_matrix), input_data)

            # LSTM前向传播
            lstm_output = lstm_model(gcn_output.unsqueeze(0))  # 添加时间维度


            if time_step == 17:
                df = pd.read_excel("Trade movements as LNG/2018.xlsx")
                labels = list(df.columns.values)

                for ii in range(df.shape[0]):
                    for jj in range(df.shape[1]):
                        if jj != 0:
                            if df.iloc[ii, jj] > 0:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(1))
                            else:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(0))

            if time_step == 18:
                df = pd.read_excel("Trade movements as LNG/2019.xlsx")
                labels = list(df.columns.values)

                for ii in range(df.shape[0]):
                    for jj in range(df.shape[1]):
                        if jj != 0:
                            if df.iloc[ii, jj] > 0:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(1))
                            else:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(0))

            if time_step == 19:
                df = pd.read_excel("Trade movements as LNG/2020.xlsx")
                labels = list(df.columns.values)

                for ii in range(df.shape[0]):
                    for jj in range(df.shape[1]):
                        if jj != 0:
                            if df.iloc[ii, jj] > 0:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(1))
                            else:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(0))

            if time_step == 20:
                df = pd.read_excel("Trade movements as LNG/2021.xlsx")
                labels = list(df.columns.values)

                for ii in range(df.shape[0]):
                    for jj in range(df.shape[1]):
                        if jj != 0:
                            if df.iloc[ii, jj] > 0:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(1))
                            else:
                                X.append(lstm_output.squeeze(0)[fmp[df.iloc[ii, 0]]].detach().numpy() +
                                         lstm_output.squeeze(0)[fmp[labels[jj]]].detach().numpy())
                                y.append(torch.tensor(0))

            if time_step == 21:
                from sklearn.ensemble import GradientBoostingRegressor

                G_1 = 0
                G_2 = 0
                G_3 = 0
                G_4 = 0
                for n_estimators in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275,
                                     300]:
                    for learning_rate in [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
                                          1]:
                        for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            model1 = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                                               max_depth=max_depth)
                            model1.fit(X, y)
                            tp = 0
                            fp = 0
                            num_neg = 0
                            num_pos = 0

                            for ii in range(df231.shape[0]):
                                for jj in range(df231.shape[1]):
                                    if jj != 0:
                                        if df231.iloc[ii, jj] == 0:
                                            num_neg += 1
                                            if np.around(model1.predict(
                                                    [lstm_output.squeeze(0)[
                                                         fmp[df231.iloc[ii, 0]]].detach().numpy() +
                                                     lstm_output.squeeze(0)[
                                                         fmp[labels231[jj]]].detach().numpy()])) == 0:
                                                fp += 1
                                        else:
                                            num_pos += 1
                                            if np.around(model1.predict(
                                                    [lstm_output.squeeze(0)[
                                                         fmp[df231.iloc[ii, 0]]].detach().numpy() +
                                                     lstm_output.squeeze(0)[
                                                         fmp[labels231[jj]]].detach().numpy()])) != 0:
                                                tp += 1

                            if tp + fp > G_1:
                                G_1 = tp + fp
                                G_2 = num_pos + num_neg
                                G_3 = tp
                                G_4 = num_pos

                ans.append([G_4, G_2 - G_4, G_3, G_1 - G_3])
print(ans)

