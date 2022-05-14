import numpy as np
import pandas as pd
import torch
from torch import nn
import pickle
from tqdm import tqdm
import re
import json
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter

from pensmodule.UserEncoder.model import *
from pensmodule.UserEncoder.data import *
from pensmodule.UserEncoder.utils import *

import os
# os. environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0')

lr = 0.0001
batch_size=128

news_vert = np.load('../../data2/news_vert.npy')
news_title = np.load('../../data2/news_title.npy')
news_body = np.load('../../data2/news_body.npy')

with open('../../data2/TrainUsers.pkl', 'rb') as f:
    TrainUsers = pickle.load(f)
with open('../../data2/TrainSamples.pkl', 'rb') as f:
    TrainSamples = pickle.load(f)

with open('../../data2/dict.pkl', 'rb') as f:
    _, category_dict, word_dict = pickle.load(f)
with open('../../data2/news.pkl', 'rb') as f:
    news = pickle.load(f)
embedding_matrix = np.load('../../data2/embedding_matrix.npy')

torch.cuda.empty_cache()
model = NRMS(embedding_matrix)
model = model.to(device)

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot
optimizer = optim.Adam(model.parameters(), lr=0.0001) #lr = 0.0001

min_train_loss = 100.0
for ep in range(1,4):
    loss = 0.0
    accuary = 0.0
    cnt = 1
    dset = TrainDataset(TrainUsers, TrainSamples, news_title, news_vert, news_body)
    data_loader = DataLoader(dset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    tqdm_util = tqdm(data_loader)
    for user_feature, news_feature, label in tqdm_util:
        user_feature = [i.to(device) for i in user_feature]
        news_feature = [i.to(device) for i in news_feature]
        label = label.to(device)
        bz_loss, y_hat = model(user_feature, news_feature, label)
        loss += bz_loss.data.float()
        accuary += acc(label, y_hat)

        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()

        if cnt % 10 == 0:
            tqdm_util.set_description('ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(cnt * batch_size, loss.data / cnt, accuary / cnt))
        cnt += 1
    loss /= cnt
    print(ep, loss)
    torch.save(model.state_dict(), '../../runs/userencoder2/NAML-{}.pkl'.format(ep))


with open('../../data2/ValidUsers.pkl', 'rb') as f:
    ValidUsers = pickle.load(f)
with open('../../data2/ValidSamples.pkl', 'rb') as f:
    ValidSamples = pickle.load(f)

for ep in range(1, 4):
    model = NRMS(embedding_matrix)
    model = model.to(device)
    model.load_state_dict(torch.load('../../runs/userencoder2/NAML-{}.pkl'.format(ep)))
    model.eval()
    # save new embedding matrix
    np.save('../../data2/embedding_matrix2{}.npy'.format(ep), model.embed.weight.data.cpu().numpy())

    n_dset = news_dataset(news_title, news_vert, news_body)
    news_data_loader = DataLoader(n_dset, batch_size=512, collate_fn=news_collate_fn, shuffle=False)

    news_scoring = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for news_feature in tqdm(news_data_loader):
            news_feature = [i.to(device) for i in news_feature]
            news_vec = model.news_encoder(news_feature)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)
    news_scoring = np.array(news_scoring)
    np.save('../../data2/news_scoring2{}.npy'.format(ep), news_scoring)

    u_dset = UserDataset(news_scoring, ValidUsers)
    user_data_loader = DataLoader(u_dset, batch_size=128, shuffle=False)

    user_scoring = []
    with torch.no_grad():
        for user_feature in tqdm(user_data_loader):
            user_feature = user_feature.to(device)
            user_vec = model.user_encoder(user_feature)
            user_vec = user_vec.to(torch.device("cpu")).detach().numpy()
            user_scoring.extend(user_vec)
    user_scoring = np.array(user_scoring)
    np.save('../../data2/global_user_embed2{}.npy'.format(ep), np.mean(user_scoring, axis=0))
    g = evaluate(user_scoring, news_scoring, ValidSamples)
    print(ep)
    print('AUC\t', 'MRR\t', 'nDCG5\t', 'nDCG10\t', 'CTR1\t', 'CTR10\t')
    print(g)
