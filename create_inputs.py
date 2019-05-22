import os
import numpy as np
from tqdm import tqdm

feature_list = []
with open('./list/train.list', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        dirname = line.split()[0]
        video_features = np.load(dirname + '.npy')
        feature_list.append(video_features)

features = np.concatenate(feature_list)
np.save('train_features.npy', features)

feature_list = []
with open('./list/test.list', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        dirname = line.split()[0]
        video_features = np.load(dirname + '.npy')
        feature_list.append(video_features)

features = np.concatenate(feature_list)
np.save('test_features.npy', features)