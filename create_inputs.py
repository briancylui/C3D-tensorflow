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
np.save(features, 'train_features.npy')

feature_list = []
with open('./list/test.list', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        dirname = line.split()[0]
        video_features = np.load(dirname + '.npy')
        feature_list.append(video_features)

features = np.concatenate(feature_list)
np.save(features, 'test_features.npy')