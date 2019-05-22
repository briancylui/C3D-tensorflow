import os
import numpy as np

feature_list = []
with open('./list/train.list', 'r') as f:
    lines = f.readlines()
    for line in lines:
        dirname = line.split()[0]
        video_features = np.load(dirname + '.npy')
        feature_list.append(video_features)

features = np.concatenate(feature_list)
np.save('features.npy')