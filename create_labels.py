import os
import numpy as np
from tqdm import tqdm

label_list = []
with open('./list/train.list', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        dirname = line.split()[0]
        label = int(line.split()[1])
        label_list.append(label)

labels = np.array(label_list)
np.save('train_labels.npy', labels)

label_list = []
with open('./list/test.list', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        dirname = line.split()[0]
        label = int(line.split()[1])
        label_list.append(label)

labels = np.array(label_list)
np.save('test_labels.npy', labels)