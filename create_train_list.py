import os

with open('./list/train.list', 'r') as f:
    lines = f.readlines()
    for line in lines:
        dirname = line.split()[0]
        if not os.path.exists(dirname + '.npy'):
            print(dirname)