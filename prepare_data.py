import os
from shutil import copyfile

import numpy as np
import pandas as pd

data = pd.read_csv('images_labelling.csv')

print('found ' + str(len(pd.unique(data['label']))) + ' unique labels')

print('found ' + str(len(pd.unique(data['class_']))) + ' unique classes')

mask = np.random.rand(len(data)) < 0.8

train = data[mask]
val = data[~mask]

if not os.path.exists('images_train'):
    os.mkdir('images_train')
    print('Making training set')
    for index, row in train.iterrows():
        src = os.getcwd() + '\\images\\' + str(row['boxid']) + '.png'
        if not os.path.exists('images_train\\' + str(row['label'])):
            os.mkdir('images_train\\' + str(row['label']))
        dst = os.getcwd() + '\\images_train\\' + str(row['label']) + '\\' + str(row['boxid']) + '.png'
        copyfile(src, dst)

if not os.path.exists('images_val'):
    os.mkdir('images_val')
    print('Making val set')
    for index, row in val.iterrows():
        src = os.getcwd() + '\\images\\' + str(row['boxid']) + '.png'
        if not os.path.exists('images_val\\' + str(row['label'])):
            os.mkdir('images_val\\' + str(row['label']))
        dst = os.getcwd() + '\\images_val\\' + str(row['label']) + '\\' + str(row['boxid']) + '.png'
        copyfile(src, dst)

print('Done')
