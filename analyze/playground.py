import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from os import listdir
from os.path import isfile, join
sys.path.append('E:/Projects/accelerometer-project/assessments2/extensions')
import statistical_extensions
import pickle
import csv


with open(r"correlation_dict_MET.pickle", "rb") as input_file:
    e = pickle.load(input_file)

keys_remove = []
for key, value in e.items():
    if len(value) == 0:
        keys_remove.append(key)

for remove_key in keys_remove:
    e.pop(remove_key, 0)

with open("correlation_dict_MET.csv", "w") as outfile:
   writer = csv.writer(outfile)
   writer.writerow(e.keys())
   writer.writerows(zip(*e.values()))