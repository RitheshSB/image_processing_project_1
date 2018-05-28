# =============================================================================
# Installing Packages
# =============================================================================
import os
import pandas as pd 
import matplotlib.pyplot as plt
import imageio
import numpy as np
# =============================================================================
# Setting the working directory
# =============================================================================
os.getcwd()
os.chdir(r'C:\Users\Rithesh\Documents\GitHub\Age Prediction from Images')

# =============================================================================
# Importing the required files
# =============================================================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

class_dummies = pd.get_dummies(train['Class'])
class_dummies.columns = 'Class_'+ class_dummies.columns

train = pd.merge(train,class_dummies,left_index=True,right_index=True).drop(['Class'],axis = 1)

train_dir = os.getcwd()+'//Train'
test_dir = os.getcwd()+'//Test'


train_dict = {}
for fname in os.listdir(train_dir):
    train_dict[fname] = np.asarray(imageio.imread(train_dir + '\\'+fname))
    print(train_dict[fname].shape)

