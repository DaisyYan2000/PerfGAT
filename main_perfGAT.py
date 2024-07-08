import os.path

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import dgl
import torch
import torch.nn as nn
import torch.optim as optim

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, balanced_accuracy_score
from dgl.dataloading import GraphDataLoader

from utils import GraphImageData
from train import cRT_training

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Now device using: {device}')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = f'./log_{current_date}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
sys.stdout = open(f'{log_dir}/output_joint.txt', 'w')

# load dataset
graph_dir = '../merged_data_csv1'
img_dir = '../PWI_tumour_cropped'
clinic_dir = './IDH_mask_avail_clinic.csv'
img_size = (54, 54, 54)
time_len = 45
dataset = GraphImageData(graph_dir=graph_dir, imgs_dir=img_dir, clinic_dir=clinic_dir,
                         img_size=img_size, time_len=time_len)
print(f"\tLoaded Dataset from: graphs '{graph_dir}' & images '{img_dir}' of size: {len(dataset)} ...\n")

gnn_model = 'PerfGAT'
train_mode = 'joint'
aug = True
for node_hidden_dim in [16, 32, 64]:
    for edge_hidden_dim in [8, 16, 32]:
        if edge_hidden_dim > node_hidden_dim:
            continue
        result_dir = f'{log_dir}/{gnn_model}_edge/{train_mode}/aug_{aug}/feat_{node_hidden_dim}_{edge_hidden_dim}/'
        cRT_training(dataset=dataset, gnn_name=gnn_model, clf_name='dot', batch_size=1, seed=seed, train_mode=train_mode, patience=20,
                    node_hidden_dim=node_hidden_dim, edge_hidden_dim=edge_hidden_dim, result_dir=result_dir, edge_feat=True)

sys.stdout.close()
