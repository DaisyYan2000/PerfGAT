import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, balanced_accuracy_score, \
    matthews_corrcoef, accuracy_score

import nibabel as nib



def my_transform(g, keep_node=True):
    
    device = g.device

    if not 'raw_ID' in g.ndata.keys():
        g.ndata['raw_ID'] = g.nodes()
    
    if 'temporal' in g.etypes:
        g_t = dgl.edge_type_subgraph(g, ['temporal'])
    else:
        g_t = g
    # if not keep_node:
    #     no_in_degree = torch.where(g_t.in_degrees()==0)[0]
    #     if g_t.nodes()[-1] in no_in_degree:
    #         if 'temporal' in g.etypes:
    #             g.add_edges(g.nodes()[-1], g.nodes()[-1], etype='temporal')
    #         else:
    #             g.add_edges(g.nodes()[-1], g.nodes()[-1])

    #     nodes_with_connections = torch.unique(torch.cat([g.edges()[0], g.edges()[1]])).to(device)
        
    #     all_nodes = g.nodes()
    #     nodes_without_connections = torch.tensor(list(set(all_nodes.cpu().numpy()) - set(nodes_with_connections.cpu().numpy())), device=device,
    #                                             dtype=torch.int64)
            
    #     g = dgl.transforms.remove_nodes(g, nodes_without_connections, store_ids=False)
        

    no_in_degree = torch.where(g_t.in_degrees()==0)[0]
    if no_in_degree.shape[0] != 0:
        if 'temporal' in g.etypes:
            g.add_edges(no_in_degree, no_in_degree, etype='temporal')
        else:
            g.add_edges(g.nodes()[-1], g.nodes()[-1])
        # if not 'raw_ID' in g.ndata.keys():
        #     g.ndata['raw_ID'] = torch.tensor(range(g.num_nodes())).to(device)

    assert 423 in g.ndata['raw_ID'], 'tumour node deleted!'

    # # Add a step to change the node feature length to 45 and interpolate
    # new_feature_length = 45
    # if g.ndata.get('feat') is not None:
    #     old_features = g.ndata['feat']
    #     if old_features.shape[1] != new_feature_length:

    #         # Linear interpolation
    #         new_features = F.interpolate(old_features.unsqueeze(0), new_feature_length, mode='linear').squeeze(0)

    #         g.ndata['feat'] = new_features

    return g
    

class GraphImageData(Dataset):

    def __init__(self, graph_dir, imgs_dir, clinic_dir, img_size, time_len, keep_node=False):
        self.graph_dir = graph_dir

        self.graph_dataset = dgl.data.CSVDataset(self.graph_dir, transform=lambda x: my_transform(x, keep_node))
        
        self.imgs_dir = imgs_dir
        self.clinic_dir = clinic_dir
        self.img_path = os.listdir(self.imgs_dir)
                
        self.clinic_data = pd.read_csv(self.clinic_dir)
        self.id = list(self.clinic_data['ID'])
        
        idh = list(self.clinic_data['IDH'])
        label_mapping = {'Wildtype': 0, 'Mutated': 1}
        numerical_labels = [label_mapping[label] for label in idh]

        self.idh = torch.tensor(numerical_labels, dtype=torch.float32)
        
        self.img_size = img_size
        self.time_len = time_len
    
    def target(self):
        return self.idh

    def __getitem__(self, idx):

        uni_img_size = torch.nn.Upsample(size=self.img_size)

        patient_id = self.id[idx]
        img_name = f'{patient_id}_cropped.nii.gz'
        img_item_path = os.path.join(self.imgs_dir, img_name)
        img = nib.load(img_item_path)   # (h, w, d, t)

        img_tensor = torch.from_numpy(img.get_fdata()).to(torch.float32)
        img_tensor = F.interpolate(img_tensor, (img_tensor.shape[-1], self.time_len), mode='bilinear')
        img_tensor = img_tensor.permute(3, 0, 1, 2)  # [t, h, w, d]
        t, h, w, d = img_tensor.shape
        img_tensor = torch.reshape(img_tensor, (t, 1, h, w, d))  # [time, channel, height, width, depth]
        # img_tensor = img_tensor[:, :, 44:192, 42:215, 39:122]
        img_tensor = uni_img_size(img_tensor)

        label = self.idh[idx]

        graph, g_label = self.graph_dataset[idx]

        assert label == g_label, 'Graph dataset and Image dataset not match!'

        return graph, img_tensor, label

    def __len__(self):
        return len(self.id)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path = path
        self.trace_func = trace_func
        # self.save = save

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # if self.save:
            #     self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # if self.save:
            #     self.save_checkpoint(val_loss, model)
            self.counter = 0


def plot_metrics(train_losses, val_losses, val_accuracies, fpr, tpr, roc_auc, result_dir, kfold=None, save_flag=True,
                 show_flag=False):
    # Plotting
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.3f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_flag:
        # Save the plots with the current date in the filename
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}/'
        # if not os.path.exists(folder_dir):
        #     os.makedirs(folder_dir)
        if kfold:
            filename = f'{result_dir}/training_plot/fold{kfold}.png'
        else:
            filename = f'{result_dir}/training_plot.png'
        # Save the plots
        plt.savefig(filename)
    if show_flag:  
        plt.show()
    plt.close()



def save_metrics(true_labels, predicted_scores, train_losses, val_losses, val_accuracies,
                 result_dir, kfold=None):

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc, optim_thresh = roc_threshold(true_labels, predicted_scores)
    predicted_labels = (predicted_scores > optim_thresh).astype(int)

    # roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(true_labels, predicted_labels)
    b_acc = balanced_accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, zero_division=np.nan)
    sens = recall_score(true_labels, predicted_labels, zero_division=np.nan)
    spec = recall_score(true_labels, predicted_labels, zero_division=np.nan, pos_label=0)
    f1 = f1_score(true_labels, predicted_labels)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'B-Accuracy': [b_acc],
        'Precision': [prec],
        'Sensitivity': [sens],
        'Specificity': [spec],
        'F1 Score': [f1],
        'ROC AUC': [roc_auc],
    })

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(folder_dir):
    #     os.makedirs(folder_dir)
    if kfold:
        metrics_filename = f'{result_dir}/metrics/fold{kfold}.csv'
    else:
        metrics_filename = f'{result_dir}/metrics.csv'
    
    metrics_df.to_csv(metrics_filename, index=False)

    save_results(true_label=true_labels, predicted_label=predicted_labels, predicted_scores=predicted_scores,
                 train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accuracies,
                 result_dir=result_dir, kfold=kfold)
    plot_metrics(train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accuracies,
                 fpr=fpr, tpr=tpr, roc_auc=roc_auc, show_flag=False, result_dir=result_dir)
    
    return metrics_df



def save_results(true_label, predicted_label, predicted_scores, train_losses, val_losses, val_accuracies, result_dir,
                 kfold=None):
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(val_accuracies) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accuracies,
    })

    results_df = pd.DataFrame({
        'True Label': true_label,
        'Predicted Label': predicted_label,
        'Predicted Scores': predicted_scores
    })

    # current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # folder_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    if kfold:
        metrics_filename = f'{result_dir}/results/fold{kfold}.csv'
        result_filename = f'{result_dir}/predictions/fold{kfold}.csv'
    else:
        metrics_filename = f'{result_dir}/results.csv'
        result_filename = f'{result_dir}/predictions.csv'
    metrics_df.to_csv(metrics_filename, index=False)
    results_df.to_csv(result_filename, index=False)


# modify threshold
from sklearn.metrics import roc_auc_score, roc_curve

def optimal_thresh(fpr, tpr, thresholds, p=0):
   loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
   idx = np.argmin(loss, axis=0)
   return fpr[idx], tpr[idx], thresholds[idx]


def roc_threshold(label, prediction):
   fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
   fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
   c_auc = roc_auc_score(label, prediction)
   return c_auc, threshold_optimal

