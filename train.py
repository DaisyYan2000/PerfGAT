import os
import sys
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
                            auc, balanced_accuracy_score, matthews_corrcoef
import time

from utils import EarlyStopping, plot_metrics, save_metrics, save_results, roc_threshold
# from cRT_gnn import *
# from dynamic_edge_gnn import E_GAT
# from cnn_gnn import CNN_EGAT
from Perf_GNN import PerfGAT, dual_attn_fusion
from dataloaders import my_dataloader, graph_img_dataloader
import datetime

from classifier.DotProductClassifier import DotProduct_Classifier
from classifier.CosNormClassifier import CosNorm_Classifier
from classifier.KNNClassifier import KNNClassifier
from classifier.MetaEmbeddingClassifier import MetaEmbedding_Classifier
from classifier.TauNormClassifier import TauNorm_Classifier

# Three mode: Joint and cRT and tau
def cRT_training(dataset, gnn_name, clf_name, batch_size=10, input_dim=45, node_hidden_dim=128, edge_hidden_dim=64,
                num_classes=2, num_layer=3, dropout_prob=0.5, lr=1e-4, epochs=500, train_mode='cRT', patience=50, device='cuda', seed=42,
                result_dir=None, edge_feat=False, keep_node=False, rm_perc=0.4, add_perc=0.01, aug=True):
    
    assert train_mode in ['cRT', 'joint', 'tau'], "Train Mode can only be 'cRT', 'tau', or 'joint'!!!"
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if result_dir == None:
        result_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
        data_loader = graph_img_dataloader
    else:
        data_loader = my_dataloader
    if train_mode=='cRT' or train_mode=='tau':
        retrain=True
    elif train_mode=='joint':
        retrain=False
    train_dataloader, retrain_dataloader, val_dataloader, test_dataloader = data_loader(dataset=dataset,
                                                                                        retrain=retrain,
                                                                                        aug=aug,
                                                                                        train_batch_size=batch_size,
                                                                                        val_batch_size=batch_size,
                                                                                        test_batch_size=batch_size,
                                                                                        seed=seed)
    print('Done Loading dataset.')
    sys.stdout.flush()
    if gnn_name == 'PerfGAT':
        gnn = PerfGAT

    if gnn_name == 'EGAT' or gnn_name == 'CEGAT' or 'PerfGAT':
        feature_extractor = gnn(input_dim, 1, node_hidden_dim, edge_hidden_dim, num_layer=num_layer, dropout_prob=dropout_prob,
                                edge_feat=edge_feat).to(device)
        fusion = dual_attn_fusion().to(device)
    else:
        feature_extractor = gnn(input_dim, node_hidden_dim, dropout_prob=dropout_prob).to(device)
        
    if clf_name == 'dot':
        clf = DotProduct_Classifier
    if clf_name == 'cos':
        clf = CosNorm_Classifier
    if clf_name == 'meta':
        clf = MetaEmbedding_Classifier
    if clf_name == 'knn':
        clf = KNNClassifier
    if train_mode == 'tau':
        clf = TauNorm_Classifier
    if gnn_name == 'EGAT' or gnn_name == 'GAT' or gnn_name == 'E_GAT' or gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
        classifier = clf(node_hidden_dim*3, num_classes).to(device)
    elif gnn_name == 'GIN':
        classifier = clf(input_dim, num_classes).to(device)
    else:
        classifier = clf(node_hidden_dim, num_classes).to(device)
        
    # Initialize optimizer and loss function
    feat_optimizer = optim.Adam(feature_extractor.parameters(), lr=lr)
    fusion_optimizer = optim.Adam(fusion.parameters(), lr=lr)
    clf_optimizer = optim.Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    earlystop = EarlyStopping(patience=patience, verbose=False)

    train_losses = []
    retrain_losses = []
    
    val_losses = []
    reval_losses = []
    
    val_accuracies = []
    reval_accuracies = []
    
    best_val_loss = float('inf')
    best_feat_state = None
    best_clf_state = None
    
    start_time = time.time()
    # First Training loop (if Joint Training, this is the only training loop)
    print('...Training the entire model without resampling...')
    if train_mode == 'cRT':
        print('(Losses shown only for monitoring purpose...)')
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        for batch in train_dataloader:
            batched_graph = batch[0].to(device)
            if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
                imgs = batch[1].to(device)
                labels = batch[2].unsqueeze(1).to(device)
                node_feat = batched_graph.ndata['feat']
                edge_feat = torch.unsqueeze(batched_graph.edges['temporal'].data['feat'], 1)
                feature = feature_extractor(batched_graph, node_feat, edge_feat, imgs, keep_node=keep_node,
                                            rm_perc=rm_perc, add_perc=add_perc)
                feature = fusion(feature)
            else:
                labels = batch[1].to(device)
                if gnn_name == 'EGAT':
                    feature = feature_extractor(batched_graph, batched_graph.ndata["feat"], torch.unsqueeze(batched_graph.edata["feat"], 1))
                else:
                    feature = feature_extractor(batched_graph, batched_graph.ndata["feat"])
            pred = classifier(feature)

            loss = loss_fn(pred, labels)
            feat_optimizer.zero_grad()
            fusion_optimizer.zero_grad()
            clf_optimizer.zero_grad()
            loss.backward()
            feat_optimizer.step()
            fusion_optimizer.step()
            clf_optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_dataloader))
    
        # num_correct = 0
        # num_tests = 0
        epoch_val_loss = 0.0

        true_labels = []
        # predicted_labels = []
        predicted_scores = []

        sys.stdout.flush()
        # Validation
        with torch.no_grad():
            for batch in val_dataloader:
                batched_graph = batch[0].to(device)
                if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
                    imgs = batch[1].to(device)
                    labels = batch[2].unsqueeze(1).to(device)
                    node_feat = batched_graph.ndata['feat']
                    edge_feat = torch.unsqueeze(batched_graph.edges['temporal'].data['feat'], 1)
                    feature = feature_extractor(batched_graph, node_feat, edge_feat, imgs, keep_node=keep_node,
                                                rm_perc=rm_perc, add_perc=add_perc)
                    feature = fusion(feature)
                else:
                    labels = batch[1].to(device)
                    if gnn_name == 'EGAT':
                        feature = feature_extractor(batched_graph, batched_graph.ndata["feat"], torch.unsqueeze(batched_graph.edata["feat"], 1))
                    else:
                        feature = feature_extractor(batched_graph, batched_graph.ndata["feat"])
                pred = classifier(feature)
                
                loss = loss_fn(pred, labels)
                epoch_val_loss += loss.item()
                true_labels.extend(labels.squeeze(dim=1).cpu().numpy())
                # predicted_labels.extend(pred.round().cpu().numpy())
                predicted_scores.extend(pred.squeeze(dim=1).cpu().numpy())
                # num_correct += (pred.round() == labels).float().sum().item()
                # num_tests += len(labels)

            val_losses.append(epoch_val_loss / len(val_dataloader))
            _, optim_thresh = roc_threshold(true_labels, predicted_scores)
            predicted_labels = (predicted_scores > optim_thresh).astype(int)
            val_accuracy = accuracy_score(true_labels, predicted_labels)
            # val_accuracy = num_correct / num_tests
            val_accuracies.append(val_accuracy)

            if (epoch+1)%5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss / len(train_dataloader)} | Validate Loss: {epoch_val_loss / len(val_dataloader)}")
                sys.stdout.flush()
            # Check for early stopping
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_feat_state = feature_extractor.state_dict()
                best_fusion_state = fusion.state_dict()
                best_clf_state = classifier.state_dict()

                # Save the best model if joint training
                if train_mode == 'joint' or train_mode == 'tau':
                    # log_dir = f'./log_{current_date}/{gnn_name}/{train_mode}'
                    # if not os.path.exists(log_dir):
                    #     os.makedirs(log_dir)
                    torch.save(best_feat_state, f'{result_dir}/best_extractor.pth')
                    torch.save(best_fusion_state, f'{result_dir}/best_fusion.pth')
                    torch.save(best_clf_state, f'{result_dir}/best_classifier.pth')

            earlystop(epoch_val_loss)

            if earlystop.early_stop:
                # save_image(test_dir, output=outputs, epoch=i + 1, batch_size=batch_size, mode=2)
                print("\tEarly stopping...")
                print(f'First Training Loop Ends, time cost: {(time.time() - start_time):.2f}')
                break
    
    sys.stdout.flush()
    if train_mode == 'cRT':
        # Freeze the feature extractor and re-initialize the classifier
        feature_extractor.load_state_dict(best_feat_state)
        fusion.load_state_dict(best_fusion_state)
        for param in feature_extractor.parameters():
            param.requires_grad = False
        for param in fusion.parameters():
            param.required_grad = False
        
        classifier.reset_parameters()

        print('...Re-training the classifier with class-balanced sampling...')
        best_val_loss = float('inf')

        for epoch in range(30):
            classifier.train()
            epoch_retrain_loss = 0.0
            
            for batch in retrain_dataloader:
                batched_graph = batch[0].to(device)
                with torch.no_grad():
                    if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
                        imgs = batch[1].to(device)
                        labels = batch[2].unsqueeze(1).to(device)
                        node_feat = batched_graph.ndata['feat']
                        edge_feat = torch.unsqueeze(batched_graph.edges['temporal'].data['feat'], 1)
                        feature = feature_extractor(batched_graph, node_feat, edge_feat, imgs, keep_node=keep_node,
                                                rm_perc=rm_perc, add_perc=add_perc)
                        feature = fusion(feature)
                    else:
                        labels = batch[1].to(device)
                        if gnn_name == 'EGAT':
                            feature = feature_extractor(batched_graph, batched_graph.ndata["feat"], torch.unsqueeze(batched_graph.edata["feat"], 1))
                        else:
                            feature = feature_extractor(batched_graph, batched_graph.ndata["feat"])
                pred = classifier(feature)

                loss = loss_fn(pred, labels)
                clf_optimizer.zero_grad()
                loss.backward()
                clf_optimizer.step()

                epoch_retrain_loss += loss.item()

            retrain_losses.append(epoch_retrain_loss / len(retrain_dataloader))
            
            # num_correct = 0
            # num_tests = 0
            epoch_val_loss = 0.0
            true_labels = []
            # predicted_labels = []
            predicted_scores = []
            
            sys.stdout.flush()
            # Validation
            with torch.no_grad():
                for batch in val_dataloader:
                    batched_graph = batch[0].to(device)
                    if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
                        imgs = batch[1].to(device)
                        labels = batch[2].unsqueeze(1).to(device)
                        node_feat = batched_graph.ndata['feat']
                        edge_feat = torch.unsqueeze(batched_graph.edges['temporal'].data['feat'], 1)
                        feature = feature_extractor(batched_graph, node_feat, edge_feat, imgs, keep_node=keep_node,
                                                rm_perc=rm_perc, add_perc=add_perc)
                        feature = fusion(feature)
                    else:
                        labels = batch[1].to(device)
                        if gnn_name == 'EGAT':
                            feature = feature_extractor(batched_graph, batched_graph.ndata["feat"], torch.unsqueeze(batched_graph.edata["feat"], 1))
                        else:
                            feature = feature_extractor(batched_graph, batched_graph.ndata["feat"])
                    pred = classifier(feature)

                    loss = loss_fn(pred, labels)
                    
                    epoch_val_loss += loss.item()
                    true_labels.extend(labels.squeeze(dim=1).cpu().numpy())
                    # predicted_labels.extend(pred.round().cpu().numpy())
                    predicted_scores.extend(pred.squeeze(dim=1).cpu().numpy())
                    # num_correct += (pred.round() == labels).float().sum().item()
                    # num_tests += len(labels)

                reval_losses.append(epoch_val_loss / len(val_dataloader))
                _, optim_thresh = roc_threshold(true_labels, predicted_scores)
                predicted_labels = (predicted_scores > optim_thresh).astype(int)
                val_accuracy = accuracy_score(true_labels, predicted_labels)
                # val_accuracy = num_correct / num_tests
                reval_accuracies.append(val_accuracy)

                if (epoch+1)%2 == 0:
                    print(f"Epoch {epoch + 1}/{30} | Train Loss: {epoch_retrain_loss / len(retrain_dataloader)} | Validate Loss: {epoch_val_loss / len(val_dataloader)}")
                    sys.stdout.flush()
                if reval_losses[-1] < best_val_loss:
                    best_val_loss = reval_losses[-1]
                    best_feat_state = feature_extractor.state_dict()
                    best_clf_state = classifier.state_dict()

                    # Save the best model
                    # log_dir = f'./log_{current_date}/{gnn_name}/cRT'
                    # if not os.path.exists(log_dir):
                    #     os.makedirs(log_dir)
                    torch.save(best_feat_state, f'{result_dir}/best_extractor.pth')
                    torch.save(best_fusion_state, f'{result_dir}/best_fusion.pth')
                    torch.save(best_clf_state, f'{result_dir}/best_classifier.pth')
        print(f'Training ends, time cost: {(time.time() - start_time):.2f}')

    # Test
    # num_correct = 0
    # num_tests = 0
    
    sys.stdout.flush()
    true_labels = []
    # predicted_labels = []
    predicted_scores = []
    feature_extractor.load_state_dict(best_feat_state)
    fusion.load_state_dict(best_fusion_state)
    classifier.load_state_dict(best_clf_state)

    with torch.no_grad():
        for batch in test_dataloader:
            batched_graph = batch[0].to(device)
            if gnn_name == 'CEGAT' or gnn_name == 'PerfGAT':
                imgs = batch[1].to(device)
                labels = batch[2].unsqueeze(1).to(device)
                node_feat = batched_graph.ndata['feat']
                edge_feat = torch.unsqueeze(batched_graph.edges['temporal'].data['feat'], 1)
                feature = feature_extractor(batched_graph, node_feat, edge_feat, imgs, keep_node=keep_node,
                                                rm_perc=rm_perc, add_perc=add_perc)
                feature = fusion(feature)
            else:
                labels = batch[1].to(device)
                if gnn_name == 'EGAT':
                    feature = feature_extractor(batched_graph, batched_graph.ndata["feat"], torch.unsqueeze(batched_graph.edata["feat"], 1))
                else:
                    feature = feature_extractor(batched_graph, batched_graph.ndata["feat"])
            pred = classifier(feature)
            
            # num_correct += (pred.round() == labels).float().sum().item()
            # num_tests += len(labels)

            true_labels.extend(labels.squeeze(dim=1).cpu().numpy())
            # predicted_labels.extend(pred.round().cpu().numpy())
            predicted_scores.extend(pred.squeeze(dim=1).cpu().numpy())
    
    sys.stdout.flush()
    
    if train_mode == 'cRT':
        train_losses = retrain_losses
        val_losses = reval_losses
        val_accuracies = reval_accuracies
    
    metrics = save_metrics(true_label=true_labels, predicted_label=predicted_labels, predicted_scores=predicted_scores,
                 train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accuracies,
                 result_dir=result_dir)

    # Restore the best model state
    # model.load_state_dict(best_model_state)
    print(f"Summary of Training:")
    print(metrics)
    print()

    
    

