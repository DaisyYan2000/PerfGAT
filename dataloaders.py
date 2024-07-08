# Joint class_balanced Dataloader
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader

from collections import Counter
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl


def collate_fn(data):
    graphs = []
    imgs = []
    labels = []

    for unit in data:
        graphs.append(unit[0])
        imgs.append(unit[1])
        labels.append(unit[2])
    
    graphs = dgl.batch(graphs)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)

    return graphs, imgs, labels


def class_balanced_sampler(labels, num_classes):
    class_counts = Counter(labels)
    class_weights = [1.0 / class_counts[i] for i in range(num_classes)]
    weights = [class_weights[labels[i].item()] for i in range(len(labels))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def my_dataloader(dataset, retrain:bool, train_batch_size=10, val_batch_size=10, test_batch_size=10,
                  test_size=0.15, val_size=0.15, seed=42):
    labels = dataset.labels.squeeze().numpy().astype('int')
    val_size = val_size/(1-test_size)
    train_mask, test_mask = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed, stratify=labels)
    train_mask, val_mask = train_test_split(train_mask, test_size=val_size, random_state=seed, stratify=labels[train_mask])

    train_dataset = [dataset[i] for i in train_mask]
    val_dataset = [dataset[i] for i in val_mask]
    test_dataset = [dataset[i] for i in test_mask]

    train_sampler = class_balanced_sampler(labels[train_mask], 2)

    if retrain:
        train_loader = GraphDataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        retrain_loader = GraphDataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    else:
        train_loader = GraphDataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # val_sampler = class_balanced_sampler(labels[val_mask], 2)
    val_loader = GraphDataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)


    # test_sampler = class_balanced_sampler(labels[test_mask], 2)
    test_loader = GraphDataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if retrain:
        out = train_loader, retrain_loader, val_loader, test_loader
    else:
        out = train_loader, val_loader, test_loader

    return out


def graph_img_dataloader(dataset, retrain:bool, aug:bool, train_batch_size=10, val_batch_size=10, test_batch_size=10,
                         test_size=0.15, val_size=0.15, seed=42):
    
    labels = dataset.target().numpy().astype('int')
    val_size = val_size/(1-test_size)
    train_mask, test_mask = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed, stratify=labels)
    train_mask, val_mask = train_test_split(train_mask, test_size=val_size, random_state=seed, stratify=labels[train_mask])

    train_dataset = [dataset[i] for i in train_mask]
    if aug:
        retrain_dataset = recombine_dataset(train_dataset)
    
    val_dataset = [dataset[i] for i in val_mask]
    test_dataset = [dataset[i] for i in test_mask]

    train_sampler = class_balanced_sampler(labels[train_mask], 2)

    if retrain:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
        if aug:
            retrain_loader = DataLoader(retrain_dataset, train_batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            retrain_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_fn)
    else:
        if aug:
            train_loader = DataLoader(retrain_dataset, train_batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_fn)

    # val_sampler = class_balanced_sampler(labels[val_mask], 2)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    # test_sampler = class_balanced_sampler(labels[test_mask], 2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    if retrain:
        return train_loader, retrain_loader, val_loader, test_loader
    else:
        return train_loader, None, val_loader, test_loader



def recombine_dataset(dataset):
    
    graphs = []
    imgs = []
    labels = []
    # original_dataset = []
    
    for g, img, label in dataset:
        graphs.append(g)
        imgs.append(img)
        labels.append(label)
        # original_dataset.append((g, img, label))
    
    minority_index = list(torch.where(torch.tensor(labels)==1)[0])
    
    recombine_sample = []

    for i in range(len(minority_index)):
        for j in range(len(minority_index)):
            if i == j:
                continue
            g_index = minority_index[i]
            img_index = minority_index[j]
            recombine_sample.append((graphs[g_index], imgs[img_index], labels[g_index]))
    
    import random
    num_to_select = len(labels) - len(minority_index) - len(minority_index)
    recombine_sample = random.sample(recombine_sample, num_to_select)
    
    recombine_sample = dataset + recombine_sample
    
    return recombine_sample
    