"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import *
from os import path


class TauNorm_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000, act=nn.Sigmoid(), *args):
        super(TauNorm_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        if num_classes == 2:
            num_classes = 1
        self.fc = nn.Linear(feat_dim, num_classes)
        self.act = act
        self.scales = Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, *args):
        x = self.fc(x)
        x *= self.scales
        x = self.act(x)
        return x
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = TauNorm_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            if log_dir is not None:
                subdir = log_dir.strip('/').split('/')[-1]
                subdir = subdir.replace('stage2', 'stage1')
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading classifier weights from %s' % weight_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf