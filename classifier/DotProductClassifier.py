"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch.nn as nn
from utils import *
from os import path

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000, act_fn=nn.Sigmoid(), *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        if num_classes == 2:
            num_classes = 1
        self.fc = nn.Linear(feat_dim, num_classes)
        self.act = act_fn
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x, *args):
        x = self.fc(x)
        x = self.act(x)
        return x
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(feat_dim, num_classes)

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