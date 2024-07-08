import torch
import torch.nn as nn

import dgl
import numpy as np
import torch.nn.functional as F
from dgl.nn import EGATConv, GATConv
import scipy.sparse as sp
from utils import my_transform
from classifier.DotProductClassifier import DotProduct_Classifier

from resnet import ResNet, BasicBlock
from convlstm.convlstm_py import ConvLSTMCell

class DyGATConv(nn.Module):
    
    def __init__(self, in_dim, in_edge_dim=1, out_dim=8, out_edge_dim=None, num_heads=3, dropout_prob=0.5):
        
        super(DyGATConv, self).__init__()
        self.conv = GATConv(in_dim, out_dim, num_heads)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, g_batch, in_feat, in_edge_feat=None, get_attention=False, keep_node=False, rm_perc=0.2, add_perc=0.05):
        
        device = g_batch.device
        graphs = dgl.unbatch(g_batch)
        batch_num_node = g_batch.batch_num_nodes()

        feats = []
        i = 0
        for num_node in batch_num_node:
            feats.append(in_feat[i:(i + num_node)])
            i += num_node
            
        edge_attentions = []
        for i, g in enumerate(graphs):
            h, edge_attention = self.conv(g, feats[i], get_attention=True)
            h = h.flatten(1)
            h = F.relu(h)
            h = self.dropout(h)
            g.ndata['h'] = h

            edge_attention = self.sigmoid(torch.mean(edge_attention, 1)[:, 0])

            # Negative graph
            u, v = g.edges()
            adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
            adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())

            neg_u, neg_v = np.where(adj_neg != 0)

            neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
            neg_g = neg_g.to(device)
            
            neg_g.ndata['feat'] = feats[i]
            neg_g.ndata['raw_ID'] = g.ndata['raw_ID']
            neg_g = my_transform(neg_g, keep_node=keep_node)
            
            
            _, neg_edge_attention = self.conv(neg_g, neg_g.ndata['feat'], get_attention=True)
            neg_edge_attention = self.sigmoid(torch.mean(neg_edge_attention, 1)[:, 0])

            # Delete last 20% edges based on edge attention
            _, indices = edge_attention.sort(descending=False)
            num_edges_to_delete = int(rm_perc * len(indices))
            edge_attention = edge_attention[indices[num_edges_to_delete:].sort()[0]]
            edges_to_delete = indices[:num_edges_to_delete]
            g.remove_edges(edges_to_delete, store_ids=False)


            # Add top 10% edges from negative graph to original graph
            _, neg_indices = neg_edge_attention.sort(descending=True)
            num_edges_to_add = int(add_perc * len(neg_indices))
            edges_to_add = neg_g.edges()[0][neg_indices[:num_edges_to_add]], neg_g.edges()[1][neg_indices[:num_edges_to_add]]
            
            g.add_edges(edges_to_add[0], edges_to_add[1])

            g = my_transform(g, keep_node=keep_node)
            edge_attentions.append(edge_attention)
        
        g_batch = dgl.batch(graphs)
        edge_attentions = torch.concat(edge_attentions)

        if get_attention:
            return g_batch, edge_attention
        
        else:
            return g_batch
        

class DyEGATConv(nn.Module):
    
    def __init__(self, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, num_heads, dropout_prob=0.5):
        
        super(DyEGATConv, self).__init__()
        self.conv = EGATConv(in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, num_heads)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, g_batch, in_node_feat, in_edge_feat,  get_attention=False, keep_node=False, rm_perc=0.2, add_perc=0.05):
        
        device = g_batch.device
        
        graphs = dgl.unbatch(g_batch)
        batch_num_node = g_batch.batch_num_nodes()
        batch_num_edge = g_batch.batch_num_edges(etype='temporal')

        node_feats = []
        i = 0
        for num_node in batch_num_node:
            node_feats.append(in_node_feat[i:(i + num_node)])
            i += num_node
        
        edge_feats = []
        i = 0
        for num_edge in batch_num_edge:
            edge_feats.append(in_edge_feat[i:(i + num_edge)])
            i += num_edge
            
        edge_attentions = []
        for i, g in enumerate(graphs):
            g_t = dgl.edge_type_subgraph(g, ['temporal'])
            h_node, h_edge, edge_attention = self.conv(g_t, node_feats[i], edge_feats[i], get_attention=True)
            h_node = self.dropout(F.relu(h_node.flatten(1)))
            h_edge = self.dropout(F.relu(h_edge.flatten(1)))
            
            g.ndata['ht'] = h_node
            g.edges['temporal'].data['h'] = h_edge

            edge_attention = self.sigmoid(torch.mean(edge_attention, 1)[:, 0])

            # Negative graph
            u, v = g_t.edges()
            adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
            adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())

            neg_u, neg_v = np.where(adj_neg != 0)

            neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
            neg_g = neg_g.to(device)
            
            neg_g.ndata['feat'] = node_feats[i]
            neg_g.ndata['raw_ID'] = g.ndata['raw_ID']
            neg_g.edata['feat'] = torch.zeros((neg_g.number_of_edges(), edge_feats[i].shape[1])).to(device)
            neg_g = my_transform(neg_g, keep_node=keep_node)
            
            
            _, neg_h_edge, neg_edge_attention = self.conv(neg_g, neg_g.ndata['feat'], neg_g.edata['feat'], get_attention=True)
            neg_h_edge = F.relu(neg_h_edge.flatten(1))
            neg_edge_attention = self.sigmoid(torch.mean(neg_edge_attention, 1)[:, 0])

            # Delete last 20% edges based on edge attention
            _, indices = edge_attention.sort(descending=False)
            num_edges_to_delete = int(rm_perc * len(indices))
            edge_attention = edge_attention[indices[num_edges_to_delete:].sort()[0]]
            edges_to_delete = indices[:num_edges_to_delete]
            g.remove_edges(edges_to_delete, store_ids=False, etype='temporal')


            # Add top 5% edges from negative graph to original graph
            _, neg_indices = neg_edge_attention.sort(descending=True)
            num_edges_to_add = int(add_perc * len(neg_indices))
            edges_to_add = neg_g.edges()[0][neg_indices[:num_edges_to_add]], neg_g.edges()[1][neg_indices[:num_edges_to_add]]
            edge_feat = neg_edge_attention[neg_indices[:num_edges_to_add]]
            edge_hidden = neg_h_edge[neg_indices[:num_edges_to_add]]
            g.add_edges(edges_to_add[0], edges_to_add[1], {'feat': edge_feat, 'h': edge_hidden}, etype='temporal')

            g = my_transform(g, keep_node=keep_node)

            edge_attentions.append(edge_attention)
        
        g_batch = dgl.batch(graphs)
        edge_attentions = torch.concat(edge_attentions)

        if get_attention:
            return g_batch, edge_attention
        
        else:
            return g_batch




def get_inplanes():
    return [8, 16, 32, 64]

class ResConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, img_size, num_classes=2,
                 batch_first=False, bias=True, return_all_layers=False, no_max_pool=True):
        super(ResConvLSTM, self).__init__()
        
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=input_dim,
                             no_max_pool=no_max_pool)

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = get_inplanes()[-1]
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # if num_classes == 2:
        #     num_classes = 1
        # self.num_classes = num_classes

        _, _, _, t = img_size
        self.fc1 = nn.Linear(hidden_dim[-1] * 7 * 7 * 7 * t, 256)
        self.fc2 = nn.Linear(256, 48)
        # self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.LeakyReLU()


        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):

        b, t, _, _, _, _ = input_tensor.size()

        resnet_output = []
        for timestep in range(t):
            input_t = input_tensor[:, timestep, ...]
            resnet_output.append(self.resnet(input_t))
        resnet_output = torch.stack(resnet_output, dim=1)
        b, _, _, h, w, d = resnet_output.size()
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w, d))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = resnet_output

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:][0]
            last_state_list = last_state_list[-1:][0]

        out = self.relu(self.fc1(layer_output_list.view(layer_output_list.size(0), -1)))
        out = self.relu(self.fc2(out))
        # logits = self.fc3(out)
        # logits = self.fc4(logits)
        # prob = torch.sigmoid(logits)

        return out

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class dual_attn_fusion(nn.Module):
    def __init__(self, in_size=48, hidden_size=128):
        super(dual_attn_fusion, self).__init__()
        
        self.cosinesim = nn.CosineSimilarity(dim=1)
        
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        
    def forward(self, out):
        # img_feats: [batch, 48], graph_feats: [batch, nodes, 48]
        img_feats, spatial_feats, temporal_feats = out
        # Node Attention
        similarity = []
        for img_feat, spatial_feat in zip(img_feats, spatial_feats):
            
            sim = self.cosinesim(spatial_feat, img_feat)
            similarity.append(sim)
        
        similarity = torch.stack(similarity)
        
        weighted_sfeats = torch.stack([similarity]*spatial_feats.shape[-1], dim=2) * spatial_feats
        
        z = torch.stack([weighted_sfeats, temporal_feats], dim=1)
        
        # Semantic Attention
        w_d = self.project(z).mean(2)
        beta = torch.softmax(w_d, dim=1)
        beta = beta.unsqueeze(2)
        out = torch.mean(torch.sum((z*beta), dim=1), dim=1)
        
        return out
        
    
    
    
class PerfGAT(nn.Module):
    def __init__(self, in_feats, in_edge_feats=1, h_feats=16, h_edge_feats=16, img_size=(54, 54, 54), time_len=45,
                 num_classes=2, num_heads=3, num_layer=2, dropout_prob=0.5, edge_feat=False):
        super(PerfGAT, self).__init__()

        self.edge_feat = edge_feat
        if edge_feat:
            dyconv = DyEGATConv
        else:
            dyconv = DyGATConv
        
        self.num_layer = num_layer
        
        if not isinstance(h_feats, list):
            h_feats = [h_feats//(2**i) for i in range(num_layer - 1, -1, -1)]
        else:
            assert len(h_feats) == num_layer, "Length of node feature lists should be equal to number of layers!"
        if not isinstance(h_edge_feats, list):
            h_edge_feats = [h_edge_feats//(2**i) for i in range(num_layer - 1, -1, -1)]
        else:
            assert len(h_edge_feats) == num_layer, "Length of edge feature lists should be equal to number of layers!"
        
        self.gconv = nn.ModuleList([dyconv(in_feats, in_edge_feats, h_feats[0], h_edge_feats[0], num_heads, dropout_prob)]
                                   + [dyconv(h_feats[i]*num_heads, h_edge_feats[i]*num_heads, h_feats[i+1], h_edge_feats[i+1], num_heads, dropout_prob)
                                      for i in range(num_layer - 1)])
        
        self.gatconv = nn.ModuleList([GATConv(in_feats, h_feats[0], num_heads, feat_drop =dropout_prob)]
                                     + [GATConv(h_feats[i]*num_heads, h_feats[i+1], num_heads, feat_drop =dropout_prob)
                                        for i in range(num_layer - 1)])
        # self.conv1 = dyconv(in_feats, in_edge_feats, h_feats//2, h_edge_feats//2, num_heads, dropout_prob=dropout_prob)
        # self.conv2 = dyconv(h_feats//2*num_heads, h_edge_feats//2*num_heads, h_feats, h_edge_feats, num_heads, dropout_prob=dropout_prob)
        # self.conv3 = dyconv(h_feats*num_heads, h_edge_feats*num_heads, h_feats, h_edge_feats, num_heads, dropout_prob=dropout_prob)

        self.resconvlstm = ResConvLSTM(input_dim=1, hidden_dim=[128, 64, 64], kernel_size=(3, 3, 3),
                                       num_layers=3, img_size=img_size+(time_len,), batch_first=True, bias=True)

        self.featFC = nn.Linear(h_feats[-1]*num_heads*2, h_feats[-1] * num_heads)

        self.fc = DotProduct_Classifier(h_feats[-1]*num_heads, num_classes)
        
    
    def forward(self, g, in_feat, in_edge_feat=None, x=None, get_attention=False, cRT=True, keep_node=False,
                rm_perc=0.2, add_perc=0.05):
        
        g = self.gconv[0](g, in_feat, in_edge_feat, keep_node=keep_node, rm_perc=rm_perc, add_perc=add_perc)

        if not self.edge_feat:
            for i in range(1, self.num_layer - 1):
                g = self.gconv[i](g, g.ndata['ht'], keep_node=keep_node, rm_perc=rm_perc, add_perc=add_perc)
            if get_attention:
                g, edge_attention = self.gconv[-1](g, g.ndata['ht'], get_attention=get_attention, keep_node=keep_node,
                                                   rm_perc=rm_perc, add_perc=add_perc)
            else:
                g = self.gconv[-1](g, g.ndata['ht'], keep_node=keep_node, rm_perc=rm_perc, add_perc=add_perc)
        else:
            for i in range(1, self.num_layer - 1):
                edge_temp_feat = g.edges['temporal'].data['h']
                g = self.gconv[i](g, g.ndata['ht'], edge_temp_feat, keep_node=keep_node, rm_perc=rm_perc, add_perc=add_perc)
            if get_attention:
                g, edge_attention = self.gconv[-1](g, g.ndata['ht'], edge_temp_feat, get_attention=get_attention, keep_node=keep_node,
                                                   rm_perc=rm_perc, add_perc=add_perc)
            else:
                edge_temp_feat = g.edges['temporal'].data['h']
                g = self.gconv[-1](g, g.ndata['ht'], edge_temp_feat, keep_node=keep_node, rm_perc=rm_perc, add_perc=add_perc)

        gs = dgl.edge_type_subgraph(g, ['spatial'])
        hs = F.relu(self.gatconv[0](gs, in_feat)).flatten(1)
        for i in range(1, self.num_layer):
            hs = F.relu(self.gatconv[i](gs, hs)).flatten(1)
        g.ndata['hs'] = hs

        graphs = dgl.unbatch(g)
        temporal_graph_out = torch.stack([graph.ndata['ht'] for graph in graphs])
        spatial_graph_out = torch.stack([graph.ndata['hs'] for graph in graphs])
        
        # graph_out = dgl.mean_nodes(g, "ht")

        img_out = self.resconvlstm(x)

        output = img_out, temporal_graph_out, spatial_graph_out
        # output = torch.concat([temporal_graph_out, spatial_graph_out, img_out], dim=1)
        # output = self.featFC(output)

        if not cRT:
            output = self.fc(output)

        if get_attention:
            return output, g, edge_attention
        else:
            return output
