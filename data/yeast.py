#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:00:26 2020

@author: root
"""
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
<<<<<<< HEAD
#from progressbar import progressbar
from tqdm import tqdm
import csv
from scipy.sparse import coo_matrix
=======

import csv
>>>>>>> b22de18d594b91c11a1b339f6b81cd3d27d0e221

import dgl

from scipy import sparse as sp


# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']




class YeastDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)
    
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data['y'])
        self.num_graphs = self.n_samples
        self._prepare()
        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
<<<<<<< HEAD
        adj = torch.tensor(self.data['GSO'])
        edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
        edge_idxs_in_adj = edge_list.split(1, dim=1)
        edge_features = adj[edge_idxs_in_adj].reshape(-1).long() 
        print("Creating graph from sparse GSO")
        t0 = time.time()
        sparse_adj = coo_matrix(self.data['GSO'])
        print(f"Percentage of non-zero values in adj : {100*sparse_adj.nnz/self.data['GSO'].size}")
        g = dgl.from_scipy(sparse_adj)
        tf = time.time()
        print(f"Graph Created in {tf-t0}s")
        for cow, feno in tqdm(zip(torch.tensor(self.data['X']), torch.tensor(self.data['y']))):
            node_features = cow.long()
            g = dgl.from_scipy(sparse_adj)
            g.ndata['feat'] = node_features
            g.edata['feat'] = edge_features            
=======
        adj = self.data['GSO']
        edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
        edge_idxs_in_adj = edge_list.split(1, dim=1)
        edge_features = adj[edge_idxs_in_adj].reshape(-1).long() 
        
        for cow, feno in zip(self.data['X'], self.data['y']):
            node_features = cow.long()
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(adj.shape[0])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
>>>>>>> b22de18d594b91c11a1b339f6b81cd3d27d0e221
            self.graph_lists.append(g)
            self.graph_labels.append(feno)
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
class YeastDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Yeast'):
        t0 = time.time()
        self.name = name
        
        data_dir='./data/yeast'
              
        self.train = YeastDGL(data_dir, 'train')
        self.val = YeastDGL(data_dir, 'val')
        self.test = YeastDGL(data_dir, 'test')
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    return g    

class YeastDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/yeast/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = 1
            self.num_bond_type = 1
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type + self.num_bond_type)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_label in enumerate(g.edata['feat']):
                adj_with_edge_feat[edge_label.item()+1+self.num_atom_type][us[idx]][vs[idx]] = 1

            for node, node_label in enumerate(g.ndata['feat']):
                adj_with_edge_feat[node_label.item()+1][node][node] = 1
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels
        
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata['feat']):
                adj_no_edge_feat[node_label.item()+1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]




