# -*- coding: utf-8 -*-
"""GNN_experiments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kaIJof8cE6dl1tQX721JrXrhMTgwncwu

This notebook shows the steps of extracting ASTs, building the dataset on Pytorch Geometric and then applying GNN model for graph embedding as well as predicting
"""


import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score
from torch_geometric.data import Dataset

import clang.cindex
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'], 'size'   : 24})
rc('text', usetex=True)

vdisc = pd.read_csv("cvd/dataset vdisc CWE.csv")
vdisc["bug"] = vdisc["bug"].astype(int)

vdisc.info()

def save_ast(node):

    node.children = list(node.get_children())

    for child in node.children:
        counter = save_ast(child)

def numbering_ast_nodes(node, counter=1):

    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = numbering_ast_nodes(child, counter)

    return counter

def generate_edgelist(ast_root):

    edges = [[],[]]

    def walk_tree_and_add_edges(node):
        for child in node.children:
            # edges.append([node.identifier, child.identifier])
            # walk_tree_and_add_edges(child)
            edg_0 = (node.identifier)-1
            edg_1 = (child.identifier)-1
            # edges[0].append(node.identifier)
            # edges[1].append(child.identifier)
            edges[0].append(edg_0)
            edges[1].append(edg_1)
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)
    return  torch.tensor(edges, dtype=torch.long)

def generate_features(ast_root):

    features = []

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        #in_degree = 1
        #degree = out_degree + in_degree
        degree = out_degree
        node_id = node.identifier
        features.append([node_id, degree])

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    features_array = np.asarray(features)
    # nodes_tensor = torch.from_numpy(features_array).float()
    nodes_tensor = torch.tensor(features_array, dtype=torch.float)
    # nodes_tensor = torch.LongTensor(features).unsqueeze(1)
    return nodes_tensor

def clang_process(testcase, **kwargs):

    parse_list = [
        (testcase.filename, testcase.code)

    ]

    # source_file= get_source_file(testcase)

    # Parsing the source code and extracting AST using clang
    index = clang.cindex.Index.create()
    translation_unit = index.parse(
        path=testcase.filename,
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    save_ast(ast_root)
    numbering_ast_nodes(ast_root)

    graphs_embedding = generate_edgelist(ast_root)

    nodes_embedding = generate_features(ast_root)


    y = torch.tensor([testcase.bug], dtype=torch.int64)



    # delete clang objects
    del translation_unit
    del ast_root
    del index

    return Data(x=nodes_embedding, edge_index=graphs_embedding, y=y)

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'not_implemented.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        self.data = pd.read_csv("cvd/dataset vdisc CWE.csv")
        for index, vuln in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            data = clang_process(vuln)
            torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
        return data

dataset = MyOwnDataset(root='./cvd/')

len(dataset)

print(f'Number of features: {dataset.num_features}')

data1 = dataset[2]  # Get the first graph object.
print(f'Number of nodes: {data1.num_nodes}')
print(f'Number of edges: {data1.num_edges}')

print(dataset[2].edge_index.t())

print(dataset[2].x)

print(dataset[2].y)

dataset = dataset.shuffle()
#one_tenth_length = int(len(dataset) * 0.1)
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length*8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length*9:]
#test_dataset = dataset[one_tenth_length*8:one_tenth_length * 10]
len(train_dataset), len(val_dataset), len(test_dataset)
#len(train_dataset), len(test_dataset)

from torch_geometric.data import DataLoader
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH,drop_last=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=NUM_GRAPHS_PER_BATCH,drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH,drop_last=True, shuffle=True)

import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 128
class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(dataset.num_features, embedding_size) #to  translate our node features into the size of the embedding
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        # pooling layer
        #self.pool = TopKPooling(embedding_size, ratio=0.8)
        #dropout layer
        #self.dropout = Dropout(p=0.2)

        # Output layer
        self.lin1 = Linear(embedding_size*2, 128) # linear output layer ensures that we get a continuous unbounded output value. It input is the flattened vector (embedding size *2) from the pooling layer (mean and max)
        self.lin2 = Linear(128, 128)
        self.lin3 = Linear(128, 1)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        #hidden = self.dropout(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
        # Apply a final (linear) classifier.
        out = self.lin1(hidden)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.act2(out)
        #out = F.dropout(out, p=0.5, training=self.training)
        out = self.lin3(out)
        out = torch.sigmoid(out)

        # return out, hidden
        return out

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x.float(), data.edge_index, data.batch)
        label = data.y.to(device)
        #loss = torch.sqrt(loss_fn(output, label))
        loss = loss_fn(output.squeeze(), label.float())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

from sklearn.metrics import roc_auc_score
def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            # pred = model(data.x.float(), data.edge_index, data.batch).detach().cpu().numpy()
            pred = model(data.x.float(), data.edge_index, data.batch)
            label_true = data.y.to(device)
            label = data.y.detach().cpu().numpy()
            # predictions.append(pred)
            # labels.append(label)
            predictions.append(np.rint(pred.cpu().detach().numpy()))
            labels.append(label)
            loss = loss_fn(pred.squeeze(), label_true.float())
    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)
    predictions = np.concatenate(predictions).ravel()
    labels = np.concatenate(labels).ravel()

    # print(predictions)
    # print(labels)
    return accuracy_score(labels, predictions), loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fn = torch.nn.BCELoss()


print("Starting training...")
train_losses = []
val_losses = []
val_acc_list= []
train_acc_list= []
best_loss = 1000
early_stopping_counter = 0
for epoch in range(200):
    if early_stopping_counter <=  10: # = x * 5
        loss = train()
        train_losses.append(loss)
        train_acc, train_loss = evaluate(train_loader)
        #val_acc = evaluate(val_loader)
        val_acc, val_loss = evaluate(val_loader)
        val_losses.append(val_loss)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        if float(val_loss) < best_loss:
            best_loss = val_loss
            # Save the currently best model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        print(f"Epoch {epoch} | Train Loss {loss} | Train Accuracy{train_acc} | Validation Accuracy{val_acc} | Validation loss{val_loss}")

    else:
        print("Early stopping due to no improvement.")
        break
print(f"Finishing training with best val loss: {best_loss}")

NUM_GRAPHS_PER_BATCH_1 = 4835
test_loader_all = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH_1,drop_last=True, shuffle=True)

import torch

# Save model
torch.save(model.state_dict(), "gnn_model.pth")

# Load model in deployment script
model.load_state_dict(torch.load("gnn_model.pth"))
model.eval() 
