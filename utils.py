import pickle
import numpy as np 
import scipy
import networkx as nx 
import random
from collections import defaultdict
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def load_noesis_data(folder, name):

    a = scipy.io.loadmat(folder+name+"/"+name+".mat")

    net = a["net"]

    if scipy.sparse.issparse(net):
        net = net.A
            

    g = nx.from_numpy_array(net)
    feat = [0.1] * 10
    X = dict()
    for i,j in nx.clustering(g).items():
        X[i] = [j]

    fun = [nx.betweenness_centrality,nx.closeness_centrality]
    for f in fun:
        for i,j in f(g).items():
            X[i].append(j)
            #X[i] = feat

    nx.set_node_attributes(g,X,"X")
    data = from_networkx(g,group_node_attrs=["X"])
    return data

def generate_graph(folder, data_name):
    if data_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='__init__', name=data_name)
        data = dataset[0]
    elif data_name in ['ADV', 'BUP', 'Celegans', 'EML', 'GRQ', 'HPD', 'KHN', 'LDG', 'NSC', 'Power', 'SMG', 'USAir', 'YST', 'ZWL']:
        data = load_noesis_data(folder, data_name)

    G = to_networkx(data)
    
    node_features = {}

    edge_dict = defaultdict(list)
    for i, (src, tgt) in enumerate(data.edge_index.t()):
        edge_dict[i].extend([src.item(), tgt.item()])
        node_features[i] = torch.cat((data.x[src.item()], data.x[tgt.item()]), dim=0)
    
    negative_dict = {}
    max_node_index = data.edge_index.max().item()
    index = 1+max(list(edge_dict.keys()))
    iter = 0
    for i in range(len(data.edge_index[0].unique().tolist())):
        node = data.edge_index[0].unique().tolist()[i]
        ego_graph_radius_1 = list(nx.ego_graph(G, node).nodes())
        ego_graph_radius_1.remove(node)
        
        degree = len(ego_graph_radius_1)
        nodes_ego_graph = list(nx.ego_graph(G, node, radius=4).nodes())
        nodes_ego_graph.remove(node)
        intersection = [x for x in nodes_ego_graph if x not in ego_graph_radius_1]
        for j in range(degree):
            if intersection:
                tgz = random.choice(intersection)
                intersection.remove(tgz)
            else:
                tgz = random.randint(0, max_node_index)
            negative_dict[iter+index] = [node, tgz]
            node_features[iter+index] = torch.cat((data.x[node], data.x[tgz]), dim=0)
            iter+=1
    # labels
    y = []
    inverse = {}
    for key, values in edge_dict.items():
        for value in values:
            inverse.setdefault(value, []).append(key)
        y.append(1)

    for key, values in negative_dict.items():
        for value in values:
            inverse.setdefault(value, []).append(key)
        y.append(0)
    
    edge_dict.update(negative_dict)
    
    # edge index
    edge_index = torch.tensor([[combo[0], combo[1]] for values in inverse.values() if len(values) > 1 for combo in itertools.combinations(values, 2)])
    
    x = torch.stack(list(node_features.values()))
    
    data_object = Data(x=x, edge_index = edge_index.t(), y=torch.tensor(y), num_classes=2, num_features=x.shape[1], edge_dict_original=edge_dict, edge_index_original=data.edge_index, x_original=data.x)
    
    with open("line_graphs/"+data_name+"_line_graph.pkl", "wb") as file:
        pickle.dump(data_object, file)
        

def load_graph(name):
    with open("line_graphs/"+name+"_line_graph.pkl", "rb") as file:
        data = pickle.load(file)
    return data

def split(data):
    tgts  = data.y
    node_ids = [i for i in range(len(data.y))]
    tgts = tgts.cpu()

    train_ids, test_ids, train_labels, test_labels = train_test_split(node_ids, tgts, test_size=0.1, stratify=tgts, random_state=4)
    train_ids, val_ids, train_labels, val_labels = train_test_split(train_ids, train_labels, test_size=0.10, stratify=train_labels, random_state=4)
    train_boolean_mask = np.zeros(data.x.size(0), dtype=bool)
    val_boolean_mask = np.zeros(data.x.size(0), dtype=bool)
    test_boolean_mask = np.zeros(data.x.size(0), dtype=bool)


    train_boolean_mask[train_ids] = True
    val_boolean_mask[val_ids] = True
    test_boolean_mask[test_ids] = True
    tg = [0] * data.x.size(0)
    for i in range(len(node_ids)):
        tg[i] = tgts[i]

    
    data.train_mask = train_boolean_mask
    data.val_mask = val_boolean_mask
    data.test_mask = test_boolean_mask
    data.y_ = torch.tensor(tg)

    #class weights
    tgts_tensor = tgts.clone().detach().to(dtype=torch.int)
    label_counts = torch.bincount(tgts_tensor)

    # Calcolo dei pesi inversamente proporzionali ai conteggi
    total_samples = len(tgts_tensor)
    weights = total_samples / (len(label_counts) * label_counts.float())
    data.weights = weights
    return data


def train(data, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()  
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  
    optimizer.step()  
    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask] 
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
    train_auc = roc_auc_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu(), average = "macro")
    return loss, train_acc, train_auc

def val(data, model, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    val_auc = roc_auc_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu(), average = "macro")
    return val_acc, val_auc, val_loss

def test(data, best_model, criterion):
    best_model.eval()
    out = best_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    test_auc = roc_auc_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average = "macro")
    return test_acc, test_auc