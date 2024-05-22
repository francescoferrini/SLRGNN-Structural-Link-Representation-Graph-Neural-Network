import torch

from tqdm import tqdm
import numpy as np

import copy
import argparse
from models import GIN
from utils import *

seeds = [1, 20, 54, 34, 65]

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")


    
def grid_search(args):
    args_dict = vars(args)
    del args_dict['folder']

    keys = list(args_dict)  
    values = args_dict.values()
    combination = list(values)
    
    best_results = {}
    
    param_config = {keys[i]: combination[i] for i in range(len(keys))}
    dataset_name = param_config.pop('dataset')
    accuracies = []
    
    data = load_graph(dataset_name)
    print("Dataset loaded...")
    
    for seed in tqdm(seeds, desc="Training on different seeds"): 
        seed_everything(seed=seed)
        data = split(data)
        data = data.to(device)
        test_accuracy = main(param_config, data)
        accuracies.append(test_accuracy)
            
    mean = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    if dataset_name not in best_results or mean > best_results[dataset_name]['accuracy']:
        best_results[dataset_name] = {'config': param_config, 'accuracy': mean, 'std': std_deviation}

    return best_results


def main(params, data):
    model = GIN(in_channels=data.x.shape[1], 
                mlp_hidden_channels=params['mlp_hidden_dimension'], 
                mlp_output_channels=params['mlp_output_dimension'],
                mlp_num_layers=params['mlp_num_layers'],
                gin_hidden_channels=params['gin_hidden_dimension_1'],
                gin_output_channels=params['gin_hidden_dimension_2'],
                gin_num_layers=params['gin_num_layers'],
                out_channels=torch.unique(data.y).shape[0],
                dropout_rate=params['dropout_rate'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()


    best_model = None
    prev_val = 0
    count = 0
    prev_epoch = 0
    for epoch in range(1, 501):
        train_loss, train_acc, train_auc = train(data, model, criterion, optimizer)
        val_acc, val_auc, val_loss = val(data, model, criterion)
        
        if val_auc > prev_val: 
                best_model = copy.deepcopy(model)
                prev_val = val_auc
                count = 0
        else:
                count += 1 
        if count == 40:
                break
                
    test_acc, test_auc = test(data, best_model, criterion)      
    return test_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRLGNN')
    parser.add_argument("--folder", type=str, required=True,
            help="folder")
    parser.add_argument("--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--mlp_hidden_dimension", type=int, required=True,
            help="mlp_hidden_dimension")
    parser.add_argument("--mlp_output_dimension", type=int, required=True,
            help="mlp_output_dimension")
    parser.add_argument("--mlp_num_layers", type=int, required=True,
            help="mlp_num_layers")
    parser.add_argument("--gin_hidden_dimension_1", type=int, required=True,
            help="gin_hidden_dimension_1")
    parser.add_argument("--gin_hidden_dimension_2", type=int, required=True,
            help="gin_hidden_dimension_2")
    parser.add_argument("--gin_num_layers", type=int, required=True,
            help="gin_num_layers")
    parser.add_argument("--lr", type=float, required=True,
            help="lr")
    parser.add_argument("--weight_decay", type=float, required=True,
            help="weight_decay")
    parser.add_argument("--dropout_rate", type=float, required=True,
            help="dropout value")
    args = parser.parse_args()
    best_auc = 0
    folder = args.folder
    generate_graph(args.folder, args.dataset)
    best_results = grid_search(args)
    
    with open(folder+args.dataset+"/"+"results.txt", "w") as file:
        for dataset, result in best_results.items():
            file.write(f"Dataset: {dataset}\n")
            file.write(f"Configuration: {result['config']}\n")
            file.write(f"Avg tet accuracy 5 seeds: {result['accuracy']}\n\n")
            file.write(f"Std deviation: {result['std']}\n\n")
    print("Results file saved!")
        
    