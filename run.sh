#!/bin/bash
dataset="BUP"
folder="/.../SLRGNN-Structural-Link-Representation-Graph-Neural-Network/data/"

lr=0.001
hidden_dimension=32
weight_decay=0.0005
num_mlp_layers=2
num_initial_gin_layers=2
num_gin_layers=2
dropout_rate=0.5

python3 main.py --folder ${folder} --dataset ${dataset} --lr ${lr} --hidden_dimension ${hidden_dimension} --weight_decay ${weight_decay} --num_mlp_layers ${num_mlp_layers} --num_initial_gin_layers ${num_initial_gin_layers} --num_gin_layers ${num_gin_layers} --dropout_rate ${dropout_rate}