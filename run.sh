#!/bin/bash
dataset="BUP"
folder="/.../SLRGNN-Structural-Link-Representation-Graph-Neural-Network/data/"
file_name="configuration.txt"
file_path="${folder}${dataset}/${file_name}"

mlp_hidden_dimension=$(grep 'mlp_h_dim=' "$file_path" | awk -F'=' '{print $2}')
mlp_output_dimension=$(grep 'mlp_o_dim=' "$file_path" | awk -F'=' '{print $2}')
mlp_num_layers=$(grep 'mlp_num_layers=' "$file_path" | awk -F'=' '{print $2}')

gin_hidden_dimension_1=$(grep 'gin_h_dim_1=' "$file_path" | awk -F'=' '{print $2}')
gin_hidden_dimension_2=$(grep 'gin_h_dim_2=' "$file_path" | awk -F'=' '{print $2}')
gin_num_layers=$(grep 'gin_num_layers=' "$file_path" | awk -F'=' '{print $2}')

lr=$(grep 'lr=' "$file_path" | awk -F'=' '{print $2}')
weight_decay=$(grep 'lr=' "$file_path" | awk -F'=' '{print $2}')
dropout_rate=$(grep 'lr=' "$file_path" | awk -F'=' '{print $2}')


python3 main.py --folder ${folder} --dataset ${dataset} --mlp_hidden_dimension ${mlp_hidden_dimension} --mlp_output_dimension ${mlp_output_dimension} --mlp_num_layers ${mlp_num_layers} --gin_hidden_dimension_1 ${gin_hidden_dimension_1} --gin_hidden_dimension_2 ${gin_hidden_dimension_2} --gin_num_layers ${gin_num_layers} --lr ${lr} --weight_decay ${weight_decay} --dropout_rate ${dropout_rate}