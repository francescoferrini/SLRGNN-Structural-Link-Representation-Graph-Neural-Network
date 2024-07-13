# SLRGNN-Structural-Link-Representation-Graph-Neural-Network

This is the official repository for the paper "A Simple and Expressive Graph Neural Network Based Method for Structural Link Representation", which was accepted at the **GRaM** workshop at ICML24 (Geometry-grounded Representation Learning and Generative Modeling Workshop). Here, you will find the official implementation of **SLRGNN**.  

[A Simple and Expressive Graph Neural Network Based Method for Structural Link Representation](https://openreview.net/pdf?id=EGGSCLyVrz) 



## Requirements

To install the requirements for the project, if you are using CPU, run:
```bash
conda env create -f environment_cpu.yml
```
Otherwise, if GPU is available:
```bash
conda env create -f environment_gpu.yml
```
Then activate the environment:
```bash
conda activate slrgnn
```
## Training

To train our model, navigate to the main folder and change the following variables in the `run.sh` file with:

```sh
dataset="BUP"
folder="/.../SLRGNN-Structural-Link-Representation-Graph-Neural-Network/data/"
```

then:
```bash
bash run.sh
```

## Cite

Please cite our [paper](https://openreview.net/pdf?id=EGGSCLyVrz) if you find **SLRGNN** useful in your work:

```
@inproceedings{lachi2024a,
title={A Simple and Expressive Graph Neural Network Based Method for Structural Link Representation},
author={Veronica Lachi and Francesco Ferrini and Antonio Longa and Bruno Lepri and Andrea Passerini},
booktitle={ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling},
year={2024},
url={https://openreview.net/forum?id=EGGSCLyVrz}
}
```

## Contacts

If you have any question:

 - Veronica Lachi &rarr;  vlachi (at) fbk (dot) eu
 - Francesco Ferrini &rarr; francesco (dot) ferrini (at) unitn (dot) it

