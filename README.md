# SLRGNN-Structural-Link-Representation-Graph-Neural-Network

Official repository for SLRGNN model.

### Requirements

To install the requirements for the project, if you are using CPU, run:
```bash
conda env create -f environment_cpu.yml
```
Otherwise, if GPU is available:
```bash
conda env create -f environment_GPU.yml
```
Then activate environment:
```bash
conda activate slrgnn
```
### Training

To train our model, navigate to the main folder and change the following variables in the `run.sh` file:

```sh
dataset="BUP"
folder="/.../SLRGNN-Structural-Link-Representation-Graph-Neural-Network/data/"
```

then:
```bash
bash run.sh
```