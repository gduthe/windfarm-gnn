# Wind Farm GNNs

This repository is dedicated to the development and exploration of multivariate, system-wide approaches based on Graph Neural Networks (GNN) for predicting wind farm-wide power, local flow variables and loads. Our models aim to predict average power, rotor averaged wind speed and effective turbulence intensity, along with damage equivalent moments (DEL) in blade flap- and edgewise directions, tower top tortional and bottom fore-aft and side-to-side. The use of GNNs allows for a layout-agnostic multivariate model.

## Features
- **Robust Data Generation**: We use Sobol sampling to generate robust datasets with realistic inflow conditions and randomized layouts.
- **Parallelized Data Generation**: We parallelize [PyWake](https://topfarm.pages.windenergy.dtu.dk/PyWake/) to quickly generate some training data.
- **Multivariate GNN Models**: Leverage the power of graph neural networks to predict wind farm performance and loads. Three default message-passing implementations are provided.
- **Layout-Agnostic**: Capable of working with any wind farm layout.

## Getting Started

To get started with this project, clone the repository and install the required dependencies.

### Prerequisites

Ensure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/) or using [Anaconda](https://www.anaconda.com/) ([Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) is the best for a barebones lightweight install).

### Installation

1. Clone the repository:
`git clone https://github.com/GDuthe/windfarm-gnn.git`

2. Install dependencies: `pip install -r requirements.txt`

## Project structure
```
windfarm-gnn/
│
├── gnn_framework/
│ ├── train.py              # script to train the GNN model
│ ├── config.yml            # template training config file
│ ├── predict.py            # evaluation of the GNN model
│ ├── finetuning/           # fine-tuning submodule
│ │    ├── finetune.py      # fine-tuning script
│ │    ├── ft_config.yml    # fine-tuning config file
│ │ 
│ ├── data/                 # data submodule (dataset class & more)
│ ...
│
├── graphfarms/
│ ├── generate_graphs.py    # script to generate wind farm graphs
│ ├── config.yml            # template data generation config file
│ ...
│
├── notebooks/              # example usage notebooks
│ ...
│
├── plotting/               # plotting functions
│ ...
```
## Usage

### Generating wind farm graph data

Navigate to the `graph_farms` folder.

To generate some training graphs with random layouts using PyWake and realistic random inflow conditions, you must first set up a config yaml file, please take a look the provided template. Then, you can launch the graph generation script. For example you can use: 

`python3 generate_graphs.py -c custom_config.yml -nl 200 -ni 15 -d ./your/custom/dataset/path -t 6` 

This command will generate a dataset with 200 layouts, each being simulated with 15 different inflows, using 6 threads. Some other arguments can be set and are detailed in `generate_graphs.py`.


### Training a Model

Navigate into the `gnn_framework` directory.

To train a model, you must first set up a config yaml file. Please use the provided `config.yml` template as a starting point. Then, use the following command to launch the training script:

`python3 train.py -c your_config.yml`


### Finetuning a Model

If you would like to finetune your trained model on different data (higher fidelity, new turbines, etc.) or a specific layout, you can use the provided finetuning submodule. Three different finetuning methods are implemented (LoRA, all-layers, decoder only).

Again you must first set up a config file, have a look at the template provided in `finetuning/ft_config.yml`. To launch the fintuning process, stay in the `gnn_framework` directory and run: 

`python3 -m finetuning.finetune -y your_ft_config.yml`

### Testing a Model

We provide a function to gather prediction results for a given test dataset in `predict.py`. 


## Citing

If you are using the code, please consider citing: 

> Duthé, G., de Nolasco Santos, F., Abdallah, I., Réthore, P.É., Weijtjens, W., Chatzi, E. and Devriendt, C., 2023, May. Local flow and loads estimation on wake-affected wind turbines using graph neural networks and PyWake. In Journal of Physics: Conference Series (Vol. 2505, No. 1, p. 012014). IOP Publishing.
