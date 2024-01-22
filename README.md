NeuroSynt: A Neuro-symbolic Portfolio Solver for Reactive Synthesis
===================================================================

We introduce NeuroSynt, a neuro-symbolic portfolio solver framework for reactive synthesis. At the core of the solver lies a seamless integration of neural and symbolic approaches to solving the reactive synthesis problem. To ensure soundness, the neural engine is coupled with model checkers verifying the predictions of the underlying neural models. 

The open-source implementation of NeuroSynt provides an integration framework for reactive synthesis in which new neural and state-of-the-art symbolic approaches can be seamlessly integrated. Extensive experiments demonstrate its efficacy in handling challenging specifications, enhancing the state-of-the-art reactive synthesis solvers, with NeuroSynt contributing novel solves in the current SYNTCOMP benchmarks.

Our tool relies on the open-source [Python library ML2](https://github.com/reactive-systems/ml2), in which we integrated our neural solver for reactive synthesis.


Requirements
------------

- [Docker](https://www.docker.com>)
- [Python 3.8](https://www.python.org/dev/peps/pep-0569/)

Note on Docker: For integrating a variety of symbolic solvers and tools, we use Docker. Each tool is encapsulated in a Docker container that is automatically pulled and started when the tool is needed. Thus, we require Docker to be installed and running.


Installation
------------

To install NeuroSynt, clone the git repo and install with pip as follows:

```shell
    git clone https://github.com/reactive-systems/neurosynt.git && \
    cd neurosynt && \
    pip install .
```

For development, install with pip in editable mode and include the development dependencies as follows:

```shell
    pip install -e .[dev]
```

Synthesis
---------

In this mode, we select a single instance for solving. We use the ```configs/small_local.yaml``` config file, the ```simple_arbiter2.json``` (or ```.tlsf``` for using SyFCo as translator). The flag ```all-results``` indicates that we want to wait for both tools to finish. It can be omitted for only showing the fastest result.

To check the installation, run a small version of the model:

```shell
python -m neurosynt.main synthesize --spec simple_arbiter2.tlsf --config configs/small_local.yaml --all-results 
```

After startup messages, you should see the following (with varying times):

```
symbolic_solver
0.6009421348571777
REALIZABLE
aag 3 2 1 2 0
[...]
neural_solver
8.540999174118042
SATISFIED
REALIZABLE
aag 3 2 1 2 0
[...]
```

After the run is finished, the script kills running instances of symbolic and neural solvers. That can lead to exceptions printed to the command line. 

For a full model evaluation (needs appropriate compute resources) run

```shell
python -m neurosynt.main synthesize --spec simple_arbiter2.tlsf --config configs/strix_local.yaml --all-results 
```

Config Files
------------

We provide some exemplary config files for this artifact.

We use config files to configure the run arguments of the different tools.

 - ```configs/small_*```: Run a small version of the neural solver that should work on most setups. Yields worse results than the standard model reported in the paper. Uses [Spot](https://spot.lre.epita.fr/) as model-checker and [Strix](https://github.com/meyerphi/strix) as symbolic solver. ```docker``` runs the neural solver in a docker container, ```local``` directly in the current python environment and ```standalone``` just configures the port on which NeuroSynt expects the neural solver to run (50072).
 - ```configs/strix_*```: Run the full version of the neural solver. Needs appropriate compute resources.  Uses [Spot](https://spot.lre.epita.fr/) as model-checker and [Strix](https://github.com/meyerphi/strix) as symbolic solver. ```docker``` runs the neural solver in a docker container, ```local``` directly in the current python environment and ```standalone``` just configures the port on which NeuroSynt expects the neural solver to run (50072).
 - ```configs/bosy_*```: Run the full version of the neural solver. Needs appropriate compute resources.  Uses [Spot](https://spot.lre.epita.fr/) as model-checker and [BoSy](https://github.com/reactive-systems/bosy) as symbolic solver. ```docker``` runs the neural solver in a docker container, ```local``` directly in the current python environment and ```standalone``` just configures the port on which NeuroSynt expects the neural solver to run (50072).


 We encourage to change the config files file by adjusting ```num_properties```, ```length_properties``` and ```beam_size```. The larger these values, the better the performance but the more memory is needed. We refer to the paper for an explanation of ```num_properties```, ```length_properties```. ```beam_size``` controls the number of beams in the beam search of the Transformer model.


Benchmarking
------------

In this mode, we select a dataset ```sc-1-f``` (SYNTCOMP 2022 benchmark) that is automatically downloaded and read from ```~/ml2-storage/ltl-spec```. We write an output CSV file to ```neurosynt-bm``` (in ```~/ml2-storage/ltl-syn```). We prepared a small dataset (```sc-1-small```) of 20 samples from SYNTCOMP that is likely to run on most setups with the ```configs/small_*.yaml``` config files.

```python -m neurosynt.main benchmark --dataset sc-1-small --config configs/small_local.yaml --save-as neurosynt-bm-small```

Alternatively, choose 20 random instances from the SYNTCOMP benchmark.

```python -m neurosynt.main benchmark --dataset sc-1-f --config configs/small_local.yaml --save-as neurosynt-bm-sample  --sample 20```


The full benchmark can be reproduced with the following commands:

```python -m neurosynt.main benchmark --dataset sc-1-f --config configs/strix_local.yaml --save-as neurosynt-bm-full-strix```
```python -m neurosynt.main benchmark --dataset sc-1-f --config configs/bosy_local.yaml --save-as neurosynt-bm-full-bosy```