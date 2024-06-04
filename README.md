# Simultaneously Detecting Node and Edge Level Anomalies on Heterogeneous Attributed Graphs

This repository contains the experimental source code of the **"Simultaneously Detecting Node and Edge Level Anomalies on Heterogeneous Attributed Graphs"** paper presented at the [IEEE World Congress on Computational Intelligence (WCCI) 2024](https://2024.ieeewcci.org/). 

Authors: [Rizal Fathony](mailto:rizal.fathony@grab.com), [Jenn Ng](mailto:jenn.ng@grab.com), and [Jia Chen](mailto:jia.chen@grab.com).

## Abstract

In complex systems like social media and financial transactions, diverse entities (users, groups, products) interact through a multitude of relationships (friendships, comments, purchases). These interactions can be represented by heterogeneous graphs (graphs with many node and edge types). In many real-world applications, these graphs may contain unusual patterns or anomalies. Detecting anomalies, both entity (node) level and interaction (edge) level anomalies, in these graphs are important, as their occurrence may have serious implications. Node-level anomalies may indicate abnormal behavior from a specific entity, such as unexpected activity that could suggest fraud. Edge-level anomalies may signify unusual interactions, like unexpected changes in interaction frequency or pattern, potentially indicating collaborative fraud like collusion.

Unfortunately, existing graph neural network anomaly detection models focus only on homogeneous graphs and consider only node-level detection, rendering them incapable of harnessing the full complexity of heterogeneous graph data. To address this limitation, we present a new graph neural network model that capable of simultaneously detecting node-level and edge-level anomalies on heterogeneous graphs, by harnessing the rich information in the entities and relations. We develop our model as a type of graph autoencoder with a customized architecture design to enable the detection of node-level and edge-level anomalies simultaneously. Our graph neural network structure is scalable, facilitating its application in large real-world scenarios. Finally, our method outperforms previous anomaly detection methods in the experiments.

## Setup

1. Install the required packages using:
    ```
    pip install -r requirements.txt
    ```
2. Download the datasets.

    - `telecom`: https://snap.stanford.edu/data/telecom-graph.html
    - `reddit`:  https://snap.stanford.edu/data/web-Reddit.html
    - `gowalla`: https://snap.stanford.edu/data/loc-Gowalla.html
    - `brightkite`: https://snap.stanford.edu/data/loc-Brightkite.html


## Construct Graph Datasets

Please check `data_{dataset_name}.py` for the graph construction. It contains two functions:
- `create_graph()`: read the raw data and convert it to a heterogeneous graph representation in Pytorch Geometric's `HeteroData`.
- `synth_random_anomalies()`: inject anomalies into the graph using `anomaly_insert.py` functions, to create multiple copies of the graph for multi-run experiments.

## Run Experiment

To run the experiments, please execute the corresponding file for each model. 

1. `HeagNet-C`: 
    ```
    python train_{dataset}.py --id 0
    ```

1. `HeagNet-A` (attention mechanism): 
    ```
    python train_{dataset}_att.py --id 0
    ```

1. `IsolationForest`: 
    ```
    python isoforest_experiment.py --name {dataset} --id 0
    ```

1. Homogeneous GNN models: 
    ```
    python pygod_experiment.py --name {dataset} --method {model_name} --id 0
    ```

    Note: `transform_graph.py` code transforms the heterogeneous graphs to a homogeneous ones.


The argument `--id` indicates the instance of anomaly injected graph [0-9] used in training.


## License

This repository is licensed under the [MIT License](LICENSE).

## Citation

If you use this repository for academic purpose, please cite the following paper:


> R. Fathony, J. Ng and J. Chen, "Simultaneously Detecting Node and Edge Level Anomalies on Heterogeneous Attributed Graphs". 2024. IEEE World Congress on Computational Intelligence (WCCI) 2024.