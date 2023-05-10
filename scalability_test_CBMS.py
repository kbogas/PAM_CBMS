import gc
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pam_creation import get_sparsity
from utils import get_prime_map_from_rel

# Path to projects
# Please download them and change the paths accordingly.
# In this script they are all expected to be in the ./data folder in the same directory
project_to_path = {
    "DDB14": "./data/DDB14/train.txt",
    "Hetionet": "./data/Hetionet/hetionet-v1.0-edges.tsv",
    "PharmKG": "./data/PharmKG/PharmKG-180K.zip",
    "DRKG": "./data/DRKG/train.tsv",
    "PrimeKG": "./data/PrimeKG/train.csv",
}

res = []
max_order = 4

for project_name, path in project_to_path.items():

    print(project_name)

    # specific loaders for each dataset
    if project_name == "Hetionet":
        df_train = pd.read_csv(path, sep="\t")
        df_train.dropna(inplace=True)
        df_train.columns = ["head", "rel", "tail"]
    elif project_name == "PharmKG":
        df_train = pd.read_csv(
            path, usecols=["Entity1_name", "relationship_type", "Entity2_name"]
        )
        df_train.dropna(inplace=True)
        df_train.columns = ["head", "rel", "tail"]
    elif project_name == "DRKG":
        df_train = pd.read_csv(path, sep="\t")
        df_train.dropna(inplace=True)
        df_train.columns = ["head", "rel", "tail"]
    elif project_name == "PrimeKG":
        df_train = pd.read_csv(path, sep=",", usecols=["x_name", "relation", "y_name"])
        df_train.dropna(inplace=True)
        df_train = df_train[["x_name", "relation", "y_name"]]
        df_train.columns = ["head", "rel", "tail"]
    elif project_name == "ogbl-wikikg2":
        dataset = LinkPropPredDataset(name="ogbl-wikikg2")
        graph = dataset[0]
        df_train = pd.DataFrame(graph["edge_index"]).T
        df_train.columns = ["head", "tail"]
        df_train["rel"] = graph["edge_reltype"]
        df_train.shape
    else:
        raise NotImplementedError(f"{project_name} not understood...")

    # Statistics
    unique_rels = sorted(list(df_train["rel"].unique()))
    unique_nodes = sorted(
        set(df_train["head"].values.tolist() + df_train["tail"].values.tolist())
    )
    print(
        f"# of unique rels: {len(unique_rels)} \t | # of unique nodes: {len(unique_nodes)}"
    )

    # Map nodes to indices of PAM
    node2id = {}
    id2node = {}
    for i, node in enumerate(unique_nodes):
        node2id[node] = i
        id2node[i] = node

    time_s = time.time()

    # Map the relations to primes
    rel2id, id2rel = get_prime_map_from_rel(
        unique_rels, starting_value=2, spacing_strategy="step_1"
    )

    # Create the adjacency matrix
    df_train["rel_mapped"] = df_train["rel"].map(rel2id)
    df_train["head_mapped"] = df_train["head"].map(node2id)
    df_train["tail_mapped"] = df_train["tail"].map(node2id)
    A_big = csr_matrix(
        (df_train["rel_mapped"], (df_train["head_mapped"], df_train["tail_mapped"])),
        shape=(len(unique_nodes), len(unique_nodes)),
    )

    # Calculate sparsity
    sparsity = get_sparsity(A_big)
    print(A_big.shape, f"Sparsity: {sparsity:.2f} %")

    time_prev = time.time()
    time_setup = time_prev - time_s
    print(f"Total setup: {time_setup:.5f} secs ({time_setup/60:.2f} mins)")

    # Generate the PAM^k matrices
    power_A = [A_big]
    for ii in range(1, max_order):
        updated_power = power_A[-1] * A_big

        power_A.append(updated_power)
        latest_sparsity = get_sparsity(updated_power)
        print(f"Sparsity {ii + 1}-hop: {latest_sparsity:.2f} %")

    time_stop = time.time()
    time_calc = time_stop - time_prev
    print(f"A^k calc time: {time_calc:.5f} secs ({time_calc/60:.2f} mins)")

    time_all = time_stop - time_s
    print(f"All time: {time_all:.5f} secs ({time_all/60:.2f} mins)")
    res.append(
        {
            "dataset": project_name,
            "nodes": len(unique_nodes),
            "rels": len(unique_rels),
            "edges": df_train.shape[0],
            "sparsity_start": sparsity,
            "sparsity_end": get_sparsity(power_A[-1]),
            "setup_time": time_setup,
            "cacl_time": time_calc,
            "total_time": time_all,
        }
    )
    print(res[-1])
    print("\n\n")
    del df_train
    del A_big
    del power_A
    gc.collect()

res = pd.DataFrame(res)
res.sort_values("edges", inplace=True)
print(res.to_string())
print("\n\n")
