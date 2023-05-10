import functools
import os
from typing import Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
from sympy import nextprime, primefactors
from sympy.ntheory import factorint


@functools.lru_cache(maxsize=None)
def get_primefactors(value: float) -> Tuple[int]:
    """Wrapper functiom that gets a value and returns the list
       of prime factors of the value. It is used as a wrapper around
       primefactors in ordet ot use memoization with cache for speed.

    Args:
        value (float): The float value to decompose

    Returns:
        Tuple[int]: A list of the unique prime factors
    """
    return primefactors(value)


def get_prime_map_from_rel(
    list_of_rels: list,
    starting_value=1,
    spacing_strategy="step_1",
) -> Tuple[dict, dict]:
    """
    Helper function that given a list of relations returns the mappings to and from the prime numbers used.
    Different strategies to map the numbers are available.
    "step_X", increases the step between two prime numbers by adding X to the current prime
    "factor_X", increases the step between two prime numbers by multiplying the current prime with X

    Args:
        list_of_rels (list): iterable, contains a list of the relations that need to be mapped.
        starting_value (int, optional): Starting value of the primes. Defaults to 1.
        spacing_strategy (str, optional):  Spacing strategy for the primes. Defaults to "step_1".

    Returns:
        rel2prime: dict, relation to prime dictionary e.g. {"rel1":2}.
        prime2rel: dict, prime to relation dictionary e.g. {2:"rel1"}.
    """

    rel2prime = {}
    prime2rel = {}
    current_int = starting_value
    for relid in list_of_rels:
        cur_prime = nextprime(current_int)
        rel2prime[relid] = cur_prime
        prime2rel[cur_prime] = relid
        if "step" in spacing_strategy:
            step = float(spacing_strategy.split("_")[1])
            current_int = cur_prime + step
        elif "factor" in spacing_strategy:
            factor = float(spacing_strategy.split("_")[1])
            current_int = cur_prime * factor
        else:
            raise NotImplementedError(
                f"Spacing strategy : {spacing_strategy}  not understood!"
            )
    return rel2prime, prime2rel


def load_data(
    path_to_folder: str, project_name: str, add_inverse_edges: str = "NO"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Helper function that loads the data in pd.DataFrames and returns them.
    Args:
        path_to_folder (str): path to folder with train.txt, valid.txt, test.txt
        project_name (str): name of the project
        add_inverse_edges (str, optional):  Whether to add the inverse edges.
        Possible values "YES", "YES__INV", "NO". Defaults to "NO".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]: [description]
    """
    PROJECT_DETAILS = {
        "lc-neo4j": {"skiprows": 1, "sep": "\t"},
        "codex-s": {"skiprows": 0, "sep": "\t"},
        "WN18RR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10-DR": {"skiprows": 0, "sep": "\t"},
        "YAGO3-10": {"skiprows": 0, "sep": "\t"},
        "FB15k-237": {"skiprows": 0, "sep": "\t"},
        "NELL995": {"skiprows": 0, "sep": "\t"},
        "DDB14": {"skiprows": 0, "sep": "\t"},
    }

    df_train = pd.read_csv(
        os.path.join(path_to_folder, "train.txt"),
        sep=PROJECT_DETAILS[project_name]["sep"],
        header=None,
        dtype="str",
        skiprows=PROJECT_DETAILS[project_name]["skiprows"],
    )
    df_train.columns = ["head", "rel", "tail"]
    df_train_orig = df_train.copy()
    if "YES" in add_inverse_edges:
        print(f"Will add the inverse train edges as well..")
        df_train["rel"] = df_train["rel"].astype(str)
        df_train_inv = df_train.copy()
        df_train_inv["head"] = df_train["tail"]
        df_train_inv["tail"] = df_train["head"]
        if add_inverse_edges == "YES__INV":
            df_train_inv["rel"] = df_train["rel"] + "__INV"
        df_train = df_train.append(df_train_inv)
    if project_name in ["lc-neo4j"]:
        df_eval = None
        df_test = None
        already_seen_triples = set(df_train.to_records(index=False).tolist())
    else:
        try:
            df_eval = pd.read_csv(
                os.path.join(path_to_folder, "valid.txt"),
                sep=PROJECT_DETAILS[project_name]["sep"],
                header=None,
                dtype="str",
                skiprows=PROJECT_DETAILS[project_name]["skiprows"],
            )
            df_eval.columns = ["head", "rel", "tail"]
        except FileNotFoundError:
            df_eval = df_train.copy()
        df_test = pd.read_csv(
            os.path.join(path_to_folder, "test.txt"),
            sep=PROJECT_DETAILS[project_name]["sep"],
            header=None,
            dtype="str",
            skiprows=PROJECT_DETAILS[project_name]["skiprows"],
        )
        df_test.columns = ["head", "rel", "tail"]
        if "YAGO" in project_name:
            for cur_df in [df_train, df_eval, df_test]:
                for col in cur_df.columns:
                    cur_df[col] = cur_df[col]  # + "_YAGO"

        already_seen_triples = set(
            df_train.to_records(index=False).tolist()
            + df_eval.to_records(index=False).tolist()
        )
    print(f"Total: {len(already_seen_triples)} triples in train + eval!)")
    print(f"In train: {len(df_train)}")
    print(f"In valid: {len(df_eval)}")
    print(f"In test: {len(df_test)}")
    return df_train_orig, df_train, df_eval, df_test, already_seen_triples


def get_egograph(
    whole_df: pd.DataFrame, root: str, radius: int = 2, keep_direction: bool = True
):
    """
    Simple function to generate a (un)directed ego-graph of N-hops from a starting node
    given a KG.
    Input:
    - whole_df: pd.DataFrame,
      KG in a dataframe with columns [head, rel, tail]
    - root: str,
      the label of the root node
    - radius: int,
      the size of the neighborhood around the root node to fetch
    - keep_direction: boolean,
      whether to take directionality into account.
    """
    steps = radius
    seeds = [root]
    subgraph = ()

    while steps > 0:
        if keep_direction:
            cur_subgraph = whole_df[whole_df["head"].isin(seeds)]
            seeds = cur_subgraph["tail"].unique().tolist()
        else:
            cur_subgraph_head = whole_df[whole_df["head"].isin(seeds)]
            seed_head = cur_subgraph_head["tail"].unique().tolist()
            cur_subgraph_tail = whole_df[whole_df["tail"].isin(seeds)]
            seed_tail = cur_subgraph_tail["head"].unique().tolist()
            cur_subgraph = pd.concat((cur_subgraph_head, cur_subgraph_tail), axis=0)
            seeds = list(set(seed_tail + seed_head))
        if len(subgraph) == 0:
            subgraph = cur_subgraph
        else:
            subgraph = pd.concat((subgraph, cur_subgraph), axis=0)
        steps -= 1
    subgraph.drop_duplicates(inplace=True)
    return subgraph.values.tolist()


def set_all_seeds(seed: int = 0):
    """Fix random seeds
    Args:
        seed (int): Random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    return 1


def ILP_solver(denominations, target_value, max_number_of_coins):
    """
    Solver using GLPK_MI for boolean integer linear programming
    :param denominations: list, list of avaialble denominations to break the target value into
    :param target_value: int, target value that needs to be broken into a sum of denominations
    :param max_number_of_coins: int, the number of denominations used to create the target value
    :return: dict, {'denomination_1':times_used1, 'denomination2':times_used2, ...}
    """
    w = cp.Constant(
        denominations,
    )
    CASH = cp.Constant(target_value)
    max_number_of_coins = cp.Constant(max_number_of_coins)

    x = cp.Variable((1, w.shape[0]), integer=True)

    # We want to minimize the total number of coins returned
    objective = cp.Minimize(cp.abs(max_number_of_coins - cp.sum(x)))
    # print(CASH, max_number_of_coins, w)
    # The constraints
    constraints = [
        w @ x.T == CASH,
        # cp.sum(x) == max_number_of_coins, #
        x >= 0,  # semi-positive coins
    ]
    # Form and solve problem.
    prob = cp.Problem(objective, constraints)
    # Need the GLPK_MI solver because the ECOS_BB is not working correctly.
    prob.solve(solver="GLPK_MI")  # Returns the optimal value.
    if prob.status == "infeasible":
        return {}
    else:
        return dict(zip([w_ for w_ in w.value], x.value.flatten()))


def get_sparsity(A: scipy.sparse.csr_matrix) -> float:
    """Calculate sparsity % of scipy sparse matrix.
    Args:
        A (scipy.sparse): Scipy sparse matrix
    Returns:
        (float)): Sparsity as a float
    """

    return 100 * (1 - A.nnz / (A.shape[0] ** 2))
