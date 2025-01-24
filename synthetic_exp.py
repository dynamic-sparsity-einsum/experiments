import sqlite3
from collections import defaultdict
from timeit import default_timer as timer

import einsum_benchmark
import numpy as np
import opt_einsum
import sesum.sr as sr
import torch


db_path = "synth_results.db"
table_name = "results"


def store_results(db_path, table_name, n, dense_runtime, sparse_runtime):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        n INTEGER,
        dense_runtime REAL,
        sparse_runtime REAL
    )
    """
    )

    # Insert results
    cursor.execute(
        f"""
    INSERT INTO {table_name} (n, dense_runtime, sparse_runtime)
    VALUES (?, ?, ?)
    """,
        (n, dense_runtime, sparse_runtime),
    )

    conn.commit()
    conn.close()


def var_ij(i, j):
    return n * i + j + 1


# Mappging simplified from einsum_benchmark package
def clause_to_tensor(clause):
    input = ""  # start a new clause
    index_tuple = ()  # index tuple for clause
    index_clause = []
    for variable in clause:
        input += opt_einsum.get_symbol(abs(variable))
        index_tuple += (1 if variable < 0 else 0,)
        index_clause.append(variable)
    tensor = torch.ones([2] * len(index_tuple), dtype=torch.float32)
    tensor[index_tuple] = 0
    return input, tensor


def get_grid_tn(n):
    # Generate the grid clauses from the paper
    clauses = [
        c
        for i in range(n - 1)
        for j in range(n)
        for c in [[var_ij(i + 1, j), -var_ij(i, j)], [-var_ij(i + 1, j), var_ij(i, j)]]
    ]
    clauses.extend(
        [
            c
            for i in range(n)
            for j in range(n - 1)
            for c in [
                [var_ij(i, j + 1), -var_ij(i, j)],
                [-var_ij(i, j + 1), var_ij(i, j)],
            ]
        ]
    )
    inputs, tensors = zip(*[clause_to_tensor(clause) for clause in clauses])

    # Merge each pear of neighboring tensors
    merged_tensors = []
    merged_inputs = []
    for i in range(0, len(tensors) - 1, 2):
        merged_tensors.append(torch.einsum("ab,ab->ab", tensors[i], tensors[i + 1]))
        merged_inputs.append(inputs[i])
    eq = ",".join(merged_inputs) + "->"

    return eq, merged_tensors


def get_path(n):
    high = n * (n - 1)
    path = [(0, high)]
    ssa_id = len(merged_tensors)
    for i in range(1, high):
        path.append((ssa_id, i))
        ssa_id += 1
        path.append((ssa_id, high + i))
        ssa_id += 1
    return path


if __name__ == "__main__":
    for n in range(2, 129):
        print("====>", n, flush=True)
        for i in range(5):
            eq, merged_tensors = get_grid_tn(n)
            path = get_path(n)
            annotated_ssa_path = einsum_benchmark.meta.runtime.to_annotated_ssa_path(
                eq,
                ssa_path=path,
            )
            if n < 31:
                tic = timer()
                result = einsum_benchmark.meta.runtime.jensum(
                    annotated_ssa_path, *merged_tensors, debug=False
                )
                dense_runtime = timer() - tic
                print(result, dense_runtime, flush=True)
            else:
                dense_runtime = None

            tic = timer()
            result = sr.sesum(
                eq,
                *merged_tensors,
                path=path,
                backend="sparse",
                debug=False,
                dtype=np.float32,
            )
            sparse_runtime = timer() - tic
            print(result, sparse_runtime, flush=True)

            # Store results
            store_results(db_path, table_name, n, dense_runtime, sparse_runtime)

            # Example usage
