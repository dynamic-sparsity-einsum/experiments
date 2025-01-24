import os

import pickle
import random
import sqlite3

import time
from timeit import default_timer as timer

import einsum_benchmark
import torch as pt

from utils import annotate_ssa_path, sort_by_size

db_file_name = "benchmark_results.db"

with sqlite3.connect(db_file_name) as conn:
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS dense_results
                (
                id INTEGER PRIMARY KEY,
                instance_id INTEGER,
                name TEXT, 
                flops REAL,
                size REAL,
                dense_runtime REAL,
                path TEXT)"""
    )

    conn.commit()


def main():

    instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    factor = 168
    for instance_id in range(factor * instance_id, factor * instance_id + factor):
        with open("./instance_id_to_name.pkl", "rb") as f:
            instance_names = pickle.load(f)

        instance = einsum_benchmark.instances[instance_names[instance_id]]

        if instance.name in [
            "mc_2021_arjun_007",
            "mc_2020_175",
            "mc_2022_arjun_069",
            "gm_Promedas_33",
            "mc_2020_175",
            "mc_2022_arjun_037",
            "mc_2023_arjun_071",
            "mc_2022_085",
            "mc_rw_c1908.isc",
        ]:
            path_meta = instance.paths.opt_size
            path_type = "size"
        else:
            path_meta = instance.paths.opt_flops
            path_type = "flops"
        flops = path_meta.flops
        size = path_meta.size
        print(instance_id, instance.name, flops, size, flush=True)

        path = path_meta.path
        format_string = instance.format_string
        tensors = [pt.tensor(t) for t in instance.tensors]

        ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
        annotated_ssa_path = annotate_ssa_path(format_string, ssa_path, tensors)
        sorted_ssa_path = sort_by_size(annotated_ssa_path, tensors, False)

        start_time = timer()
        result = einsum_benchmark.meta.runtime.jensum(
            sorted_ssa_path, *tensors  # , debug=True
        )
        end_time = timer()
        dense_runtime = end_time - start_time

        print("Dense runtime", dense_runtime)

        with sqlite3.connect(db_file_name) as conn:
            c = conn.cursor()
            # Insert the results
            while True:
                try:
                    c.execute(
                        """INSERT INTO dense_results(instance_id,name,flops,size,dense_runtime,path) VALUES 
                            (?, ?, ?, ?, ?, ?)""",
                        (
                            instance_id,
                            instance.name,
                            flops,
                            size,
                            dense_runtime,
                            path_type,
                        ),
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    print("Error inserting into database, trying again")
                    time.sleep(random.uniform(0.1, 1.0))


if __name__ == "__main__":
    main()
