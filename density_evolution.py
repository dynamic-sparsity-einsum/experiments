import os

import pickle
import random
import sqlite3

import time
import einsum_benchmark
import numpy as np
import torch as pt

from utils import annotate_ssa_path, sort_by_size

db_file_name = "benchmark_results.db"

with sqlite3.connect(db_file_name) as conn:
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS density_evolution
            (
                name TEXT,
                flops REAL,
                size REAL,
                avg_density REAL,
                step INT,
                step_out_size REAL,
                jump BOOLEAN,
                num_elements INT,
                remaining_tensors INT
            )"""
    )


def store_result(results):
    with sqlite3.connect(db_file_name) as conn:
        c = conn.cursor()
        # Insert the results
        while True:
            try:
                c.executemany(
                    """INSERT INTO density_evolution VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    results,
                )
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                print("Error inserting into database, trying again")
                print(e)
                time.sleep(random.uniform(0.1, 1.0))


def main():
    instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    factor = 168
    for instance_id in range(factor * instance_id, factor * instance_id + factor):
        with open("./instance_id_to_name.pkl", "rb") as f:
            instance_names = pickle.load(f)

        instance = einsum_benchmark.instances[instance_names[instance_id]]
        # Check if all tensor shapes only have dimenstions of size 2

        # Buggy flops paths (memory issues (oom, segfaul), extremely bad performance), so take size path
        if instance.name in [
            "mc_2021_arjun_007",
            "mc_2020_175",
            "mc_2022_arjun_069",
            "gm_Promedas_33",
            "mc_2022_arjun_037",
            "mc_2023_arjun_071",
            "mc_2022_085",
            "mc_rw_c1908.isc",
        ]:
            path_meta = instance.paths.opt_size
        else:
            path_meta = instance.paths.opt_flops
        flops = path_meta.flops
        size = path_meta.size
        print(
            instance_id, instance.name, flops, size, len(instance.tensors), flush=True
        )

        name = instance.name
        path = path_meta.path
        format_string = instance.format_string
        tensors = [pt.tensor(t) for t in instance.tensors]

        ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
        annotated_ssa_path = annotate_ssa_path(format_string, ssa_path, tensors)
        annotated_ssa_path = sort_by_size(annotated_ssa_path, tensors)

        remaining_tensors = len(tensors)

        step = 0
        max_log2_size = 0

        num_elements = 0
        num_nonzeroes = 0
        for tensor in tensors:
            numel = tensor.numel()
            num_elements += numel
            if (max_log2_size) < np.log2(numel):
                max_log2_size = np.log2(numel)
            non_zero = tensor.count_nonzero()
            num_nonzeroes += non_zero

        avg_density = (num_nonzeroes / num_elements).item()
        print(step, "of", len(annotated_ssa_path), "", avg_density, flush=True)
        results = []
        results.append(
            (
                name,
                flops,
                size,
                avg_density,
                step,
                0,
                False,
                num_elements,
                remaining_tensors,
            )
        )

        step += 1
        original_dtype = tensors[0].dtype

        for (
            first,
            second,
            expression,
            t3_log_size,
            *_,
        ) in annotated_ssa_path:
            t1, t2 = tensors[first], tensors[second]
            t3 = pt.einsum(expression, t1, t2)
            non_zero = t3.count_nonzero()
            num_nonzeroes = num_nonzeroes + non_zero
            num_elements = num_elements + t3.numel()
            if t3.dtype != original_dtype:
                # cast to original dtype
                t3 = t3.type(original_dtype)
            tensors.append(t3)
            tensors[first] = None
            tensors[second] = None
            jump = False
            if (max_log2_size) < t3_log_size:
                max_log2_size = t3_log_size
                jump = True

            avg_density = (num_nonzeroes / num_elements).item()
            remaining_tensors -= 1
            results.append(
                (
                    name,
                    flops,
                    size,
                    avg_density,
                    step,
                    t3_log_size,
                    jump,
                    num_elements,
                    remaining_tensors,
                )
            )
            step += 1
        store_result(results)


if __name__ == "__main__":
    main()
