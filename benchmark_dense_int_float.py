import os
import random
import sqlite3

import time
from timeit import default_timer as timer

import einsum_benchmark
import torch as pt

from utils import annotate_ssa_path

db_file_name = "benchmark_results.db"

with sqlite3.connect(db_file_name) as conn:
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS dense_results_float64
                (
                id INTEGER PRIMARY KEY,
                name TEXT, 
                flops REAL,
                size REAL,
                dense_runtime REAL,
                path TEXT)"""
    )

    conn.commit()


def main():
    instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    int_instances = [
        "mc_2020_017",
        "mc_2020_062",
        "mc_2020_arjun_042",
        "mc_2020_arjun_046",
        "mc_2020_arjun_057",
        "mc_2021_027",
        "mc_2021_065",
        "mc_2021_074",
        "mc_2022_085",
        "mc_2022_167",
        "mc_2022_arjun_069",
        "mc_2023_199",
        "mc_2023_arjun_071",
        "mc_rw_32.sk_4_38",
        "mc_rw_blasted_case3",
        "mc_rw_blasted_squaring12",
        "mc_rw_blockmap_05_01.net",
        "mc_rw_c1908.isc",
        "mc_rw_or-70-5-6-UC-10",
        "mc_rw_s510.bench",
        "mc_rw_s510_15_7",
    ]
    factor = 21
    for instance_id in range(factor * instance_id, factor * instance_id + factor):
        instance = einsum_benchmark.instances[int_instances[instance_id]]

        # Buggy flops patha (memory issues, ridicously bad performance), so take size path
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
        tensors = [pt.tensor(t) for t in instance.tensors]

        ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)
        annotated_ssa_path = einsum_benchmark.meta.runtime.to_annotated_ssa_path(
            format_string=instance.format_string, ssa_path=ssa_path
        )
        del tensors
        tensors = [pt.tensor(t).type(pt.float64) for t in instance.tensors]

        start_time = timer()
        result = einsum_benchmark.meta.runtime.jensum(annotated_ssa_path, *tensors)
        end_time = timer()
        dense_runtime = end_time - start_time
        print(result.sum(), instance.result_sum)
        print("Dense runtime", dense_runtime)

        with sqlite3.connect(db_file_name) as conn:
            c = conn.cursor()
            # Insert the results
            while True:
                try:
                    c.execute(
                        """INSERT INTO dense_results_float64(name,flops,size,dense_runtime,path) VALUES 
                            (?, ?, ?, ?, ?)""",
                        (
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
