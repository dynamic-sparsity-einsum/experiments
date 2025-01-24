import os
import random
import signal
import sqlite3
import time
from timeit import default_timer as timer

import einsum_benchmark
import numpy as np
import sesum as sr

from utils import rewrite_to_dim2

db_file_name = "benchmark_results.db"

with sqlite3.connect(db_file_name) as conn:
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS sparse_results_float64
                (
                id INTEGER PRIMARY KEY,
                name TEXT,
                flops REAL,
                size REAL,
                sparse_runtime REAL,
                path TEXT)"""
    )

    conn.commit()


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def main():
    instance_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

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
    factor = 11
    for instance_id in range(factor * instance_id, factor * instance_id + factor):
        instance = einsum_benchmark.instances[int_instances[instance_id]]

        # Buggy flops patha (memory issues, ridicously bad performance), so take size path
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
            path_type = "size"
        else:
            path_meta = instance.paths.opt_flops
            path_type = "flops"
        flops = path_meta.flops
        size = path_meta.size
        print(instance_id, instance.name, flops, size, flush=True)
        path = path_meta.path
        ssa_path = einsum_benchmark.meta.runtime.to_ssa_path(path)

        tensors = instance.tensors
        format_string = instance.format_string

        # Check if all tensor shapes only have dimenstions of size 2
        if not all(
            all(dim == 2 for dim in tensor.shape) for tensor in instance.tensors
        ):
            format_string, tensors = rewrite_to_dim2(
                instance.format_string, instance.tensors
            )

        tensors = [t.astype(np.float64) for t in tensors]
        print(tensors[0].dtype)

        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            # Start the timer
            signal.alarm(14400)
            start_time = timer()
            result = sr.sesum(
                format_string,
                *tensors,
                path=ssa_path,
                backend="sparse",
                dtype=np.float64,
            )
            end_time = timer()
            sparse_runtime = end_time - start_time
            print("Sparse runtime", sparse_runtime, flush=True)
            # Disable the alarm
            signal.alarm(0)
        except TimeoutError as e:
            print("Function timed out")
            sparse_runtime = None
            # Handle timeout case
        finally:
            # Ensure alarm is disabled
            signal.alarm(0)

        print("Inserting result", flush=True)
        with sqlite3.connect(db_file_name) as conn:
            c = conn.cursor()
            # Insert the results
            while True:
                try:
                    c.execute(
                        """INSERT INTO sparse_results_float64(name,flops,size,sparse_runtime,path) VALUES 
                            (?, ?, ?, ?, ?)""",
                        (
                            instance.name,
                            flops,
                            size,
                            sparse_runtime,
                            path_type,
                        ),
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError:
                    print("Error inserting into database, trying again", flush=True)
                    time.sleep(random.uniform(0.1, 1.0))


if __name__ == "__main__":
    main()
