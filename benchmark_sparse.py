import os

import pickle
import random
import signal
import sqlite3

import time
from timeit import default_timer as timer

import einsum_benchmark
import sesum as sr

from utils import rewrite_to_dim2

db_file_name = "benchmark_results.db"

with sqlite3.connect(db_file_name) as conn:
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS sparse_results
                (
                id INTEGER PRIMARY KEY,
                instance_id INTEGER,
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
    key = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # for instance_id in range(4 * instance_id, 4 * instance_id + 4):
    factor = 168
    for instance_id in range(factor * key, factor * key + factor):
        with open("./instance_id_to_name.pkl", "rb") as f:
            instance_names = pickle.load(f)

        instance = einsum_benchmark.instances[instance_names[instance_id]]

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
                dtype=instance.tensors[0].dtype,
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
                        """INSERT INTO sparse_results(instance_id,name,flops,size,sparse_runtime,path) VALUES 
                            (?, ?, ?, ?, ?, ?)""",
                        (
                            instance_id,
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
