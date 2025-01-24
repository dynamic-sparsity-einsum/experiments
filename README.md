# Experiments for Exploiting Dynamic Sparsity in Einsum

We recommend to use `uv` to install the required packages. To install `uv`, see their installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Afterwards you can install all dependencies by running:
```bash
uv sync
```

However you may use any other package manager to install the required packages, that supports the `pyproject.toml` file. If you do not want to use `uv`, you need to replace the `uv run` command with `python` in the following commands after activating the right virtual environment.

## Experiments


### Synthetic Data

To run the experiments on synthetic data, you can use the following command:
```bash
uv run synthetic_exp.py
```
The results will be written to a sqlite databsae, `synth_results.db`. The file with our results is already included in the `results/` folder.


### Real Data (Einsum Benchmark)

All scripts in this section will write the results into a sqlite database, `benchmark_results.db`. The file with our results is already included in the `results/` folder. For ease of use we provide the results in a csv file as well. Note that the experimental scripts only run each instance once. In our experiments we ran them several times, but in the CSV file we only report the median runtimes.

#### Density Evolution

To run the density evolution experiments, you can use the following command:
```bash
uv run density_evolution.py
```

#### Speedup Comparison

The benchmark installed are downloaded by the dependency `einsum_benchmark` package, on first run. There exists a seperate benchmark script for each implementation (dense, sparse, hybrid). Moreover, for sparse and dense there is a script that only considers integer instances and converts them to float for the float speedup comparison in the paper. 

To run the speedup comparison for the dense implementation, you can use the following command:
```bash
uv run benchmark_dense.py
```
To run the speedup comparison for the sparse implementation, you can use the following command:
```bash
uv run benchmark_sparse.py
```
To run the speedup comparison for the hybrid implementation, you can use the following command:
```bash
uv run benchmark_hybrid.py
```

#### Speedup Comparison with Integer Instances Converted to Float
To run the speedup comparison for the dense implementation with integer instances converted to float, you can use the following command:
```bash
uv run benchmark_dense_int_to_float.py
```

To run the speedup comparison for the sparse implementation with integer instances converted to float, you can use the following command:
```bash
uv run benchmark_sparse_int_to_float.py
```
