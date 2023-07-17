# Benchmark Suite for Hyperplane-based Iteratively Optimized Binarization (HIOB) for the SISAP'23 Challenge

## Prupose of this repository

This repository is a submission to the [2023 SISAP Indexing Challenge](https://sisap-challenges.github.io/) which focuses on the "quality" of binary sketches as well as the similarity search performance of various approaches.
The focus of this submission is the optimization of binary sketches to improve the $k@n$-recall using the Hamming distance as a proxy for the dot product on normalized data. Whilst we only employ brute-force search and thus not expect the query performance to be en par with more involved approaches, we also submit this approach as an additional baseline to the challenge and to see how far off it is.

The repository is generally based on the [Python example](https://github.com/sisap-challenges/sisap23-laion-challenge-faiss-example) provided by the challenge.
Accordingly, there is an option to run all tests from within Python, yet, the calls copy the dataset.
At least for the largest dataset considered here (~300GB), this is not viable for the 512GB RAM limit.
The `search.py` file has not been included in the last few updates so the usage is disencouraged and we suggest to use the `rust-search` binary instead.

## Usage

The challenge includes three tasks and multiple datasets which can be looked up at [the challenge website](https://sisap-challenges.github.io/tasks/).
We provide predefined benchmarks for the `clip768` datasets of sizes `100K`, `300K`, `10M`, `30M`, and `100M`.
Each benchmark can be executed using one of the GitHub Action workflows of this repository (all using manual triggers).
Since Tasks A and C are identical in regards to our submission, they share the same task for each dataset.
On a reference system with 32 CPU cores, each workflow executed in at most 2 hours (download times of the datasets not considered).
Each benchmark reports the individually required time of most steps as well as the total build and query time (including loading of the dataset, excluding writing results to disk).

Each benchmark trains a Stochastic HIOB once (publication pending) and executes queries for two different probe/candidate set sizes.
To change the number of probe set sizes, the `--probe-steps` option of the benchmark calls can be decreased or increased appropriately.
In case of `--probe-steps 1`, only the `--probe-max` value is considered.
The choice of two different probe set sizes is to ensure that each task produces one run that is elligible to the 90% recall limit for both the public and private gold standards.
Using the public gold standards, the smaller probe set size should result in a recall of roughly 90% to 92% and the larger should lead to a recall between 91% and 93%.
The exact recall values may vary depending on the random initialization.

For information on how to run the `rust-search` binary, please refer to the `--help` print of the binary.
