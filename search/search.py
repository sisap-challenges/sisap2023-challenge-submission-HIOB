import argparse
import hiob
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time
import itertools
import tqdm

PRODUCTION_MODE = True

import multiprocessing
hiob.limit_threads(multiprocessing.cpu_count()-2)

# Download a single h5 file to the specified destination if not already available
def download_if_missing(file_url, file_path):
	if not os.path.exists(file_path):
		os.makedirs(Path(file_path).parent, exist_ok=True)
		print("downloading '{:}' -> '{:}'...".format(file_url, file_path))
		urlretrieve(file_url, file_path)
	else:
		print("file '{:}' already at '{:}'".format(file_url, file_path))

# Download all missing files for a specified format and size
def ensure_files_available(kind, size):
	url = (
		"https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
		if PRODUCTION_MODE else
		"http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"
	)
	task = {
		"query": f"{url}/public-queries-10k-{kind}.h5",
		"dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
	}
	for version, url in task.items():
		download_if_missing(url, os.path.join("data", kind, size, f"{version}.h5"))

# Store the output values as required in the task specification
def store_results(
	out_file,
	kind,
	size,
	alg_name,
	parameter_string,
	neighbor_dists,
	neighbor_ids,
	build_time,
	query_time,
):
	os.makedirs(Path(out_file).parent, exist_ok=True)
	f = h5py.File(out_file, 'w')
	f.attrs['algo'] = alg_name
	f.attrs['data'] = kind
	f.attrs['buildtime'] = build_time
	f.attrs['querytime'] = query_time
	f.attrs['size'] = size
	f.attrs['params'] = parameter_string
	f.create_dataset('knns', neighbor_ids.shape, dtype=neighbor_ids.dtype)[:] = neighbor_ids
	f.create_dataset('dists', neighbor_dists.shape, dtype=neighbor_dists.dtype)[:] = neighbor_dists
	f.close()

# Run 
def run(kind, key, size="100K", k=30, ram_mode=False, n_bitss=[1024], n_its=30_000, sample_size=10_000, its_per_sample=1024, noise_std=None):
	print("Running", kind)
	if not kind.startswith("clip"):
		raise ValueError(f"unsupported input type {kind}")
	ensure_files_available(kind, size)

	data_file = os.path.join("data", kind, size, "dataset.h5")
	queries_file = os.path.join("data", kind, size, "query.h5")
	data_shape = h5py.File(data_file)[key].shape
	queries_shape = h5py.File(queries_file)[key].shape

	print(f"Training index on {data_shape} with {n_bitss} bits")
	index_identifier = "StochasticHIOB(n_bits={:},n_its={:},n_samples={:},batch_its={:},noise_std={:.4})".format(n_bitss, n_its, sample_size, its_per_sample, 0. if noise_std is None else noise_std)
	start = time.time()
	hs = []
	data_bins = []
	for n_bits in n_bitss:
		if not ram_mode:
			h = hiob.StochasticHIOB.from_h5_file(
				file=data_file,
				dataset=key,
				n_bits=n_bits,
				scale=.1,
				sample_size=sample_size,
				its_per_sample=its_per_sample,
				init_ransac=True,
				input_type=np.float32,
				noise_std=noise_std,
			)
			with tqdm.tqdm(total=n_its) as bar:
				batch_size=100
				for _ in range(n_its//batch_size):
					h.run(batch_size)
					bar.update(batch_size)
					bar.refresh()
			data_bin = h.binarize_h5(data_file, key, 500_000)
		else:
			data = np.array(h5py.File(data_file)[key]).astype(np.float32)
			h = hiob.StochasticHIOB.from_ndarray(
				X=data,
				n_bits=n_bits,
				scale=.1,
				sample_size=sample_size,
				its_per_sample=its_per_sample,
				init_ransac=True,
				noise_std=noise_std,
			)
			with tqdm.tqdm(total=n_its) as bar:
				batch_size=100
				for _ in range(n_its//batch_size):
					h.run(batch_size)
					bar.update(batch_size)
					bar.refresh()
			data_bin = h.binarize(data)
		hs.append(h)
		data_bins.append(data_bin)
	end = time.time()
	elapsed_build = end - start
	print(f"Done training in {elapsed_build:.3}s.")

	nprobe_vals = np.round(np.exp(np.linspace(np.log(k), np.log(1000*k), 21))).astype(int)
	# nprobe_vals = np.round(np.exp(np.linspace(np.log(10_000), np.log(800_000), 11))).astype(int)
	nprobe_groups = list(map(list,itertools.combinations(nprobe_vals,len(n_bitss))))
	# nprobe_groups = [
	# 	[a,b]
	# 	for a in np.round(np.exp(np.linspace(np.log(250), np.log(400), 6))).astype(int)
	# 	for b in np.round(np.exp(np.linspace(np.log(100_000), np.log(800_000), 11))).astype(int)
	# ]
	# for nprobe in nprobe_vals:
	# for nprobe in [50, 100, 150, 250, 450, 500, 600]:
	for nprobes in nprobe_groups:
		print(f"Starting search on {queries_shape} with nprobes={nprobes}")
		nprobes = nprobes[::-1] # Revert order to filter the largest set first
		start = time.time()
		queries = np.array(h5py.File(queries_file)[key]).astype(np.float32)
		end = time.time()
		elapsed_search = end - start
		print(f"Queries loaded after {elapsed_search:.3}s.")
		queries_bins = [h.binarize(queries) for h in hs]
		end = time.time()
		elapsed_search = end - start
		print(f"Queries binarized after {elapsed_search:.3}s.")
		bin_eval = hiob.BinarizationEvaluator()
		# Query nearest neighbors as per cosine similarity
		if not ram_mode:
			neighbor_dots, neighbor_ids = bin_eval.query_cascade_h5(
				data_file, key, data_bins,
				queries, queries_bins,
				k, nprobes,
				chunk_size=max(10, int(np.ceil(queries.shape[0] / hiob.num_threads() / 2)))
			)
		else:
			neighbor_dots, neighbor_ids = bin_eval.query_cascade(
				data, data_bins,
				queries, queries_bins,
				k, nprobes,
				chunk_size=int(np.ceil(queries.shape[0] / hiob.num_threads()))
			)
		# Translate cosines into distances
		neighbor_dists = np.sqrt(np.maximum(0, 2-2*neighbor_dots))
		end = time.time()
		elapsed_search = end - start
		print(f"Done searching in {elapsed_search:.3}s.")
		# HIOB is 0-indexed, groundtruth is 1-indexed
		neighbor_ids += 1
		# Generate parameter string for this call
		param_string = "index_params=({:}),query_params=(nprobe={:})".format(
			",".join([
				"{:}={:d}".format(a,int(b))
				for a,b in [
					["scale%", h.scale*100],
					["its_per_sample", h.its_per_sample],
				]
			]),
			nprobes,
		)
		# Store results for this run
		store_results(
			os.path.join("result/", kind, size, index_identifier, f"{param_string}.h5"),
			kind,
			size,
			index_identifier+" + brute-force",
			param_string,
			neighbor_dists,
			neighbor_ids,
			elapsed_build,
			elapsed_search,
		)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--size",
		default="100K",
		type=str,
	)
	parser.add_argument(
		"--k",
		default=30,
		type=int,
	)
	parser.add_argument(
		"--ram",
		default=True,
		help="Whether or not to load the entire dataset into RAM. If false, will only load as much data from disk at any time as required for execution",
		type=bool,
		action=argparse.BooleanOptionalAction,
	)
	parser.add_argument(
		"--its",
		default=30_000,
		help="How many iterations to run the Stochastic HIOB",
		type=int,
	)
	parser.add_argument(
		"--bits",
		default="1024",
		help="How many bits to use in binarization",
		type=str,
	)
	parser.add_argument(
		"--batch_its",
		default=1024,
		help="How many iterations to run per stochastic sub sample",
		type=int,
	)
	parser.add_argument(
		"--samples",
		default=10_000,
		help="How many samples to use in each stochastic batch",
		type=int,
	)
	parser.add_argument(
		"--noise",
		default=None,
		help="Standard deviation of noise to add to stochastic subsamples",
		type=float,
	)

	args = parser.parse_args()

	assert args.size in ["100K", "300K", "10M", "30M", "100M"]

	print("RAM Mode={:}".format(args.ram))
	# run("pca32v2", "pca32", args.size, args.k)
	# run("pca96v2", "pca96", args.size, args.k)
	# run("hammingv2", "hamming", args.size, args.k)
	run(
		"clip768v2", "emb",
		args.size, args.k,
		ram_mode=args.ram,
		n_its=args.its,
		n_bitss=[int(s) for s in args.bits.split(",")],
		sample_size=args.samples,
		its_per_sample=args.batch_its,
		noise_std=args.noise,
	)
