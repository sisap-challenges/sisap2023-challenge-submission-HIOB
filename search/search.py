import argparse
import hiob
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time

PRODUCTION_MODE = True

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
def run(kind, key, size="100K", k=30, ram_mode=False, n_bits=1024, n_its=30_000, sample_size=10_000, its_per_sample=1024):
	print("Running", kind)
	if not kind.startswith("clip"):
		raise ValueError(f"unsupported input type {kind}")
	ensure_files_available(kind, size)

	data_file = os.path.join("data", kind, size, "dataset.h5")
	queries_file = os.path.join("data", kind, size, "query.h5")
	data_shape = h5py.File(data_file)[key].shape
	queries_shape = h5py.File(queries_file)[key].shape

	print(f"Training index on {data_shape} with {n_bits} bits")
	index_identifier = "StochasticHIOB(n_bits={:},n_its={:},n_samples={:},batch_its={:})".format(n_bits, n_its, sample_size, its_per_sample)
	start = time.time()
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
		)
		h.run(n_its)
		data_bin = h.binarize_h5(data_file, key, 300_000)
	else:
		data = np.array(h5py.File(data_file)[key]).astype(np.float32)
		h = hiob.StochasticHIOB.from_ndarray(
			X=data,
			n_bits=n_bits,
			scale=.1,
			sample_size=sample_size,
			its_per_sample=its_per_sample,
			init_ransac=True,
			input_type=np.float32,
		)
		h.run(n_its)
		data_bin = h.binarize(data)
	end = time.time()
	elapsed_build = end - start
	print(f"Done training in {elapsed_build}s.")

	for nprobe in np.round(np.exp(np.linspace(np.log(k), np.log(1000*k), 21))).astype(int):
	# for nprobe in [50, 100, 150, 250, 450, 500, 600]:
		print(f"Starting search on {queries_shape} with nprobe={nprobe}")
		start = time.time()
		queries = np.array(h5py.File(queries_file)[key]).astype(np.float32)
		queries_bin = h.binarize(queries)
		bin_eval = hiob.BinarizationEvaluator()
		# Query nearest neighbors as per cosine similarity
		if not ram_mode:
			neighbor_dots, neighbor_ids = bin_eval.query_h5(data_file, key, data_bin, queries, queries_bin, k, nprobe, chunk_size=300_000)
		else:
			neighbor_dots, neighbor_ids = bin_eval.query(data, data_bin, queries, queries_bin, k, nprobe)
		# Translate cosines into distances
		neighbor_dists = np.sqrt(np.maximum(0, 2-2*neighbor_dots))
		end = time.time()
		elapsed_search = end - start
		print(f"Done searching in {elapsed_search}s.")
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
			nprobe,
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
	)
	parser.add_argument(
		"--its",
		default=30_000,
		help="How many iterations to run the Stochastic HIOB",
		type=int,
	)
	parser.add_argument(
		"--bits",
		default=1024,
		help="How many bits to use in binarization",
		type=int,
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

	args = parser.parse_args()

	assert args.size in ["100K", "300K", "10M", "30M", "100M"]

	# run("pca32v2", "pca32", args.size, args.k)
	# run("pca96v2", "pca96", args.size, args.k)
	# run("hammingv2", "hamming", args.size, args.k)
	run(
		"clip768v2", "emb",
		args.size, args.k,
		ram_mode=args.ram,
		n_its=args.its,
		n_bits=args.bits,
		sample_size=args.samples,
		its_per_sample=args.batch_its,
	)
