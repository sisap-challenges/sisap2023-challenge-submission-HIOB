#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

use hiob::{
	limit_threads,
	num_threads,
	pydata::H5PyDataset,
	data::MatrixDataSource,
	binarizer::StochasticHIOB,
	eval::BinarizationEvaluator,
};
use ndarray::{Axis, Slice, Array2, ArrayView2};
use clap::Parser;
use std::str::FromStr;
use std::time::Instant;
use itertools::Itertools;
use num_traits::cast::NumCast;

mod fs_fun;
mod hdf5_fun;
mod cli;

use crate::fs_fun::{download_if_missing};
use crate::hdf5_fun::{H5Builder};//, open_hdf5};
use crate::cli::{Cli};

const PRODUCTION_MODE: bool = true;

type Res<T> = Result<T, Box<dyn std::error::Error>>;
type NoRes = Res<()>;

// Download all missing files for a specified format and size
fn ensure_files_available(in_base_path: &str, kind: &str, size: &str) -> NoRes {
	let base_url = if PRODUCTION_MODE {
		"https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
	} else {
		"http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"
	};
	let versions = vec!["query", "dataset"];
	let urls = vec![
		format!("{}/public-queries-10k-{}.h5", base_url, kind),
		format!("{}/laion2B-en-{}-n={}.h5", base_url, kind, size),
	];
	for (version,url) in versions.iter().zip(urls.iter()) {
		download_if_missing(
			url,
			format!("{}/{}/{}/{}.h5", in_base_path, kind, size, version).as_str()
		)?
	};
	Ok(())
}
fn dataset_path(in_base_path: &str, kind: &str, size: &str) -> String {
	format!("{}/{}/{}/dataset.h5", in_base_path, kind, size)
}
fn queries_path(in_base_path: &str, kind: &str, size: &str) -> String {
	format!("{}/{}/{}/query.h5", in_base_path, kind, size)
}
fn result_path(out_base_path: &str, kind: &str, size: &str, index_identifier: &str, param_string: &str) -> String {
	format!("{}/{}/{}/{}/{}.h5", out_base_path, kind, size, index_identifier, param_string)
}

// Store the output values as required in the task specification
fn store_results<T: hdf5::H5Type>(
	out_file: &str,
	kind: &str,
	size: &str,
	alg_name: &str,
	parameter_string: &str,
	neighbor_dists: &Array2<T>,
	neighbor_ids: &Array2<usize>,
	build_time: f64,
	query_time: f64,
) -> NoRes {
	H5Builder::new(out_file)?
	.with_dataset("dists", neighbor_dists)?
	.with_dataset("knns", neighbor_ids)?
	.with_str_attr("algo", alg_name)?
	.with_str_attr("data", kind)?
	.with_str_attr("size", size)?
	.with_str_attr("params", parameter_string)?
	.with_num_attr("buildtime", build_time)?
	.with_num_attr("querytime", query_time)?;
	Ok(())
}


struct Timer {
	start: Instant
}
impl Timer {
	fn new() -> Self {
		Timer{start: Instant::now()}
	}
	fn elapsed_s(&self) -> f64 {
		self.start.elapsed().as_secs_f64()
	}
}


fn linspace<T: NumCast+Copy+Clone>(start: T, end: T, n_vals: usize) -> Vec<T> {
	let fstart: f64 = <f64 as NumCast>::from(start.clone()).unwrap();
	let fend: f64 = <f64 as NumCast>::from(end.clone()).unwrap();
	let fstep = (fend-fstart)/((n_vals-1) as f64);
	let mut vals: Vec<T> = (0..n_vals)
	.map(|i_val| fstart + fstep * (i_val as f64))
	.map(|fval| <T as NumCast>::from(fval).unwrap())
	.collect();
	/* Fix limits to guarantee start and end included */
	vals[0] = start;
	vals[n_vals-1] = end;
	vals
}
fn logspace<T: NumCast+Copy+Clone>(start: T, end: T, n_vals: usize) -> Vec<T> {
	let fstart: f64 = <f64 as NumCast>::from(start.clone()).unwrap();
	let fend: f64 = <f64 as NumCast>::from(end.clone()).unwrap();
	let lstart = fstart.ln();
	let lend = fstart.ln();
	let mut vals: Vec<T> = linspace(lstart, lend, n_vals)
	.iter()
	.map(|lval| lval.exp())
	.map(|fval| <T as NumCast>::from(fval).unwrap())
	.collect();
	/* Fix limits to guarantee start and end included */
	vals[0] = start;
	vals[n_vals-1] = end;
	vals
}


fn read_h5py_source(source: &H5PyDataset<f32>, batch_size: usize) -> Array2<f32> {
	let data_shape = [source.n_rows(), source.n_cols()];
	let mut data = Array2::from_elem(data_shape, 0f32);
	let n_batches = (data_shape[0]+(batch_size-1)) / batch_size;
	(0..n_batches).for_each(|i_batch| {
		let batch_start = batch_size * i_batch;
		let batch_end = (batch_start + batch_size).min(data_shape[1]);
		let batch = source.get_rows_slice(batch_start, batch_end);
		batch.axis_iter(Axis(0))
		.zip(data.slice_axis_mut(Axis(0), Slice::from(batch_start..batch_end)).axis_iter_mut(Axis(0)))
		.for_each(|(from, mut to)| {
			to.assign(&from);
		});
	});
	data
}


fn run_experiment(
	in_base_path: &str,
	out_base_path: &str,
	kind: &str,
	key: &str,
	size: &str,
	k: usize,
	ram_mode: bool,
	n_bitss: &Vec<usize>,
	n_its: usize,
	sample_size: usize,
	its_per_sample: usize,
	noise_std: f32,
) -> NoRes {
	println!("Running {}", kind);
	ensure_files_available(in_base_path, kind, size)?;

	assert!(ram_mode);

	let data_file: H5PyDataset<f32> = H5PyDataset::new(dataset_path(in_base_path, kind, size).as_str(), key);
	let queries_file: H5PyDataset<f32> = H5PyDataset::new(queries_path(in_base_path, kind, size).as_str(), key);
	let data_shape = [data_file.n_rows(), data_file.n_cols()];
	let queries_shape = [queries_file.n_rows(), queries_file.n_cols()];

	/* Training */
	println!("Training index on {:?} with {:?} bits",data_shape,n_bitss);
	let build_timer = Timer::new();
	/* Creating string handle for experiment */
	let index_identifier = format!(
		"StochasticHIOB(n_bits={:?},n_its={:},n_samples={:},batch_its={:},noise_std={:.4})",
		n_bitss,
		n_its,
		sample_size,
		its_per_sample,
		noise_std
	);
	/* Loading data */
	let data_load_timer = Timer::new();
	let data = read_h5py_source(&data_file, 300_000);
	println!("Data loaded in {:.3}s", data_load_timer.elapsed_s());
	/* Training HIOBs */
	let mut hs = vec![];
	let mut data_bins = vec![];
	(0..n_bitss.len()).for_each(|i_hiob| {
		let init_timer = Timer::new();
		let mut h: StochasticHIOB<f32, u64, ArrayView2<f32>> = StochasticHIOB::new(
			data.view(),
			sample_size,
			its_per_sample,
			*n_bitss.get(i_hiob).unwrap(),
			None,
			None,
			None,
			None,
			Some(true),
			None,
			None,
			if noise_std > 0f32 { Some(noise_std) } else { None },
		);
		println!("Stochastic HIOB {} initialized in {:.3}s", i_hiob+1, init_timer.elapsed_s());
		let init_timer = Timer::new();
		h.run(n_its);
		println!("Stochastic HIOB {} trained in {:.3}s", i_hiob+1, init_timer.elapsed_s());
		let init_timer = Timer::new();
		let data_bin = h.binarize(&data);
		println!("Data binarized with HIOB {} in {:.3}s", i_hiob+1, init_timer.elapsed_s());
		hs.push(h);
		data_bins.push(data_bin);
	});
	let build_time = build_timer.elapsed_s();
	println!("Done training in {:.3}s.", build_time.clone());
	
	let nprobe_vals = logspace(k, 1000*k, 21);
	let nprobe_groups: Vec<Vec<usize>> = nprobe_vals.clone().into_iter().rev().combinations(n_bitss.len()).collect();
	let bin_eval = BinarizationEvaluator::new();
	for nprobes in nprobe_groups.iter() {
		println!("Starting search on {:?} with nprobes={:?}", queries_shape, nprobes);
		let query_timer = Timer::new();
		/* Loading queries */
		let queries_load_timer = Timer::new();
		let queries = read_h5py_source(&queries_file, 300_000);
		println!("Queries loaded in {:.3}s", queries_load_timer.elapsed_s());
		/* Binarize queries */
		let queries_bin_timer = Timer::new();
		let queries_bins: Vec<Array2<u64>> = hs.iter()
		.map(|h| h.binarize(&queries))
		.collect();
		println!("Queries binarized in {:.3}s", queries_bin_timer.elapsed_s());
		/* Perform query */
		let query_call_timer = Timer::new();
		let chunk_size = (queries_shape[0]+num_threads()*2-1)/(num_threads()*2);
		let (mut neighbor_dists, mut neighbor_ids) = bin_eval.query_cascade(
			&data,
			&data_bins,
			&queries,
			&queries_bins,
			k,
			nprobes,
			Some(chunk_size),
		);
		println!("Queries executed in {:.3}s", query_call_timer.elapsed_s());
		/* Modify dot products to euclidean distances and change to 1-based index */
		neighbor_dists.mapv_inplace(|v| 0f32.max(2f32-2f32*v).sqrt());
		neighbor_ids.mapv_inplace(|v| v+1);
		let query_time = query_timer.elapsed_s();
		println!("Overall query time: {:.3}s", query_time.clone());
		/* Create parameter string and store results */
		let param_string = format!(
			"index_params=({:}),query_params=(nprobe={:?})",
			format!(
				"scale%={:},its_per_sample={:}",
				<usize as NumCast>::from(hs.get(0).unwrap().get_scale()*100f32).unwrap(),
				its_per_sample,
			),
			nprobes,
		);
		let out_file = result_path(out_base_path, kind, size, index_identifier.as_str(), param_string.as_str());
		let storage_timer = Timer::new();
		store_results(
			out_file.as_str(),
			kind,
			size,
			format!("{} + brute-force", index_identifier).as_str(),
			param_string.as_str(),
			&neighbor_dists,
			&neighbor_ids,
			build_time,
			query_time,
		)?;
		println!("Wrote results to disk in {:.3}s", storage_timer.elapsed_s());
	}
	Ok(())
}

fn main() -> NoRes {
	let _ = limit_threads(num_cpus::get()-1);
	let args = Cli::parse();
	run_experiment(
		args.in_path.as_str(),
		args.out_path.as_str(),
		"clip768v2",
		"emb",
		args.size.as_str(),
		args.k,
		args.ram,
		&args.bits.split(",").map(|v| usize::from_str(v.trim()).unwrap()).collect::<Vec<usize>>(),
		args.its,
		args.samples,
		args.batch_its,
		args.noise,
	)?;
	Ok(())
}
