use hdf5::{File, Group, H5Type, types::{VarLenUnicode}};
use std::ops::Deref;
use ndarray::{Array2};

use crate::{Res, NoRes};
use crate::fs_fun::{create_path_to_file};

pub fn create_str_attr<T: Deref<Target=Group>>(location: &T, name: &str, value: &str) -> NoRes {
	let attr = location.new_attr::<VarLenUnicode>().create(name)?;
	let value_: VarLenUnicode = value.parse()?;
	attr.write_scalar(&value_)?;
	Ok(())
}
pub fn create_num_attr<T: Deref<Target=Group>, F: H5Type>(location: &T, name: &str, value: F) -> NoRes {
	let attr = location.new_attr::<VarLenUnicode>().create(name)?;
	attr.write_scalar(&value)?;
	Ok(())
}

pub struct H5Builder {
	file: File
}
impl H5Builder {
	pub fn new(file: &str) -> Res<H5Builder> {
		create_path_to_file(file)?;
		Ok(H5Builder{
			file: File::create(file)?
		})
	}
	pub fn with_str_attr(mut self, attr_name: &str, attr_value: &str) -> Res<Self> {
		create_str_attr(&self.file, attr_name, attr_value)?;
		Ok(self)
	}
	pub fn with_num_attr<F: H5Type>(mut self, attr_name: &str, attr_value: F) -> Res<Self> {
		create_num_attr(&self.file, attr_name, attr_value)?;
		Ok(self)
	}
	pub fn with_dataset<F: H5Type>(mut self, dataset_name: &str, dataset: &Array2<F>) -> Res<Self> {
		self.file.new_dataset_builder().with_data(dataset.view()).create(dataset_name)?;
		Ok(self)
	}
}

pub fn open_hdf5(file_path: &str) -> Res<File> {
	let result = File::open(file_path)?;
	Ok(result)
}
