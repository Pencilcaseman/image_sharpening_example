#![feature(portable_simd)]

#[cfg(feature = "cuda_compute")]
pub mod conv_cu;

#[cfg(feature = "opencl_compute")]
pub mod conv_cl;

pub mod filter;
