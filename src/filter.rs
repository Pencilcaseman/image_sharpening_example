use image::{GenericImageView, ImageBuffer, Pixel, Rgba, Rgba32FImage, SubImage};
use lazy_static::lazy_static;
use std::simd::{num::SimdFloat, Simd};

#[cfg(feature = "cuda_compute")]
use super::conv_cu::apply_conv_gpu;

#[cfg(feature = "opencl_compute")]
use super::conv_cl::apply_conv_gpu;

type SimdPix = Simd<f32, 4>;

const RADIUS: u32 = 8;
const CONV_SIZE: u32 = RADIUS * 2 + 1;
const NORM: f32 = ((2 * RADIUS - 1) * (2 * RADIUS - 1)) as f32;
const SCALE: f32 = 2.0;

fn filter(i: i32, j: i32) -> f32 {
    const D4: f32 = 4.0;
    const SIGMA_D4: f32 = 1.4;
    const FILTER_0: f32 = -40.0;
    const R_D4_SQ: f32 = D4 * D4;
    const SIGMA_D4_SQ: f32 = SIGMA_D4 * SIGMA_D4;
    const RSQ: f32 = (RADIUS * RADIUS) as f32;
    const SIGMA_SQ: f32 = SIGMA_D4_SQ * (RSQ / R_D4_SQ);

    let i = i as f32;
    let j = j as f32;

    let rsq = i * i + j * j;
    let delta = rsq / (2.0 * SIGMA_SQ);

    FILTER_0 * (1.0 - delta) * (-delta).exp()
}

lazy_static! {
    static ref KERNEL: [[f32; CONV_SIZE as usize]; CONV_SIZE as usize] = {
        let mut data = [[0.0f32; CONV_SIZE as usize]; CONV_SIZE as usize];
        let d = RADIUS as i32;
        for i in 0..=2 * RADIUS as usize {
            for j in 0..=2 * RADIUS as usize {
                data[i][j] = filter(j as i32 - d, i as i32 - d);
            }
        }
        data
    };
}

lazy_static! {
    static ref KERNEL_VEC: Vec<f32> = {
        let mut tmp = Vec::with_capacity((RADIUS * RADIUS * 4) as usize);
        let d = RADIUS as i32;
        for i in 0..=2 * RADIUS as usize {
            for j in 0..=2 * RADIUS as usize {
                // data[i][j] = filter(j as i32 - d, i as i32 - d);
                tmp.push(filter(j as i32 - d, i as i32 - d));
            }
        }
        tmp
    };
}

type ImageView<'a> = SubImage<&'a ImageBuffer<Rgba<f32>, Vec<f32>>>;

/// View must be a (diameter * 2) + 1 sized squre, centered on the target
/// pixel.
pub fn apply_conv<'a>(view: &'a ImageView<'a>) -> Rgba<f32>
where
    SimdPix: std::ops::AddAssign<SimdPix>,
{
    let mut rgba = SimdPix::splat(0.0f32);

    for i in 0..=2 * RADIUS {
        for j in 0..=2 * RADIUS {
            let pix = SimdPix::from_slice(view.get_pixel(j, i).channels());
            rgba += pix * SimdPix::splat(KERNEL[j as usize][i as usize]);
        }
    }

    let middle = SimdPix::from_slice(
        view.get_pixel(view.width() / 2, view.height() / 2)
            .channels(),
    );

    rgba = middle - SimdPix::splat(SCALE) / SimdPix::splat(NORM) * rgba;

    Rgba::<f32>::from(rgba.to_array())
}

pub fn sharpen(image: &Rgba32FImage) -> Rgba32FImage {
    let width = image.width();
    let height = image.height();

    let in_vec = image.to_vec();
    let out = apply_conv_gpu(
        &in_vec,
        &KERNEL_VEC,
        width as i32,
        height as i32,
        CONV_SIZE as i32,
    );

    let mut convolution = Rgba32FImage::from_vec(width, height, out).unwrap();

    // Find the min and max values
    let mut min = 999.9f32;
    let mut max = 0.0f32;
    for i in RADIUS..height - RADIUS {
        for j in RADIUS..width - RADIUS {
            let pix = SimdPix::from_slice(convolution.get_pixel(j, i).channels());
            max = pix.reduce_max().max(max);
            min = pix.reduce_min().min(min);
        }
    }

    // Map all values to the interval [0, 1]
    let min = SimdPix::splat(min);
    let max = SimdPix::splat(max);
    let one = SimdPix::splat(1.0f32);
    let inv_range = SimdPix::splat(1.0f32) / (max - min);
    for i in 0..height {
        for j in 0..width {
            let pix = SimdPix::from_slice(convolution.get_pixel(j, i).channels());
            let norm = one - (pix - min) * inv_range;
            norm.copy_to_slice(convolution.get_pixel_mut(j, i).channels_mut());
        }
    }

    convolution
}
