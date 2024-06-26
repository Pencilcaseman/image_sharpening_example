use cudarc::driver::LaunchAsync;
use lazy_static::lazy_static;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

static CONV_KERNEL_SRC_NAME: &'static str = "conv_kernel_f32";
static CONV_KERNEL_SRC: &'static str = r#"
    extern "C" __global__ void conv_kernel_f32(
        float* const out,
        const float* const source,
        const float* const kern,
        const int width,
        const int height,
        int kernel_size
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int x = idx % width;
        const int y = idx / width;

        if (x < width && y < height) {
            float red = 0.0f;
            float green = 0.0f;
            float blue = 0.0f;
            float alpha = 0.0f;

            const int half_kernel_size = kernel_size / 2;

            for (int ky = -half_kernel_size; ky <= half_kernel_size; ++ky) {
                for (int kx = -half_kernel_size; kx <= half_kernel_size; ++kx) {
                    const int src_x = x + kx;
                    const int src_y = y + ky;

                    if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                        const int src_index = src_y * width + src_x;
                        const int kernel_index = (ky + half_kernel_size) * kernel_size + (kx + half_kernel_size);

                        red   += source[src_index * 4 + 0] * kern[kernel_index];
                        green += source[src_index * 4 + 1] * kern[kernel_index];
                        blue  += source[src_index * 4 + 2] * kern[kernel_index];
                        alpha += source[src_index * 4 + 3] * kern[kernel_index];
                    }
                }
            }

            const int out_index = y * width + x;
            out[out_index * 4 + 0] = red;   // Red
            out[out_index * 4 + 1] = green; // Green
            out[out_index * 4 + 2] = blue;  // Blue
            out[out_index * 4 + 3] = alpha; // Alpha
        }
    }
"#;

lazy_static! {
    static ref CUDA_DEVICE: Arc<cudarc::driver::CudaDevice> =
        cudarc::driver::CudaDevice::new(0).expect("Failed to create CUDA device");
}

static LOADED_KERNELS: LazyLock<Mutex<HashMap<&'static str, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn load_kernel(name: &'static str, source: &'static str) {
    // if *LOADED_KERNELS.lock().unwrap().entry(name).or_insert(true) {
    if !LOADED_KERNELS.lock().unwrap().contains_key(name) {
        let ptx = cudarc::nvrtc::compile_ptx(source).expect("Failed to compile kernel source");
        CUDA_DEVICE
            .load_ptx(ptx, "sharpen", &[name])
            .expect("Failed to load kernel");
    }

    match LOADED_KERNELS.lock().unwrap().entry(name) {
        Entry::Vacant(v) => {
            v.insert(true);
        }
        _ => {}
    }
}

pub fn apply_conv_gpu(
    image: &[f32],
    kernel: &[f32],
    width: i32,
    height: i32,
    kernel_size: i32,
) -> Vec<f32> {
    load_kernel(CONV_KERNEL_SRC_NAME, CONV_KERNEL_SRC);

    let source_image = CUDA_DEVICE
        .htod_sync_copy(image)
        .expect("Failed to copy image to device");

    let source_kernel = CUDA_DEVICE
        .htod_sync_copy(kernel)
        .expect("Failed to copy kernel to device");

    let mut result_buffer = CUDA_DEVICE
        .alloc_zeros::<f32>((width * height * 4) as usize)
        .expect("Failed to allocate result buffer");

    let kern = CUDA_DEVICE.get_func("sharpen", "conv_kernel_f32").unwrap();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems((width * height) as u32);

    unsafe {
        kern.launch(
            cfg,
            (
                &mut result_buffer,
                &source_image,
                &source_kernel,
                width,
                height,
                kernel_size,
            ),
        )
    }
    .expect("Failed to launch kernel");

    CUDA_DEVICE
        .dtoh_sync_copy(&result_buffer)
        .expect("Failed to copy result back to host")
}
