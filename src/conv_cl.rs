static CONV_KERNEL_SRC: &'static str = r#"
    __kernel void conv_kernel_f32(
        __global float* const out,
        __global const float* const source,
        __global const float* const kern,
        __private const int width,
        __private const int height,
        __private int kernel_size
    ) {
        const int y = get_global_id(0);
        const int x = get_global_id(1);

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
            out[out_index * 4 + 1] = green; // Red
            out[out_index * 4 + 2] = blue;  // Red
            out[out_index * 4 + 3] = alpha; // Alpha
        }
    }
"#;

pub fn apply_conv_gpu(
    image: &[f32],
    kernel: &[f32],
    width: i32,
    height: i32,
    kernel_size: i32,
) -> Vec<f32> {
    let ocl_queue = ocl::ProQue::builder()
        .src(CONV_KERNEL_SRC)
        .dims((height, width))
        .build()
        .expect("Failed to build queue");

    let source_image = ocl::Buffer::builder()
        .queue(ocl_queue.queue().clone())
        .flags(ocl::MemFlags::new().read_write())
        .len(width * height * 4) // * 4 because RGBA
        .copy_host_slice(&image)
        .build()
        .expect("Failed to allocate device memory for image");

    let source_kernel = ocl::Buffer::builder()
        .queue(ocl_queue.queue().clone())
        .flags(ocl::MemFlags::new().read_write())
        .len(kernel_size * kernel_size)
        .copy_host_slice(&kernel)
        .build()
        .expect("Failed to allocate device memory for image");

    let mut vec_result = vec![0.0f32; (width * height * 4) as usize];

    let result_buffer = ocl::Buffer::builder()
        .queue(ocl_queue.queue().clone())
        .flags(ocl::MemFlags::new().read_write())
        .len(width * height * 4)
        .build()
        .expect("Failed to allocate device memory for result");

    let kern = ocl_queue
        .kernel_builder("conv_kernel_f32")
        .global_work_size((height, width))
        .arg(None::<&ocl::Buffer<f32>>)
        .arg(None::<&ocl::Buffer<f32>>)
        .arg(None::<&ocl::Buffer<f32>>)
        .arg(0i32)
        .arg(0i32)
        .arg(0i32)
        .build()
        .expect("Failed to construct kernel");

    kern.set_arg(0, &result_buffer).expect("Failed to set arg0");
    kern.set_arg(1, &source_image).expect("Failed to set arg1");
    kern.set_arg(2, &source_kernel).expect("Failed to set arg2");
    kern.set_arg(3, &width).expect("Failed to set arg3");
    kern.set_arg(4, &height).expect("Failed to set arg4");
    kern.set_arg(5, &kernel_size).expect("Failed to set arg5");

    unsafe {
        kern.enq().expect("Failed to enqueue");
    }

    result_buffer
        .read(&mut vec_result)
        .enq()
        .expect("Failed to copy back to host");

    vec_result
}
