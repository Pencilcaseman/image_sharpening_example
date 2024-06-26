use clap::Parser;
use image::{buffer::ConvertBuffer, ImageError, RgbImage};
use image_sharpen_rs::filter::sharpen;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long)]
    num_iters: Option<u32>,
}

fn main() -> Result<(), ImageError> {
    let args = Args::parse();

    println!("Reading file: {}", args.file);
    let img = image::open(args.file).unwrap();

    let iters = args.num_iters.unwrap_or(1);
    println!("Sharpening image {iters} times(s)");

    let rgba_img = img.to_rgba32f();

    let start = std::time::Instant::now();
    let mut res = sharpen(&rgba_img);
    for _ in 0..iters - 1 {
        res = sharpen(&rgba_img);
    }
    println!("Image sharpened {iters} time(s) in {:?}", start.elapsed());
    println!("Average: {:?}", start.elapsed() / iters);

    println!("Saving image");
    let res2: RgbImage = res.convert();
    res2.save(args.output.unwrap_or("image.png".to_string()))
}
