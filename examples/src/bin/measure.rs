use pmj::{generate, Sample};
use rand::prelude::*;
use std::f32::consts::PI;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::result;

#[derive(Debug)]
enum Error {
    Io(io::Error),
    Fmt(fmt::Error),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}
impl From<fmt::Error> for Error {
    fn from(err: fmt::Error) -> Self {
        Error::Fmt(err)
    }
}

type Result<T> = result::Result<T, Error>;

fn disc(x: f32, y: f32) -> f32 {
    if x * x + y * y < 2.0 / PI {
        1.0
    } else {
        0.0
    }
}

fn bilinear(x: f32, y: f32) -> f32 {
    x * y
}

fn accumulate_error(
    samples: &[Sample],
    errors: &mut [f64],
    func: &impl Fn(f32, f32) -> f32,
    reference: f64,
) {
    let mut sum = 0.0;
    for (i, (sample, error)) in samples.iter().zip(errors.iter_mut()).enumerate() {
        sum += f64::from(func(sample.x(), sample.y()));
        let count = i + 1;
        *error += (sum / (count as f64) - reference).abs();
    }
}

fn write_error<P: AsRef<Path>>(path: P, errors: &[f64], seed_count: usize) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut w = io::BufWriter::new(file);

    for (i, error) in errors.iter().enumerate() {
        writeln!(w, "{}, {}", 1 + i, error / (seed_count as f64))?;
    }

    Ok(())
}

fn main() -> Result<()> {
    const SAMPLE_COUNT: usize = 1 << 12;
    const BLUE_NOISE_RESAMPLE_COUNT: u32 = 0;
    const SEED_COUNT: usize = 100;

    let mut disc_results = vec![0.0; SAMPLE_COUNT];
    let mut bilinear_results = vec![0.0; SAMPLE_COUNT];

    let disc_reference = 0.5;
    let bilinear_reference = 0.25;

    for seed in 0..SEED_COUNT {
        let mut rng = SmallRng::seed_from_u64(seed as u64);
        let samples = generate(SAMPLE_COUNT, BLUE_NOISE_RESAMPLE_COUNT, &mut || {
            rng.gen::<u32>()
        });

        accumulate_error(&samples, &mut disc_results, &disc, disc_reference);
        accumulate_error(
            &samples,
            &mut bilinear_results,
            &bilinear,
            bilinear_reference,
        );
    }

    write_error("results/disc_error.csv", &disc_results, SEED_COUNT)?;
    write_error("results/bilinear_error.csv", &bilinear_results, SEED_COUNT)?;
    Ok(())
}
