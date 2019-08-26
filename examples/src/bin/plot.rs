use pmj::{generate, PairClass, QuadClass, Sample};
use rand::prelude::*;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
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

fn write_samples_svg(
    w: &mut impl io::Write,
    samples: &[Sample],
    sample_col: impl Fn(Sample) -> &'static str,
    (x, y): (f32, f32),
    grid_size: f32,
    (grid_x, grid_y): (u32, u32),
) -> Result<()> {
    const SAMPLE_RADIUS: f32 = 2.0;
    writeln!(
        w,
        r#"<rect x="{x}" y="{y}" width="{size}" height="{size}" stroke="black" fill="none"/>"#,
        x = x,
        y = y,
        size = grid_size
    )?;
    for i in 1..grid_x {
        writeln!(
            w,
            r#"<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="black"/>"#,
            x1 = x,
            x2 = x + grid_size,
            y = y + grid_size * (i as f32) / (grid_x as f32)
        )?;
    }
    for i in 1..grid_y {
        writeln!(
            w,
            r#"<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="black"/>"#,
            y1 = y,
            y2 = y + grid_size,
            x = x + grid_size * (i as f32) / (grid_y as f32)
        )?;
    }
    for &sample in samples.iter() {
        writeln!(
            w,
            r#"<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}"/>"#,
            cx = x + sample.x() * grid_size,
            cy = y + sample.y() * grid_size,
            r = SAMPLE_RADIUS,
            col = sample_col(sample),
        )?;
    }
    Ok(())
}

fn write_partitions_svg() -> Result<()> {
    const MAX_LOG2_SAMPLE_COUNT: u32 = 5;
    const BLUE_NOISE_RESAMPLE_COUNT: u32 = 4;
    const SEED: u64 = 0;
    let mut rng = SmallRng::seed_from_u64(SEED);
    let samples = generate(
        1 << MAX_LOG2_SAMPLE_COUNT,
        BLUE_NOISE_RESAMPLE_COUNT,
        &mut || rng.gen::<u32>(),
    );

    const GRID_SIZE: f32 = 128.0;
    const ORIGIN: f32 = 0.5;
    const MARGIN: f32 = 10.0;
    let file = fs::File::create("results/partitions.svg")?;
    let mut w = io::BufWriter::new(file);
    writeln!(
        w,
        r#"<svg width="{width}" height="{height}">"#,
        width = (MAX_LOG2_SAMPLE_COUNT + 1) as f32 * (GRID_SIZE + MARGIN) + MARGIN,
        height = (MAX_LOG2_SAMPLE_COUNT - 1) as f32 * (GRID_SIZE + MARGIN) + MARGIN
    )?;
    let mut y = ORIGIN + MARGIN;
    for log2_sample_count in 2..=MAX_LOG2_SAMPLE_COUNT {
        let sample_count = 1 << log2_sample_count;
        let mut x = ORIGIN
            + MARGIN
            + 0.5 * ((MAX_LOG2_SAMPLE_COUNT - log2_sample_count) as f32) * (GRID_SIZE + MARGIN);
        for log2_grid_x in 0..=log2_sample_count {
            let log2_grid_y = log2_sample_count - log2_grid_x;
            write_samples_svg(
                &mut w,
                &samples[0..sample_count],
                |_| "blue",
                (x, y),
                GRID_SIZE,
                (1 << log2_grid_x, 1 << log2_grid_y),
            )?;
            x += GRID_SIZE + MARGIN;
        }
        y += GRID_SIZE + MARGIN;
    }
    writeln!(w, "</svg>")?;
    Ok(())
}

fn write_classes_svg() -> Result<()> {
    const SAMPLE_COUNT: usize = 1 << 10;
    const BLUE_NOISE_RESAMPLE_COUNT: u32 = 4;
    const SEED: u64 = 0;

    const GRID_SIZE: f32 = 384.0;
    const ORIGIN: f32 = 0.5;
    const MARGIN: f32 = 10.0;
    let file = fs::File::create("results/classes.svg")?;
    let mut w = io::BufWriter::new(file);
    writeln!(
        w,
        r#"<svg width="{width}" height="{height}">"#,
        width = 2.0 * GRID_SIZE + 3.0 * MARGIN,
        height = GRID_SIZE + 2.0 * MARGIN,
    )?;

    let mut x = ORIGIN + MARGIN;
    let y = ORIGIN + MARGIN;

    let mut rng = SmallRng::seed_from_u64(SEED);
    let samples = generate(SAMPLE_COUNT, BLUE_NOISE_RESAMPLE_COUNT, &mut || {
        rng.gen::<u32>()
    });

    write_samples_svg(
        &mut w,
        &samples,
        |sample| match sample.pair_class() {
            PairClass::A => "blue",
            PairClass::B => "red",
        },
        (x, y),
        GRID_SIZE,
        (1, 1),
    )?;
    x += GRID_SIZE + MARGIN;

    write_samples_svg(
        &mut w,
        &samples,
        |sample| match sample.quad_class() {
            QuadClass::A => "blue",
            QuadClass::B => "red",
            QuadClass::C => "green",
            QuadClass::D => "orange",
        },
        (x, y),
        GRID_SIZE,
        (1, 1),
    )?;

    writeln!(w, "</svg>")?;
    Ok(())
}

fn main() -> Result<()> {
    write_partitions_svg()?;
    write_classes_svg()?;
    Ok(())
}
