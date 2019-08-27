use pmj::generate;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::time::{Duration, Instant};

fn generate_bench(count: usize) -> Duration {
    let mut rng = SmallRng::seed_from_u64(0);
    let start = Instant::now();
    let _samples = generate(count, 0, &mut rng);
    start.elapsed()
}

fn secs_from_duration(d: Duration) -> f64 {
    (d.as_secs() as f64) + f64::from(d.subsec_nanos()) * (0.000_000_001)
}

fn repeat_measure<F: Fn() -> Duration>(f: &F, name: &str) {
    const COUNT: usize = 16;
    let mut sum = 0.0;
    let mut sq_sum = 0.0;
    for _ in 0..COUNT {
        let d = secs_from_duration(f());
        sum += d;
        sq_sum += d * d;
    }
    let avg = sum / (COUNT as f64);
    let var = sum * sum - sq_sum;
    println!("{}: avg {} (var: {})", name, avg, var);
}

fn main() {
    repeat_measure(&|| generate_bench(256), "generate(256)");
    repeat_measure(&|| generate_bench(1024), "generate(1024)");
    repeat_measure(&|| generate_bench(4096), "generate(4096)");
}
