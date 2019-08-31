#![deny(missing_docs)]

//! # Overview
//!
//! A crate to generate pmj02 and pmj02b sample sequences as described in the following papers:
//! * [Progressive Multi-Jittered Sample Sequences](https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/) by Christensen et al.
//! * [Efficient Generation of Points that Satisfy Two-Dimensional Elementary Intervals](http://jcgt.org/published/0008/01/04/) by Matt Pharr.

use rand_core::RngCore;
use std::fmt;
use std::num::NonZeroU64;

/// Divides a set of samples into 2 classes.
#[derive(Debug, Copy, Clone)]
pub enum PairClass {
    /// Class 1 of 2.
    A,
    /// Class 2 of 2.
    B,
}

/// Divides a set of samples into 4 classes.
#[derive(Debug, Copy, Clone)]
pub enum QuadClass {
    /// Class 1 of 4.
    A,
    /// Class 2 of 4.
    B,
    /// Class 3 of 4.
    C,
    /// Class 4 of 4.
    D,
}

/// Represents a 2D sample coordinate and classes.
#[derive(Copy, Clone)]
pub struct Sample(NonZeroU64);

impl Sample {
    const COORD_BIT_COUNT: u32 = 23;
    const COORD_MASK: u32 = 0x007f_ffff;
    const CLASS_MASK: u32 = 0x0000_0003;
    const X_SHIFT: u32 = 1;
    const Y_SHIFT: u32 = Self::X_SHIFT + Self::COORD_BIT_COUNT;
    const CLASS_SHIFT: u32 = Self::Y_SHIFT + Self::COORD_BIT_COUNT;

    fn new(x_bits: u32, y_bits: u32, class_bits: u32) -> Self {
        debug_assert_eq!(x_bits & Self::COORD_MASK, x_bits);
        debug_assert_eq!(y_bits & Self::COORD_MASK, y_bits);
        debug_assert_eq!(class_bits & Self::CLASS_MASK, class_bits);
        let all_bits = 1
            | (u64::from(x_bits) << Self::X_SHIFT)
            | (u64::from(y_bits) << Self::Y_SHIFT)
            | (u64::from(class_bits) << Self::CLASS_SHIFT);
        unsafe { Self(NonZeroU64::new_unchecked(all_bits)) }
    }

    #[inline]
    fn x_bits(self, bit_count: u32) -> u32 {
        debug_assert!(bit_count <= Self::COORD_BIT_COUNT);
        let x_bits = ((self.0.get() >> Self::X_SHIFT) as u32) & Self::COORD_MASK;
        x_bits >> (Self::COORD_BIT_COUNT - bit_count)
    }

    #[inline]
    fn y_bits(self, bit_count: u32) -> u32 {
        debug_assert!(bit_count <= Self::COORD_BIT_COUNT);
        let y_bits = ((self.0.get() >> Self::Y_SHIFT) as u32) & Self::COORD_MASK;
        y_bits >> (Self::COORD_BIT_COUNT - bit_count)
    }

    #[inline]
    fn class_bits(self) -> u32 {
        ((self.0.get() >> Self::CLASS_SHIFT) as u32) & Self::CLASS_MASK
    }

    #[inline]
    fn grid_index(self, x_bit_count: u32, y_bit_count: u32) -> u32 {
        (self.y_bits(y_bit_count) << x_bit_count) | self.x_bits(x_bit_count)
    }

    /// Returns the x coordinate of this sample in [0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// # use rand::prelude::*;
    /// # use rand::rngs::SmallRng;
    /// # use pmj::*;
    /// let mut rng = SmallRng::seed_from_u64(0);
    /// let samples = generate(1, 0, &mut rng);
    /// let x = samples[0].x();
    /// assert!(0.0 <= x && x < 1.0);
    /// ```
    #[inline]
    pub fn x(self) -> f32 {
        f32::from_bits(0x3f80_0000 | self.x_bits(Self::COORD_BIT_COUNT)) - 1f32
    }

    /// Returns the y coordinate of this sample in [0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// # use rand::prelude::*;
    /// # use rand::rngs::SmallRng;
    /// # use pmj::*;
    /// let mut rng = SmallRng::seed_from_u64(0);
    /// let samples = generate(1, 0, &mut rng);
    /// let y = samples[0].y();
    /// assert!(0.0 <= y && y < 1.0);
    /// ```
    #[inline]
    pub fn y(self) -> f32 {
        f32::from_bits(0x3f80_0000 | self.y_bits(Self::COORD_BIT_COUNT)) - 1f32
    }

    /// Returns which class this sample belongs to if samples are divided into 2 classes.
    #[inline]
    pub fn pair_class(self) -> PairClass {
        if (self.class_bits() & 0x2) == 0 {
            PairClass::A
        } else {
            PairClass::B
        }
    }

    /// Returns which class this sample belongs to if samples are divided into 4 classes.
    #[inline]
    pub fn quad_class(self) -> QuadClass {
        match self.class_bits() {
            0 => QuadClass::A,
            1 => QuadClass::B,
            2 => QuadClass::C,
            _ => QuadClass::D,
        }
    }
}

impl fmt::Debug for Sample {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Sample")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("pair_class", &self.pair_class())
            .field("quad_class", &self.quad_class())
            .finish()
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "({}, {})", self.x(), self.y())
    }
}

#[inline]
fn generate_sample_bits<R: RngCore + ?Sized>(
    high_bit_count: u32,
    high_bits: u32,
    rng: &mut R,
) -> u32 {
    let low_bit_count = Sample::COORD_BIT_COUNT - high_bit_count;
    let rng_shift = 32 - low_bit_count;
    (high_bits << low_bit_count) | (rng.next_u32() >> rng_shift)
}

#[inline]
fn generate_index<R: RngCore + ?Sized>(len: usize, rng: &mut R) -> usize {
    let u = rng.next_u32();
    let prod = u64::from(u) * (len as u64);
    (prod >> 32) as usize
}

#[derive(Default)]
struct SampleCoordSet {
    valid_x: Vec<u32>,
    valid_y: Vec<u32>,
    bit_count: u32,
}

impl SampleCoordSet {
    fn clear(&mut self, bit_count: u32) {
        self.valid_x.clear();
        self.valid_y.clear();
        self.bit_count = bit_count;
    }

    fn sample<R: RngCore + ?Sized>(
        &self,
        rng: &mut R,
        class_bits: u32,
    ) -> Sample {
        let x_index = generate_index(self.valid_x.len(), rng);
        let y_index = generate_index(self.valid_y.len(), rng);
        let valid_x = self.valid_x.get(x_index).cloned().expect("no valid x");
        let valid_y = self.valid_y.get(y_index).cloned().expect("no valid y");
        let x_bits = generate_sample_bits(self.bit_count, valid_x, rng);
        let y_bits = generate_sample_bits(self.bit_count, valid_y, rng);
        Sample::new(x_bits, y_bits, class_bits)
    }
}

struct BitArray {
    blocks: Box<[u32]>,
    count: usize,
}

impl BitArray {
    fn new(count: usize) -> Self {
        let block_count = ((count + 31) / 32).max(1);
        Self {
            blocks: vec![0; block_count].into_boxed_slice(),
            count,
        }
    }

    fn set(&mut self, index: usize) {
        assert!(index < self.count);
        self.blocks[index / 32] |= 1u32 << (index % 32);
    }

    fn is_set(&self, index: usize) -> bool {
        assert!(index < self.count);
        (self.blocks[index / 32] & (1u32 << (index % 32))) != 0
    }

    fn is_all_set(&self) -> bool {
        self.blocks.iter().enumerate().all(|(block_index, block)| {
            let end_index = 32 * (block_index + 1);
            let skip_count = end_index.max(self.count) - self.count;
            let mask = 0xffff_ffff >> skip_count;
            (block & mask) == mask
        })
    }
}

struct StratificationAccel {
    levels: Vec<BitArray>,
    bit_count: u32,
}

impl StratificationAccel {
    fn new(bit_count: u32, samples: &[Sample]) -> Self {
        let mut levels = Vec::new();
        for x_bit_count in 0..=bit_count {
            let y_bit_count = bit_count - x_bit_count;
            let mut level = BitArray::new(1 << bit_count);
            for sample in samples {
                let index = sample.grid_index(x_bit_count, y_bit_count);
                debug_assert!(!level.is_set(index as usize));
                level.set(index as usize);
            }
            levels.push(level);
        }
        Self { levels, bit_count }
    }

    fn filter_x(&self, x_bits: u32, y_bits: u32, low_bit_count: u32) -> bool {
        (0..low_bit_count).all(|y_bit_count| {
            let x_bit_count = self.bit_count - y_bit_count;
            let x_mask = (1 << x_bit_count) - 1;
            let index = (y_bits & !x_mask) | (x_bits >> y_bit_count);
            !self.levels[x_bit_count as usize].is_set(index as usize)
        })
    }

    fn filter_y(&self, x_bits: u32, y_bits: u32, low_bit_count: u32) -> bool {
        (0..low_bit_count).all(|x_bit_count| {
            let y_bit_count = self.bit_count - x_bit_count;
            let x_mask = (1 << x_bit_count) - 1;
            let index = (y_bits & !x_mask) | (x_bits >> y_bit_count);
            !self.levels[x_bit_count as usize].is_set(index as usize)
        })
    }

    fn get_valid_coords(
        &self,
        high_bit_count: u32,
        high_x: u32,
        high_y: u32,
        coord_set: &mut SampleCoordSet,
    ) {
        let low_bit_count = self.bit_count - high_bit_count;
        let partial_x = high_x << low_bit_count;
        let partial_y = high_y << low_bit_count;
        let low_count = 1 << low_bit_count;

        coord_set.clear(self.bit_count);

        coord_set.valid_x.extend((0..low_count).filter_map(|low_x| {
            // build coordinates
            let x_bits = partial_x | low_x;
            let y_bits = partial_y;

            // brute force check this location is valid
            if self.filter_x(x_bits, y_bits, low_bit_count) {
                Some(x_bits)
            } else {
                None
            }
        }));

        coord_set.valid_y.extend((0..low_count).filter_map(|low_y| {
            // build coordinates
            let x_bits = partial_x;
            let y_bits = partial_y | low_y;

            // brute force check this location is valid
            if self.filter_y(x_bits, y_bits, low_bit_count) {
                Some(y_bits)
            } else {
                None
            }
        }));
    }

    fn set(&mut self, sample: Sample) {
        for (x_bit_count, level) in self.levels.iter_mut().enumerate() {
            let x_bit_count = x_bit_count as u32;
            let y_bit_count = self.bit_count - x_bit_count;
            let index = sample.grid_index(x_bit_count, y_bit_count) as usize;
            debug_assert!(!level.is_set(index));
            level.set(index);
        }
    }

    fn is_all_set(&self) -> bool {
        self.levels.iter().all(|level| level.is_all_set())
    }
}

struct BlueNoiseAccel {
    grid: Vec<Option<Sample>>,
    bit_count: u32,
}

impl BlueNoiseAccel {
    fn new(bit_count: u32, samples: &[Sample]) -> Self {
        let grid_size = 1 << (2 * bit_count);
        let mut grid = vec![None; grid_size];
        for sample in samples {
            let index = sample.grid_index(bit_count, bit_count);
            let elem = unsafe { grid.get_unchecked_mut(index as usize) };
            debug_assert!(elem.is_none());
            *elem = Some(*sample);
        }
        Self { grid, bit_count }
    }

    fn get_min_distance_sq(&self, sample: Sample) -> f32 {
        let grid_size = 1 << (2 * self.bit_count);
        let grid_stride = 1 << self.bit_count;
        let centre_index = sample.grid_index(self.bit_count, self.bit_count);

        // check all neighbouring grid cells
        let ref_x = sample.x();
        let ref_y = sample.y();
        [
            grid_size - grid_stride - 1,
            grid_size - grid_stride,
            grid_size - grid_stride + 1,
            grid_size - 1,
            // skip centre sample, we know it is empty
            grid_size + 1,
            grid_size + grid_stride - 1,
            grid_size + grid_stride,
            grid_size + grid_stride + 1,
        ]
        .iter()
        .filter_map(|offset| {
            let index = (centre_index + offset) & (grid_size - 1);
            unsafe { self.grid.get_unchecked(index as usize) }.map(|s| {
                let xd = (s.x() - ref_x).abs();
                let yd = (s.y() - ref_y).abs();
                let xd = xd.min(1f32 - xd);
                let yd = yd.min(1f32 - yd);
                xd * xd + yd * yd
            })
        })
        .fold(2f32, |m, x| m.min(x))
    }

    fn set(&mut self, sample: Sample) {
        let index = sample.grid_index(self.bit_count, self.bit_count);
        let elem = unsafe { self.grid.get_unchecked_mut(index as usize) };
        debug_assert!(elem.is_none());
        *elem = Some(sample);
    }
}

fn pick_sample<R: RngCore + ?Sized>(
    strat_result: &SampleCoordSet,
    bn_accel: Option<&BlueNoiseAccel>,
    blue_noise_retry_count: u32,
    rng: &mut R,
    class_bits: u32,
) -> Sample {
    let mut sample = strat_result.sample(rng, class_bits);
    if let Some(bn_accel) = bn_accel {
        let mut dist_sq = bn_accel.get_min_distance_sq(sample);
        for _ in 0..blue_noise_retry_count {
            let other_sample = strat_result.sample(rng, class_bits);
            let other_dist_sq = bn_accel.get_min_distance_sq(other_sample);
            if dist_sq < other_dist_sq {
                sample = other_sample;
                dist_sq = other_dist_sq;
            }
        }
    }
    sample
}

/// Generates samples of a pmj02 sequence.
///
/// If `blue_noise_retry_count` is non-zero, then this many additional candidates
/// are considered as each sample is generated, and the candidate that is the greatest
/// distance from all previous samples is selected.
///
/// The random number generator can be any type that implements the RngCore trait
/// from the rand_core crate.
///
/// # Example
///
/// ```
/// # use rand::prelude::*;
/// # use rand::rngs::SmallRng;
/// # use pmj::*;
/// let mut rng = SmallRng::seed_from_u64(0);
/// let samples = generate(1024, 0, &mut rng);
/// ```
pub fn generate<R: RngCore + ?Sized>(
    sample_count: usize,
    blue_noise_retry_count: u32,
    rng: &mut R,
) -> Vec<Sample> {
    // first sample is anywhere
    let mut samples = Vec::with_capacity(sample_count);
    {
        let x_bits = generate_sample_bits(0, 0, rng);
        let y_bits = generate_sample_bits(0, 0, rng);
        samples.push(Sample::new(x_bits, y_bits, 0));
    }

    // sample to next power of 2, with stratification
    loop {
        if samples.len() >= sample_count {
            break;
        }

        let quadrant_count = samples.len();
        let current_bit_count = (quadrant_count as u32).trailing_zeros();
        let mut strat_result = SampleCoordSet::default();

        let high_bit_count = current_bit_count / 2 + 1;
        let mut bn_accel = if blue_noise_retry_count > 0 {
            Some(BlueNoiseAccel::new(high_bit_count, &samples))
        } else {
            None
        };

        // Generate a set of samples in the diagonally opposite quadrants to existing samples
        let mut strat_accel = StratificationAccel::new(current_bit_count + 1, &samples);
        for old_sample_index in 0..quadrant_count {
            let old_sample = samples[old_sample_index];
            strat_accel.get_valid_coords(
                high_bit_count,
                old_sample.x_bits(high_bit_count) ^ 1,
                old_sample.y_bits(high_bit_count) ^ 1,
                &mut strat_result,
            );
            let sample = pick_sample(
                &strat_result,
                bn_accel.as_ref(),
                blue_noise_retry_count,
                rng,
                old_sample.class_bits() ^ 0x1,
            );
            samples.push(sample);
            strat_accel.set(sample);
            if let Some(bn_accel) = bn_accel.as_mut() {
                bn_accel.set(sample);
            }
        }
        debug_assert!(strat_accel.is_all_set());

        if samples.len() >= sample_count {
            break;
        }

        // Now pick one of the two remaining quadrants to existing samples
        /*
            Currently we pick a quadrant by flipping in x relative to the initial
            quadrant, as this results in a 02 sequence for this sub-sequence, which
            gets good convergence in the test integrals.

            It is possible to choose the quadrant randomly, but this has worse
            convergence on the test integrals.  So far it does not seem possible to
            pick randomly and keep this sub-sequence a 02 sequence, this greedy
            algorithm seems to get stuck where there are no valid samples for a quadrant.
        */
        let mut strat_accel = StratificationAccel::new(current_bit_count + 2, &samples);
        let mut sub_strat_check = StratificationAccel::new(current_bit_count, &[]);
        for old_sample_index in 0..quadrant_count {
            let old_sample = samples[old_sample_index];
            let choice = 1;
            strat_accel.get_valid_coords(
                high_bit_count,
                old_sample.x_bits(high_bit_count) ^ choice,
                old_sample.y_bits(high_bit_count) ^ choice ^ 1,
                &mut strat_result,
            );
            let sample = pick_sample(
                &strat_result,
                bn_accel.as_ref(),
                blue_noise_retry_count,
                rng,
                old_sample.class_bits() ^ 0x2,
            );
            samples.push(sample);
            strat_accel.set(sample);
            sub_strat_check.set(sample);
            if let Some(bn_accel) = bn_accel.as_mut() {
                bn_accel.set(sample);
            }
        }
        debug_assert!(sub_strat_check.is_all_set());

        // Finally pick the remaining quadrent (diagonally opposite the previous set)
        for old_sample_index in 0..quadrant_count {
            let old_sample = samples[old_sample_index + 2 * quadrant_count];
            strat_accel.get_valid_coords(
                high_bit_count,
                old_sample.x_bits(high_bit_count) ^ 1,
                old_sample.y_bits(high_bit_count) ^ 1,
                &mut strat_result,
            );
            let sample = pick_sample(
                &strat_result,
                bn_accel.as_ref(),
                blue_noise_retry_count,
                rng,
                old_sample.class_bits() ^ 0x1,
            );
            samples.push(sample);
            strat_accel.set(sample);
            if let Some(bn_accel) = bn_accel.as_mut() {
                bn_accel.set(sample);
            }
        }
        debug_assert!(strat_accel.is_all_set());
    }

    samples.truncate(sample_count);
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_debug() {
        let s = Sample::new(0x40_0000, 0x40_0000, 0x2);
        assert_eq!(
            format!("{:?}", s),
            "Sample { x: 0.5, y: 0.5, pair_class: B, quad_class: C }"
        );
    }

    #[test]
    fn fmt_display() {
        let s = Sample::new(0x40_0000, 0x40_0000, 0x2);
        assert_eq!(format!("{}", s), "(0.5, 0.5)");
    }
}
