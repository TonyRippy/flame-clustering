// ===================================================================
//  The original C library:
//  Copyright(C) 2007, Fu Limin (phoolimin@gmail.com).
//  All rights reserved.
//
//  Rust port:
//  Copyright (c) 2023, Tony Rippy (tonyrippy@gmail.com).
//
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//  2. The origin of this software must not be misrepresented; you must
//     not claim that you wrote the original software. If you use this
//     software in a product, an acknowledgment in the product
//     documentation would be appreciated but is not required.
//  3. Altered source versions must be plainly marked as such, and must
//     not be misrepresented as being the original software.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// ===================================================================

//! A collection of useful distance measures.
//!
//! These methods use generic traits to allow use of this library with
//! whatever numerical precision is best for the task at hand. It results in
//! more complicated type signatures, but usage is straightforward:
//!
//! # Example
//!
//! ```
//! # extern crate flame_clustering;
//! use flame_clustering::distance;
//!
//! # fn main() {
//! assert_eq!(distance::manhattan(&[1, 2, 3], &[-4, 5, -6]), 17);
//! assert_eq!(distance::manhattan(&[1.0, 0.0, 0.0], &[0.0, -3.0, 4.0]), 8.0);
//! # }
//! ```

use std::ops::{Add, Div, Sub};

use num_traits::sign::Signed;
use num_traits::{Float, FromPrimitive, Zero};

/// The distance between two points on a coordinate plane.
///
/// See: <https://en.wikipedia.org/wiki/Euclidean_distance>
pub fn euclidean<F, V>(x: &V, y: &V) -> F
where
    F: Float,
    V: AsRef<[F]>,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .fold(F::zero(), |sum, (a, b)| {
            let d = *a - *b;
            d.mul_add(d, sum)
        })
        .sqrt()
}

/// A distance measure where the distance between two points is the sum of the
/// absolute differences of their Cartesian coordinates.
///
/// See: <https://en.wikipedia.org/wiki/Taxicab_geometry>
pub fn manhattan<F, V>(x: &V, y: &V) -> F
where
    F: Copy + Signed + Sub + Zero,
    V: AsRef<[F]>,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .fold(F::zero(), |sum, (a, b)| sum + (*a - *b).abs())
}

/// A measure of distance between two vectors, calculated from the
/// dot product of the vectors divided by the product of their lengths.
///
/// See: <https://en.wikipedia.org/wiki/Cosine_similarity>
pub fn cosine<F, V>(x: &V, y: &V) -> F
where
    F: Float + FromPrimitive,
    V: AsRef<[F]>,
{
    // Convert measure of similarity into a distance measure.
    F::one() - _cosine(x.as_ref(), y.as_ref())
}

fn _cosine<F>(x: &[F], y: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let (r, x2, y2) = x
        .iter()
        .zip(y.iter())
        .fold((F::zero(), F::zero(), F::zero()), |(r, x2, y2), (x, y)| {
            (x.mul_add(*y, r), x.mul_add(*x, x2), y.mul_add(*y, y2))
        });
    r / ((x2 * y2).sqrt() + F::from_f64(crate::EPSILON).unwrap())
}

fn average<F>(v: &[F]) -> F
where
    F: Zero + Add + Div<Output = F> + Copy + FromPrimitive,
{
    let mut it = v.iter();
    match it.next() {
        None => F::zero(),
        Some(x) => {
            let sum = it.fold(*x, |sum, x| sum + *x);
            let n = F::from_usize(v.len()).unwrap();
            sum / n
        }
    }
}

/// A measure of distance between two samples, calculated using
/// Pearson correlation.
///
/// See: <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>
pub fn pearson<F, V>(x: &V, y: &V) -> F
where
    F: Float + FromPrimitive,
    V: AsRef<[F]>,
{
    // Convert measure of similarity into a distance measure.
    F::one() - _pearson(x.as_ref(), y.as_ref())
}

fn _pearson<F>(x: &[F], y: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let xavg = average(x);
    let yavg = average(y);
    let (r, x2, y2) =
        x.iter()
            .zip(y.iter())
            .fold((F::zero(), F::zero(), F::zero()), |(r, x2, y2), (x, y)| {
                let dx = *x - xavg;
                let dy = *y - yavg;
                (dx.mul_add(dy, r), dx.mul_add(dx, x2), dy.mul_add(dy, y2))
            });
    r / ((x2 * y2).sqrt() + F::from_f64(crate::EPSILON).unwrap())
}

pub fn uc_pearson<F, V>(x: &V, y: &V) -> F
where
    F: Float + FromPrimitive,
    V: AsRef<[F]>,
{
    // Convert measure of similarity into a distance measure.
    F::one() - _uc_pearson(x.as_ref(), y.as_ref())
}

fn _uc_pearson<F>(x: &[F], y: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let xavg = average(x);
    let yavg = average(y);
    let (r, x2, y2) =
        x.iter()
            .zip(y.iter())
            .fold((F::zero(), F::zero(), F::zero()), |(r, x2, y2), (x, y)| {
                let dx = *x - xavg;
                let dy = *y - yavg;
                (x.mul_add(*y, r), dx.mul_add(dx, x2), dy.mul_add(dy, y2))
            });

    r / ((x2 * y2).sqrt() + F::from_f64(crate::EPSILON).unwrap())
}

pub fn sq_pearson<F, V>(x: &V, y: &V) -> F
where
    F: Float + FromPrimitive,
    V: AsRef<[F]>,
{
    // Convert measure of similarity into a distance measure.
    F::one() - _sq_pearson(x.as_ref(), y.as_ref())
}

fn _sq_pearson<F>(x: &[F], y: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let xavg = average(x);
    let yavg = average(y);
    let (r, x2, y2) =
        x.iter()
            .zip(y.iter())
            .fold((F::zero(), F::zero(), F::zero()), |(r, x2, y2), (x, y)| {
                let dx = *x - xavg;
                let dy = *y - yavg;
                (dx.mul_add(dy, r), dx.mul_add(dx, x2), dy.mul_add(dy, y2))
            });
    r * r / (x2 * y2 + F::from_f64(crate::EPSILON).unwrap())
}

/// A measure of distance between two samples, calculated using their covariance.
///
/// See: <https://en.wikipedia.org/wiki/Covariance>
pub fn covariance<F, V>(x: &V, y: &V) -> F
where
    F: Float + FromPrimitive,
    V: AsRef<[F]>,
{
    // Convert measure of similarity into a distance measure.
    F::one() - _covariance(x.as_ref(), y.as_ref())
}

fn _covariance<F>(x: &[F], y: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let it = x.iter().zip(y.iter());
    let m = it.len();
    if m <= 1 {
        return F::zero();
    }
    let xavg = average(x);
    let yavg = average(y);
    it.fold(F::zero(), |r, (x, y)| {
        let dx = *x - xavg;
        let dy = *y - yavg;
        dx.mul_add(dy, r)
    }) / F::from_usize(m - 1).unwrap()
}

mod tests {

    #[test]
    pub fn euclidean() {
        assert_eq!(super::euclidean(&[0.0, 0.0, 0.0], &[0.0, 3.0, 4.0]), 5.0);
    }

    #[test]
    pub fn euclidean_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert_eq!(super::euclidean(x, x), 0.0);
    }

    #[test]
    pub fn manhattan() {
        assert_eq!(super::manhattan(&[0, 0, 0], &[0, 3, 4]), 7);
        assert_eq!(super::manhattan(&[1.0, 0.0, 0.0], &[0.0, -3.0, 4.0]), 8.0);
    }

    #[test]
    pub fn manhattan_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert_eq!(super::manhattan(x, x), 0.0);
    }

    #[test]
    pub fn average() {
        assert_eq!(super::average::<f32>(&[]), 0.0);
        assert_eq!(super::average(&[1, 10, 22]), 11);
    }

    #[test]
    pub fn cosine_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert!(super::cosine(x, x) < crate::EPSILON);
    }

    #[test]
    pub fn pearson_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert!(super::pearson(x, x) < crate::EPSILON);
    }

    #[test]
    pub fn uc_pearson_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert!(super::uc_pearson(x, x) < crate::EPSILON);
    }

    #[test]
    pub fn sq_pearson_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert!(super::sq_pearson(x, x) < crate::EPSILON);
    }

    #[test]
    pub fn covariance_distance_to_self() {
        let x = &[1.1, 2.2, 3.3];
        assert!(super::covariance(x, x) < crate::EPSILON);
    }
}
