// ===================================================================
// The original C library:
// Copyright(C) 2007, Fu Limin (phoolimin@gmail.com).
// All rights reserved.
//
// Rust port:
// Copyright (c) 2023, Tony Rippy (tonyrippy@gmail.com).
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. The origin of this software must not be misrepresented; you must
//    not claim that you wrote the original software. If you use this
//    software in a product, an acknowledgment in the product
//    documentation would be appreciated but is not required.
// 3. Altered source versions must be plainly marked as such, and must
//    not be misrepresented as being the original software.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// ===================================================================

//! An implementation of the FLAME data clustering algorithm.
//!
//! **F**uzzy clustering by **L**ocal **A**pproximation of **ME**mberships (FLAME)
//! is a data clustering algorithm that defines clusters in the dense parts of
//! a dataset and performs cluster assignment solely based on the neighborhood
//! relationships among objects.
//!
//! The algorithm was first described in:
//! "FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data",
//! BMC Bioinformatics, 2007, 8:3.
//! Available from: <http://www.biomedcentral.com/1471-2105/8/3>
//!
//! This Rust library was adapted from the original C implementation:
//! <https://code.google.com/archive/p/flame-clustering/>
//!
//! # Example
//!
//! ```
//! # extern crate flame_clustering;
//! use flame_clustering::Flame;
//!
//! # fn main() {
//! let data: Vec<f64> = vec![0.12, 0.23, 0.15, 0.19, 100.0];
//! let mut flame = Flame::new(&data, |a, b| (a - b) * (a - b));
//! flame.define_supports(2, -2.0);
//! flame.local_approximation(100, 1e-6);
//! let (clusters, outliers) = flame.make_clusters(-1.0);
//!
//! assert_eq!(format!("{clusters:?}"), "[[0, 3, 1, 2, 4]]");
//! assert_eq!(format!("{outliers:?}"), "[]");
//! # }
//! ```

pub mod distance;

/// Data for clustering are usually noisy, so it is not very necessary
/// to have EPSILON extremely small.
const EPSILON: f64 = 1e-9;

/// For sorting and storing the orignal indices.
#[derive(Clone, Copy, Debug)]
struct IndexFloat {
    index: usize,
    value: f64,
}

/// Sort until the smallest "part" items are sorted.
///
/// Based on algorithm as presented in Adam Drozdek's
/// "Data Structures and Algorithms in C++", 2nd Edition.
fn partial_quicksort(data: &mut [IndexFloat], first: usize, last: usize, part: usize) {
    if first >= last {
        return;
    }
    data.swap(first, (first + last) / 2);
    let pivot = data[first].value;

    let mut lower = first + 1;
    let mut upper = last;
    while lower <= upper {
        while lower <= last && data[lower].value < pivot {
            lower += 1;
        }
        while pivot < data[upper].value {
            upper -= 1;
        }
        if lower < upper {
            data.swap(lower, upper);
            upper -= 1;
        }
        lower += 1;
    }
    data.swap(first, upper);
    if upper > 0 && first < (upper - 1) {
        partial_quicksort(data, first, upper - 1, part);
    }
    if upper >= part {
        return;
    }
    if (upper + 1) < last {
        partial_quicksort(data, upper + 1, last, part);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObjectType {
    Normal,
    Support,
    Outlier,
}

pub struct Flame {
    /// Number of objects
    n: usize,

    /// Number of K-Nearest Neighbors
    k: usize,

    /// Upper bound for K defined as: sqrt(N)+10
    kmax: usize,

    /// Stores the KMAX nearest neighbors instead of K nearest neighbors
    /// for each objects, so that when K is changed, weights and CSOs can be
    /// re-computed without referring to the original data.
    graph: Vec<Vec<usize>>,

    /// Distances to the KMAX nearest neighbors.
    dists: Vec<Vec<f64>>,

    /// Nearest neighbor count.
    /// it can be different from K if an object has nearest neighbors with
    /// equal distance.
    nncounts: Vec<usize>,
    weights: Vec<Vec<f64>>,

    /// Number of identified Cluster Supporting Objects
    pub cso_count: usize,
    obtypes: Vec<ObjectType>,

    pub fuzzyships: Vec<Vec<f64>>,
}

impl Flame {
    /// Start a new instance of the FLAME clustering algortim,
    /// given a set of objects and a distance function.
    pub fn new<'a, V, F>(data: &'a [V], distfunc: F) -> Flame
    where
        F: Fn(&'a V, &'a V) -> f64,
    {
        let n = data.len();
        let mut kmax: usize = (n as f64).sqrt() as usize + 10;
        if kmax >= n {
            kmax = n - 1;
        }

        let mut flame = Flame {
            n,
            k: 0,
            kmax,
            graph: Vec::with_capacity(n),
            dists: Vec::with_capacity(n),
            nncounts: vec![0; n],
            weights: Vec::with_capacity(n),
            cso_count: 0,
            obtypes: vec![ObjectType::Normal; n],
            fuzzyships: Vec::with_capacity(n),
        };

        let mut vals = Vec::with_capacity(n - 1);
        for i in 0..n {
            // Store MAX number of nearest neighbors.
            vals.clear();
            for j in 0..n {
                if j == i {
                    continue;
                }
                vals.push(IndexFloat {
                    index: j,
                    value: distfunc(&data[i], &data[j]),
                });
            }
            partial_quicksort(&mut vals, 0, n - 2, kmax);
            let mut indexes = Vec::with_capacity(kmax);
            let mut values = Vec::with_capacity(kmax);
            for x in vals.iter().take(kmax) {
                indexes.push(x.index);
                values.push(x.value);
            }
            flame.graph.push(indexes);
            flame.dists.push(values);
            flame.weights.push(vec![0.0; kmax]);
        }
        flame
    }

    /// Define knn-nearest neighbors for each object
    /// and the Cluster Supporting Objects (CSO).
    ///
    /// The actual number of nearest neighbors could be larger than knn,
    /// if an object has neighbors of the same distances.
    ///
    /// Based on the distances of the neighbors, a density can be computed
    /// for each object. Objects with local maximum density are defined as
    /// CSOs. The initial outliers are defined as objects with local minimum
    /// density which is less than mean( density ) + thd * stdev( density );
    pub fn define_supports(&mut self, mut knn: usize, mut thd: f64) {
        let mut density = vec![0.0; self.n];

        if knn > self.kmax {
            knn = self.kmax;
        }
        self.k = knn;
        for i in 0..self.n {
            /* To include all the neighbors that have distances equal to the
             * distance of the most distant one of the K-Nearest Neighbors */
            let mut k = knn;
            let d = self.dists[i][knn - 1];
            for j in knn..self.kmax {
                if self.dists[i][j] == d {
                    k += 1;
                } else {
                    break;
                }
            }
            self.nncounts[i] = k;

            // The definition of weights in this implementation is
            // different from the previous implementations where distances
            // or similarities often have to be transformed in some way.
            //
            // But in this definition, the weights are only dependent on
            // the ranking of distances of the neighbors, so it is more
            // robust against distance transformations.
            let mut sum = ((k * (k + 1)) / 2) as f64;
            for j in 0..k {
                self.weights[i][j] = (k - j) as f64 / sum;
            }
            sum = 0.0;
            for j in 0..k {
                sum += self.dists[i][j];
            }
            density[i] = 1.0 / (sum + EPSILON);
        }
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for d in &density {
            sum += d;
            sum2 += d * d;
        }
        sum /= self.n as f64;

        // Density threshold for possible outliers.
        thd = sum + thd * (sum2 / (self.n as f64) - sum * sum).sqrt();

        self.obtypes = vec![ObjectType::Normal; self.n];
        self.cso_count = 0;
        for i in 0..self.n {
            let k = self.nncounts[i];
            let mut fmax = 0.0;
            let mut fmin = density[i] / density[self.graph[i][0]];
            for j in 1..k {
                let d = density[i] / density[self.graph[i][j]];
                if d > fmax {
                    fmax = d;
                }
                if d < fmin {
                    fmin = d;
                }
                // To avoid defining neighboring objects or objects close
                // to an outlier as CSOs.
                if self.obtypes[self.graph[i][j]] != ObjectType::Normal {
                    fmin = 0.0;
                }
            }
            if fmin >= 1.0 {
                self.cso_count += 1;
                self.obtypes[i] = ObjectType::Support;
            } else if fmax <= 1.0 && density[i] < thd {
                self.obtypes[i] = ObjectType::Outlier;
            }
        }
    }

    /// Local Approximation of fuzzy memberships.
    /// Stopped after the maximum steps of iterations;
    /// Or stopped when the overall membership difference between
    /// two iterations become less than epsilon.
    pub fn local_approximation(&mut self, steps: usize, epsilon: f64) {
        let m = self.cso_count;

        let mut k = 0;
        for i in 0..self.n {
            let mut fuzzy: Vec<f64>;
            match self.obtypes[i] {
                ObjectType::Support => {
                    // Full membership to the cluster represented by itself.
                    fuzzy = vec![0.0; m + 1];
                    fuzzy[k] = 1.0;
                    k += 1;
                }
                ObjectType::Outlier => {
                    // Full membership to the outlier group.
                    fuzzy = vec![0.0; m + 1];
                    fuzzy[m] = 1.0;
                }
                _ => {
                    // Equal memberships to all clusters and the outlier group.
                    // Random initialization does not change the results.
                    let frac = ((m + 1) as f64).recip();
                    fuzzy = vec![frac; m + 1];
                }
            }
            self.fuzzyships.push(fuzzy);
        }
        let mut fuzzyships2 = self.fuzzyships.clone();

        let mut dev: f64;
        let mut even = true;
        for _ in 0..steps {
            dev = 0.0;
            for i in 0..self.n {
                if self.obtypes[i] != ObjectType::Normal {
                    continue;
                }
                let knn = self.nncounts[i];
                let ids = &self.graph[i];
                let wt = &self.weights[i];
                let (fuzzy, fuzzy2) = if even {
                    (&mut self.fuzzyships[i], &fuzzyships2)
                } else {
                    (&mut fuzzyships2[i], &self.fuzzyships)
                };
                let mut sum: f64 = 0.0;
                // Update membership of an object by a linear combination of
                // the memberships of its nearest neighbors.
                for j in 0..=m {
                    fuzzy[j] = 0.0;
                    for k in 0..knn {
                        fuzzy[j] += wt[k] * fuzzy2[ids[k]][j];
                    }
                    let d = fuzzy[j] - fuzzy2[i][j];
                    dev += (d * d) as f64;
                    sum += fuzzy[j] as f64;
                }
                for j in 0..=m {
                    fuzzy[j] /= sum as f64;
                }
            }
            even = !even;
            if dev < epsilon {
                break;
            }
        }
        // update the membership of all objects to remove clusters
        // that contains only the CSO.
        for i in 0..self.n {
            let knn = self.nncounts[i];
            let ids = &self.graph[i];
            let wt = &self.weights[i];
            let fuzzy = &mut self.fuzzyships[i];
            let fuzzy2 = &fuzzyships2;
            for j in 0..=m {
                fuzzy[j] = 0.0;
                for k in 0..knn {
                    fuzzy[j] += wt[k] * fuzzy2[ids[k]][j];
                }
                //dev += (fuzzy[j] - fuzzy2[i][j]) * (fuzzy[j] - fuzzy2[i][j]);
            }
        }
    }

    /// Construct clusters.
    /// If 0<thd<1:
    ///   each object is assigned to all clusters in which
    ///   it has membership higher than thd; if it can not be assigned
    ///   to any clusters, it is then assigned to the outlier group.
    /// Else:
    ///   each object is assigned to the group (clusters/outlier group)
    ///   in which it has the highest membership. */
    pub fn make_clusters(&self, thd: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
        // Sort objects based on the "entropy" of fuzzy memberships.
        let mut vals = Vec::with_capacity(self.n);
        for index in 0..self.n {
            let mut value = 0.0;
            for j in 0..=self.cso_count {
                let fs = self.fuzzyships[index][j];
                if fs > EPSILON {
                    value -= fs * fs.ln();
                }
            }
            vals.push(IndexFloat { index, value });
        }
        vals.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        let mut clusters = Vec::with_capacity(self.cso_count + 1);
        for _ in 0..=self.cso_count {
            clusters.push(Vec::new());
        }

        if !(0.0..=1.0).contains(&thd) {
            // Assign each object to the cluster
            // in which it has the highest membership.
            for id in vals.iter().map(|x| x.index) {
                //             for (id, fuzzy) in self.fuzzyships.iter().enumerate() {
                let (imax, _) = self.fuzzyships[id]
                    .iter()
                    .enumerate()
                    .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                    .unwrap();
                clusters[imax].push(id);
            }
        } else {
            // Assign each object to all the clusters
            // in which it has membership higher than thd,
            // otherwise, assign it to the outlier group.
            for id in vals.iter().map(|x| x.index) {
                let mut assigned = false;
                for j in 0..self.cso_count {
                    if self.fuzzyships[id][j] > thd {
                        clusters[j].push(id);
                        assigned = true;
                    }
                }
                if !assigned {
                    clusters[self.cso_count].push(id);
                }
            }
        }
        // keep the outlier group, even if its empty
        let outliers = clusters.pop().unwrap();
        // remove empty clusters
        clusters.retain(|v| !v.is_empty());

        (clusters, outliers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn partial_sort_in_order() {
        let mut v = vec![
            IndexFloat {
                index: 0,
                value: 0.0,
            },
            IndexFloat {
                index: 1,
                value: 1.0,
            },
            IndexFloat {
                index: 2,
                value: 2.0,
            },
            IndexFloat {
                index: 3,
                value: 3.0,
            },
            IndexFloat {
                index: 4,
                value: 4.0,
            },
        ];
        let last = v.len() - 1;
        partial_quicksort(&mut v, 0, last, 2);
        for i in 0..2 {
            assert!(
                v[i].value < v[i + 1].value,
                "Expected v[{}] ({}) to be less than v[{}] ({})",
                i,
                v[i].value,
                i + 1,
                v[i + 1].value
            );
        }
    }

    #[test]
    pub fn partial_sort_in_reverse() {
        let mut v = vec![
            IndexFloat {
                index: 0,
                value: 4.0,
            },
            IndexFloat {
                index: 1,
                value: 3.0,
            },
            IndexFloat {
                index: 2,
                value: 2.0,
            },
            IndexFloat {
                index: 3,
                value: 1.0,
            },
            IndexFloat {
                index: 4,
                value: 0.0,
            },
        ];
        let last = v.len() - 1;
        partial_quicksort(&mut v, 0, last, 2);
        for i in 0..2 {
            assert!(
                v[i].value < v[i + 1].value,
                "Expected v[{}] ({}) to be less than v[{}] ({})",
                i,
                v[i].value,
                i + 1,
                v[i + 1].value
            );
        }
    }
}
