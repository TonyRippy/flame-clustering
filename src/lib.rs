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
//! use flame_clustering::DistanceGraph;
//!
//! # fn main() {
//! let data: Vec<f64> = vec![0.12, 0.23, 0.15, 0.19, 100.0];
//! let (clusters, outliers) = DistanceGraph::build(&data, |a, b| (a - b).abs())
//!     .find_supporting_objects(2, -1.0)
//!     .approximate_fuzzy_memberships(100, 1e-6)
//!     .make_clusters(-1.0);
//!
//! assert_eq!(format!("{clusters:?}"), "[[0, 1, 3, 2]]");
//! assert_eq!(format!("{outliers:?}"), "[4]");
//! # }
//! ```

pub mod distance;

#[macro_use]
extern crate derivative;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjectType {
    Normal,
    Support,
    Outlier,
}

#[derive(Debug)]
pub struct DistanceGraph {
    /// Number of objects
    n: usize,

    /// Upper bound for K defined as: sqrt(N)+10
    kmax: usize,

    /// Stores the kmax nearest neighbors instead of K nearest neighbors
    /// for each objects, so that when K is changed, weights and CSOs can be
    /// re-computed without referring to the original data.
    neighbors: Vec<Vec<usize>>,

    /// Distances to the KMAX nearest neighbors.
    distances: Vec<Vec<f64>>,
}

impl DistanceGraph {
    /// Start a new instance of the FLAME clustering algortim,
    /// given a set of objects and a distance function.
    pub fn build<'a, V, F>(data: &'a [V], distfunc: F) -> DistanceGraph
    where
        F: Fn(&'a V, &'a V) -> f64,
    {
        let n = data.len();
        let mut kmax: usize = (n as f64).sqrt() as usize + 10;
        if kmax >= n {
            kmax = n - 1;
        }
        let mut neighbors = Vec::<Vec<usize>>::with_capacity(n);
        let mut distances = Vec::<Vec<f64>>::with_capacity(n);
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
            neighbors.push(indexes);
            distances.push(values);
        }
        DistanceGraph {
            n,
            kmax,
            neighbors,
            distances,
        }
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
    pub fn find_supporting_objects(
        &self,
        mut knn: usize,
        mut thd: f64,
    ) -> ClusterSupportingObjects {
        if knn > self.kmax {
            knn = self.kmax;
        }
        let mut nncounts = Vec::<usize>::with_capacity(self.n);
        let mut weights = Vec::<Vec<f64>>::with_capacity(self.n);
        let mut density = Vec::<f64>::with_capacity(self.n);
        for i in 0..self.n {
            let dists = &self.distances[i];
            // To include all the neighbors that have distances equal to the
            // distance of the most distant one of the K-Nearest Neighbors
            let mut k = knn;
            let d = dists[knn - 1];
            k += dists[knn..self.kmax].iter().filter(|&x| *x == d).count();
            nncounts.push(k);

            // The definition of weights in this implementation is
            // different from the previous implementations where distances
            // or similarities often have to be transformed in some way.
            //
            // But in this definition, the weights are only dependent on
            // the ranking of distances of the neighbors, so it is more
            // robust against distance transformations.
            let mut sum = ((k * (k + 1)) / 2) as f64;
            weights.push((0..k).map(|j| (k - j) as f64 / sum).collect());

            sum = dists.iter().take(k).sum();
            density.push((sum + EPSILON).recip());
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

        let mut obtypes = vec![ObjectType::Normal; self.n];
        let mut cso_count = 0;
        for i in 0..self.n {
            let k = nncounts[i];
            let mut fmax = 0.0;
            let mut fmin = density[i] / density[self.neighbors[i][0]];
            for j in 1..k {
                let d = density[i] / density[self.neighbors[i][j]];
                if d > fmax {
                    fmax = d;
                }
                if d < fmin {
                    fmin = d;
                }
                // To avoid defining neighboring objects or objects close
                // to an outlier as CSOs.
                if obtypes[self.neighbors[i][j]] != ObjectType::Normal {
                    fmin = 0.0;
                }
            }
            if fmin >= 1.0 {
                cso_count += 1;
                obtypes[i] = ObjectType::Support;
            } else if fmax <= 1.0 && density[i] < thd {
                obtypes[i] = ObjectType::Outlier;
            }
        }

        let mut k = 0;
        let fuzzyships = obtypes
            .iter()
            .map(|obtype| match obtype {
                ObjectType::Support => {
                    // Full membership to the cluster represented by itself.
                    let mut fuzzy = vec![0.0; cso_count + 1];
                    fuzzy[k] = 1.0;
                    k += 1;
                    fuzzy
                }
                ObjectType::Outlier => {
                    // Full membership to the outlier group.
                    let mut fuzzy = vec![0.0; cso_count + 1];
                    fuzzy[cso_count] = 1.0;
                    fuzzy
                }
                _ => {
                    // Equal memberships to all clusters and the outlier group.
                    // Random initialization does not change the results.
                    let frac = ((cso_count + 1) as f64).recip();
                    vec![frac; cso_count + 1]
                }
            })
            .collect::<Vec<Vec<f64>>>();
        let fuzzyships2 = fuzzyships.clone();

        ClusterSupportingObjects {
            graph: self,
            nncounts,
            weights,
            cso_count,
            obtypes,
            fuzzyships,
            fuzzyships2,
        }
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ClusterSupportingObjects<'a> {
    graph: &'a DistanceGraph,

    /// Nearest neighbor count.
    /// it can be different from K if an object has nearest neighbors with
    /// equal distance.
    nncounts: Vec<usize>,
    weights: Vec<Vec<f64>>,

    /// Number of identified Cluster Supporting Objects
    cso_count: usize,

    /// The classification of objects.
    obtypes: Vec<ObjectType>,

    /// The fuzzy measure of how likely it is that an object is a member
    /// of a cluster, iteratively computed.
    fuzzyships: Vec<Vec<f64>>,

    #[derivative(Debug = "ignore")]
    fuzzyships2: Vec<Vec<f64>>,
}

impl<'a> ClusterSupportingObjects<'a> {
    /// Returns the number of cluster supporting objects.
    pub fn count(&self) -> usize {
        self.cso_count
    }

    /// Local Approximation of fuzzy memberships.
    /// Stopped after the maximum steps of iterations;
    /// Or stopped when the overall membership difference between
    /// two iterations become less than epsilon.
    pub fn approximate_fuzzy_memberships(mut self, steps: usize, epsilon: f64) -> Self {
        for _ in 0..steps {
            // Move previous iteration to fuzzyships2, update fuzzyships.
            std::mem::swap(&mut self.fuzzyships, &mut self.fuzzyships2);
            if self.step(ObjectType::Normal) < epsilon {
                break;
            }
        }
        // Update memberships to remove clusters that only contain the CSO.
        self.step(ObjectType::Support);
        self
    }

    /// Attempt to assign outliers to the nearest cluster.
    pub fn assign_outliers(mut self) -> Self {
        self.step(ObjectType::Outlier);
        self
    }

    /// Perform one iteration of the local approximation algorithm.
    fn step(&mut self, obtype: ObjectType) -> f64 {
        let fuzzy2 = &self.fuzzyships2;
        let mut dev = 0.0;
        for (i, fuzzy) in self.fuzzyships.iter_mut().enumerate() {
            if self.obtypes[i] != obtype {
                continue;
            }
            let knn = self.nncounts[i];
            let ids = &self.graph.neighbors[i];
            let wt = &self.weights[i];
            let mut sum = 0.0;
            // Update membership of an object by a linear combination of
            // the memberships of its nearest neighbors.
            for (j, fv) in fuzzy.iter_mut().enumerate() {
                let value = (0..knn).map(|k| wt[k] * fuzzy2[ids[k]][j]).sum();
                *fv = value;
                sum += value;
                let d = value - fuzzy2[i][j];
                dev += d * d;
            }
            for value in fuzzy.iter_mut() {
                *value /= sum;
            }
        }
        dev
    }

    /// Construct clusters.
    /// If 0<thd<1:
    ///   each object is assigned to all clusters in which
    ///   it has membership higher than thd; if it can not be assigned
    ///   to any clusters, it is then assigned to the outlier group.
    /// Else:
    ///   each object is assigned to the group (clusters/outlier group)
    ///   in which it has the highest membership.
    pub fn make_clusters(&self, thd: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
        // Sort objects based on the "entropy" of fuzzy memberships.
        let mut vals = self
            .fuzzyships
            .iter()
            .enumerate()
            .map(|(index, fuzzy)| IndexFloat {
                index,
                value: fuzzy.iter().fold(0.0, |value, &fs| {
                    if fs > EPSILON {
                        value - fs * fs.ln()
                    } else {
                        value
                    }
                }),
            })
            .collect::<Vec<IndexFloat>>();
        vals.sort_unstable_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        let mut clusters = Vec::with_capacity(self.cso_count + 1);
        for _ in 0..=self.cso_count {
            clusters.push(Vec::new());
        }

        if !(0.0..=1.0).contains(&thd) {
            // Assign each object to the cluster
            // in which it has the highest membership.
            for id in vals.iter().map(|x| x.index) {
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
                for (j, cluster) in clusters.iter_mut().enumerate() {
                    if self.fuzzyships[id][j] > thd {
                        cluster.push(id);
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
