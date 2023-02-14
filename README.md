# flame-clustering

## Description

**F**uzzy clustering by **L**ocal **A**pproximation of **ME**mberships (FLAME)
is a data clustering algorithm that defines clusters in the dense parts of
a dataset and performs cluster assignment solely based on the neighborhood
relationships among objects.

The algorithm was first described in:
"FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data",
BMC Bioinformatics, 2007, 8:3.
Available from: http://www.biomedcentral.com/1471-2105/8/3

This Rust library was adapted from the
[original C implementation](https://code.google.com/archive/p/flame-clustering/).

## Usage Example

The following is a simplified example of how one would use the library to
cluster data:

```rust
use flame_clustering::DistanceGraph;

let data: Vec<f64> = vec![0.12, 0.23, 0.15, 0.19, 100.0];
let (clusters, outliers) = DistanceGraph::build(&data, |a, b| (a - b).abs())
    .find_supporting_objects(2, -1.0)
    .approximate_fuzzy_memberships(100, 1e-6)
    .make_clusters(-1.0);

assert_eq!(format!("{clusters:?}"), "[[0, 1, 3, 2]]");
assert_eq!(format!("{outliers:?}"), "[4]");
```

The output is a sequence of indexes into the original dataset. In the example
above, one cluster was identified that contains the numbers
`[0.12, 0.15, 0.19, 0.23]`, and `100.0` was identified as an outlier.

See the library documentation for more information about method parameters.

## Compatibility Notes

The Rust implementation differs from the original library in the following ways:

- The single `Flame` struct was replaced with a builder-style API. See library
  documentation for details.
- The code was modified to accept user-supplied distance functions.
- The ability to specify a distance matrix directly is not yet available.
- Outliers are no longer automatically assigned to the nearest cluster. By
  default outliers remain outliers; in order to preserve the original behavior
  one will need to call a new `assign_outliers()` method.
- The clustering algorithm uses 64-bit floating point numbers, whereas the
  original mostly used 32-bit numbers. This results in minor numerical
  differences when compared to the C implementation.
- The original library provided `Flame_DotProduct` and `Flame_DotProductDist`
  distance methods, but I have not included them here.
- There were no tests included with the original implementation. A few unit tests
  have been added as a development aid when when porting the code, but test
  coverage is incomplete.
- Minor grammatical corrections have been made to comments in the code.
