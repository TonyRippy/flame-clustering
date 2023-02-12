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

## Compatibility Notes

The Rust implementation differs from the original library in the following ways:

- The code was modified to accept user-supplied distance functions.
- The clustering algorithm uses 64-bit floating point numbers, whereas the
  original mostly used 32-bit numbers. This results in minor numerical
  differences when compared to the C implementation.
- The original library provided `Flame_DotProduct` and `Flame_DotProductDist`
  distance methods, but I have not included them here.
- There were no tests included with the original implementation. A few unit tests
  have been added as a development aid when when porting the code, but test
  coverage is incomplete.
- Minor grammatical corrections have been made to comments in the code.
