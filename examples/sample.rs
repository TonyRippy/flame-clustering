// An example that uses FLAME to analyze multi-dimensional data.
//
// To run the example, execute the following from the project root:
//   cargo run --example sample examples/matrix.txt

extern crate flame_clustering;

use flame_clustering::{distance, DistanceGraph};
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::process::exit;

fn read_data(reader: impl io::BufRead) -> io::Result<Vec<Vec<f64>>> {
    let mut lines = reader
        .lines()
        .map(|x| x.unwrap().trim().to_string())
        .filter(|x| !x.is_empty());

    let header = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|n| n.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();
    let n = header[0];
    let m = header[1];
    println!("Reading dataset with {} rows and {} columns", n, m);

    let mut out = Vec::with_capacity(n);
    for line in lines {
        let v = line
            .split_whitespace()
            .map(|n| n.parse::<f64>().unwrap())
            .collect::<Vec<f64>>();
        assert_eq!(v.len(), m);
        out.push(v);
    }
    Ok(out)
}

fn print_cluster(cluster: &[usize]) {
    for (j, v) in cluster.iter().enumerate() {
        if j > 0 {
            print!(",");
            if j % 10 == 0 {
                println!();
            }
        }
        print!("{:5}", v);
    }
    println!();
}

fn main() -> io::Result<()> {
    let filename = env::args().nth(1);
    if filename.is_none() {
        eprintln!("No input file");
        exit(1);
    }
    let data = read_data(io::BufReader::new(File::open(filename.unwrap())?))?;
    let flame = DistanceGraph::build(&data, distance::euclidean);

    print!("Detecting Cluster Supporting Objects ...");
    io::stdout().flush()?;
    let supports = flame.find_supporting_objects(10, -2.0);
    println!("done, found {}", supports.count());

    print!("Propagating fuzzy memberships ... ");
    io::stdout().flush()?;
    let fuzzyships = supports
        .approximate_fuzzy_memberships(500, 1e-6)
        .assign_outliers();
    println!("done");

    print!("Defining clusters from fuzzy memberships ... ");
    io::stdout().flush()?;
    let (clusters, outliers) = fuzzyships.make_clusters(-1.0);
    println!("done");

    for (i, cluster) in clusters.iter().enumerate() {
        print!("\nCluster {:3}, with {:6} members:\n", i + 1, cluster.len());
        print_cluster(cluster);
    }
    print!("\nCluster outliers, with {:6} members:\n", outliers.len());
    print_cluster(&outliers);

    Ok(())
}
