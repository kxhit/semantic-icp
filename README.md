# About

**Name:** Semantic ICP

**Summary:** Novel point cloud registration algorithm that directly incorporates
pixelated semantic measurements into the estimation of the relative
transformation between two point clouds.

**Description:** The algorithm uses an Iterative Closest Point (ICP)-like scheme
and performs joint semantic and geometric inference using the
Expectation-Maximization technique in which semantic labels and point
associations between two point clouds are treated as latent random variables.
The minimization of the expected cost on the three-dimensional special Euclidean
group, i.e., SE(3), yields the rigid body transformation between two point
clouds.


**License:** BSD 3-Clause

# Requirements


## libraries
* PCL
* Ceres-Solver
* Sophus
* gtest

# Install

```
$ mdkir build
$ cd build
$ cmake ../
$ make
```

# Maintainers

Steven Parkison (sparki@umich.edu)

# Paper

```
@INPROCEEDINGS { sparkison-2018a,
  AUTHOR = { Steven A Parkison and Lu Gan and Maani Ghaffari Jadidi and Ryan
    M. Eustice },
  TITLE = { Semantic Iterative Closest Point through
    Expectation-Maximization },
  BOOKTITLE = { Proceedings of the British Machine Vision Conference},
  YEAR = { 2018 },
  MONTH = { September },
  ADDRESS = { Newcastle, UK },
  PAGES = { 1--17 },
}
```
