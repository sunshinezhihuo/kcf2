# Object tracker based on kernel correlation filter

This repo is based on the below paper.

> High-Speed Tracking with Kernelized Correlation Filters
>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista
>
> TPAMI 2015

There are two foldes that are different with the paper in this repo.

1. The tracker self-adaptively changes the size of tracking window by using the idea of image scale pyramid.
2. we comprehensively consider the advantages and disadvantages about HOG feature and color names feature, and merge the two together to make more robust results.

We evaluate our algorithm on VOT2015 datasets and Visual Tracker Benchmark datasets, and show our results by success plot and precision plot as below.

precision plot:

![precision plot](https://raw.githubusercontent.com/zhr01/kcf2/master/read_res/pp_all.png)

Success Plot:

![success plot](https://raw.githubusercontent.com/zhr01/kcf2/master/read_res/sp_all.png)

## Installation Dependencies

* Python 3.4+
* NumPy
* Numba
* opencv

