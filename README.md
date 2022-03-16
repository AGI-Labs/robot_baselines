# Robot Baselines
Impelmentation for baseline agents used in the [RB2 benchmark](https://rb2.info/). Specifically, this repo contains tuned training code for various versions of behavior cloning (e.g. Closed Loop vs Open Loop vs LSTM) and an implementation of [NDPs](https://shikharbahl.github.io/neural-dynamic-policies/), a representation learning method for predicting smooth trajectory distributions.

If you find this codebase useful please consider citing our paper
```
@inproceedings{dasari2021rb2,
    title={RB2: Robotic Manipulation Benchmarking with a Twist},
    author={Sudeep Dasari and Jianren Wang and Joyce Hong and Shikhar Bahl and Yixin Lin and Austin Wang and Abitha Thankaraj and Karanbir Chahal and Berk Calli and Saurabh Gupta and David Held and Lerrel Pinto and Deepak Pathak and Vikash Kumar and Abhinav Gupta},
    year={2021},
    eprint={2203.08098},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    booktitle={NeurIPS 2021 Datasets and Benchmarks Track}
}
```

## Installation
This python packages uses [PyTorch](https://pytorch.org/) for GPU accelerated machine learning. Please refer to that guide for workstation setup and CUDA installation tips. 

The easiest way to install this package is with [Anaconda](https://www.anaconda.com/). Simply run `conda env create -f environment.yml` to install dependencies, and install the package itself with `python setup.py`.

## Usage
We assume access to the following data:
* `pretrain.npz` - dataset for the pretext localization task (e.g. train CNN to predict location form image) 
* `expert.npz` - expert trajectories of robot solving tasks in RB2. Should contain RGB images, proprioceptive (state) information, and actions.

For a more thorough discussion of how collect to this data, please refer to the RB2 [website](https://rb2.info/) and paper.

Once the data is collected, use `pretrain.py` to obtain pretrained representations. This step improves performance versus starting with a randomly initialized visual model. Then use `train_bc.py` and `train_ndp.py` to train behavior cloning and NDP policies respectively.
