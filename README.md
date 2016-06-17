# Improve Variational Inference with Inverse Autoregressive Flow

Code for reproducing key results in the paper [Improving Variational Inference with Inverse Autoregressive Flow](http://arxiv.org/abs/1606.04934) by Diederik P. Kingma, Tim Salimans and Max Welling.

## Prerequisites

1. Make sure that recent versions installed of:
    - Python (version 2.7 or higher)
    - Numpy (e.g. `pip install numpy`)
    - Theano (e.g. `pip install Theano`)

2. Set `floatX = float32` in the `[global]` section of Theano config (usually `~/.theanorc`). Alternatively you could prepend `THEANO_FLAGS=floatX=float32 ` to the python commands below. 

3. Clone this repository, e.g.:
```sh
git clone https://github.com/openai/iaf.git
```

4. Download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (get the *Python* version) and create an environment variable `CIFAR10_PATH` that points to the subdirectory with CIFAR-10 data. For example:
```sh
export CIFAR10_PATH="$HOME/cifar-10"
```

## Syntax of train.py

Example:
```sh
python train.py with problem=cifar10 n_z=32 n_h=64 depths=[2,2,2] margs.depth_ar=1 margs.posterior=down_iaf2_NL margs.kl_min=0.25
```

`problem` is the problem (dataset) to train on. I only tested `cifar10` for this release.

`n_z` is the number of stochastic featuremaps in each layer.

`n_h` is the number of deterministic featuremaps used throughout the model.

`depths` is an array of integers that denotes the depths of the *levels* in the model. Each level is a sequence of layers. Each subsequent level operates over spatially smaller featuremaps. In case of CIFAR-10, the first level operates over 16x16 featuremaps, the second over 8x8 featuremaps, etc.

Some possible choices for `margs.posterior` are:
- `up_diag`: bottom-up factorized Gaussian
- `up_iaf1_nl`: bottom-up IAF, mean-only perturbation
- `up_iaf2_nl`: bottom-up IAF
- `down_diag`: top-down factorized Gaussian
- `down_iaf1_nl`: top-down IAF, mean-only perturbation
- `down_iaf2_nl`: top-down IAF

`margs.depth_ar` is the number of hidden layers within IAF, and can be any non-negative integer.

`margs.kl_min`: the minimum information constraint. Should be a non-negative float (where 0 is no constraint).

## Results of Table 3

(3.28 bits/dim)

```sh
python train.py with problem=cifar10 n_h=160 depths=[10,10] margs.depth_ar=2 margs.posterior=down_iaf2_nl margs.prior=diag margs.kl_min=0.25
```

More instructions will follow.
