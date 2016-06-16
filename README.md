# Inverse Autoregressive Flow

Code for reproducing key results in the paper "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans and Max Welling.


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

## To reproduce best result on CIFAR-10

```sh
python train.py with problem=cifar10 n_h=160 depths=[10,10] margs.depth_ar=2 margs.posterior=down_iaf2_nl margs.prio
r=diag margs.kl_min=0.25
```

