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


## Multi-GPU TensorFlow implementation

### Prerequisites

Make sure that recent versions installed of:
- Python (version 2.7 or higher)
- TensorFlow
- tqdm
   
`CIFAR10_PATH` environment variable should point to the dataset location.

### Syntax of tf_train.py

Training script:
```sh
python tf_train.py --logdir <logdir> --hpconfig depth=1,num_blocks=20,kl_min=0.1,learning_rate=0.002,batch_size=32 --num_gpus 8 --mode train
```

It will run the training procedure on a given number of GPUs. Model checkpoints will be stored in `<logdir>/train` directory along with TensorBoard summaries that are useful for monitoring and debugging issues.

Evaluation script:
```sh
python tf_train.py --logdir <logdir> --hpconfig depth=1,num_blocks=20,kl_min=0.1,learning_rate=0.002,batch_size=32 --num_gpus 1 --mode eval_test
```

It will run the evaluation on the test set using a single GPU and will produce TensorBoard summary with the results and generated samples.

To start TensorBoard:
```sh
tensorboard --logdir <logdir>
```

For the description of hyper-parameters, take a look at `get_default_hparams` function in `tf_train.py`.


### Loading from the checkpoint

The best IAF model trained on CIFAR-10 reached 3.15 bits/dim when evaluated with a single sample. With 1,0000 samples, the estimation of the log likelihood is 3.111 bits/dim.
The checkpoint is available at [link](https://drive.google.com/file/d/0B-pv8mYT4p0OOXFfWElyeUs0bUk/view?usp=sharing).
Steps to use it:
- download the file
- create directory `<logdir>/train/` and copy the checkpoint there
- run the following command:
```sh
python tf_train.py --logdir <logdir> --hpconfig depth=1,num_blocks=20,kl_min=0.1,learning_rate=0.002,batch_size=32 --num_gpus 1 --mode eval_test
```

The script will run the evaluation on the test set and generate samples stored in the events file that can be accessed using TensorBoard.