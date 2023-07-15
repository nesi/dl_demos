# Deep Learning demos

This repository showcases how to run deep learning code on NeSI.


## Installation

After logging on NeSI, clone the repository:

```
git clone https://github.com/nesi/dl_demos
```

then create the Conda environment using:

```
cd dl_demos

# unload all environment models then load miniconda
module purge
module load Miniconda3/22.11.1-1

# ensure conda can activate environments
source $(conda info --base)/etc/profile.d/conda.sh

# ensure conda environments don't load user's local packages
export PYTHONNOUSERSITE=1

# create conda environment with advanced solver
conda env create -f environment.lock.yml -p ./venv \
    --solver=libmamba
```


## Scenarios

### Batch job

Submit a Slurm job to train a CNN model on CIFAR-10 (using [train_model.py](train_model.py)) as follows

```
sbatch --account=PROJECTID train_model.sl
```

where `PROJECTID` is your NeSI project identifier.

Check that the job is running with `squeue --me`.

While the job is running, log in the node with `ssh NODE` and use `nvidia-smi` to check that the script is using the GPU.

*Note: For an HGX-A100 node, use the complete name of the node, e.g. `ssh wmg001.ib.hpcf.nesi.org.nz` instead of `ssh wmg001`.*

Standard and error outputs of the script are captured in a log file in the `logs` folder.
Its content can be examined *live* using the `tail -f` command:

```
tail -f logs/JOBID-train_model.sl.out
```

where `JOBID` is the job identifier previously returned by the `sbatch` command.

From a Jupyter session, start a `tensorboard` instance pointing at the results folder

```
module purge && module load Miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ./venv

tensorboard --logdir=results/JOBID-train_model.sl/logs
```

and use the Jupyter Server Proxy to access it at https://jupyter.nesi.org.nz/user-redirect/proxy/6006/

*Note: It is also possible to run `tensorboard` on a login node and use an SSH tunnel to access it.*


### Interactive job

Alternatively, start an interactive session

```
srun --account=PROJECTID --time=10 --cpus-per-task=2 --mem=8GB \
    --gpus-per-node=A100-1g.5gb:1 --pty bash
```

then, on the compute node, load the modules and activate the Conda environment

```
module purge
module load Miniconda3/22.11.1-1 cuDNN/8.6.0.163-CUDA-11.8.0

# ensures XLA can find CUDA libraries
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH

source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda deactivate
conda activate ./venv
```

and run the script on the node

```
python train_model.py results/scratch
```

*Note: Add `--qos=debug` when using `sbatch` or `srun` to get higher priority for short-lived low resources jobs.*

*Note: To access an HGX-A100 GPU, make sure to run `srun` from a Mahuika login node, not from a Jupyter session. You can run `ssh mahuika01` from a Jupyter terminal to access a login node.*


### Jupyter notebook

To use the notebook, create a Jupyter kernel using the Conda environment and loading the CUDA/cuDNN toolboxes

```
module purge && module load JupyterLab
nesi-add-kernel -p ./venv -- dl_demos cuDNN/8.6.0.163-CUDA-11.8.0
```

then (re)start a Jupyter session, making sure to select a GPU, and run the [train_model.ipynb](train_model.ipynb) notebook.


## References

- [Slurm Reference Sheet](https://support.nesi.org.nz/hc/en-gb/articles/360000691716-Slurm-Reference-Sheet)
- [Available GPUs on NeSI](https://support.nesi.org.nz/hc/en-gb/articles/4963040656783-Available-GPUs-on-NeSI)
- [GPU usage support page](https://support.nesi.org.nz/hc/en-gb/articles/360001471955-GPU-use-on-NeSI)
- [Miniconda support page](https://support.nesi.org.nz/hc/en-gb/articles/360001580415-Miniconda3)
- [TensorFlow support page](https://support.nesi.org.nz/hc/en-gb/articles/360000990436-TensorFlow-on-GPUs)
- [Jupyter kernel tool support page](https://support.nesi.org.nz/hc/en-gb/articles/4414958674831-Jupyter-kernels-Tool-assisted-management)
- [TensorFlow compatibility matrix](https://www.tensorflow.org/install/source?hl=fr#gpu)
