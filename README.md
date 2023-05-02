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

module purge && module load Miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

conda env create -f environment.yml -p ./venv
```


## Scenarios

### Batch job

Submit a Slurm job to train a CNN model on CIFAR-10 using

```
sbatch train_model.sl
```

and check that the job is running with `squeue --me`.

While the job is running, log in the node with `ssh NODE` and use `nvidia-smi` to check that the script is using the GPU.

From a Jupyter session, start a `tensorboard` instance pointing at the results folder

```
module purge && module load Miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

tensorboard --logdir=results/JOBID-train_model.sl/logs
```

and use the Jupyter Server Proxy to access it at https://jupyter.nesi.org.nz/user-redirect/proxy/6006/


### Interactive job

Alternatively, start an interactive session

```
srun --account=nesi99999 --time=10 --cpus-per-task=2 --mem=8GB \
    --gpus-per-node=A100-1g.5gb:1 --pty bash
```

then, on the compute node, load the modules and activate the Conda environment

```
module purge
module load Miniconda3/22.11.1-1 cuDNN/8.1.1.33-CUDA-11.2.0

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


### Jupyter notebook

To use the notebook, create a Jupyter kernel using the Conda environment and loading the CUDA/cuDNN toolboxes

```
module purge && module load JupyterLab
nesi-add-kernel -p ./venv -- dl_demos cuDNN/8.1.1.33-CUDA-11.2.0
```

then (re)start a Jupyter session, making sure to select a GPU, and run the [train_model.ipynb](train_model.ipynb) notebook.


## References

- [GPU usage support page](https://support.nesi.org.nz/hc/en-gb/articles/360001471955-GPU-use-on-NeSI)
- [Miniconda support page](https://support.nesi.org.nz/hc/en-gb/articles/360001580415-Miniconda3)
- [TensorFlow support page](https://support.nesi.org.nz/hc/en-gb/articles/360000990436-TensorFlow-on-GPUs)
- [Jupyter kernel tool support page](https://support.nesi.org.nz/hc/en-gb/articles/4414958674831-Jupyter-kernels-Tool-assisted-management)
- [TensorFlow compatibility matrix](https://www.tensorflow.org/install/source?hl=fr#gpu)
