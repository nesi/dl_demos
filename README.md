# Deep Learning demos

This repository showcases how to run deep learning code on NeSI.


## Installation

```
module purge && module load Miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
```

```
conda env create -f environment.yml -p ./venv
```


## Scenarios

```
sbatch train_model.sl
```

TODO nvidia-smi

TODO tensorboard

```
tensorboard --logdir=results/35044479-train_model.sl/logs
```

https://jupyter.nesi.org.nz/user-redirect/proxy/6006/

```
srun --account=nesi99999 --time=10 --cpus-per-task=2 --mem=8GB \
    --gpus-per-node=A100-1g.5gb:1 --pty bash
```

--qos=debug

TODO notebook


## References

- [GPU usage support page](https://support.nesi.org.nz/hc/en-gb/articles/360001471955-GPU-use-on-NeSI)
- [Miniconda support page](https://support.nesi.org.nz/hc/en-gb/articles/360001580415-Miniconda3)
- [TensorFlow support page](https://support.nesi.org.nz/hc/en-gb/articles/360000990436-TensorFlow-on-GPUs)
- [TensorFlow compatibility matrix](https://www.tensorflow.org/install/source?hl=fr#gpu)
