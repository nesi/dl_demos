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
sbatch --qos=debug train_model.sl
```

TODO nvidia-smi

TODO tensorboard

```
srun --account=nesi99999 --time=10 --cpus-per-task=2 --mem=8GB \
    --gpus-per-node=A100-1g.5gb:1 --pty bash
```

--qos=debug

TODO notebook


## References

