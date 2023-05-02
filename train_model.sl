#!/usr/bin/env bash
#SBATCH --account=nesi99999
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gpus-per-node=A100-1g.5gb:1
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

# load required environment modules
module purge
module load Miniconda3/22.11.1-1 cuDNN/8.1.1.33-CUDA-11.2.0

# activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda deactivate
conda activate ./venv

# create a folder for results
RESULTS_FOLDER="results/${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
mkdir -p "$RESULTS_FOLDER"

# execute the script
python train_model.py "$RESULTS_FOLDER"
