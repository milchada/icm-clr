#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J params_opt
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=eisert@mpia.de
#SBATCH --time=23:00:00

module load anaconda/3

module load intel/19.1.3
module load impi/2019.9
module load fftw-serial
module load hdf5-serial
module load gsl

module load cuda/11.2
module load cudnn/8.1.0
module load pytorch/gpu-cuda-11.2/1.8.1
conda activate simclr

srun python -m scripts.util.params_opt