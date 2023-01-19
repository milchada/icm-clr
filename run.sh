#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./temp/job.out.%j
#SBATCH -e ./temp/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ergo
#
#SBATCH --partition=p.gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=eisert@mpia.de
#SBATCH --time=2:00:00

module purge
module load anaconda/3/2021.11
module load cuda/11.6
module load cudnn/8.4.1
module load pytorch/gpu-cuda-11.6/1.13.0
conda activate ergo

srun dvc repro
