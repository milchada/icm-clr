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
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=eisert@mpia.de
#SBATCH --time=23:00:00

module load anaconda/3
module load cuda/11.4   
module load cudnn/8.2.4
module load pytorch/gpu-cuda-11.4/1.11.0  
conda activate ergo

srun python -m scripts.model.params_opt
