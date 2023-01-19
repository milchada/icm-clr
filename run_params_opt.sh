#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ergo_params_opt
#
#SBATCH --partition=p.gpu
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=eisert@mpia.de
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2021.11
module load cuda/11.6
module load cudnn/8.4.1
module load pytorch/gpu-cuda-11.6/1.13.0
conda activate ergo

ssh -nNTf -L /u/leisert/mysql/run/mysqld/mysqld1.sock:/u/leisert/mysql/run/mysqld/mysqld.sock leisert@vera01
srun python -m scripts.model.params_opt
rm /u/leisert/mysql/run/mysqld/mysqld1.sock
