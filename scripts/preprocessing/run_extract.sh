#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/leisert/simclr/temp/job.out.%j
#SBATCH -e /u/leisert/simclr/temp/job.err.%j
# Initial working directory:
#SBATCH -D /u/leisert/simclr
# Job name
#SBATCH -J extract
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=eisert@mpia.de
#SBATCH --time=23:00:00

module load anaconda/3
module load hdf5-serial
conda activate simclr

srun python -m scripts.preprocessing.extract