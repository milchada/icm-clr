#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./temp/job.out.%j
#SBATCH -e ./temp/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J ergo_params_opt
#
#SBATCH --partition=p.gpu
#SBATCH --ntasks=11
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

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2021.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2021.11/etc/profile.d/conda.sh" ]; then
        . "/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/mpcdf/soft/SLE_15/packages/x86_64/anaconda/3/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ergo

srun nvidia-smi
srun python -m scripts.model.params_opt

