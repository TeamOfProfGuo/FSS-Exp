#!/bin/bash

#SBATCH --job-name=fs-seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=40:00:00
#SBATCH --gres=gpu

module purge

# training config
dataset=$1
model_name=$2
exp_name=$3
exp_dir=exp/${dataset}_${model_name}_${exp_name}
save_dir=exp/${dataset}_${model_name}_${exp_name}/checkpoints
config_dir=config/${dataset}/${dataset}_${model_name}_${exp_name}.yaml
mkdir -p ${save_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp ${config_dir} ${exp_dir}

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --overlay ${ext3_path}:ro ${sif_path} \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/FSS-Exp
python train.py --config=${config_dir} > ${exp_dir}/train-${now}.log 2>&1
"