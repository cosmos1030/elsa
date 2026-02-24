#!/bin/sh
#SBATCH -J  elsa      # Job name
#SBATCH -o output/%j.out      # Name of stdout output file (%j expands to %jobId)
#SBATCH -t 3-00:00:00         # Run time (hh:mm:ss) 

#### Select  GPU
#SBATCH -p A100               # partiton
#SBATCH   --nodes=1           # number of nodes
#SBATCH   --ntasks=1           # number of tasks
#SBATCH   --ntasks-per-node=1
#SBATCH --cpus-per-task=8     # number of cpus

# >>> Number of GPUs <<< #df
#SBATCH  --gres=gpu:1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## Load modules
module load anaconda3/2022.05
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh

echo "conda command executing"

# Execute python file using conda run for robustness
# Hyperparameters are set to match the original admm_lora_cifar100_fisher.py notebook

# 타임아웃을 2시간(7200초)으로 연장
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon

conda run -n elsa python main.py \
    --sparsity_ratio=0.5 \
    --sparsity_type="unstructured" \
    --admm_steps=4096 \
    --admm_batch_size=2 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lr=1e-5 \
    --admm_lmda=0.01 \
    --admm_interval=32 \
    --eval_zero_shot=True \
    --seed=0 \
    --wandb=True \
    --wandb_project="pruning_reasoning"
date

squeue --job $SLURM_JOBID

echo  "##### END #####"
