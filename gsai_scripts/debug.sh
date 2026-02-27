#!/bin/sh
#SBATCH -J  elsa      # Job name
#SBATCH -o output/%j.out      # Name of stdout output file (%j expands to %jobId)
#SBATCH -t 1-00:00:00         # Run time (hh:mm:ss) 

#### Select  GPU
#SBATCH -p RTX3090               # partiton
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
# module load anaconda3/2022.05
# source /opt/anaconda3/2022.05/etc/profile.d/conda.sh

echo "conda command executing"

# Execute python file using conda run for robustness
# Hyperparameters are set to match the original admm_lora_cifar100_fisher.py notebook

conda run -n elsa python -u main.py \
    --sparsity_ratio=0.5 \
    --sparsity_type="unstructured" \
    --admm_steps=10 \
    --admm_batch_size=2 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lr=1e-5 \
    --admm_lmda=0.01 \
    --admm_interval=32 \
    --admm_nonuniform_sparsity=True \
    --admm_lmda=0.02 \
    --do_distill=True \
    --eval_zero_shot=True \
    --seed=0 \
    --wandb=True \
    --wandb_project="pruning_reasoning"
date

squeue --job $SLURM_JOBID

echo  "##### END #####"
