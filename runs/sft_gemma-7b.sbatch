#!/bin/sh
#SBATCH --job-name=sft_gemma-7b
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
torchrun --nproc_per_node=1 --nnode=1 --node_rank=0 sft.py --model gemma-7b --per_device_train_batch_size 2 --learning_rate 1e-6 --gradient_checkpointing --deepspeed
