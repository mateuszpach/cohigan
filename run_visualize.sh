#!/bin/bash

#SBATCH --account=plgiris-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hierarchical-gan
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2800

cd /net/pr2/projects/plgrid/plggicv/mateuszpach/stylegan2-ada-pytorch
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate gan

#network_params=(
#  "00058-cifar10-cond-mirror-cifar/network-snapshot-023587.pkl flat l1_mean cifar10 16"
#  "00039-cifar100-cond-mirror-cifar/network-snapshot-017942.pkl flat l1_mean cifar100 32"
#  "00048-imagenetdogs-cond-mirror-paper256/network-snapshot-005846.pkl flat l1_mean imagenetdogs 16"
#  "00045-stl10-cond-mirror-cifar/network-snapshot-012096.pkl flat l1_mean stl10 16"
#  "00051-afhq-cond-mirror-paper512-gamma10/network-snapshot-001814.pkl flat l1_mean afhq 8"
#)

network_params=(
  "00057-cifar10-cond-mirror-cifar/network-snapshot-023587.pkl cube linf_mean cifar10 16"
  "00040-cifar100-cond-mirror-cifar/network-snapshot-017942.pkl cube linf_mean cifar100 32"
  "00049-imagenetdogs-cond-mirror-paper256/network-snapshot-005846.pkl cube linf_mean imagenetdogs 16"
  "00046-stl10-cond-mirror-cifar/network-snapshot-012096.pkl cube linf_mean stl10 16"
  "00050-afhq-cond-mirror-paper512-gamma10/network-snapshot-001814.pkl cube linf_mean afhq 8"
)

sampling_strategies=(
  "pure_aligned"
  "pure_random"
  "none"
)

trunc_values=(
  1.0
)

# Iterate over each combination of network, sampling strategy, and truncation value
for params in "${network_params[@]}"; do
  IFS=' ' read -r network condition_encoding mix_strategy ac_pkl ac_class <<< "$params"

  for sampling_strategy in "${sampling_strategies[@]}"; do
    for trunc in "${trunc_values[@]}"; do
      python visualize.py \
        --network=$network \
        --trunc=$trunc \
        --condition-encoding=$condition_encoding \
        --ac-pkl=$ac_pkl \
        --labels=False \
        --ac-classes=$ac_class \
        --sampling-strategy=$sampling_strategy \
        --mix-strategy=$mix_strategy \
        --outdir=.
    done
  done
done
