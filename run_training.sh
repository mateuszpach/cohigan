#!/bin/bash

#SBATCH --account=plgiris-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --job-name=hierarchical-gan
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2800

# set mem to gpus-per-node*64GB
# if training fails, try reducing number of gpus

cd /net/pr2/projects/plgrid/plggicv/mateuszpach/stylegan2-ada-pytorch
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate gan

# Datasets, classes, and configurations
#datasets=("imagenetdogs" "imagenet10" "stl10" "cifar10" "cifar100" "afhq")
#ac_classes=(16 16 16 16 32 8)
#cfgs=("paper256" "paper256" "cifar" "cifar" "cifar" "paper512")
#condition_encodings=("positive-cube" "cube" "flat")

# Iterate over datasets
for i in "${!datasets[@]}"; do
  dataset=${datasets[$i]}
  ac_class=${ac_classes[$i]}
  cfg=${cfgs[$i]}
  gamma_flag=""

  # Special setting for afhq
  if [ "$dataset" == "afhq" ]; then
    gamma_flag="--gamma=10.0"
  fi

  # Iterate over condition encodings
  for encoding in "${condition_encodings[@]}"; do
    echo "Running training for dataset: $dataset with condition encoding: $encoding"
    python train.py --outdir=. \
      --data=datasets/${dataset}.zip \
      --cond=1 \
      --gpus=4 \
      --cfg=${cfg} --mirror=1 \
      --metrics=fid50k_full \
      --ac-pkl=${dataset} \
      --ac-mode=pre-labeling \
      --condition-encoding=${encoding} \
      --ac-classes=${ac_class} \
      ${gamma_flag}
  done
done

echo "All trainings completed!"


