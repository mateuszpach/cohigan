#!/bin/bash
#SBATCH --job-name=hierarchical-gan
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --partition=dgx

cd $HOME/stylegan2-ada-pytorch
source activate tcgan

#python calc_metrics.py \
#--network=00044-afhq-cond-mirror-paper512-gamma10/network-snapshot-000000.pkl \
#--data=datasets/afhq.zip \
#--verbose=True \
#--condition-encoding=cube \
#--n-classes=8 \

#python calc_metrics.py \
#--network=00006-afhq-cond-mirror-paper512-gamma10/network-snapshot-001008.pkl \
#--data=datasets/afhq_labeled \
#--verbose=True \
#--condition-encoding=cube \
#--n-classes=8 \
#--metrics=mix_fid50k_full

python calc_metrics.py \
--network=00013-afhq-cond-mirror-paper512-gamma10/network-snapshot-001411.pkl \
--data=datasets/afhq_labeled \
--verbose=True \
--condition-encoding=cube \
--n-classes=8 \
--metrics=mix_fid50k_full

# cube-positive
# 00014-afhq-cond-mirror-paper512-gamma10/network-snapshot-001008.pkl

# ac-cube
# 00013-afhq-cond-mirror-paper512-gamma10/network-snapshot-001411.pkl

# flat
# 00011-afhq-cond-mirror-paper512-gamma10/network-snapshot-001008.pkl

# cube
# 00006-afhq-cond-mirror-paper512-gamma10/network-snapshot-001008.pkl
