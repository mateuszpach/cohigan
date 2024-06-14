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

#echo cifar100
#echo flat
#python calc_mix_metric.py \
#--network=00039-cifar100-cond-mirror-cifar/network-snapshot-017942.pkl \
#--condition-encoding=flat \
#--mix-strategy=l1_mean \
#--n-classes=32 \
#--load-ec=False
#
#echo cifar100
#echo cube
#python calc_mix_metric.py \
#--network=00040-cifar100-cond-mirror-cifar/network-snapshot-017942.pkl \
#--condition-encoding=cube \
#--mix-strategy=linf_mean \
#--n-classes=32 \
#--load-ec=False
#
#echo imagenetdogs
#echo flat
#python calc_mix_metric.py \
#--network=00048-imagenetdogs-cond-mirror-paper256/network-snapshot-005846.pkl \
#--condition-encoding=flat \
#--mix-strategy=l1_mean \
#--n-classes=16 \
#--load-ec=False
#
#echo imagenetdogs
#echo cube
#python calc_mix_metric.py \
#--network=00049-imagenetdogs-cond-mirror-paper256/network-snapshot-005846.pkl \
#--condition-encoding=cube \
#--mix-strategy=linf_mean \
#--n-classes=16 \
#--load-ec=False
#
#echo afhq
#echo flat
#python calc_mix_metric.py \
#--network=00051-afhq-cond-mirror-paper512-gamma10/network-snapshot-001814.pkl \
#--condition-encoding=flat \
#--mix-strategy=l1_mean \
#--n-classes=8 \
#--load-ec=False
#
#echo afhq
#echo cube
#python calc_mix_metric.py \
#--network=00050-afhq-cond-mirror-paper512-gamma10/network-snapshot-001814.pkl \
#--condition-encoding=cube \
#--mix-strategy=linf_mean \
#--n-classes=8 \
#--load-ec=False
#
#echo stl10
#echo flat
#python calc_mix_metric.py \
#--network=00045-stl10-cond-mirror-cifar/network-snapshot-012096.pkl \
#--condition-encoding=flat \
#--mix-strategy=l1_mean \
#--n-classes=16 \
#--load-ec=False
#
#echo stl10
#echo cube
#python calc_mix_metric.py \
#--network=00046-stl10-cond-mirror-cifar/network-snapshot-012096.pkl \
#--condition-encoding=cube \
#--mix-strategy=linf_mean \
#--n-classes=16 \
#--load-ec=False

echo cifar10
echo flat
python calc_mix_metric.py \
--network=00058-cifar10-cond-mirror-cifar/network-snapshot-023587.pkl \
--condition-encoding=flat \
--mix-strategy=l1_mean \
--n-classes=16 \
--load-ec=False

echo cifar10
echo cube
python calc_mix_metric.py \
--network=00057-cifar10-cond-mirror-cifar/network-snapshot-023587.pkl \
--condition-encoding=cube \
--mix-strategy=linf_mean \
--n-classes=16 \
--load-ec=False
