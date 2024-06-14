# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""
import glob
import os

import torch
import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal

from condition_encodings import to_binary
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_mix_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    if opts.rank != 0:
        return float('nan')

#    n_real_samples = []
#    for subdir in os.listdir(opts.dataset_kwargs.path):
#        subdirectory_path = os.path.join(opts.dataset_kwargs.path, subdir)
#        if os.path.isdir(subdirectory_path):
#            files_in_subdirectory = glob.glob(os.path.join(subdirectory_path, '*'))
#            n_real_samples.append(len(files_in_subdirectory))
#    max_real = min(n_real_samples)
#    print(n_real_samples, max_real)

    h = round(np.log2(opts.n_classes))
    n = 2 * opts.n_classes - 1
    print(h, n)

    # Compute parameters for all nodes (generated).
    all_mu_gen, all_sigma_gen, all_features_gen = [None] * n, [None] * n, [None] * n
    for node in range(n):
        gen_c = to_binary(torch.tensor(node), h) * 2 - 1
        print(node, gen_c)

        stats = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, capture_all=True, max_items=10000, gen_c=gen_c)
        # 50000

        features_gen = stats.get_all()
        mu_gen, sigma_gen = stats.get_mean_cov()

        all_features_gen[node] = features_gen
        all_mu_gen[node] = mu_gen
        all_sigma_gen[node] = sigma_gen

    all_mu_wanted, all_sigma_wanted = [None] * n, [None] * n
    for node in range(n - opts.n_classes, n):
        all_mu_wanted[node] = all_mu_gen[node]
        all_sigma_wanted[node] = all_sigma_gen[node]
        #all_sigma_wanted[node] += 0.000001 * np.eye(all_sigma_wanted[node].shape[0])

    for node in range(n - opts.n_classes - 1, -1, -1):
        l = 2 * node + 1
        r = 2 * node + 2
        #all_mu_wanted[node] = (all_mu_wanted[l] + all_mu_wanted[r]) / 2
        #all_sigma_wanted[node] = (all_sigma_wanted[l] +
        #                        all_mu_wanted[l].T @ all_mu_wanted[l] +
        #                        all_sigma_wanted[r]  +
        #                        all_mu_wanted[r].T @ all_mu_wanted[r]) / 2 - all_mu_wanted[node].T @ all_mu_wanted[node]
        #all_sigma_wanted[node] += 0.1 * np.eye(all_sigma_wanted[node].shape[0])

        all_mu_wanted[node] = (all_mu_gen[l] + all_mu_gen[r]) / 2
        all_sigma_wanted[node] = (all_sigma_gen[l] +
                                all_mu_gen[l].T @ all_mu_gen[l] +
                                all_sigma_gen[r]  +
                                all_mu_gen[r].T @ all_mu_gen[r]) / 2 - all_mu_wanted[node].T @ all_mu_wanted[node]
        #all_sigma_wanted[node] += 0.000001 * np.eye(all_sigma_wanted[node].shape[0])


    all_features_left = [None] * n
    for node in range(n - opts.n_classes, n):
        all_features_left[node] = all_features_gen[node]

    for node in range(n - opts.n_classes - 1, -1, -1):
        l = 2 * node + 1
        r = 2 * node + 2
        all_features_left[node] = all_features_gen[l]

    all_features_right = [None] * n
    for node in range(n - opts.n_classes, n):
        all_features_right[node] = all_features_gen[node]

    for node in range(n - opts.n_classes - 1, -1, -1):
        l = 2 * node + 1
        r = 2 * node + 2
        all_features_right[node] = all_features_gen[r]

    all_features_fixed = [None] * n
    for node in range(0, n):
        all_features_fixed[node] = all_features_gen[n - 1]

#    # Compute parameters for leaves (real).
#    all_mu_real, all_sigma_real = [None] * n, [None] * n
#    for node in range(n - opts.n_classes, n):
#        label = node - opts.n_classes + 1
#        print(label)
#        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
#            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
#            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, label=label).get_mean_cov()
#
#        all_mu_real[node] = mu_real
#        all_sigma_real[node] = sigma_real

#    # Compute parameters for other nodes (real)
#    for node in range(n - opts.n_classes - 1, -1, -1):
#        l = 2 * node + 1
#        r = 2 * node + 2
#        all_mu_real[node] = (all_mu_real[l] + all_mu_real[r]) / 2
#        all_sigma_real[node] = (all_sigma_real[l] ** 2 +
#                                all_mu_real[l] ** 2 +
#                                all_sigma_real[r] ** 2 +
#                                all_mu_real[r] ** 2) / 2 - all_mu_real[node]

#    # Compute parameters (leaves avg)
#    all_mu_real_leaves_avg, all_sigma_real_leaves_avg = [None] * n, [None] * n
#    for node in range(n - opts.n_classes, n):
#        all_mu_real_leaves_avg[node] = all_mu_real[node]
#        all_sigma_real_leaves_avg[node] = all_sigma_real[node]
#
#    for node in range(n - opts.n_classes - 1, -1, -1):
#        l = 2 * node + 1
#        r = 2 * node + 2
#        all_mu_real_leaves_avg[node] = (all_mu_real_leaves_avg[l] + all_mu_real_leaves_avg[r]) / 2
#        all_sigma_real_leaves_avg[node] = (all_sigma_real_leaves_avg[l] + all_sigma_real_leaves_avg[r]) / 2

#    # Compute parameters (children avg)
#    all_mu_real_children_avg, all_sigma_real_children_avg = [None] * n, [None] * n
#    for node in range(n - opts.n_classes, n):
#        all_mu_real_children_avg[node] = all_mu_real[node]
#        all_sigma_real_children_avg[node] = all_sigma_real[node]
#
#    for node in range(n - opts.n_classes - 1, -1, -1):
#        l = 2 * node + 1
#        r = 2 * node + 2
#        all_mu_real_children_avg[node] = (all_mu_real[l] + all_mu_real) / 2
#        all_sigma_real_children_avg[node] = (all_sigma_real[l] + all_sigma_real[r]) / 2

#    # Compute distances for all nodes and return mean (print others)
#    print('leaves_avg')
#    print('node \t fid')
#    fids_leaves_avg = []
#    for i, (mu_gen, sigma_gen, mu_real, sigma_real) in enumerate(zip(all_mu_gen,
#                                                                     all_sigma_gen,
#                                                                     all_mu_real_leaves_avg,
#                                                                     all_sigma_real_leaves_avg)):
#        m = np.square(mu_gen - mu_real).sum()
#        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#        print(f'{i} \t {fid}')
#        fids_leaves_avg.append(fid)
#    print(fids_leaves_avg)
#
#    print('children_avg')
#    print('node \t fid')
#    fids_children_avg = []
#    for i, (mu_gen, sigma_gen, mu_real, sigma_real) in enumerate(zip(all_mu_gen,
#                                                                     all_sigma_gen,
#                                                                     all_mu_real_children_avg,
#                                                                     all_sigma_real_children_avg)):
#        m = np.square(mu_gen - mu_real).sum()
#        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#        print(f'{i} \t {fid}')
#        fids_children_avg.append(fid)
#    print(fids_children_avg)
#
#    print()
#    print('REAL BASELINE')
#    print('leaves_avg')
#    print('node \t fid')
#    fids_leaves_avg_base = []
#    for i, (mu_gen, sigma_gen, mu_real, sigma_real) in enumerate(zip(all_mu_real,
#                                                                     all_sigma_real,
#                                                                     all_mu_real_leaves_avg,
#                                                                     all_sigma_real_leaves_avg)):
#        m = np.square(mu_gen - mu_real).sum()
#        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#        print(f'{i} \t {fid}')
#        fids_leaves_avg_base.append(fid)
#    print(fids_leaves_avg_base)
#
#    print('children_avg')
#    print('node \t fid')
#    fids_children_avg_base = []
#    for i, (mu_gen, sigma_gen, mu_real, sigma_real) in enumerate(zip(all_mu_real,
#                                                                     all_sigma_real,
#                                                                     all_mu_real_children_avg,
#                                                                     all_sigma_real_children_avg)):
#        m = np.square(mu_gen - mu_real).sum()
#        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#        print(f'{i} \t {fid}')
#        fids_children_avg_base.append(fid)
#    print(fids_children_avg_base)

####################

#    fids = []
#    for i, (mu_gen, sigma_gen, mu_real, sigma_real) in enumerate(zip(all_mu_gen,
#                                                                     all_sigma_gen,
#                                                                     all_mu_wanted,
#                                                                     all_sigma_wanted)):
#        m = np.square(mu_gen - mu_real).sum()
#        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#        print(f'{i} \t {fid}')
#        fids.append(fid)
#    print(fids)

    print('SCORES')
    likelihoods = []
    for i, (mu_wanted, sigma_wanted, features_gen) in enumerate(zip(all_mu_wanted, 
                                                                    all_sigma_wanted, 
                                                                    all_features_gen)):

        mv_normal = multivariate_normal(mean=mu_wanted, cov=sigma_wanted)
        likelihood = np.mean(mv_normal.logpdf(features_gen))
        print(f'{i} \t {likelihood}')
        likelihoods.append(likelihood)

    print('Likelihood of the samples:', sum(likelihoods)/len(likelihoods))

    print('LEFT')
    for i, (mu_wanted, sigma_wanted, features_gen) in enumerate(zip(all_mu_wanted, 
                                                                    all_sigma_wanted, 
                                                                    all_features_left)):

        mv_normal = multivariate_normal(mean=mu_wanted, cov=sigma_wanted)
        likelihood = np.mean(mv_normal.logpdf(features_gen))
        print(f'{i} \t {likelihood}')

    print('RIGHT')
    for i, (mu_wanted, sigma_wanted, features_gen) in enumerate(zip(all_mu_wanted,
                                                                    all_sigma_wanted,
                                                                    all_features_right)):

        mv_normal = multivariate_normal(mean=mu_wanted, cov=sigma_wanted)
        likelihood = np.mean(mv_normal.logpdf(features_gen))
        print(f'{i} \t {likelihood}')

    print('FIXED')
    for i, (mu_wanted, sigma_wanted, features_gen) in enumerate(zip(all_mu_wanted,
                                                                    all_sigma_wanted,
                                                                    all_features_fixed)):

        mv_normal = multivariate_normal(mean=mu_wanted, cov=sigma_wanted)
        likelihood = np.mean(mv_normal.logpdf(features_gen))
        print(f'{i} \t {likelihood}')



    return float(sum(likelihoods) / len(likelihoods))

#----------------------------------------------------------------i------------
