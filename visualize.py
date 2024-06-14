import os
from datetime import datetime
import random

import click
import numpy as np
import torch
from matplotlib import pyplot as plt
import io

import dnnlib
import tree_classifier.model
import legacy

from condition_encodings import get_encoding
from PIL import Image

from calc_mix_metric import EvaluationClassifier
from sklearn.metrics import label_ranking_average_precision_score


# All added by the authors


def visualize_trees_for_each_mix_strategy(G, gen_z, mixed_gen_c_all, truncation_psi, AC, labels, ac_classes,
                                          k=5, chosen_mix_strategy='all', EC=None, sampling_strategy='none'):
    """Generate tree visualizations for all legal mix strategies"""
    tree_visualizations = []
    for mix_strategy, nodes_gen_c in mixed_gen_c_all.items():
        if chosen_mix_strategy != 'all' and mix_strategy != chosen_mix_strategy:
            continue
        tree_img = visualize_tree(G, nodes_gen_c, gen_z, sampling_strategy, truncation_psi, AC, EC, labels,
                                  ac_classes, k)
        tree_visualizations.append((mix_strategy, tree_img))
    return tree_visualizations


def visualize_tree(G, nodes_gen_c, gen_z, sampling_strategy, truncation_psi, AC, EC, labels, ac_classes, k):
    """Generate tree visualization"""
    nodes_imgs = generate_nodes_imgs(G, nodes_gen_c, gen_z, sampling_strategy, truncation_psi, EC, k)
    nodes_ac_labels, nodes_gt_labels = None, None
    if labels:
        nodes_gt_labels = [torch.cat(gen_c).tolist() for gen_c in nodes_gen_c]
        nodes_ac_labels = classify_nodes_imgs(nodes_imgs, AC, ac_classes)
    tree_img = visualize_nodes_imgs(nodes_imgs, nodes_gt_labels, nodes_ac_labels)
    return tree_img


def visualize_nodes_imgs(nodes_imgs,
                         nodes_gt_labels,
                         nodes_ac_labels):
    """Visualize all sets of images in nodes"""

    # calculate parameters
    n_nodes = len(nodes_imgs)
    tree_height = int(np.floor(np.log2(n_nodes + 1)))
    tree_leaves = (n_nodes + 1) // 2
    k = int(round(np.sqrt(nodes_imgs[0].shape[0])))
    positions = [(j, i) for j in range(tree_height) for i in range(0, tree_leaves, tree_leaves // 2 ** j)]

    # build the plot
    fig, ax = plt.subplots(tree_height, tree_leaves, figsize=(50, 25))

    # fill the plot with images
    for node, imgs in enumerate(nodes_imgs):
        # convert to 256-bit pixels
        imgs = np.asarray(imgs.cpu(), dtype=np.float32)
        imgs = (imgs + 1) * 255 / 2
        imgs = np.rint(imgs).clip(0, 255).astype(np.uint8)

        # align in a grid
        imgs = [imgs[i, :, :, :] for i in range(k * k)]
        grid = np.concatenate([np.concatenate(imgs[j * k:j * k + k], axis=1) for j in range(k)], axis=2)
        grid = np.transpose(grid, (1, 2, 0))

        # display the grid on the plot
        ax[positions[node]].imshow(grid)
        ax[positions[node]].axis('off')

    # attach labels to the plot
    if nodes_gt_labels is not None and nodes_ac_labels is not None:
        for node, (gt_labels, ac_labels) in enumerate(zip(nodes_gt_labels, nodes_ac_labels)):
            for i in range(k * k):
                x = 1 / k * (1 + i // k)
                y = 1 - (((i % k) + 1) / k)
                label = f' gt:{gt_labels[i]} \n  ac:{ac_labels[i]}'
                ax[positions[node]].text(x, y, label, color='black', fontsize=6,
                                         verticalalignment='center', horizontalalignment='right',
                                         transform=ax[positions[node]].transAxes,
                                         bbox=dict(boxstyle='square,pad=0', facecolor='white', alpha=1,
                                                   edgecolor='none'))

    # remove empty subplots and adjust layout
    for a in ax.ravel():
        if not a.images:
            fig.delaxes(a)
        else:
            a.set_aspect('equal')

    # adjust the spaces between grids
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    img = fig2img(fig)

    return img


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def save_imgs(outdir, filename_prefix, imgs):
    """Save a list of images to files in outdir"""
    for name, img in imgs:
        filename = os.path.join(outdir, f"{filename_prefix}_{name}.png")
        img.save(filename)


def get_labels(n_nodes):
    """Generate labels of nodes to calculate the precision"""
    n_leaves = (n_nodes + 1) // 2
    labels = np.zeros((n_nodes, n_leaves), dtype=int)
    for i in range(n_leaves):
        labels[n_nodes - n_leaves + i][i] = 1
    for i in range(n_nodes - n_leaves - 1, -1, -1):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < n_nodes:
            labels[i] += labels[left_child]
        if right_child < n_nodes:
            labels[i] += labels[right_child]
    return labels


def generate_nodes_imgs(G, nodes_gen_c, gen_z, sampling_strategy, truncation_psi, EC, k):
    """Generate set of images for each node"""
    if sampling_strategy == 'none':
        nodes_imgs = [torch.cat([G(z=z,
                                   c=c,
                                   noise_mode='const',
                                   truncation_psi=truncation_psi)
                                 for z, c in zip(gen_z, gen_c)])
                      for gen_c in nodes_gen_c]
        return nodes_imgs
    elif sampling_strategy == 'pure_aligned':
        labels = get_labels(len(nodes_gen_c))
        nodes_imgs = [[] for _ in range(len(nodes_gen_c))]
        precisions = [0 for _ in range(len(gen_z))]
        for z_i, z in enumerate(gen_z):
            for node_i, gen_c in enumerate(nodes_gen_c):
                # print(f'Z: {z_i} of {len(gen_z)} Node: {node_i} of {len(nodes_gen_c)}', flush=True)
                c = gen_c[0]
                img = G(z=z,
                        c=c,
                        noise_mode='const',
                        truncation_psi=truncation_psi)
                with torch.no_grad():
                    output = EC(img)
                    # print(c)
                    # print(labels[node_i][np.newaxis, ...])
                    # print(output.cpu())
                    precision = label_ranking_average_precision_score(labels[node_i][np.newaxis, ...], output.cpu())
                    # print(precision)
                    # print('', flush=True)
                    precisions[z_i] += precision
                nodes_imgs[node_i].append(img)

        chosen_nodes_imgs = [[] for _ in range(len(nodes_gen_c))]
        for z_i in sorted(range(len(precisions)), key=lambda x: precisions[x], reverse=True):
            if len(chosen_nodes_imgs[0]) == k:
                print(f'best:{max(precisions)} last:{precisions[z_i]}')
                break
            for node_i in range(len(nodes_gen_c)):
                chosen_nodes_imgs[node_i].append(nodes_imgs[node_i][z_i])

        return [torch.cat(node_imgs) for node_imgs in chosen_nodes_imgs]
    elif sampling_strategy == 'pure_random':
        labels = get_labels(len(nodes_gen_c))
        nodes_imgs = [[] for _ in range(len(nodes_gen_c))]
        chosen_nodes_imgs = [[] for _ in range(len(nodes_gen_c))]
        for node_i, gen_c in enumerate(nodes_gen_c):
            gen_z = list(gen_z)
            random.shuffle(gen_z)
            c = gen_c[0]
            precisions = [0 for _ in range(len(gen_z))]

            for z_i, z in enumerate(gen_z):
                # print(f'Z: {z_i} of {len(gen_z)} Node: {node_i} of {len(nodes_gen_c)}', flush=True)
                img = G(z=z,
                        c=c,
                        noise_mode='const',
                        truncation_psi=truncation_psi)
                with torch.no_grad():
                    output = EC(img)
                    precision = label_ranking_average_precision_score(labels[node_i][np.newaxis, ...], output.cpu())
                    precisions[z_i] += precision

                nodes_imgs[node_i].append(img)

            for z_i in sorted(range(len(precisions)), key=lambda x: precisions[x], reverse=True):
                if len(chosen_nodes_imgs[node_i]) == k:
                    print(f'best:{max(precisions)} last:{precisions[z_i]}')
                    break
                chosen_nodes_imgs[node_i].append(nodes_imgs[node_i][z_i])

        return [torch.cat(node_imgs) for node_imgs in chosen_nodes_imgs]


def classify_imgs(imgs, AC, ac_classes):
    """Classify set of images"""
    with torch.no_grad():
        _, tree_ac_labels = AC.predict(imgs, ac_classes=ac_classes)
        return tree_ac_labels.tolist()


def classify_nodes_imgs(nodes_imgs, AC, ac_classes):
    """Classify set of images for each node"""
    nodes_labels = [classify_imgs(imgs, AC, ac_classes) for imgs in nodes_imgs]
    return nodes_labels


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--condition-encoding', 'condition_encoding', help='Encoding used in conditioning', required=True)
@click.option('--ac-pkl', 'ac_pkl', help='AC pickle filename', required=False)
@click.option('--labels', help='Whether to annotate images with labels', required=True, type=bool)
@click.option('--ac-classes', help='Number of classes in AC', type=int, metavar='INT')
@click.option('--sampling-strategy', help='Strategy for sampling images [default: none]',
              type=click.Choice(['none', 'pure_aligned', 'pure_random']), required=True)
@click.option('--mix-strategy', help='Strategy for mixing conditions [default: all]',
              type=click.Choice(['all', 'linf_mean', 'l1_mean', 'l2_mean']), required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(ctx: click.Context,
         network_pkl: str,
         truncation_psi: float,
         condition_encoding: str,
         ac_pkl: str,
         ac_classes: int,
         labels: bool,
         sampling_strategy: str,
         mix_strategy: str,
         outdir: str):
    """Generate tree visualizations from a saved network and save them"""

    # load a model
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # load AC
    AC = None
    if labels:
        if ac_pkl is None:
            ctx.fail('--ac_pkl option is required when --labels=True')
        AC = tree_classifier.model.get_model(ac_pkl, device)

    # create directory
    os.makedirs(outdir, exist_ok=True)

    # initialize condition encoding
    condition_encoding_name = condition_encoding
    condition_encoding = get_encoding(condition_encoding)

    # load EC
    ec_pkl = network_pkl[:-4] + '-ec' + network_pkl[-4:]
    EC = EvaluationClassifier(ac_classes).to(device)
    checkpoint = torch.load(ec_pkl, map_location=device)
    EC.load_state_dict(checkpoint, strict=False)
    EC.eval()

    # initialize c and z variables
    visualize_k = 3
    oversample_factor = 10 if sampling_strategy != 'none' else 1
    visualize_gen_z = torch.randn((oversample_factor * visualize_k ** 2, G.z_dim), device=device).split(1)
    visualize_mixed_gen_c_all = condition_encoding.generate_mixed_gen_c_all(oversample_factor * visualize_k ** 2,
                                                                            ac_classes)
    visualize_mixed_gen_c_all = {name: [node.to(device).split(1) for node in gen_c]
                                 for name, gen_c in visualize_mixed_gen_c_all.items()}

    print(f"{network_pkl.split('/')[0]}",
          f"{condition_encoding_name}",
          f"{sampling_strategy}",
          f"{truncation_psi}", flush=True)

    # gather tree visualizations for all mixing strategies
    tree_visualizations = visualize_trees_for_each_mix_strategy(G, visualize_gen_z, visualize_mixed_gen_c_all,
                                                                truncation_psi, AC, labels, ac_classes,
                                                                visualize_k ** 2, mix_strategy, EC, sampling_strategy)

    # save visualizations
    current_date = datetime.now().strftime("%H:%M:%S")
    path = os.path.join(outdir, f"treevis_{network_pkl.split('/')[0]}_{condition_encoding_name}")
    os.makedirs(path, exist_ok=True)
    save_imgs(path,
              f"{sampling_strategy}_{truncation_psi}_{current_date}",
              tree_visualizations)


if __name__ == "__main__":
    main()
