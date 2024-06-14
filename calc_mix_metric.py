import csv

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18, resnet50, resnet34
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score

import dnnlib
from condition_encodings import get_encoding, to_binary
import legacy

# All added by the authors


class EvaluationClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EvaluationClassifier, self).__init__()

        self.backbone_name = 'resnet18'
        backbone_model = resnet18(pretrained=True)
        self.backbone = []
        for name, module in backbone_model.named_children():
            if not isinstance(module, nn.Linear):
                self.backbone.append(module)
            # if name in ['layer1', 'layer2', 'layer3']:
            #     self.backbone.append(nn.Dropout())
        self.backbone = nn.Sequential(*self.backbone)
        self.projection = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.Sigmoid()
        )
        # 'resnet18': 512,
        # 'resnet34': 512,
        # 'resnet50': 2048

    def forward(self, x):
        x = self.backbone(x)
        features = torch.flatten(x, start_dim=1)
        output = self.projection(features)
        return output


class GeneratedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


def generate_dataset(G, condition_encoding, n_classes, n_samples, device, mix_strategy=None):
    batch_size = 4
    assert n_samples % batch_size == 0
    gen_z = torch.randn((n_samples, G.z_dim), device=device).split(batch_size)
    mixed_gen_c_all = condition_encoding.generate_mixed_gen_c_all(n_samples, n_classes)
    node = 0
    if mix_strategy:
        # mixed_gen_c = mixed_gen_c_all[mix_strategy]
        mixed_gen_c = list(mixed_gen_c_all.values())[0]
        mixed_gen_c = mixed_gen_c[:-n_classes]
    else:
        mixed_gen_c = list(mixed_gen_c_all.values())[0]
        mixed_gen_c = mixed_gen_c[-n_classes:]
        node = n_classes - 1
    mixed_gen_c = [gen_c.to(device).split(batch_size) for gen_c in mixed_gen_c]
    h = int(np.floor(np.log2(n_classes)))
    images = []
    labels = []
    for gen_c in mixed_gen_c:
        print(f'Generating {n_samples} samples for node {node}...')
        level = int(np.floor(np.log2(node + 1)))
        segment_len = depth_multiplier = 2 ** (h - level)
        left = (node + 1) * depth_multiplier - 1 - (n_classes - 1)
        right = left + segment_len
        _labels = torch.zeros([batch_size, n_classes])
        _labels[:, left:right] = 1
        # print(f'Node {node}')
        # print(f'level {level}')
        # print(f'segment_len {segment_len}')
        # print(f'left {left}')
        # print(f'right {right}')
        # print(f'_labels {_labels}')
        for c, z in zip(gen_c, gen_z):
            labels.append(_labels)
            images.append(G(z=z, c=c).cpu())
        node += 1
    images = torch.cat(images)
    labels = torch.cat(labels)

    return GeneratedDataset(images, labels)


def train(EC, G, condition_encoding, n_classes, ec_pkl, device):
    train_n_samples = 100
    test_n_samples = 100
    lr = 0.001
    momentum = 0.9
    epochs = 30
    batch_size = 16

    train_dataset = generate_dataset(G, condition_encoding, n_classes, train_n_samples, device)
    test_dataset = generate_dataset(G, condition_encoding, n_classes, test_n_samples, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(EC.parameters(), lr=lr, momentum=momentum)

    print(f'Training evaluation classifier')
    print(f'Hyperparameters:')
    print(f'backbone={EC.backbone_name}')
    print(f'lr={lr}')
    print(f'momentum={momentum}')
    print(f'epochs={epochs}')
    print(f'train_n_samples={train_n_samples}')
    print(f'test_n_samples={test_n_samples}')
    print(f'batch_size={batch_size}')

    for epoch in range(epochs):
        EC.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            # print(labels.shape)
            # print(labels)
            optimizer.zero_grad()
            outputs = EC(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}] Training Loss: {running_loss / len(train_loader)}')

        EC.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = EC(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() / n_classes

        print(f'Epoch [{epoch + 1}/{epochs}] Test Loss: {test_loss / len(test_loader)}')
        print(f'Epoch [{epoch + 1}/{epochs}] Test Accuracy: {correct / total}')

    print(f'Saving evaluation classifier in {ec_pkl}')
    torch.save(EC.state_dict(), ec_pkl)


def save_to_csv(result_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = result_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result_dict)


def eval_mix(EC, G, condition_encoding, n_classes, mix_strategy, metrics_csv, device):
    test_n_samples = 100
    batch_size = 100
    test_dataset = generate_dataset(G, condition_encoding, n_classes, test_n_samples, device, mix_strategy=mix_strategy)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.set_printoptions(precision=2, sci_mode=False, linewidth=100)

    EC.eval()
    accuracies = torch.zeros(n_classes - 1, device=device)
    maes = torch.zeros(n_classes - 1, device=device)
    wanted_maes = torch.zeros(n_classes - 1, device=device)
    unwanted_maes = torch.zeros(n_classes - 1, device=device)
    entropies = torch.zeros(n_classes - 1, device=device)
    precisions = torch.zeros(n_classes - 1, device=device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = EC(images)
            node = i

            predicted = torch.round(outputs)
            accuracy = torch.mean((predicted == labels).float())

            mae = torch.abs(outputs - labels)
            wanted_mae = torch.mean(mae[:, (labels[0].int() == 1).nonzero()])
            if node == 0:
                unwanted_mae = torch.tensor(0, device=device)
            else:
                unwanted_mae = torch.mean(mae[:, (labels[0].int() == 0).nonzero()])
            mae = torch.mean(mae)

            wanted_outputs = outputs[:, (labels[0].int() == 1).nonzero()]
            normalized = F.normalize(wanted_outputs, p=1, dim=1)
            entropy = -torch.sum(normalized * torch.log2(normalized + 1e-10), dim=1)
            entropy = torch.mean(entropy) / np.log2((labels[0].int() == 1).nonzero().size(0))

            precision = label_ranking_average_precision_score(labels.cpu(), outputs.cpu())

            accuracies[node] += accuracy
            maes[node] += mae
            wanted_maes[node] += wanted_mae
            unwanted_maes[node] += unwanted_mae
            entropies[node] += entropy
            precisions[node] += precision

            print(f'Label: \n{labels[0, :]}')
            print(f'Output: \n{outputs[0, :]}')
            print(f'Accuracy: {accuracy}')
            print(f'MAE: {mae}')
            print(f'Wanted MAE: {wanted_mae}')
            print(f'Unwanted MAE: {unwanted_mae}')
            print(f'Entropy: {entropy}')
            print(f'Precision: {precision}')
            print()

    weights = torch.zeros(n_classes - 1, device=device)
    curr_weight = n_classes
    curr_idx = 0
    while curr_weight > 1:
        for _ in range(0, 2 ** int(np.log2(n_classes / curr_weight))):
            weights[curr_idx] = curr_weight / n_classes
            curr_idx += 1
        curr_weight //= 2
    print(weights)

    metrics = {
        'mean_accuracy': torch.mean(accuracies).item(),
        'mean_mae': torch.mean(maes).item(),
        'mean_wanted_mae': torch.mean(wanted_maes).item(),
        'mean_unwanted_mae': torch.mean(unwanted_maes).item(),
        'mean_entropy': torch.mean(entropies).item(),
        'mean_precision': torch.mean(precisions).item(),
        'w_mean_accuracy': torch.mean(accuracies * weights).item(),
        'w_mean_mae': torch.mean(maes * weights).item(),
        'w_mean_wanted_mae': torch.mean(wanted_maes * weights).item(),
        'w_mean_unwanted_mae': torch.mean(unwanted_maes * weights).item(),
        'w_mean_entropy': torch.mean(entropies * weights).item(),
        'w_mean_precision': torch.mean(precisions * weights).item()
    }

    print(metrics)
    save_to_csv(metrics, metrics_csv)


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--condition-encoding', 'condition_encoding', help='Encoding used in conditioning', required=True)
@click.option('--mix-strategy', 'mix_strategy', help='Mix strategy', required=True)
@click.option('--n-classes', 'n_classes', help='Number of classes', required=True, type=int)
@click.option('--load-ec', 'load_ec', help='Load evaluation classifier from network_pkl directory', required=True, type=bool)
def main(ctx: click.Context,
         network_pkl: str,
         condition_encoding: str,
         mix_strategy: str,
         n_classes: int,
         load_ec: bool):
    """Train (or load) mixing evaluation classifier and evaluate using it"""

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    ec_pkl = network_pkl[:-4] + '-ec' + network_pkl[-4:]
    metrics_csv = network_pkl[:-4] + '-mix_metrics.csv'

    condition_encoding = get_encoding(condition_encoding)

    EC = EvaluationClassifier(n_classes).to(device)
    if load_ec:
        checkpoint = torch.load(ec_pkl, map_location=device)
        EC.load_state_dict(checkpoint, strict=False)
    else:
        train(EC, G, condition_encoding, n_classes, ec_pkl, device)

    eval_mix(EC, G, condition_encoding, n_classes, mix_strategy, metrics_csv, device)


if __name__ == "__main__":
    main()
