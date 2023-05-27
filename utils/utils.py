"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import os
import torch
import numpy as np
import errno
import torch.nn as nn
import sys
import pandas as pd
import faiss


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images, forward_pass='backbone_i')
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' % (i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def get_features_eval(val_loader, model):
    model.eval()
    targets, features, indices = [], [], []
    for i, batch in enumerate(val_loader):
        input_ = batch['image'].cuda()
        target_ = batch['target'].cuda()
        index_ = batch['index']
        feature_ = model(input_, forward_pass='backbone_i')

        targets.append(target_)
        features.append(feature_.cpu())
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    # Sort features and targets according to indices
    features_order, targets_order = torch.zeros_like(features), torch.zeros_like(targets)
    features_order[indices] = features
    targets_order[indices] = targets

    return features_order, targets_order



def get_wordnet_noun(filename):
    nouns_df = pd.read_csv(filename)
    nouns = np.array(nouns_df).flatten().tolist()
    return nouns



def mine_nearest_neighbors(features, targets, topk):
    # mine the topk nearest neighbors for every sample
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatL2(dim)  # index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included

    # evaluate
    neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)
    return indices, accuracy

@torch.no_grad()
def get_knn_indices(model, dataloader, topk):

    image_features, image_targets = get_features_eval(dataloader, model)
    image_indices, image_accuracy = mine_nearest_neighbors(image_features.numpy(), image_targets.cpu().numpy(), topk)

    return image_indices, image_accuracy