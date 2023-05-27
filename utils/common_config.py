"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
from torch.nn.functional import normalize
import clip
from PIL import Image
from torch.utils import data
from utils.semantic_filters import image_centers_filter, uniqueness_filter
from utils.utils import get_wordnet_noun

def get_criterion(p):
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'clustering':
        from losses.losses import SCANLoss
        criterion = SCANLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'confidence-cross-entropy':
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'ViT-B/32':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone, preprocess = clip.load("models/clip/ViT-B-32.pt", device=device)  
                
        """
        -- download link for clip pretrained model --
        “RN50”:“https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt”, 
        “RN101”:“https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt”, 
        “RN50x4”:“https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt”, 
        “ViT-B/32”:“https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt”
        """

    elif p['backbone'] == 'RN50':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone, preprocess = clip.load("RN50x16", device=device)

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # # Setup
    from models.models import ClusteringModel
    model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    return model, preprocess



def get_train_dataset(p, transform, to_augmented_dataset=False,
                        to_neighbors_dataset=False, split=None):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split=split, transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)

    elif p['train_db_name'] in ['imagenet_10', 'imagenet_dog', 'imagenet_tiny', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['top{}_neighbors_train_path'.format(p['num_neighbors'])])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])

    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)

    elif p['val_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)

    elif p['val_db_name'] in ['imagenet_10', 'imagenet_dog', 'imagenet_tiny', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)



def construct_semantic_space(p, image_centers, model, args):

    # Get wordnet noun set
    filename = os.path.join(os.getcwd(), 'data/noun.csv')
    nouns = get_wordnet_noun(filename)
    nouns_num = len(nouns)   # m

    # Semantic dataset \mathcal{T}
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in nouns])

    # Dataloader
    text_targets = torch.zeros(len(text_inputs))
    text_indices = torch.arange(len(text_inputs))
    text_dataset = data.TensorDataset(text_inputs, text_targets, text_indices)
    text_dataset.filename = 'text'
    text_dataloader = get_val_dataloader(p, text_dataset)

    # Construct semantic space
    target_list1 = image_centers_filter(model, text_dataloader, image_centers, args.gamma_r)  # image centers
    target_list2 = uniqueness_filter(model, text_dataloader, 1-args.gamma_u)   # uniqueness
    target_list = torch.from_numpy(np.intersect1d(target_list1.cpu().numpy(), target_list2.cpu().numpy())).cuda()

    text_class_ids = torch.arange(len(target_list))      # ids
    text_indices = torch.arange(nouns_num)[target_list]  # indices
    text_dataset = data.TensorDataset(text_inputs[target_list], text_class_ids, text_indices)
    text_dataset.filename = 'text'
    text_dataloader = get_val_dataloader(p, text_dataset)

    return text_dataloader


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_optimizer(p, model, cluster_head_only=False, prompt_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if 'head_i' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert (len(params) == 2 * p['num_heads'])
    elif prompt_only:
        for name, param in model.named_parameters():
            if 'ctx' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        for name, param in model.named_parameters():
            if 'head_i' in name or 'ctx' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def run_scheduler(p, epoch, image_optimizer, steps, e_step):

    lr = p['optimizer_image']['optimizer_kwargs']['lr']
    # print("learning rate is ",lr)
    if p['optimizer_image']['scheduler'] == 'constant':
        lr = lr
    elif p['optimizer_image']['scheduler'] == 'cosine':   # the type of cosine may need to be changed
        eta_min = lr * (p['lr_decay_rate'] ** 3)
        e_steps = (epoch - 1) * steps + e_step
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * e_steps / (p['epochs'] * steps))) / 2

    for param_group in image_optimizer.param_groups:
        param_group['lr'] = lr
