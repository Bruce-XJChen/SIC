"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp, topk, checkpoint='checkpoint.pth.tar'):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    cfg['num_neighbors'] = topk
    cfg['backbone'] = 'ViT-B/32'

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    knn_dir = os.path.join(base_dir, 'knn')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(knn_dir)

    # knn indices path
    cfg['top{}_neighbors_train_path'.format(cfg['num_neighbors'])] = os.path.join(knn_dir,
                                                    'top{}-train-neighbors.npy'.format(cfg['num_neighbors']))
    cfg['topk_neighbors_val_path'] = os.path.join(knn_dir, 'topk-val-neighbors.npy')

    # clustering dir
    clustering_dir = os.path.join(base_dir, 'clustering')
    mkdir_if_missing(clustering_dir)
    cfg['clustering_dir'] = clustering_dir
    
    # checkpoint path
    cfg['clustering_checkpoint'] = os.path.join(clustering_dir, checkpoint)
    


    return cfg 
