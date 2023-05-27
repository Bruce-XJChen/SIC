import torch
import numpy as np
import time
from utils.utils import AverageMeter, ProgressMeter
from utils.common_config import run_scheduler


def sic_train(p, args, train_loader, text_loader, image_list,
                 model, image_optimizer, criterion,
                 cpt_center, epoch, update_cluster_head_only=False):
    """
    Train w/ SICLoss
    """

    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('image consistency loss', ':.4e')
    entropy_losses = AverageMeter('entropy loss', ':.4e')
    ce_losses = AverageMeter('image-semantic consistency loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, consistency_losses, entropy_losses, ce_losses],
                             prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    # Adjusted center-based mapping
    # Get image_centers \mathcal{V} from top xi_c confident samples, generate c representative semantic centers \mathcal{H}
    image_centers, image_features = image_list
    image_centers = cpt_center.get_centers(image_features, model, args)
    text_centers  = cpt_center.search_sim_texts(args, image_centers, text_loader, model)


    for i, batch in enumerate(train_loader):
        # Forward pass
        indices = batch['index'].cuda(non_blocking=True)
        n_indices = batch['n_index'].cuda(non_blocking=True)
        anchor_features = image_features[indices].cuda(non_blocking=True)
        neighbor_features = image_features[n_indices].cuda(non_blocking=True)

        # Network output
        if update_cluster_head_only:  # Only calculate gradient for backprop of cluster head
            anchor_outputs = model(anchor_features, forward_pass='head_i')
            neighbor_outputs = model(neighbor_features, forward_pass='head_i')


        # Loss for every head_i
        total_loss, consistency_loss, entropy_loss, ce_loss = [], [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchor_outputs, neighbor_outputs):
            total_loss_, consistency_loss_, entropy_loss_, ce_loss_ = criterion(anchors_output_subhead,
                                                                                     neighbors_output_subhead, anchor_features, text_centers, epoch)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)
            ce_loss.append(ce_loss_)


        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))
        ce_losses.update(np.mean([v.item() for v in ce_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        image_optimizer.zero_grad()
        total_loss.backward()
        image_optimizer.step()

        run_scheduler(p, epoch, image_optimizer, len(train_loader), i)

        if i % 40 == 0:
            progress.display(i)
















