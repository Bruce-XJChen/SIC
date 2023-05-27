"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


EPS = 1e-8

def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SICLoss(nn.Module):
    def __init__(self, args, num_classes):
        super(SICLoss, self).__init__()
        self.entropy_weight = args.entropy_weight
        self.ce_weight = args.ce_weight
        self.class_num = num_classes
        self.CE = nn.CrossEntropyLoss(reduction="mean")


    def forward(self, image_output, image_nb_output, image_feature, text_center, epoch):

        """
        input:
            - image_output: logits for anchor images w/ shape [b, num_classes]
            - image_nb_output: logits for neighbor images w/ shape [b, num_classes]
            - image_feature: features for images w/ shape [b, 512]
            - text_center: semantic centers w/ shape [num_classes, 512]
            - epoch: the number of training rounds for dataset

        output:
            - Loss
        """
        # Softmax
        b, c = image_nb_output.size()
        image_prob = torch.softmax(image_output, dim=-1)
        image_nb_prob = torch.softmax(image_nb_output, dim=-1)
        similarity = torch.bmm(image_prob.view(b, 1, c), image_nb_prob.view(b, c, 1)).squeeze()

        # L_I: Image consistency learning loss
        consistency_loss = -torch.sum(torch.log(similarity), dim=0) / b

        # L_B: Entropy loss
        entropy_loss = entropy(torch.mean(image_prob, 0), input_as_probabilities=True)

        if epoch > 0:
            text_center = text_center.cuda()
            text_prob = torch.mm(image_feature, text_center.T).softmax(dim=-1)
            _, pseudo_label = torch.max(text_prob, dim=1)

            # L_{IS}: Image-semantic consistency learning loss
            ce_loss = self.CE(image_output, pseudo_label)

        else:
            ce_loss = torch.tensor([0.0]).cuda()

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss + self.ce_weight * ce_loss

        return total_loss, consistency_loss, - self.entropy_weight * entropy_loss, self.ce_weight * ce_loss


