import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.functional import normalize


class ComputeCenter(nn.Module):
    def __init__(self, num_cluster=10):

        super(ComputeCenter, self).__init__()
        self.num_cluster = num_cluster

    def get_image_centers(self, image_features, image_scores, xi_c):
        _, idx_max = torch.sort(image_scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        topk = int(num_per_cluster * xi_c)
        idx_max = idx_max[0:topk, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(image_features[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        return centers

    def search_sim_texts(self, args, image_centers, text_loader, model):
        model.eval()
        xi_a = args.xi_a  # xi_a nearest texts to each image center

        text_features = []
        for i, batch in enumerate(text_loader):
            input_, _, indices_ = batch
            input_ = input_.cuda()

            with torch.no_grad():
                text_feature_ = model(input_, forward_pass='backbone_t')
            text_features.append(text_feature_)

        text_features = torch.cat(text_features)
        similarity_image_texts = torch.cosine_similarity(image_centers.unsqueeze(1),
                                                         text_features.unsqueeze(0), dim=2)
        # Get nearest texts embeddings
        similarity_text_index_top = torch.topk(similarity_image_texts, xi_a)
        ind = similarity_text_index_top.indices  # the indices of texts
        ind = ind.reshape(-1)
        nearest_texts = text_loader.dataset.tensors[0][ind].cuda()
        text_features = model(nearest_texts, forward_pass='backbone_t')
        text_features = torch.cat([(torch.sum(text_features[i:i+xi_a], dim=0).unsqueeze(0))/xi_a   # Compute mean embeddings
                     for i in range(0, len(ind), xi_a)])

        text_features = normalize(text_features, dim=1)

        return text_features

    def get_centers(self, image_features, model, args):
        model.eval()
        image_outputs = []
        bs = 1024
        for i in range(0, len(image_features), bs):
            image_features_ = image_features[i:i + bs].cuda()
            with torch.no_grad():
                image_outputs_ = model(image_features_, forward_pass='head_i')
            image_outputs.append(image_outputs_[0])
        image_outputs = torch.cat(image_outputs)


        image_centers = self.get_image_centers(image_features, image_outputs, args.xi_c)
        image_centers = image_centers.cuda()
        image_centers = normalize(image_centers, dim=1)

        return image_centers

