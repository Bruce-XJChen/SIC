"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import copy

import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
_tokenizer = _Tokenizer()


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = 512
        self.nclusters = nclusters
        self.nheads = nheads
        self.prompt_learner = None
        self.cluster_head_i = nn.ModuleList([nn.Linear(self.backbone_dim, self.nclusters) for _ in range(self.nheads)])


    def forward(self, x, forward_pass='output_i'):
        if forward_pass == 'output_i':
            features = self.backbone.encode_image(x)
            features = features.float()
            out = [cluster_head_i(features) for cluster_head_i in self.cluster_head_i]
        elif forward_pass == 'head_i':
            out = [cluster_head_i(x) for cluster_head_i in self.cluster_head_i]
        elif forward_pass == 'backbone_i':
            out = self.backbone.encode_image(x)
            out = out.float()
        elif forward_pass == 'backbone_t':
            out = self.backbone.encode_text(x)
            out = out.float()
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

