import torch
import numpy as np


@torch.no_grad()
def image_centers_filter(model, text_dataloader, image_centers, gamma_r):
    model.eval()
    sim = []
    for i, batch in enumerate(text_dataloader):
        input_, _, target_ = batch
        input_ = input_.cuda()
        with torch.no_grad():
            feature_ = model(input_, forward_pass='backbone_t')
            sim_ = torch.cosine_similarity(image_centers.unsqueeze(1).cuda(), feature_.float().unsqueeze(0), dim=2)
            sim.append(sim_)
    sim = torch.cat(sim , dim=1)
    _, list_target = torch.topk(sim, gamma_r) # gamma_r: number for nearest nouns to each image center

    return list_target.reshape(-1)


@torch.no_grad()
def uniqueness_filter(model, text_dataloader, threshold):
    model.eval()
    noun_mean=[]
    for i, batch in enumerate(text_dataloader):
        input_, _, _ = batch
        input_ = input_.cuda()
        feature_ = model(input_, forward_pass='backbone_t')
        numpy_feature = feature_.cpu().numpy()
        batch_mean_ = np.mean(numpy_feature, axis=0)
        noun_mean.append(batch_mean_)

    # Get the text center
    noun_mean = np.array(noun_mean).mean(axis=0)
    noun_mean = torch.tensor(noun_mean)
    noun_mean_ = noun_mean.unsqueeze(0).float()

    list_target = []
    for i, batch in enumerate(text_dataloader):
        input_, _, target_ = batch
        input_ = input_.cuda()
        target_ = target_.cuda()
        feature_ = model(input_, forward_pass='backbone_t')
        similarity = torch.cosine_similarity(feature_.float().cpu(),noun_mean_)
        mask = similarity < threshold  # threshold = 1 - gamma_u, where gamma_u is a uniqueness score for constructing semantic space
        target_masked = torch.masked_select(target_, mask.squeeze().cuda())
        list_target.append(target_masked)

    list_target = torch.cat(list_target)
    return list_target

