import torch
import numpy as np
import time
import os
import torch.nn.functional as F

# def Intra_Sim_withinK(features):
#     num_of_trials, bag_size, feature_dims = features.shape
#     features = features.reshape(-1, feature_dims)
#     selected_elements_norm = F.normalize(features, p=2, dim=1)  # 归一化张量
#     cosine_similarities = torch.matmul(selected_elements_norm, selected_elements_norm.t())
#     return torch.mean(cosine_similarities)

# def Inter_Sim_withinK(features_a, features_n, K=30):

#     # max_k_indices = torch.topk(torch.sum(uncers_a, dim=1), k=K, largest=True)[1]
#     # selected_features_a = torch.index_select(features_a, dim=0, index=max_k_indices)

#     # min_k_indices = torch.topk(torch.sum(uncers_n, dim=1), k=K, largest=False)[1]
#     # selected_features_n = torch.index_select(features_n, dim=0, index=min_k_indices)

#     normal_trails, bag_size, feature_dims = features_a.shape
#     anomaly_trails, _, _ = features_n.shape

#     features_a = features_a.reshape(-1, feature_dims)
#     features_n = features_n.reshape(-1, feature_dims)
#     features_a = F.normalize(features_a, p=2, dim=1)  # 归一化张量
#     features_n = F.normalize(features_n, p=2, dim=1)  # 归一化张量

#     cosine_similarities = torch.matmul(features_a, features_n.t())
#     trail_list_smilarities = torch.split( cosine_similarities ,  bag_size,   dim=0 )
#     trail_list_smilarities = torch.stack(trail_list_smilarities)
#     trail_list_smilarities = trail_list_smilarities.reshape(len(trail_list_smilarities),-1)

#     topK_cosine_similarities = torch.topk(trail_list_smilarities, k=int(trail_list_smilarities.shape[-1]*0.1), largest=True)[0]
   
#     return torch.mean(topK_cosine_similarities)

def gaussian_wasserstein_distance(mu1, cov_diag1, mu2, cov_diag2):
    # 计算Wasserstein距离
    wasserstein_dist = torch.sqrt(torch.sum((mu1.unsqueeze(1) - mu2.unsqueeze(0)) ** 2 + cov_diag1.unsqueeze(1) + cov_diag2.unsqueeze(0) - 2 * torch.sqrt(cov_diag1.unsqueeze(1) * cov_diag2.unsqueeze(0)), dim=-1))
    
    # return wasserstein_dist
    return torch.exp(-wasserstein_dist/1)

def kl_divergence(mu1, cov_diag1, mu2, cov_diag2):
    # 计算KL散度
    kl_div = torch.log(torch.sqrt(cov_diag2 / cov_diag1)) + \
            (cov_diag1 + (mu1 - mu2) ** 2) / (2 * cov_diag2) - 0.5
    return kl_div.sum(dim=-1)

def js_divergence(mu1, cov_diag1, mu2, cov_diag2):
    # 计算两个分布的KL散度
    kl_div_1 = kl_divergence(mu1, cov_diag1, mu2, cov_diag2)
    kl_div_2 = kl_divergence(mu2, cov_diag2, mu1, cov_diag1)
    
    # 计算JS散度
    js_div = (kl_div_1 + kl_div_2) / 2
    
    return js_div

def Intra_Sim_withinK(dis_type, features, uncers, cur_lengths=None):
    num_of_trials, bag_size, feature_dims = features.shape

    if cur_lengths!=None:
        features_list = []
        uncers_list = []
        for trail_idx in range(num_of_trials):
            cur_features = features[trail_idx, :cur_lengths[trail_idx]]
            cur_uncers = uncers[trail_idx, :cur_lengths[trail_idx]]
            features_list.append(cur_features)
            uncers_list.append(cur_uncers)
        features = torch.cat(features_list, dim=0)
        uncers = torch.cat(uncers_list, dim=0)
    else:
        features = features.reshape(-1, feature_dims)
        uncers = uncers.reshape(-1, feature_dims)
    uncers = uncers+ 1e-6
    
    mu_expanded1 = features.unsqueeze(1)  # (B, 1, F)
    mu_expanded2 = features.unsqueeze(0)  # (1, B, F)
    cov_diag_expanded1 = uncers.unsqueeze(1)  # (B, 1, F)
    cov_diag_expanded2 = uncers.unsqueeze(0)  # (1, B, F)
    
    if dis_type == 'JS':
        js_divergence_values = js_divergence(mu_expanded1, cov_diag_expanded1, mu_expanded2, cov_diag_expanded2)
        js_divergence_values = (js_divergence_values)/ (torch.max(js_divergence_values,dim=-1,keepdim=True)[0])
        similarity_matrix = torch.exp(-js_divergence_values)
    elif dis_type == 'KL':
        kl_divergence_values = kl_divergence(mu_expanded1, cov_diag_expanded1, mu_expanded2, cov_diag_expanded2)
        kl_divergence_values = (kl_divergence_values)/ (torch.max(kl_divergence_values,dim=-1,keepdim=True)[0])
        similarity_matrix = torch.exp(-kl_divergence_values)
    elif dis_type == 'Elu':
        norm_features = F.normalize(features, p=2, dim=1)  # 归一化张量
        similarity_matrix =  torch.matmul(norm_features, norm_features.t())
    
    return torch.mean(similarity_matrix)

def Inter_Sim_withinK(dis_type, features_a, features_n, uncers_a, uncers_n, anamoly_lengths=None, normal_lengths=None):

    anomaly_trails, bag_size, feature_dims = features_a.shape
    normal_trails, _, _ = features_n.shape

    if anamoly_lengths!=None and normal_lengths!=None:
        ana_features_list = []
        ana_uncers_list = []

        nor_features_list = []
        nor_uncers_list = []

        for trail_idx in range(anomaly_trails):
            cur_features = features_a[trail_idx, :anamoly_lengths[trail_idx]]
            cur_uncers = uncers_a[trail_idx, :anamoly_lengths[trail_idx]]
            ana_features_list.append(cur_features)
            ana_uncers_list.append(cur_uncers)

        for trail_idx in range(normal_trails):
            cur_features = features_n[trail_idx, :normal_lengths[trail_idx]]
            cur_uncers = uncers_n[trail_idx, :normal_lengths[trail_idx]]
            nor_features_list.append(cur_features)
            nor_uncers_list.append(cur_uncers)

        features_a = torch.cat(ana_features_list, dim=0)
        features_n = torch.cat(nor_features_list, dim=0)
        uncers_a = torch.cat(ana_uncers_list, dim=0)
        uncers_n = torch.cat(nor_uncers_list, dim=0)
    else:
        features_a = features_a.reshape(-1, feature_dims)
        uncers_a = uncers_a.reshape(-1, feature_dims)
        features_n = features_n.reshape(-1, feature_dims)
        uncers_n = uncers_n.reshape(-1, feature_dims)

    uncers_a = uncers_a + 1e-6
    uncers_n = uncers_n + 1e-6
    
    mu_expanded1 = features_a.unsqueeze(1)  # (B, 1, F)
    mu_expanded2 = features_n.unsqueeze(0)  # (1, B, F)
    cov_diag_expanded1 = uncers_a.unsqueeze(1)  # (B, 1, F)
    cov_diag_expanded2 = uncers_n.unsqueeze(0)  # (1, B, F)

    if dis_type == 'JS':
        js_divergence_values = js_divergence(mu_expanded1, cov_diag_expanded1, mu_expanded2, cov_diag_expanded2)
        js_divergence_values = (js_divergence_values)/ (torch.max(js_divergence_values,dim=-1,keepdim=True)[0])
        similarity_matrix = torch.exp(-js_divergence_values)
    elif dis_type == 'KL':
        kl_divergence_values = kl_divergence(mu_expanded1, cov_diag_expanded1, mu_expanded2, cov_diag_expanded2)
        kl_divergence_values = (kl_divergence_values)/ (torch.max(kl_divergence_values,dim=-1,keepdim=True)[0])
        similarity_matrix = torch.exp(-kl_divergence_values)
    elif dis_type == 'Elu':
        norm_features_a = F.normalize(features_a, p=2, dim=1)  # 归一化张量
        norm_features_n = F.normalize(features_n, p=2, dim=1)  # 归一化张量
        similarity_matrix =  torch.matmul(norm_features_a, norm_features_n.t())
    
    if anamoly_lengths!=None and normal_lengths!=None:
        cur_idx = 0
        topK_similarities_list = []
        for trail_idx in range(anomaly_trails):
            cur_ana_length = anamoly_lengths[trail_idx]
            cur_trail_simis = similarity_matrix[cur_idx:cur_idx+cur_ana_length]
            cur_idx += cur_ana_length
            
            cur_topK = torch.topk(cur_trail_simis.reshape(1,-1), k=int(similarity_matrix.shape[-1]*0.1), largest=True)[0]
            topK_similarities_list.append(cur_topK)
        topK_cosine_similarities = torch.cat(topK_similarities_list, dim=0)
        
    else:
        trail_list_smilarities = torch.split( similarity_matrix ,  bag_size,   dim=0 )
        trail_list_smilarities = torch.stack(trail_list_smilarities)
        trail_list_smilarities = trail_list_smilarities.reshape(len(trail_list_smilarities),-1)

        topK_cosine_similarities = torch.topk(trail_list_smilarities, k=int(trail_list_smilarities.shape[-1]*0.1), largest=True)[0]
    
    return torch.mean(topK_cosine_similarities)  