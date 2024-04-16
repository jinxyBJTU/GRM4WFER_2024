import torch
import numpy as np
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature=0.02):
        super(NTXentLoss, self).__init__()
        # self.batch_size = batch_size
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(True)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none") # none sum

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def hand_cross_entropy_loss(self, logits, isinstance_discri_labels):
        nll_loss = torch.nn.NLLLoss(reduction='none')

        softmax = F.softmax(logits, dim=1)
        log_softmax = torch.log(softmax)
        ce_2batch_all = nll_loss( log_softmax, isinstance_discri_labels ) # B

        return ce_2batch_all

    def multi_insdic(self, logits, True_discrimination_labels):
        nll_loss = torch.nn.NLLLoss(reduction='none')

        softmax = F.softmax(logits, dim=1)
        log_softmax = torch.log(softmax)
        # ce_2batch_all = nll_loss( log_softmax, isinstance_discri_labels ) # B
        
        ce_2batch_all = torch.sum( (-log_softmax)*(True_discrimination_labels),  dim=-1 )
        return ce_2batch_all
    
    # def forward(self, feature_1, fine_labels):
    #     # feature_1: 8,30,F
    #     # fine_labels: 8,30
    #     trail_num, bag_size, feature_dim = feature_1.shape
    #     batch_size = trail_num*bag_size
    #     # neg_mask = self.get_correlated_mask(batch_size)
       
    #     feature_1 = feature_1.reshape(-1, feature_dim)
    #     fine_labels = fine_labels.reshape(-1)
    #     True_discrimination_labels = fine_labels.unsqueeze(0)==fine_labels.unsqueeze(1)

    #     diag = np.eye(batch_size)
    #     mask = torch.from_numpy((diag))
    #     mask = (1 - mask).type(torch.bool)
    #     True_discrimination_labels = True_discrimination_labels[mask].view(batch_size, -1)
    #     True_discrimination_labels = True_discrimination_labels.type(torch.FloatTensor).to(feature_1.device)

    #     representations = feature_1   
    #     similarity_matrix = self.similarity_function(representations, representations) # 2B, 2B

    #     logits_all = similarity_matrix[mask].view(batch_size, -1) # (2B, 2B-1)
    #     logits_all /= self.temperature
        
    #     # isinstance_discri_labels = torch.zeros(2 * batch_size).to(feature_1.device).long()
    #     ce_2batch_all = self.multi_insdic(logits_all, True_discrimination_labels) # # loss = self.criterion(logits, labels)
        
    #     return torch.mean(ce_2batch_all)
    
    # def forward(self, feature_1, feature_2, fine_labels):
    #     # feature_1: 8,30,F
    #     # feature_1: 8,30,F
    #     # feature_1: 8,30
    #     trail_num, bag_size, feature_dim = feature_1.shape
    #     batch_size = trail_num*bag_size
    #     neg_mask = self.get_correlated_mask(batch_size)
       
    #     feature_1 = feature_1.reshape(-1, feature_dim)
    #     feature_2 = feature_2.reshape(-1, feature_dim)
    #     fine_labels = fine_labels.reshape(-1)
    #     fine_labels = fine_labels.unsqueeze(1).repeat(1,2).reshape(-1)
    #     True_discrimination_labels = fine_labels.unsqueeze(0)==fine_labels.unsqueeze(1)
    #     diag = np.eye(2 * batch_size)
    #     mask = torch.from_numpy((diag))
    #     mask = (1 - mask).type(torch.bool)
    #     True_discrimination_labels = True_discrimination_labels[mask].view(2*batch_size, -1)
    #     True_discrimination_labels = True_discrimination_labels.type(torch.FloatTensor).to(feature_1.device)

    #     representations = torch.cat([feature_1, feature_2])       
    #     similarity_matrix = self.similarity_function(representations, representations) # 2B, 2B

    #     l_pos = torch.diag(similarity_matrix, batch_size)
    #     r_pos = torch.diag(similarity_matrix, -batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    #     negatives = similarity_matrix[neg_mask].view(2*batch_size, -1)
    #     logits_all = torch.cat((positives, negatives), dim=1) # (2B, 2B-1)
    #     logits_all /= self.temperature

    #     # isinstance_discri_labels = torch.zeros(2 * batch_size).to(feature_1.device).long()
    #     ce_2batch_all = self.multi_insdic(logits_all, True_discrimination_labels) # # loss = self.criterion(logits, labels)
        
    #     return torch.mean(ce_2batch_all)
    
    # def forward(self, feature_1, feature_2, feature_3):
    #     # representations: 2B,F
    #     # print(representations.shape)
    #     # exit()
    #     batch_size = len(feature_1)
    #     neg_mask = self.get_correlated_mask(batch_size)

    #     representations1 = torch.cat([feature_1, feature_2])
    #     representations2 = torch.cat([feature_1, feature_3])
       
    #     similarity_matrix1 = self.similarity_function(representations1, representations1) # 2B, 2B
    #     similarity_matrix2 = self.similarity_function(representations2, representations2) # 2B, 2B

    #     l_pos = torch.diag(similarity_matrix1, batch_size)
    #     r_pos = torch.diag(similarity_matrix1, -batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    #     negatives = similarity_matrix2[neg_mask].view(2*batch_size, -1)
    #     logits_all = torch.cat((positives, negatives), dim=1) # (2B, 2B-1)
    #     logits_all /= self.temperature

    #     isinstance_discri_labels = torch.zeros(2 * batch_size).to(feature_1.device).long()
    #     ce_2batch_all = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels) # # loss = self.criterion(logits, labels)
        
    #     return torch.mean(ce_2batch_all)

    # def forward(self, feature_1, feature_2):
    #     # feature_1: 8,30,F
    #     trail_num, bag_size, feature_dim = feature_1.shape
    #     feature_1 = feature_1.reshape(-1, feature_dim)
    #     feature_2 = feature_2.reshape(-1, feature_dim)

    #     batch_size = trail_num*bag_size
    #     neg_mask = self.get_correlated_mask(batch_size)

    #     representations = torch.cat([feature_1, feature_2])
       
    #     similarity_matrix = self.similarity_function(representations, representations) # 2B, 2B

    #     l_pos = torch.diag(similarity_matrix, batch_size)
    #     r_pos = torch.diag(similarity_matrix, -batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    #     negatives = similarity_matrix[neg_mask].view(2*batch_size, -1)
    #     logits_all = torch.cat((positives, negatives), dim=1) # (2B, 2B-1)
    #     logits_all /= self.temperature

    #     isinstance_discri_labels = torch.zeros(2 * batch_size).to(feature_1.device).long()
    #     ce_2batch_all = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels) # # loss = self.criterion(logits, labels)
        
    #     return torch.mean(ce_2batch_all)

    def kl_divergence(self, mu1, cov_diag1, mu2, cov_diag2):
        # 计算KL散度
        kl_div = torch.log(torch.sqrt(cov_diag2 / cov_diag1)) + \
                (cov_diag1 + (mu1 - mu2) ** 2) / (2 * cov_diag2) - 0.5
        return kl_div.sum(dim=-1)

    def js_divergence(self, mu1, cov_diag1, mu2, cov_diag2):
        # 计算两个分布的KL散度
        kl_div_1 = self.kl_divergence(mu1, cov_diag1, mu2, cov_diag2)
        kl_div_2 = self.kl_divergence(mu2, cov_diag2, mu1, cov_diag1)
        
        # 计算JS散度
        js_div = (kl_div_1 + kl_div_2) / 2
        
        return js_div
    
    def get_js_sim(self, hidden_features, hidden_uncers):
        mu_list_expanded1 = hidden_features.unsqueeze(1)  # (B, 1, F)
        mu_list_expanded2 = hidden_features.unsqueeze(0)  # (1, B, F)
        cov_diag_list_expanded1 = hidden_uncers.unsqueeze(1)  # (B, 1, F)
        cov_diag_list_expanded2 = hidden_uncers.unsqueeze(0)  # (1, B, F)
        js_divergence_values = self.js_divergence(mu_list_expanded1, cov_diag_list_expanded1, mu_list_expanded2, cov_diag_list_expanded2)

        return js_divergence_values
    
    def forward(self, hidden_features, hidden_uncers):
        
        batch_size, _ = hidden_features.shape
        similarity_matrix = self.similarity_function(hidden_features, hidden_features) # B, B
        # similarity_matrix = self.get_js_sim(hidden_features, hidden_uncers) # B,B
        print(similarity_matrix.shape)
        exit()
        # l_pos = torch.diag(similarity_matrix, batch_size)
        # r_pos = torch.diag(similarity_matrix, -batch_size)
        # positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        # negatives = similarity_matrix[neg_mask].view(2*batch_size, -1)
        # logits_all = torch.cat((positives, negatives), dim=1) # (2B, 2B-1)
        # logits_all /= self.temperature

        # isinstance_discri_labels = torch.zeros(2 * batch_size).to(instance_gains_1.device).long()
        # ce_2batch_all = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels) # # loss = self.criterion(logits, labels)
        
        # return torch.mean(ce_2batch_all)
    
    # def forward(self, instance_gains_1, instance_gains_2):
    #     # feature_1: 8,30,F
    #     batch_size, _ = instance_gains_1.shape

    #     neg_mask = self.get_correlated_mask(batch_size)

    #     representations = torch.cat([instance_gains_1, instance_gains_2])
       
    #     similarity_matrix = self.similarity_function(representations, representations) # 2B, 2B

    #     l_pos = torch.diag(similarity_matrix, batch_size)
    #     r_pos = torch.diag(similarity_matrix, -batch_size)
    #     positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    #     negatives = similarity_matrix[neg_mask].view(2*batch_size, -1)
    #     logits_all = torch.cat((positives, negatives), dim=1) # (2B, 2B-1)
    #     logits_all /= self.temperature

    #     isinstance_discri_labels = torch.zeros(2 * batch_size).to(instance_gains_1.device).long()
    #     ce_2batch_all = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels) # # loss = self.criterion(logits, labels)
        
    #     return torch.mean(ce_2batch_all)