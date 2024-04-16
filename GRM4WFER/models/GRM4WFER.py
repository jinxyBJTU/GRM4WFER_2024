import torch
from torch import nn
import torch.nn.functional as F

class Mine_case(nn.Module):
    def __init__(self, configs, input_dim, signal_length, batch_size, num_of_classes):
        super(Mine_case, self).__init__()
        self.use_uncertaiy = configs.use_uncertaiy
        self.use_crossTrail_gcn = configs.use_crossTrail_gcn

        self.repeated_times = configs.repeat_times
        self.ratio = configs.drop_ratio

        self.graph_type = configs.graph_type

        hidden_channels = [8,16,32,64,128,128]
        kernal_sizes = [signal_length//2+1, signal_length//3, signal_length//4, signal_length//8 +1, signal_length//12 +1, signal_length//8 +1]
        self.Feature_extraction_layers = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = hidden_channels[0], kernel_size = kernal_sizes[0], padding = int((kernal_sizes[0]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[0], out_channels = hidden_channels[1], kernel_size = kernal_sizes[1], padding = int((kernal_sizes[1]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[1], out_channels = hidden_channels[2], kernel_size = kernal_sizes[2], padding = int((kernal_sizes[2]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[2], out_channels = hidden_channels[3], kernel_size = kernal_sizes[3], padding = int((kernal_sizes[3]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[3], out_channels = hidden_channels[4], kernel_size = kernal_sizes[4], padding = int((kernal_sizes[4]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[4], out_channels = hidden_channels[5], kernel_size = kernal_sizes[5], padding = int((kernal_sizes[5]-1)/2)),
        )

        self.bag_classifier = nn.Linear(batch_size, num_of_classes)

        self.gcn_weight_1 = nn.Linear(128, 128)
        self.gcn_weight_2 = nn.Linear(128, 128)

        self.time_aware_proj_1 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))
        self.time_aware_proj_2 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))
        self.time_aware_proj_3 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))

    def generate_cons_graph(self, cur_lengths, weak_labels):
        
        trail_num = len(cur_lengths)
        expand_weak_labels = []
        for trail_idx in range(trail_num):
            # expand_weak_labels = weak_labels.unsqueeze(1).repeat(1,bag_size)
            # expand_weak_labels = expand_weak_labels.reshape(-1)
            expand_weak_labels.append(weak_labels[trail_idx].repeat(cur_lengths[trail_idx]))
        expand_weak_labels = torch.cat(expand_weak_labels)
        cons_graph_bool = expand_weak_labels.unsqueeze(1)==expand_weak_labels.unsqueeze(0)
        
        return cons_graph_bool.type(torch.FloatTensor).to(weak_labels.device)
    
    def message_passing(self, hidden_features, hidden_uncers, cur_lengths, weak_labels):

        trail_num, bag_size, channels = hidden_features.shape

        if self.graph_type == 'TA_adaptive':
            query  = self.time_aware_proj_1(hidden_features.permute(0,2,1))
            key    = self.time_aware_proj_2(hidden_features.permute(0,2,1))
        else:
            query = hidden_features.permute(0,2,1)
            key = hidden_features.permute(0,2,1)

        if self.use_uncertaiy:
            uncer_gate = torch.sigmoid(self.time_aware_proj_3(hidden_uncers.permute(0,2,1)))
            uncer_gate_list = []

        query_list = []
        key_list = []
        hidden_features_list = []
        
        for trail_idx in range(trail_num):
            cur_query = query.permute(0,2,1)[trail_idx,:cur_lengths[trail_idx]].reshape(-1,channels)
            cur_key = key.permute(0,2,1)[trail_idx,:cur_lengths[trail_idx]].reshape(-1,channels)
            cur_hidden_features = hidden_features[trail_idx,:cur_lengths[trail_idx]].reshape(-1,channels)
            query_list.append(cur_query)
            key_list.append(cur_key)
            hidden_features_list.append(cur_hidden_features)

            if self.use_uncertaiy:
                cur_uncer_gate = uncer_gate.permute(0,2,1)[trail_idx,:cur_lengths[trail_idx]].reshape(-1,channels)
                uncer_gate_list.append(cur_uncer_gate)
            
        part_query = torch.cat(query_list, dim=0)
        part_key = torch.cat(key_list, dim=0)
        part_hidden_features = torch.cat(hidden_features_list, dim=0)

        if self.use_uncertaiy:
            part_uncer_gate = torch.cat(uncer_gate_list, dim=0)
            part_hidden_features = part_uncer_gate*part_hidden_features

        if self.graph_type == 'TA_adaptive' or self.graph_type == 'adaptive':
            similarity_matrix = torch.softmax(torch.matmul(part_query, part_key.permute(1,0)), dim=-1)
        elif self.graph_type == 'weak':
            similarity_matrix = self.generate_cons_graph(cur_lengths, weak_labels)
        
        part_hidden_features = torch.relu( self.gcn_weight_1( torch.matmul(similarity_matrix, part_hidden_features) )  )
        part_hidden_features = self.gcn_weight_2( torch.matmul(similarity_matrix, part_hidden_features) )  
        
        cur_idx = 0
        for trail_idx in range(trail_num):
            cur_length = cur_lengths[trail_idx]
            hidden_features[trail_idx,:cur_lengths[trail_idx]] = part_hidden_features[cur_idx:cur_idx+cur_length]
            cur_idx += cur_length

        return hidden_features
    
    def forward(self, input_signals, weak_labels, cur_lengths, is_testing = False):
        
        trail_num, bag_size, input_channels, signal_length = input_signals.shape

        if self.use_uncertaiy:
            trail_features = []
            trail_uncers = []
            for trail_index in range(len(input_signals)):
                repeat_physiological_signal = input_signals[trail_index].repeat(self.repeated_times, 1, 1) # (B*repeat,c,T)

                Mask_rand = torch.rand(( bag_size*self.repeated_times, signal_length )).type(torch.FloatTensor).to(input_signals.device) # (B*repeat, T)
                Mask_zeros = Mask_rand.masked_fill( Mask_rand < self.ratio , 0. )
                Mask = Mask_zeros.masked_fill( Mask_rand >= self.ratio , 1. )
                Mask = Mask.unsqueeze(1).repeat( 1,input_channels, 1 ) # (B*repeat, c, T)

                repeat_physiological_signal = Mask*repeat_physiological_signal
                repe_instance_feature_list = torch.split( self.Feature_extraction_layers( repeat_physiological_signal ) ,  bag_size,   dim=0 )  # list[]:5; (B, C, t)
                
                mean_instance_feature = torch.mean(torch.stack(repe_instance_feature_list), dim=0)                              # B D t
                variance_of_instance = torch.mean(torch.stack(repe_instance_feature_list)**2,dim=0) - mean_instance_feature**2  # B D t
                
                cur_trail_feature, max_indices = torch.max(mean_instance_feature, dim=-1)  # B 128
                cur_trail_uncer = torch.gather(variance_of_instance, 2, max_indices.unsqueeze(2)).squeeze(-1)
                
                cur_trail_feature = torch.relu(cur_trail_feature)               # B 128
                trail_features.append(cur_trail_feature)
                trail_uncers.append(cur_trail_uncer)

            trail_features = torch.stack(trail_features, dim=0)  # 8 B 128
            trail_uncers = torch.stack(trail_uncers, dim=0)      # 8 B 128

            if self.use_crossTrail_gcn :
                after_gcn_features = self.message_passing( trail_features, trail_uncers, cur_lengths, weak_labels )
            else:
                after_gcn_features = trail_features.reshape(-1,128)

        else:
            trail_features = []
            for trail_index in range(len(input_signals)):
                mean_instance_feature = self.Feature_extraction_layers( input_signals[trail_index] )# B 128 100
                cur_trail_feature, _ = torch.max(mean_instance_feature, dim=-1)  # B 128
                cur_trail_feature = torch.relu(cur_trail_feature)                # B 128
                trail_features.append(cur_trail_feature)

            trail_features = torch.stack(trail_features, dim=0)  # 8 B 128
            trail_uncers = trail_features

            if self.use_crossTrail_gcn :
                after_gcn_features = self.message_passing( trail_features, trail_features, cur_lengths, weak_labels )
            else:
                after_gcn_features = trail_features.reshape(-1,128)

        after_gcn_features = after_gcn_features.reshape(trail_num, bag_size, -1)
        instance_gains, _ = torch.max(after_gcn_features, dim=-1)  

        if is_testing:
            return after_gcn_features, instance_gains
        else:
            bag_preds = self.bag_classifier( instance_gains )
            
            return after_gcn_features, trail_features, trail_uncers, bag_preds

class Mine_ceap(nn.Module):
    def __init__(self, configs, input_dim, signal_length, batch_size, num_of_classes):
        super(Mine_ceap, self).__init__()
        self.use_uncertaiy = configs.use_uncertaiy
        self.use_crossTrail_gcn = configs.use_crossTrail_gcn

        self.repeated_times = configs.repeat_times
        self.ratio = configs.drop_ratio

        self.graph_type = configs.graph_type

        hidden_channels = [8,16,32,64,128,128]
        kernal_sizes = [signal_length//2+1, signal_length//3, signal_length//4, signal_length//8 +1, signal_length//12 +1, signal_length//8 +1]
        self.Feature_extraction_layers = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = hidden_channels[0], kernel_size = kernal_sizes[0], padding = int((kernal_sizes[0]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[0], out_channels = hidden_channels[1], kernel_size = kernal_sizes[1], padding = int((kernal_sizes[1]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[1], out_channels = hidden_channels[2], kernel_size = kernal_sizes[2], padding = int((kernal_sizes[2]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[2], out_channels = hidden_channels[3], kernel_size = kernal_sizes[3], padding = int((kernal_sizes[3]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[3], out_channels = hidden_channels[4], kernel_size = kernal_sizes[4], padding = int((kernal_sizes[4]-1)/2)),
            nn.Conv1d(in_channels = hidden_channels[4], out_channels = hidden_channels[5], kernel_size = kernal_sizes[5], padding = int((kernal_sizes[5]-1)/2)),
        )
        self.bag_classifier = nn.Linear(batch_size, num_of_classes)
        # self.instance_classifier = nn.Linear(128, num_of_classes)

        self.gcn_weight_1 = nn.Linear(128, 128)
        self.gcn_weight_2 = nn.Linear(128, 128)

        self.time_aware_proj_1 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))
        self.time_aware_proj_2 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))
        self.time_aware_proj_3 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = int((3-1)/2))  

    def generate_cons_graph(self, weak_labels, bag_size):

        expand_weak_labels = weak_labels.unsqueeze(1).repeat(1,bag_size)
        expand_weak_labels = expand_weak_labels.reshape(-1)
        cons_graph_bool = expand_weak_labels.unsqueeze(1)==expand_weak_labels.unsqueeze(0)
        
        return cons_graph_bool.type(torch.FloatTensor).to(weak_labels.device)

    def message_passing(self, hidden_features, hidden_uncers, weak_labels):

        trail_num, bag_size, channels = hidden_features.shape

        if self.graph_type == 'TA_adaptive':
            query  = self.time_aware_proj_1(hidden_features.permute(0,2,1))
            key    = self.time_aware_proj_2(hidden_features.permute(0,2,1))
        else:
            query = hidden_features.permute(0,2,1)
            key = hidden_features.permute(0,2,1)

        query  = query.permute(0,2,1).reshape(-1,channels)
        key    = key.permute(0,2,1).reshape(-1,channels)
        hidden_features = hidden_features.reshape(-1,channels)

        if self.graph_type == 'TA_adaptive' or self.graph_type == 'adaptive':
            similarity_matrix = torch.softmax(torch.matmul(query, key.permute(1,0)), dim=-1)
        elif self.graph_type == 'weak':
            similarity_matrix = self.generate_cons_graph(weak_labels, bag_size)

        if self.use_uncertaiy:
            uncer_gate = torch.sigmoid(self.time_aware_proj_3(hidden_uncers.permute(0,2,1)))
            uncer_gate = uncer_gate.permute(0,2,1).reshape(-1,channels)
            hidden_features = uncer_gate*hidden_features
        else:
            hidden_features = hidden_features
       
        hidden_features = torch.relu( self.gcn_weight_1( torch.matmul(similarity_matrix, hidden_features) )  )
        hidden_features = self.gcn_weight_2( torch.matmul(similarity_matrix, hidden_features) )  
        
        return hidden_features
    
    def forward(self, input_signals, weak_labels, cur_lengths, is_testing = False):
        
        trail_num, bag_size, input_channels, signal_length = input_signals.shape

        if self.use_uncertaiy:
            trail_features = []
            trail_uncers = []
            for trail_index in range(len(input_signals)):
                repeat_physiological_signal = input_signals[trail_index].repeat(self.repeated_times, 1, 1) # (B*repeat,c,T)

                Mask_rand = torch.rand(( bag_size*self.repeated_times, signal_length )).type(torch.FloatTensor).to(input_signals.device) # (B*repeat, T)
                Mask_zeros = Mask_rand.masked_fill( Mask_rand < self.ratio , 0. )
                Mask = Mask_zeros.masked_fill( Mask_rand >= self.ratio , 1. )
                Mask = Mask.unsqueeze(1).repeat( 1,input_channels, 1 ) # (B*repeat, c, T)

                repeat_physiological_signal = Mask*repeat_physiological_signal
                repe_instance_feature_list = torch.split( self.Feature_extraction_layers( repeat_physiological_signal ) ,  bag_size,   dim=0 )  # list[]:5; (B, C, t)
                
                mean_instance_feature = torch.mean(torch.stack(repe_instance_feature_list), dim=0)                              # B D t
                variance_of_instance = torch.mean(torch.stack(repe_instance_feature_list)**2,dim=0) - mean_instance_feature**2  # B D t
                
                cur_trail_feature, max_indices = torch.max(mean_instance_feature, dim=-1)  # B 128
                cur_trail_uncer = torch.gather(variance_of_instance, 2, max_indices.unsqueeze(2)).squeeze(-1)
                
                cur_trail_feature = torch.relu(cur_trail_feature)               # B 128
                trail_features.append(cur_trail_feature)
                trail_uncers.append(cur_trail_uncer)

            trail_features = torch.stack(trail_features, dim=0)  # 8 B 128
            trail_uncers = torch.stack(trail_uncers, dim=0)      # 8 B 128

            if self.use_crossTrail_gcn :
                after_gcn_features = self.message_passing( trail_features, trail_uncers, weak_labels )
            else:
                after_gcn_features = trail_features.reshape(-1,128)

        else:
            trail_features = []
            for trail_index in range(len(input_signals)):
                mean_instance_feature = self.Feature_extraction_layers( input_signals[trail_index] )# B 128 100
                cur_trail_feature, _ = torch.max(mean_instance_feature, dim=-1)  # B 128
                cur_trail_feature = torch.relu(cur_trail_feature)                # B 128
                trail_features.append(cur_trail_feature)

            trail_features = torch.stack(trail_features, dim=0)  # 8 B 128
            trail_uncers = trail_features

            if self.use_crossTrail_gcn :
                after_gcn_features = self.message_passing( trail_features, trail_features, weak_labels )
            else:
                after_gcn_features = trail_features.reshape(-1,128)
                
        after_gcn_features = after_gcn_features.reshape(trail_num, bag_size, -1) # 8,30,128
        instance_gains, _ = torch.max(after_gcn_features, dim=-1)  
        
        if is_testing:
            return after_gcn_features, instance_gains
        else:
            bag_preds = self.bag_classifier( instance_gains )
            # instance_preds = self.instance_classifier( after_gcn_features.reshape(trail_num*bag_size,-1) )
            return after_gcn_features, trail_features, trail_uncers, bag_preds

    