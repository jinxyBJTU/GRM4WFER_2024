import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm

def MaxMinNorm(signals):
    # B,3,100
    max_value = np.max(signals, axis=(0,2), keepdims=True)
    min_value = np.min(signals, axis=(0,2), keepdims=True)
    
    signals = (signals-min_value)/(max_value-min_value)
    
    return signals

def Subject_Data_Prepare(signals, fine_labels, weak_labeles, lengths):

    num_of_subject, num_of_trail, num_of_segments, num_of_channels, num_of_timesteps = signals.shape
    
    signals = signals.reshape(-1, num_of_segments, num_of_channels, num_of_timesteps)
    fine_labels = fine_labels.reshape(-1, num_of_segments)
    weak_labeles = weak_labeles.reshape(-1, num_of_segments)
    lengths = lengths.reshape(-1)


    return signals, fine_labels, weak_labeles, lengths


def Trail_Data_Prepare(signals, fine_labels, weak_labeles, lengths):

    num_of_trail, num_of_segments, num_of_channels, num_of_timesteps = signals.shape
    
    new_signals = []
    new_fine_labels = []
    new_weak_labels = []
    for sample_len in range(len(lengths)):
        new_signals.append(signals[sample_len, :lengths[sample_len]])
        new_fine_labels.append(fine_labels[sample_len, :lengths[sample_len]])
        new_weak_labels.append(weak_labeles[sample_len, :lengths[sample_len]])
    new_signals = torch.cat(new_signals, dim=0)
    new_fine_labels = torch.cat(new_fine_labels, dim=0)
    new_weak_labels = torch.cat(new_weak_labels, dim=0)

    return new_signals, new_fine_labels, new_weak_labels

class Subject_kFoldGenerator():
    def __init__(self, total_signals, total_fine_labels, total_weak_labels, total_lengths):
        self.total_signals = total_signals          # 32, 8, 36, 3, 100
        self.total_fine_labels = total_fine_labels  # 32, 8, 36
        self.total_weak_labels = total_weak_labels  # 32, 8
        self.total_lengths = total_lengths          # 32, 8
        # self.if_subject_independent = if_subject_independent

        self.num_of_subject = total_signals.shape[0]
        self.num_of_trail = total_signals.shape[1]

    def getFold(self, subject_idx, log_save_file):
        train_signals, train_fine_labels, train_weak_labels, train_lengths = [], [], [], []
        test_signals, test_fine_labels, test_weak_labels, test_lengths = [], [], [], []

        for idx in range( self.num_of_subject ):
            if idx == subject_idx:
                test_signals.append( self.total_signals[idx] )
                test_fine_labels.append( self.total_fine_labels[idx] )
                test_weak_labels.append( self.total_weak_labels[idx] )
                test_lengths.append( self.total_lengths[idx] )
                
            else:
                train_signals.append( self.total_signals[idx] )
                train_fine_labels.append( self.total_fine_labels[idx] )
                train_weak_labels.append( self.total_weak_labels[idx] )
                train_lengths.append( self.total_lengths[idx] )

        train_signals = torch.from_numpy(np.stack(train_signals, axis=0)).type(torch.FloatTensor)
        train_fine_labels = torch.from_numpy(np.stack(train_fine_labels, axis=0)).type(torch.LongTensor)
        train_weak_labels = torch.from_numpy(np.stack(train_weak_labels, axis=0)).type(torch.LongTensor)
        train_lengths = torch.from_numpy(np.stack(train_lengths, axis=0)).type(torch.LongTensor)

        test_signals = torch.from_numpy(np.stack(test_signals, axis=0)).type(torch.FloatTensor)
        test_fine_labels = torch.from_numpy(np.stack(test_fine_labels, axis=0)).type(torch.LongTensor)
        test_weak_labels = torch.from_numpy(np.stack(test_weak_labels, axis=0)).type(torch.LongTensor)
        test_lengths = torch.from_numpy(np.stack(test_lengths, axis=0)).type(torch.LongTensor)

        print( 'train_signals:\t\t', train_signals.shape )
        print( 'train_fine_labels:\t',train_fine_labels.shape )
        print( 'train_weak_labels:\t',train_weak_labels.shape )
        print( 'train_lengths:\t',train_lengths.shape )
        print()
        print( 'test_signals:\t\t', test_signals.shape )
        print( 'test_fine_labels:\t',test_fine_labels.shape )
        print( 'test_weak_labels:\t',test_weak_labels.shape )
        print( 'test_lengths:\t',test_lengths.shape )
        print()
        
        # train_signals, train_fine_labels, train_weak_labels, train_lengths = Subject_Data_Prepare(train_signals, train_fine_labels, train_weak_labels, train_lengths)
        # test_signals, test_fine_labels, test_weak_labels, test_lengths = Subject_Data_Prepare(test_signals, test_fine_labels, test_weak_labels, test_lengths)
        
        print( 'train_signals:\t\t', train_signals.shape )
        print( 'train_fine_labels:\t',train_fine_labels.shape )
        print( 'train_weak_labels:\t',train_weak_labels.shape )
        print( 'train_lengths:\t',train_lengths.shape )
        print()
        print( 'test_signals:\t\t', test_signals.shape )
        print( 'test_fine_labels:\t',test_fine_labels.shape )
        print( 'test_weak_labels:\t',test_weak_labels.shape )
        print( 'test_lengths:\t',test_lengths.shape )
        print()
        
        
        train_fine_counts = [(train_fine_labels == i).sum() for i in range(3)]
        train_weak_counts = [(train_weak_labels == i).sum() for i in range(3)]
        print("current subject-{},  train fine 标签数量".format(subject_idx ), train_fine_counts)
        print("current subject-{},  train weak 标签数量".format(subject_idx), train_weak_counts)
        print()

        test_fine_counts = [(test_fine_labels == i).sum() for i in range(3)]
        test_weak_counts = [(test_weak_labels == i).sum() for i in range(3)]
        print("current subject-{},  test fine 标签数量".format(subject_idx), test_fine_counts)
        print("current subject-{},  test weak 标签数量".format(subject_idx), test_weak_counts)
        print()
        with open(log_save_file , 'a') as f:
            f.writelines( "train fine 标签数量:\t" + str(train_fine_counts) + '\n' )
            f.writelines( "train weak 标签数量:\t" + str(train_weak_counts) + '\n' )
            f.writelines( "test fine 标签数量:\t" + str(test_fine_counts) + '\n' )
            f.writelines( "test weak 标签数量:\t" + str(test_weak_counts) + '\n' )
            f.writelines( '\n' )
        
        train_dataset = torch.utils.data.TensorDataset(train_signals, train_fine_labels, train_weak_labels, train_lengths)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, drop_last=False )  # drop_last=True

        test_dataset = torch.utils.data.TensorDataset(test_signals, test_fine_labels, test_weak_labels, test_lengths)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

        return train_loader, test_loader 

class Trail_kFoldGenerator():
    def __init__(self, total_signals, total_fine_labels, total_weak_labels, total_lengths):
        self.total_signals = total_signals          # 32, 8, 36, 3, 100
        self.total_fine_labels = total_fine_labels  # 32, 8, 36
        self.total_weak_labels = total_weak_labels  # 32, 8
        self.total_lengths = total_lengths          # 32, 8

        self.num_of_subject = total_signals.shape[0]
        self.num_of_trail = total_signals.shape[1]

    def getFold(self, cur_batch_size, subject_idx, trail_idx, log_save_file):
        train_signals, train_fine_labels, train_weak_labels, train_lengths = [], [], [], []
        test_signals, test_fine_labels, test_weak_labels, test_lengths = [], [], [], []

        cur_sub_total_signals = self.total_signals[subject_idx]
        cur_sub_total_finelabels = self.total_fine_labels[subject_idx]
        cur_sub_total_weaklabels = self.total_weak_labels[subject_idx]
        cur_sub_total_lengths = self.total_lengths[subject_idx]

        for idx in range( self.num_of_trail ):
            if idx == trail_idx:
                test_signals.append( cur_sub_total_signals[idx] )
                test_fine_labels.append( cur_sub_total_finelabels[idx] )
                test_weak_labels.append( cur_sub_total_weaklabels[idx] )
                test_lengths.append( cur_sub_total_lengths[idx] )
                
            else:
                train_signals.append( cur_sub_total_signals[idx] )
                train_fine_labels.append( cur_sub_total_finelabels[idx] )
                train_weak_labels.append( cur_sub_total_weaklabels[idx] )
                train_lengths.append( cur_sub_total_lengths[idx] )

        train_signals = torch.from_numpy(np.stack(train_signals, axis=0)).type(torch.FloatTensor)
        train_fine_labels = torch.from_numpy(np.stack(train_fine_labels, axis=0)).type(torch.LongTensor)
        train_weak_labels = torch.from_numpy(np.stack(train_weak_labels, axis=0)).type(torch.LongTensor)
        train_lengths = torch.from_numpy(np.stack(train_lengths, axis=0)).type(torch.LongTensor)

        test_signals = torch.from_numpy(np.stack(test_signals, axis=0)).type(torch.FloatTensor)
        test_fine_labels = torch.from_numpy(np.stack(test_fine_labels, axis=0)).type(torch.LongTensor)
        test_weak_labels = torch.from_numpy(np.stack(test_weak_labels, axis=0)).type(torch.LongTensor)
        test_lengths = torch.from_numpy(np.stack(test_lengths, axis=0)).type(torch.LongTensor)

        print( 'train_signals:\t\t', train_signals.shape )
        print( 'train_fine_labels:\t',train_fine_labels.shape )
        print( 'train_weak_labels:\t',train_weak_labels.shape )
        print( 'train_lengths:\t',train_lengths.shape )
        print()
        print( 'test_signals:\t\t', test_signals.shape )
        print( 'test_fine_labels:\t',test_fine_labels.shape )
        print( 'test_weak_labels:\t',test_weak_labels.shape )
        print( 'test_lengths:\t',test_lengths.shape )
        print()

        train_fine_counts = [(train_fine_labels == i).sum() for i in range(3)]
        train_weak_counts = [(train_weak_labels == i).sum() for i in range(3)]
        print("current subject-{} trail-{},  train fine 标签数量".format(subject_idx, trail_idx), train_fine_counts)
        print("current subject-{} trail-{},  train weak 标签数量".format(subject_idx, trail_idx), train_weak_counts)
        print()

        test_fine_counts = [(test_fine_labels == i).sum() for i in range(3)]
        test_weak_counts = [(test_weak_labels == i).sum() for i in range(3)]
        print("current subject-{} trail-{},  test fine 标签数量".format(subject_idx, trail_idx), test_fine_counts)
        print("current subject-{} trail-{},  test weak 标签数量".format(subject_idx, trail_idx), test_weak_counts)
        print()
        with open(log_save_file , 'a') as f:
            f.writelines( "train fine 标签数量:\t" + str(train_fine_counts) + '\n' )
            f.writelines( "train weak 标签数量:\t" + str(train_weak_counts) + '\n' )
            f.writelines( "test fine 标签数量:\t" + str(test_fine_counts) + '\n' )
            f.writelines( "test weak 标签数量:\t" + str(test_weak_counts) + '\n' )
            f.writelines( '\n' )
        
        train_dataset = torch.utils.data.TensorDataset(train_signals, train_fine_labels, train_weak_labels, train_lengths)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cur_batch_size, shuffle = True )  # drop_last=True

        test_dataset = torch.utils.data.TensorDataset(test_signals, test_fine_labels, test_weak_labels, test_lengths)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = cur_batch_size, shuffle = False)

        return train_loader, test_loader 
       
def generate_weak_trail_data(data_path, task, num_of_subjects, max_length_trails):
    
    total_signals = []
    total_weak_labels = []
    total_fine_labels = []
    total_lengths = []

    for subject_idx in range( num_of_subjects ): # subjects
        subject_npz = np.load(data_path + 'subject_{}.npz'.format( subject_idx+1 ), allow_pickle=True)

        subject_wise_data = subject_npz['data']  
        subject_wise_fine_label = subject_npz['fine_label']  # valance arousal
        
        cur_subject_data = []
        cur_subject_fine_labels = []
        cur_subject_weak_labels = []
        cur_subject_length = []

        for idx_of_trail in range( len(subject_wise_data) ): # trails           
            cur_trail_signals =  np.transpose(subject_wise_data[idx_of_trail], axes=(0,2,1)) # B,100,3 -> B,3,100
            cur_trail_fine_labels = subject_wise_fine_label[idx_of_trail] # B,2
            
            # label related ------------------------------------------------------------------------------------------------
            if task == 'valence':
                digital_trail_fine_labels = cur_trail_fine_labels[:,0]
            else:
                digital_trail_fine_labels = cur_trail_fine_labels[:,1]

            digital_trail_weak_labels = np.mean(digital_trail_fine_labels)
            if (0.5 <= digital_trail_weak_labels) & (digital_trail_weak_labels <= 3.5):
                cur_trail_weak_label = 0
            elif (3.5 < digital_trail_weak_labels) & (digital_trail_weak_labels <= 6.5):
                cur_trail_weak_label = 1
            else: #(6.5 < digital_trail_weak_labels) & (digital_trail_weak_labels <= 9.5):
                cur_trail_weak_label = 2

            disc_trail_fine_labels = np.zeros_like(digital_trail_fine_labels, dtype=int)
            disc_trail_fine_labels[(0.5 <= digital_trail_fine_labels) & (digital_trail_fine_labels <= 3.5)] = 0
            disc_trail_fine_labels[(3.5 < digital_trail_fine_labels) & (digital_trail_fine_labels <= 6.5)] = 1
            disc_trail_fine_labels[(6.5 < digital_trail_fine_labels) & (digital_trail_fine_labels <= 9.5)] = 2

            cur_trail_signals = MaxMinNorm(cur_trail_signals)
            cur_subject_length.append( len(cur_trail_signals) )

            if len(cur_trail_signals) < max_length_trails:
                paddings_of_signal = np.zeros((int(max_length_trails - len(cur_trail_signals)), cur_trail_signals.shape[1], cur_trail_signals.shape[2] ))
                paddings_of_label  = np.zeros(max_length_trails-len(cur_trail_signals))

                cur_trail_signals = np.concatenate([cur_trail_signals, paddings_of_signal])
                disc_trail_fine_labels = np.concatenate([disc_trail_fine_labels, paddings_of_label])

                cur_subject_data.append(cur_trail_signals)
                cur_subject_fine_labels.append(disc_trail_fine_labels)
                
            else:
                cur_subject_data.append(cur_trail_signals)
                cur_subject_fine_labels.append(disc_trail_fine_labels)

            cur_trail_weak_label = np.repeat(cur_trail_weak_label, len(disc_trail_fine_labels))
            cur_subject_weak_labels.append(cur_trail_weak_label)
            
        cur_subject_data = np.stack(cur_subject_data)
        cur_subject_fine_labels = np.stack(cur_subject_fine_labels)
        cur_subject_weak_labels = np.stack(cur_subject_weak_labels)
        cur_subject_length = np.stack(cur_subject_length)

        total_signals.append(cur_subject_data)
        total_fine_labels.append(cur_subject_fine_labels)
        total_weak_labels.append(cur_subject_weak_labels)
        total_lengths.append(cur_subject_length)

    total_signals = np.stack(total_signals)         # 32, 8, 30, 3, 100
    total_fine_labels = np.stack(total_fine_labels) # 32, 8, 30
    total_weak_labels = np.stack(total_weak_labels) # 32, 8, 30
    total_lengths = np.stack(total_lengths)         # 32, 8

    return total_signals, total_fine_labels, total_weak_labels, total_lengths 

def data_generator(data_path, task, num_of_subjects, max_length_trails):
   
    total_signals, total_fine_labels, total_weak_labels, total_lengths = generate_weak_trail_data(data_path, task, num_of_subjects, max_length_trails)

    return total_signals, total_fine_labels, total_weak_labels, total_lengths
