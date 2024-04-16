import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm

subject_video_ids = {
    '1':[3,1,5,7, 4,2,6,8],
}

video2emotion = {
    '1':'amusing',
    '2':'amusing',

    '3':'boring',
    '4':'boring',

    '5':'relaxed',
    '6':'relaxed',

    '7':'scary',
    '8':'scary',
}

def weak_trail_data(data_path, start_idx, end_idx, task):
    total_subject_trail_wise_labels = []
    for subject_idx in range(start_idx, end_idx+1):
        subject_npz = np.load(data_path + 'subject_{}.npz'.format(subject_idx), allow_pickle=True)
        subject_wise_data = subject_npz['data']  
        subject_wise_label = subject_npz['label'] # list 8 trails, 
        trail_wise_labels = {
            'amusing':[],
            'boring':[],
            'relaxed':[],
            'scary':[],
        }
        for idx_of_trail in range(len(subject_wise_data)):
            trail_fine_2labels = subject_wise_label[idx_of_trail] # B,2
            
            digital_trail_fine_valence = trail_fine_2labels[:,0]
            digital_trail_fine_arousal = trail_fine_2labels[:,1]

            trail_mean_valence = np.mean(digital_trail_fine_valence)
            trail_mean_arousal = np.mean(digital_trail_fine_arousal)

            if (0.5 <= trail_mean_valence) & (trail_mean_valence <= 3.5):
                trail_weak_valence = 0
            elif (3.5 < trail_mean_valence) & (trail_mean_valence <= 6.5):
                trail_weak_valence = 1
            elif (6.5 < trail_mean_valence) & (trail_mean_valence <= 9.5):
                trail_weak_valence = 2
            
            if (0.5 <= trail_mean_arousal) & (trail_mean_arousal <= 3.5):
                trail_weak_arousal = 0
            elif (3.5 < trail_mean_arousal) & (trail_mean_arousal <= 6.5):
                trail_weak_arousal = 1
            elif (6.5 < trail_mean_arousal) & (trail_mean_arousal <= 9.5):
                trail_weak_arousal = 2

            disc_fine_grain_valence = np.zeros_like(digital_trail_fine_valence, dtype=int)
            disc_fine_grain_valence[(0.5 <= digital_trail_fine_valence) & (digital_trail_fine_valence <= 3.5)] = 0
            disc_fine_grain_valence[(3.5 < digital_trail_fine_valence) & (digital_trail_fine_valence <= 6.5)] = 1
            disc_fine_grain_valence[(6.5 < digital_trail_fine_valence) & (digital_trail_fine_valence <= 9.5)] = 2

            disc_fine_grain_arousal = np.zeros_like(digital_trail_fine_arousal, dtype=int)
            disc_fine_grain_arousal[(0.5 <= digital_trail_fine_arousal) & (digital_trail_fine_arousal <= 3.5)] = 0
            disc_fine_grain_arousal[(3.5 < digital_trail_fine_arousal) & (digital_trail_fine_arousal <= 6.5)] = 1
            disc_fine_grain_arousal[(6.5 < digital_trail_fine_arousal) & (digital_trail_fine_arousal <= 9.5)] = 2

            print('weak',trail_weak_valence, trail_weak_arousal)
            cur_label_tuples = []
            for fine_idx in range(len(trail_fine_2labels)):
                cur_label_tuples.append( (disc_fine_grain_valence[fine_idx],disc_fine_grain_arousal[fine_idx]) )
            exit()

            
    # fold_signal = np.concatenate(fold_signal, axis=0)
    # fold_weak_label = np.concatenate(fold_weak_label, axis=0)
    # fold_fined_label = np.concatenate(fold_fined_label, axis=0)
    # fold_trail_length = np.concatenate(fold_trail_length, axis=0)

    # return fold_signal, fold_weak_label, fold_fined_label, fold_trail_length

if __name__ == '__main__':

    folder_path = '/data2/JinXiyuan/pyspace/data/'
    dataset = 'CASE_MIL_processed'

    data_path = folder_path + dataset + '/' 

    start_idx = 1
    end_idx = 1
    task = 'arousal' # valence arousal

    weak_trail_data(data_path, start_idx, end_idx, task)