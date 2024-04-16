import os
import numpy as np
import time
import argparse

os.environ['MKL_THREADING_LAYER']='GNU'

folder_path = '/data/JinXiyuan/pyspace/data/'
cuda = 0

for dataset in {'CASE_wtonorm','Ceap360_wtonorm'}:          #   CASE_wtonorm   Ceap360_wtonorm
    for task in ['valence','arousal']:                      #   valence  arousal
        for seed in [1]: # 1,2,3,4,5
            data_path = folder_path + dataset + '/'     
            com_code = f'python subject_main.py  --data_path {data_path} --cuda {cuda} --seed {seed} --task {task}'
            start_time = time.asctime(time.localtime(time.time()))
            os.system(com_code)
            print(com_code)
            print('\nstart  at', start_time)
            print('finish at', time.asctime(time.localtime(time.time())))