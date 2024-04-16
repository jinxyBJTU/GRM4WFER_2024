import argparse


def get_args():
    parser = argparse.ArgumentParser('Train_MoCo')

    # General args
    # data loading
    parser.add_argument('--data_path',      type=str, default="/data2/JinXiyuan/pyspace/data/")
    # training setting
    parser.add_argument("--seed",           type=int,   default= 1)
    parser.add_argument('--cuda',           type=str,   default='0')  
    # task related
    parser.add_argument("--n_classes",      type=int, default= 3)
    parser.add_argument('--task',           type=str, default='valence')
    # modify related
    # parser.add_argument('--output_type',    type=str, default='Linear')

    args = parser.parse_args()

    return args
