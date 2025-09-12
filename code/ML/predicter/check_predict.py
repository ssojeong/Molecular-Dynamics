import sys
sys.path.append('../../')

import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def check_predcit(pred_list, gt_path):
    qpl_list = []
    for idx, f in enumerate(pred_list):
        print(f'===== dealing with {idx} file: {f}')
        data = torch.load(f)
        qpl = data['qpl_trajectory']
        print('qpl shape', qpl.shape)
        qpl_list.append(qpl)
    pred_pql = torch.cat(qpl)
    print(pred_pql.shape)


if __name__ == '__main__':
    file_list = [f'../../../../SavedModel/LLUF/0.02ws8_test_id{i}.pt' for i in range(32)]
    gt_path = '../../../../Data/LLUF/300k_100ktraj_gap10_valid.pt'
    check_predcit(file_list, gt_path)
