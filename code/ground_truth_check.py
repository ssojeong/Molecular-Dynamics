import torch
import time
import sys
import yaml

from einops import rearrange
import matplotlib.pyplot as plt

from utils.mydevice import mydevice
from utils.pbc import pbc
from ML.trainer.loss import loss


def main():
    tau_short = 0.002
    tau_long = 0.02
    ratio = tau_long / tau_short
    gap = int(round(ratio))
    assert abs(ratio - gap) < 1e-9, f"target_dt/origin_dt must be integer; got {ratio}"

    mass = torch.tensor([8, 1, 1] * 8)
    filename = '../../Data/LLUF/300k_new8k.pt'
    file = torch.load(filename)
    # print(file.keys())
    # print(file['atom_id'])
    # print(file['qp'].dtype)
    # quit()
    qp = rearrange(file['qp'][:, ::gap, :, :, :], 'traj tpts atom dim qp -> traj qp tpts atom dim')
    q = qp[:, 0, :, :, :]
    p = qp[:, 1, :, :, :]
    print(torch.tensor([8, 1, 1] * 8).shape, q.shape)
    mass_center = torch.sum(q * mass[:, None], dim=-2) / torch.sum(mass)
    # print(torch.sum((mass_center[:, 0, :] - mass_center[:, -1, :])**2, dim=-1))
    mass_dis = torch.sum((mass_center[:, 0, :] - mass_center[:, -1, :])**2, dim=-1)
    print(torch.sum(mass_dis < 0.1) / len(mass_dis))
    mass_dis_np = mass_dis.cpu().numpy()

    # Plot histogram
    # plt.figure(figsize=(6, 4))
    # plt.hist(mass_dis_np, bins=50, color='skyblue', edgecolor='black')
    # plt.xlabel('Squared Distance (between t=0 and t=end)')
    # plt.ylabel('Count')
    # plt.title('Histogram of mass displacement')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # quit()
    l_init = 2.2 * torch.ones(qp.size(0), qp.size(3), qp.size(4), dtype=qp.dtype, device=qp.device)
    times = file['times'][::gap]
    traj = file['traj_id']
    print('qp shape', qp.shape, 'traj shape', traj.shape, 'l shape', l_init.shape)
    # print('time', times)
    # print(times[69:72])

    loss_obj = loss(1, 0.7, 0, 0, 8)
    # time_gap = 0
    # del_vec_gt = loss_obj.q_RMSE_loss(q[:, 7, :], q[:, 7+time_gap, :], l_init)
    # print(del_vec_gt.abs().max().item())
    time_gap = 1
    del_q_gt = loss_obj.q_RMSE_loss(q[:, 7, :], q[:, 7+time_gap, :], l_init)
    max_val, max_idx = torch.max(del_q_gt.abs(), dim=0)
    print("Max value:", max_val.item())
    print("Flat index:", max_idx.item(), "Traj id", traj[max_idx])
    print("Mean value of del q:", del_q_gt.abs().mean())

    del_p_gt = loss_obj.p_RMSE_loss(p[:, 7, :], p[:, 7+time_gap, :])
    print("Mean value of del p:", del_p_gt.abs().mean())

    # max_idx = 2176
    # print("Flat index:", max_idx, "Traj id", traj[max_idx])
    # for i in range(10):
    #     print(f'~~~~ delta q {times[70+i]} to {times[70+i+time_gap]} ~~~~')
    #     # print(qp[max_idx, 0, 70, :, :])
    #     # print(qp[max_idx, 0, 70+time_gap, :, :])
    #     print(torch.norm(qp[max_idx, 0, 70+i, :, :] - qp[max_idx, 0, 70+time_gap+i, :, :], dim=-1))
    #     # print(torch.norm(qp[max_idx-1, 0, 70, :, :] - qp[max_idx-1, 0, 70 + time_gap, :, :], dim=-1))
    #     print(f'#### p {times[70+i]} ####')
    #     # print(qp[max_idx, 1, 70+i, :, :])
    #     # print(qp[max_idx, 1, 70+time_gap+i, :, :])
    #     print(torch.norm(qp[max_idx, 1, 70+i, :, :], dim=-1))
    #
    # print(f'~~~~ delta q {times[70]} to {times[70+10]} ~~~~')
    # print(torch.norm(qp[max_idx, 0, 70, :, :] - qp[max_idx, 0, 70 + 10, :, :], dim=-1))
    #
    # print(q[max_idx, 0, :, :])
    # split = int(qpl_trajectory.size(0) * 0.9)
    # data = {'qpl_trajectory': qpl_trajectory[:split].clone(),
    #         'times': times[:split].clone(),
    #         'traj_id': file['traj_id'][:split].clone(),
    #         'atom_id': file['atom_id'][:split],
    #         'tau_short': tau_short,
    #         'tau_long': tau_long}
    # # torch.save(data, f'../../../Data/LLUF/300k_100ktraj_gap{gap}_1_train.pt')
    #
    # data = {'qpl_trajectory': qpl_trajectory[split:].clone(),
    #         'times': times[split:].clone(),
    #         'traj_id': file['traj_id'][split:].clone(),
    #         'atom_id': file['atom_id'][split:],
    #         'tau_short': tau_short,
    #         'tau_long': tau_long}
    # # torch.save(data, f'../../../Data/LLUF/300k_100ktraj_gap{gap}_1_valid.pt')
    #
    # f_list = [f'../../../Data/LLUF/300k_100ktraj_gap10_train.pt',
    #           f'../../../Data/LLUF/300k_100ktraj_gap10_1_train.pt']
    # merge_pt(f_list, f'../../../Data/LLUF/300k_150ktraj_gap10_train.pt')
    #
    # f_list = [f'../../../Data/LLUF/300k_100ktraj_gap10_valid.pt',
    #           f'../../../Data/LLUF/300k_100ktraj_gap10_1_valid.pt']
    # merge_pt(f_list, f'../../../Data/LLUF/300k_150ktraj_gap10_valid.pt')


if __name__ == '__main__':
    _ = mydevice()
    main()

