import numpy as np

def check_maindict(dict):
    # 20250809
    tau_long = dict['tau_long']
    tau_traj_len_max = 64*tau_long # 64 steps times tau_long
    tau_traj_len=dict['tau_traj_len']
    saved_pair_steps = dict['saved_pair_steps']
    label_idx_max = 180
    traj_len_index = round(tau_traj_len / tau_long) * saved_pair_steps  # 20250809
    label_idx = int((traj_len_index - saved_pair_steps) + dict['window_sliding'] * saved_pair_steps)

    assert (tau_traj_len <= tau_traj_len_max), 'trajectory len must be <=64'
    assert (tau_traj_len >= tau_long), 'trajectory len must be <= tau_long'
    #print([i*tau_long for i in range(label_idx_max+1) if i*tau_long % tau_long == 0])
    assert (tau_traj_len in [i*tau_long for i in range(label_idx_max) if i*tau_long % tau_long == 0]), 'trajectory len must be multiplied by tau_long'
    assert (tau_traj_len == (traj_len_index // saved_pair_steps) * tau_long ), 'label index multiplied by tau long must be trajectory len'
    assert (isinstance(label_idx, int)), 'label idx needs to be integer'
    #assert (isinstance(tau_long, int)), 'tau_long needs to be integer'
    assert (tau_long in [0.05, 0.02, 0.01, 0.1, 0.2, 0.4, 0.5, 1, 2, 4, 8]), 'incorrect tau long' # 20250813 add 0.05
    assert (tau_traj_len % tau_long == 0), 'incompatible traj_len and tau_long'

def check_datadict(dict):

    batch_size=dict['batch_size']
    assert (dict['train_pts'] >= batch_size), 'ERROR: batch_size request more than data points'
    assert (dict['vald_pts'] >= batch_size), 'ERROR: batch_size request more than data points'

def check_traindict(dict,tau_long):

    nitr=dict['nitr']
    tau_short=dict['tau_short']
    append_strike=dict['append_strike']

    assert (nitr % append_strike == 0), 'incompatible strike and nitr'
    assert (tau_long == append_strike * tau_short), 'tau long must be same as append strike multiplied by tau short'

def check_testdict(maindict):

    tau_traj_len = maindict["tau_traj_len"]

    # traj_len_list = maindict["traj_len"]
    # tau_long_list = maindict["tau_long"]

    # end_traj_len_list = maindict["end_traj_len_list"]
    #tau_traj_len_list = [i*j for i,j in zip(traj_len_list,tau_long_list)]
    tau_max = maindict["tau_max"]

    prep_traj_len_list = [int(item)-1 for item in traj_len_list]
    pred_traj_len_list = [int(i-j) for i,j in zip(end_traj_len_list,prep_traj_len_list)]
    tau_pred_traj_len_list = [i*j for i,j in zip(tau_long_list,pred_traj_len_list)]
    tau_pred_traj_last = tau_max - sum(tau_pred_traj_len_list[:-1])
    print('tau long          ', tau_long_list)
    print('traj len          ', traj_len_list)
    print('prep traj len     ',prep_traj_len_list)
    print('pred traj len     ', pred_traj_len_list)
    print('tau pred traj len ', tau_pred_traj_len_list)
    print('tau pred traj last', tau_pred_traj_last, tau_pred_traj_len_list[-1])
    print('end traj len      ', end_traj_len_list)

    # assert ( S_ML == S_MD - S_prep ), 'ML steps not match (MD steps - prepared step)'
    assert (len(tau_long_list) == len(traj_len_list) == len(end_traj_len_list)), 'list len not match'
    assert all(isinstance(traj_len, int) for traj_len in traj_len_list), 'traj_len needs to be integer'
    assert (tau_pred_traj_last == tau_pred_traj_len_list[-1]), 'the last tau pred traj len not match the last tau pred in list'
    assert (end_traj_len_list[:-1] == [2*(int(item)-1) for item in traj_len_list[1:]]), 'tau_traj_len not match traj_len * tau_long'
    assert ([int(item) for item in end_traj_len_list]==[int(i+j) for i,j in zip(prep_traj_len_list,pred_traj_len_list)] ), 'end traj len not match prep_len + pred len'
