import torch

from torchviz import make_dot

from ML.trainer.trainer          import trainer
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from data_loader.data_loader     import data_loader
from data_loader.data_loader     import my_data

def print_compute_tree(name,node):
    dot = make_dot(node)
    #print(dot)
    dot.render(name)

def pack_data(qpl_input, qpl_label):

    q_init = qpl_input[:,0,:,:].clone().detach().requires_grad_()
    p_init = qpl_input[:,1,:,:].clone().detach().requires_grad_()
    l_init = qpl_input[:,2,:,:].clone().detach().requires_grad_()
    q_label = qpl_label[:,0,:,:].clone().detach().requires_grad_()
    p_label = qpl_label[:,1,:,:].clone().detach().requires_grad_()

    q_init = mydevice.load(q_init)
    p_init = mydevice.load(p_init)
    l_init = mydevice.load(l_init)
    q_label = mydevice.load(q_label)
    p_label = mydevice.load(p_label)

    return q_init,p_init,q_label,p_label,l_init



if __name__=='__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    #print('checking pbc method ... wait a minute ')
    #check_pbc()
    torch.manual_seed(23841)

    traindict = {"loadfile"  : None, # to load previously trained model
                 "nn_mode"   : 'hf', # 'hf' or 'ff' predict hamiltonian or predict force
                 "force_clip": 5,    # clamp the force within a range
                 "grad_clip" : 10,   # clamp the gradient for neural net parameters
                 "tau_long"  : 0.1,  # the large time step
                 "n_chain"   : 1,    # number of times to do integration before cal the loss
                 "ngrids"    : 6,    # for multibody interactions
                 "b"         : 0.2,  # grid lattice constant for multibody interactions
                 "lr"        : 1e-2, # starting learning rate
                 "alpha_lr"  : 1e-4, # lr for alpha for mbpw
                 "sch_step"  : 10,   # scheduler step
                 "sch_decay" : 0.98} # scheduler decay

    data = { "train_file": '../data_sets/n16anyrholt0.1everystps_nsamples1200000.pt',
             "valid_file": '../data_sets/n16anyrholt0.1stps_10000pts.pt',
             "test_file" : '../data_sets/n16anyrholt0.1stps_100pts.pt',
             "train_pts" : 1200,
             "vald_pts"  : 200,
             "test_pts"  : 100,
             "batch_size": 100,
             "n_chain"   : traindict["n_chain"] }

    maindict = { "start_epoch"  : 0,
                 "end_epoch"    : 100000,
                 "save_dir"     : 'saved_ff_models',
                 "ckpt_interval": 10,    # for check pointing
                 "val_interval" : 10,
                 "verb"         : 10 }    # peroid for printing out losses

    print('trainer dict ============== ')
    for key,value in traindict.items(): print(key,':',value)
    print('data dict  ==============')
    for key,value in data.items(): print(key,':',value)
    print('main dict  ==============')
    for key,value in maindict.items(): print(key,':',value)

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],data["n_chain"],
                       data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set,data["batch_size"])

    train = trainer(traindict)

    train.load_models()

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        for qpl_input,qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
    
            q_init,p_init,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)

            train.one_step(q_init,p_init,q_label,p_label,l_init)

        if e%maindict["verb"]==0: 
            train.verbose(e+1,'train')
            system_logs.record_memory_usage(e+1)
            system_logs.record_time_usage(e+1)

        if e%maindict["ckpt_interval"]==0: 
            filename = './{}/mbpw{:06d}.pth'.format(maindict["save_dir"],e+1)
            train.checkpoint(filename)

        if e%maindict["val_interval"]==0: 
            train.loss_obj.clear()
            for qpl_input,qpl_label in loader.val_loader:
                q_init,p_init,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)
                train.eval(e+1,q_init,p_init,q_label,p_label,l_init)
            train.verbose(e+1,'eval')

        train.scheduler_step()


system_logs.print_end_logs()


