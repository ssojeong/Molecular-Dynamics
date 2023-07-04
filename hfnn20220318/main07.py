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


def print_dict(name,thisdict):
    print(name,'dict ============== ')
    for key,value in thisdict.items(): print(key,':',value)

if __name__=='__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    #print('checking pbc method ... wait a minute ')
    #check_pbc()
    torch.manual_seed(23841)

    traindict = {"loadfile"  : None,  # to load previously trained model
                 "nn_mode"   : 'ff',  # 'hf' or 'ff' predict hamiltonian or predict force
                 "nnodes"    : 256,   # number of nodes in neural nets
                 "force_clip": 50,    # clamp the force within a range
                 "grad_clip" : 10,    # clamp the gradient for neural net parameters
                 "tau_long"  : 1.0,   # the large time step
                 "n_chain"   : 1,     # number of times to do integration before cal the loss
                 "ngrids"    : 6,     # for multibody interactions
                 "b"         : 0.2,   # grid lattice constant for multibody interactions
                 "lr"        : 2e-5,  # starting learning rate
                 "alpha_lr"  : 1e-3,  # lr for alpha for mbpw
                 "sch_step"  : 10,    # defunct: scheduler step
                 "sch_decay" : .99,   # 200 steps and reduce lr to 10%
                 "reset_lr"  : False} 

    n_chain_list = [1,4,8,16,32,64]

    if traindict["n_chain"] not in n_chain_list:
        print('n_chain is not valid, need ',n_chain_list)
        quit()

    lossdict = { "eweight" : 1,
                 "polynomial_degree" : 6 }

    data = { "train_file": '../data_sets/n16anyrholt1everystps_nsamples100000train.pt',
             "valid_file": '../data_sets/n16anyrholt1everystps_nsamples100000valid.pt',
             "test_file" : '../data_sets/n16anyrholt1everystps_nsamples100000valid.pt',
             "train_pts" : 100000,
             "vald_pts"  : 1000,
             "test_pts"  : 100,
             "batch_size": 200,
             "n_chain"   : traindict["n_chain"] }

    maindict = { "start_epoch"  : 0,
                 "end_epoch"    : 10000,
                 "save_dir"     : 'ffmbpwtau1.0lr_cos_nchain01npts0100000net256poly06pw020mb',
                 "ckpt_interval": 20,    # for check pointing
                 "val_interval" : 1000,  # no use of valid for now
                 "verb"         : 1 }    # peroid for printing out losses

    print_dict('trainer',traindict)
    print_dict('loss',lossdict)
    print_dict('data',data)
    print_dict('main',maindict)

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],data["n_chain"],
                       data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set,data["batch_size"])

    train = trainer(traindict,lossdict)

    train.load_models()
    if traindict["reset_lr"] is True: train.reset_opt_lr(traindict["lr"])

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        cntr = 0
        for qpl_input,qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
    
            q_init,p_init,q_label,p_label,l_init = pack_data(qpl_input,qpl_label)

            train.one_step(q_init,p_init,q_label,p_label,l_init)
            cntr += 1
            if cntr%10==0: print('.',end='',flush=True)
        print(cntr,'batches \n')

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

        if traindict["sch_step"] > 0:
            train.scheduler_step() # -- no change in learning rate for now


system_logs.print_end_logs()

