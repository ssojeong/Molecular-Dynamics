from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from HNN.data_loader                 import data_loader
from HNN.data_loader                 import my_data
from phase_space                     import phase_space
from utils.make_hamiltonian          import make_hamiltonian
from integrator.linear_integrator    import linear_integrator

import sys
import torch

if __name__=='__main__':

    argv = sys.argv
    MDjson_file = argv[1]
    MLjson_file = argv[2]

    MD_parameters.load_dict(MDjson_file)
    ML_parameters.load_dict(MLjson_file)

    # io varaiables
    train_filename = ML_parameters.train_filename
    val_filename   = train_filename # read the same data
    test_filename  = train_filename
    train_pts      = ML_parameters.train_pts
    val_pts        = train_pts
    test_pts       = train_pts

    phase_space = phase_space.phase_space()
    hamiltonian_obj = make_hamiltonian(MD_parameters.hamiltonian_type, MD_parameters.tau_long, ML_parameters)
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )

    dataset = my_data(train_filename,val_filename,test_filename,train_pts,val_pts,test_pts)
    loader  = data_loader(dataset, ML_parameters.batch_size)

    seed = ML_parameters.seed
    torch.manual_seed(seed) # cpu
    # np.random.seed(seed)  # cpu
    # random.seed(seed)  # Python

    # # cuda
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # needed
    # torch.backends.cudnn.benchmark = False

    # assert (abs(tau_long - dataset.tau_long)<1e-6), 'incorrect tau long value'

    tau_cur = MD_parameters.tau_long
    boxsize =dataset.train_set.data_boxsize
    nepoch = 20000

    # create parameters from two models in one optimizer
    optimizer = ML_parameters.opt.create(hamiltonian_obj.net_parameters())

    for e in range(nepoch):

        for step,(input, label) in enumerate(loader.train_loader):

            optimizer.zero_grad()
            # input shape, [nsamples, (q,p), nparticle, DIM]
            phase_space.set_q(input[:,0,:,:])
            phase_space.set_p(input[:,1,:,:])
            phase_space.set_boxsize(boxsize)
            qp_list = linear_integrator_obj.one_step(hamiltonian_obj, phase_space, tau_cur)

            # qp_list.shape = [nsamples,(q,p),nparticle,DIM]
            q_predict = qp_list[:,0,:,:]
            p_predict = qp_list[:,1,:,:]
            q_label   = label[:,0,:,:]
            p_label   = label[:,1,:,:]

            loss = torch.mean((q_predict - q_label)**2 + (p_predict - p_label)**2)
            loss.backward()
            optimizer.step()

        if e%(nepoch//10)==0: print('e ',e,' loss ',loss)

