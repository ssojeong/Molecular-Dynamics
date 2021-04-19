from hamiltonian.noML_hamiltonian   import noML_hamiltonian
from HNN.pairwise_HNN               import pairwise_HNN
from HNN.fields_HNN                 import fields_HNN
from HNN.models.pairwise_MLP        import pairwise_MLP
from HNN.models.fields_cnn          import fields_cnn
from utils.interpolator             import interpolator

def make_hamiltonian(hamiltonian_type, integrator, tau_short, tau_long, ML_param):

    if hamiltonian_type  == 'noML':
        hamiltonian_obj = noML_hamiltonian()

    elif hamiltonian_type == 'pairwise_HNN':
        net1 = pairwise_MLP(ML_param.MLP_input, ML_param.MLP_nhidden)
        net2 = pairwise_MLP(ML_param.MLP_input, ML_param.MLP_nhidden)
        hamiltonian_obj = pairwise_HNN(net1, net2)
        hamiltonian_obj.set_tau(tau_long)

    elif hamiltonian_type == 'fields_HNN':
        net = fields_cnn(ML_param.gridL) # SJ
        interpolator_obj = interpolator() # SJ
        hamiltonian_obj = fields_HNN(net,integrator, interpolator_obj) # HK, here can make noML_hamiltonian() obj?
        hamiltonian_obj.set_tau(tau_short)

    else:
        assert (False), 'invalid hamiltonian type given'


    return hamiltonian_obj
