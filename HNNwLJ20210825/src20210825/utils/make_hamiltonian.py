from hamiltonian.noML_hamiltonian   import noML_hamiltonian
from HNN.pairwise_HNN               import pairwise_HNN
from HNN.fields_HNN                 import fields_HNN
from HNN.models.mlp_net             import mlp_net

def make_hamiltonian(hamiltonian_type, tau_long, ML_param):

    if hamiltonian_type  == 'noML':
        hamiltonian_obj = noML_hamiltonian()

    elif hamiltonian_type == 'pairwise_HNN':
        net1 = mlp_net(ML_param.layer_list,ML_param.dropout_list)
        net2 = mlp_net(ML_param.layer_list,ML_param.dropout_list)
        hamiltonian_obj = pairwise_HNN(net1, net2, ML_param.on_off_noML)
        hamiltonian_obj.set_tau_long(tau_long)

    elif hamiltonian_type == 'fields_HNN':
        net1 = mlp_net(ML_param.layer_list,ML_param.dropout_list)
        net2 = mlp_net(ML_param.layer_list,ML_param.dropout_list)
        hamiltonian_obj = fields_HNN(net1, net2, ML_param.dgrid, ML_param.ngrid, ML_param.on_off_noML)
        hamiltonian_obj.set_tau_long(tau_long)

    else:
        assert (False), 'invalid hamiltonian type given'


    return hamiltonian_obj
