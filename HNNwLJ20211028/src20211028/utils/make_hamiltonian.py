from hamiltonian.noML_hamiltonian   import noML_hamiltonian
from HNN.pairwise_HNN               import pairwise_HNN
from HNN.fields_HNN                 import fields_HNN
from HNN.models.mlp_net             import mlp_net
from HNN.fields_pairwise_HNN        import fields_pairwise_HNN

def make_hamiltonian(hamiltonian_type, tau_long, ML_param):

    pwnet1 = mlp_net(ML_param.pw_layer_list, ML_param.dropout_list)
    pwnet2 = mlp_net(ML_param.pw_layer_list, ML_param.dropout_list)
    pwhamiltonian_obj = pairwise_HNN(pwnet1, pwnet2, ML_param.on_off_noML)
    pwhamiltonian_obj.set_tau_long(tau_long)
    fnet1 = mlp_net(ML_param.f_layer_list, ML_param.dropout_list)
    fnet2 = mlp_net(ML_param.f_layer_list, ML_param.dropout_list)
    fhamiltonian_obj = fields_HNN(fnet1, fnet2, ML_param.dgrid, ML_param.ngrid, ML_param.on_off_noML)
    fhamiltonian_obj.set_tau_long(tau_long)

    if hamiltonian_type == 'noML':
        return noML_hamiltonian()
    elif hamiltonian_type == 'pairwise_HNN':
        return pwhamiltonian_obj
    elif hamiltonian_type == 'fields_HNN':
        return fhamiltonian_obj
    elif hamiltonian_type == 'fields_pairwise_HNN':
        fpwhamiltonian_obj = fields_pairwise_HNN(fhamiltonian_obj, pwhamiltonian_obj)
        return fpwhamiltonian_obj
    else:
        assert (False), 'invalid hamiltonian type given'

    # return hamiltonian_obj